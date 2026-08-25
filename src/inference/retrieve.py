import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Dense Sub-Graph Retrieval")
    parser.add_argument("--corpus", type=str, default="data/processed/universal_corpus.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mhsa_best.pt")
    parser.add_argument("--trigger_index", type=int, default=150)
    parser.add_argument("--top_k", type=int, default=25, help="Number of logs to retrieve")
    parser.add_argument("--output_context", type=str, default="data/results/retrieved_context.txt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=10000, help="Number of logs to encode around the trigger")
    return parser.parse_args()

# ==========================================
# Model Definitions
# ==========================================
class Time2Vec(nn.Module):
    def __init__(self, output_dim=768):
        super(Time2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, output_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(1, output_dim - 1))
        
    def forward(self, t):
        v1 = self.w0 * t + self.b0
        v2 = torch.sin(self.w * t + self.b)
        return torch.cat([v1, v2], dim=-1)

class CrossModalLogEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", hidden_dim=768):
        super(CrossModalLogEncoder, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.time2vec = Time2Vec(output_dim=hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, input_ids, attention_mask, timestamps):
        h_i = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        t_emb = self.time2vec(timestamps)
        v_i = h_i + t_emb
        q = self.W_Q(v_i)
        k = self.W_K(v_i)
        return v_i, q, k

# ==========================================
# Main Retrieval Logic
# ==========================================
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_context), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    # 1. Load Data
    print("📂 Loading full corpus into memory...")
    full_df = pd.read_csv(args.corpus)
    
    print("🔍 Searching for a real exception/error to use as the trigger...")
    # Search for critical keywords indicating a real fault
    error_candidates = full_df[full_df['message'].str.contains('Exception|Error|Fail|Traceback|Timeout|Panic', case=False, na=False)]
    
    if not error_candidates.empty:
        global_trigger_idx = error_candidates.index[0]
        print(f"🚨 Auto-detected actual ERROR log at global index {global_trigger_idx}.")
    else:
        global_trigger_idx = args.trigger_index
        print(f"⚠️ No explicit errors found in entire corpus. Falling back to index {global_trigger_idx}.")

    # Slice a temporal window around the trigger
    start_idx = max(0, global_trigger_idx - (args.window_size // 2))
    end_idx = min(len(full_df), global_trigger_idx + (args.window_size // 2))
    
    df = full_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    local_trigger_idx = global_trigger_idx - start_idx
    
    trigger_row = df.iloc[local_trigger_idx]
    print(f"🎯 Trigger Anchor: [{trigger_row['service']}] {trigger_row['message']}")
    
    # Time Normalization MUST use the global corpus bounds to match training!
    min_time, max_time = full_df['observed_timestamp'].min(), full_df['observed_timestamp'].max()
    time_range = max_time - min_time if max_time > min_time else 1.0
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # 2. Load Model
    print(f"🧠 Loading trained checkpoint from {args.checkpoint}...")
    model = CrossModalLogEncoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 3. Stage 1: Vectorized Dense Encoding (Batch Processing)
    print(f"🔄 Encoding {len(df)} logs in the local temporal window...")
    all_k = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), args.batch_size), desc="Encoding Batches"):
            batch_df = df.iloc[i:i+args.batch_size]
            
            encoded = tokenizer(batch_df['message'].tolist(), padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
            t_norm = ((batch_df['observed_timestamp'].values - min_time) / time_range).astype(np.float32)
            t_tensor = torch.tensor(t_norm).unsqueeze(1).to(device)
            
            _, _, batch_k = model(encoded['input_ids'], encoded['attention_mask'], t_tensor)
            all_k.append(batch_k.cpu())
            
    K_matrix = torch.cat(all_k, dim=0).to(device)
    
    # 4. Stage 2: Graph Attention Threading
    encoded_trig = tokenizer([trigger_row['message']], padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
    t_trig = torch.tensor([[(trigger_row['observed_timestamp'] - min_time) / time_range]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _, q_trig, _ = model(encoded_trig['input_ids'], encoded_trig['attention_mask'], t_trig)
    
    d_k = q_trig.size(-1)
    attention_scores = torch.softmax((q_trig @ K_matrix.T) / np.sqrt(d_k), dim=-1).squeeze(0)

    # 5. Stage 3: Sub-Graph Extraction
    print(f"🕸️ Retrieving Top-{args.top_k} semantically connected threads...")
    top_k_scores, top_k_indices = torch.topk(attention_scores, args.top_k)
    
    retrieved_df = df.iloc[top_k_indices.cpu().numpy()].copy()
    retrieved_df = retrieved_df.sort_values(by="observed_timestamp")
    
    # 6. Save Extracted Context
    print(f"💾 Saving extracted sub-graph context to {args.output_context}")
    with open(args.output_context, "w") as f:
        f.write("--- RETRIEVED LOG CONTEXT FOR GENERATIVE LLM ---\n")
        f.write(f"TRIGGER LOG: [{trigger_row['service']}] {trigger_row['message']}\n")
        f.write("-" * 50 + "\n")
        for _, row in retrieved_df.iterrows():
            f.write(f"[{row['service']}] {row['message']}\n")
            
    print("✅ Inference Complete! The unwoven context is ready for the RAG Generative Diagnosis.")

if __name__ == "__main__":
    main()