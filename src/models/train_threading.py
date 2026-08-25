import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/processed/universal_corpus.csv")
    parser.add_argument("--golden_key", type=str, default="data/processed/golden_key.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin_alpha", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # For initial testing, we'll train on a subset so it doesn't take weeks on 6 million logs
    parser.add_argument("--train_subset", type=int, default=100000) 
    return parser.parse_args()

# ==========================================
# 1. Time2Vec (Continuous Temporal Embedding)
# ==========================================
class Time2Vec(nn.Module):
    def __init__(self, output_dim=768):
        super(Time2Vec, self).__init__()
        self.output_dim = output_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, output_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(1, output_dim - 1))
        
    def forward(self, t):
        # t shape: (batch_size, 1)
        v1 = self.w0 * t + self.b0
        v2 = torch.sin(self.w * t + self.b)
        return torch.cat([v1, v2], dim=-1) # shape: (batch_size, output_dim)

# ==========================================
# 2. Cross-Modal Bi-Encoder Model
# ==========================================
class CrossModalLogEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", hidden_dim=768):
        super(CrossModalLogEncoder, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.time2vec = Time2Vec(output_dim=hidden_dim)
        
        # MHSA Projection Matrices (from Eq 3 in your paper)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, input_ids, attention_mask, timestamps):
        # Extract text embedding (CLS token)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_i = text_outputs.last_hidden_state[:, 0, :] # Shape: (batch, 768)
        
        # Extract time embedding
        t_emb = self.time2vec(timestamps) # Shape: (batch, 768)
        
        # Eq 2: Modal Fusion
        v_i = h_i + t_emb
        
        # Projection
        q = self.W_Q(v_i)
        k = self.W_K(v_i)
        
        return v_i, q, k

# ==========================================
# 3. Contrastive Triplet Dataset
# ==========================================
class LogTripletDataset(Dataset):
    def __init__(self, corpus_df, golden_key_list, tokenizer, max_length=128, max_traces=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("⏳ Mapping Golden Key to Corpus...")
        # Convert corpus to dict for fast lookup by log_id
        self.corpus_dict = corpus_df.set_index('log_id').to_dict('index')
        
        # Group log_ids by their specific fault injection incident (System + Fault + Trial)
        self.trace_groups = {}
        for entry in golden_key_list:
            # Use the folder origin as the Ground Truth grouping key
            incident_id = f"{entry['system']}_{entry['fault_type']}_{entry['trial']}"
            
            if incident_id not in self.trace_groups:
                self.trace_groups[incident_id] = []
            self.trace_groups[incident_id].append(entry['log_id'])
            
        # Filter out incidents that only have 1 log (we need at least 2 for an Anchor-Positive pair)
        self.valid_traces = [t for t, logs in self.trace_groups.items() if len(logs) > 1]
        
        # PROPER SUBSETTING: Sample incidents/traces, not individual logs
        if max_traces and len(self.valid_traces) > max_traces:
            self.valid_traces = list(np.random.choice(self.valid_traces, max_traces, replace=False))
            
        self.all_log_ids = list(self.corpus_dict.keys())
        self.min_time = corpus_df['observed_timestamp'].min()
        self.max_time = corpus_df['observed_timestamp'].max()
        
        print(f"✅ Successfully mapped {len(self.valid_traces)} fault incidents for Contrastive Training.")

    def __len__(self):
        return len(self.valid_traces)

    def tokenize_log(self, text):
        return self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )

    def __getitem__(self, idx):
        incident_id = self.valid_traces[idx]
        positive_logs = self.trace_groups[incident_id]
        
        # Sample Anchor and Positive from the SAME fault incident (Eq 1: u_a, u_p \in B)
        anchor_id, pos_id = np.random.choice(positive_logs, 2, replace=False)
        
        # Sample Negative from a COMPLETELY DIFFERENT incident (Eq 1: u_n \notin B)
        neg_id = np.random.choice(self.all_log_ids)
        while neg_id in positive_logs:
            neg_id = np.random.choice(self.all_log_ids)

        anchor_log = self.corpus_dict[anchor_id]
        pos_log = self.corpus_dict[pos_id]
        neg_log = self.corpus_dict[neg_id]

        anchor_tok = self.tokenize_log(anchor_log['message'])
        pos_tok = self.tokenize_log(pos_log['message'])
        neg_tok = self.tokenize_log(neg_log['message'])

        return {
            "anchor_ids": anchor_tok['input_ids'].squeeze(0),
            "anchor_mask": anchor_tok['attention_mask'].squeeze(0),
            "anchor_time": torch.tensor([(anchor_log['observed_timestamp'] - self.min_time) / (self.max_time - self.min_time)], dtype=torch.float32),
            
            "pos_ids": pos_tok['input_ids'].squeeze(0),
            "pos_mask": pos_tok['attention_mask'].squeeze(0),
            "pos_time": torch.tensor([(pos_log['observed_timestamp'] - self.min_time) / (self.max_time - self.min_time)], dtype=torch.float32),
            
            "neg_ids": neg_tok['input_ids'].squeeze(0),
            "neg_mask": neg_tok['attention_mask'].squeeze(0),
            "neg_time": torch.tensor([(neg_log['observed_timestamp'] - self.min_time) / (self.max_time - self.min_time)], dtype=torch.float32),
        }

# ==========================================
# 4. Main Training Loop
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    # Load Data
    print("📂 Loading datasets...")
    corpus_df = pd.read_csv(args.corpus)
    
    with open(args.golden_key, "r") as f:
        golden_key = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # We pass the max_traces limit (e.g. 5000 traces) directly to the Dataset so it samples correctly
    dataset = LogTripletDataset(corpus_df, golden_key, tokenizer, max_traces=5000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Model & Optimizer
    model = CrossModalLogEncoder(model_name="roberta-base").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Triplet Margin Loss (Matches Eq 1 in paper)
    criterion = nn.TripletMarginLoss(margin=args.margin_alpha, p=2)

    print("🔥 Starting Contrastive Training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Forward passes
            _, q_a, _ = model(batch['anchor_ids'].to(device), batch['anchor_mask'].to(device), batch['anchor_time'].to(device))
            _, _, k_p = model(batch['pos_ids'].to(device), batch['pos_mask'].to(device), batch['pos_time'].to(device))
            _, _, k_n = model(batch['neg_ids'].to(device), batch['neg_mask'].to(device), batch['neg_time'].to(device))
            
            # Calculate Loss
            loss = criterion(q_a, k_p, k_n)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")
        
        ckpt_path = os.path.join(args.checkpoint_dir, f"mhsa_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved to {ckpt_path}")

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "mhsa_best.pt"))
    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()