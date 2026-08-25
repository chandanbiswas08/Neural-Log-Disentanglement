import os
import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IR Metrics (Recall@K, MRR)")
    parser.add_argument("--corpus", type=str, default="data/processed/universal_corpus.csv")
    parser.add_argument("--golden_key", type=str, default="data/processed/golden_key.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mhsa_best.pt")
    parser.add_argument("--num_tests", type=int, default=50, help="Number of random triggers to test")
    parser.add_argument("--window_size", type=int, default=10000)
    return parser.parse_args()

# Model Definitions
class Time2Vec(torch.nn.Module):
    def __init__(self, output_dim=768):
        super(Time2Vec, self).__init__()
        self.w0, self.b0 = torch.nn.Parameter(torch.randn(1, 1)), torch.nn.Parameter(torch.randn(1, 1))
        self.w, self.b = torch.nn.Parameter(torch.randn(1, output_dim - 1)), torch.nn.Parameter(torch.randn(1, output_dim - 1))
    def forward(self, t): return torch.cat([self.w0 * t + self.b0, torch.sin(self.w * t + self.b)], dim=-1)

class CrossModalLogEncoder(torch.nn.Module):
    def __init__(self, model_name="roberta-base", hidden_dim=768):
        super(CrossModalLogEncoder, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.time2vec = Time2Vec(output_dim=hidden_dim)
        self.W_Q, self.W_K = torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim)
    def forward(self, input_ids, attention_mask, timestamps):
        h_i = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        v_i = h_i + self.time2vec(timestamps)
        return v_i, self.W_Q(v_i), self.W_K(v_i)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📂 Loading Corpus & Golden Key...")
    full_df = pd.read_csv(args.corpus)
    
    # SAFETY FIX: Force all messages to be strings and fill NaNs
    full_df['message'] = full_df['message'].fillna("").astype(str)
    
    with open(args.golden_key, "r") as f:
        golden_key = json.load(f)
        
    # Find testable triggers
    trace_groups = {}
    for entry in golden_key:
        incident_id = f"{entry['system']}_{entry['fault_type']}_{entry['trial']}"
        if incident_id not in trace_groups: trace_groups[incident_id] = []
        trace_groups[incident_id].append(entry['log_id'])
        
    valid_triggers = [log_ids[0] for logs in trace_groups.values() if len(logs) > 1 for log_ids in [logs]]
    test_triggers = np.random.choice(valid_triggers, min(args.num_tests, len(valid_triggers)), replace=False)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = CrossModalLogEncoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    recalls = {5: [], 15: [], 25: []}
    mrrs = []

    print(f"📊 Running Evaluation on {len(test_triggers)} triggers...")
    
    for trig_id in tqdm(test_triggers):
        global_trig_idx = full_df[full_df['log_id'] == trig_id].index[0]
        
        incident = next(k for k, v in trace_groups.items() if trig_id in v)
        ground_truth_ids = set(trace_groups[incident])
        
        start_idx = max(0, global_trig_idx - (args.window_size // 2))
        end_idx = min(len(full_df), global_trig_idx + (args.window_size // 2))
        df = full_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        local_trig_idx = global_trig_idx - start_idx
        
        min_t, max_t = full_df['observed_timestamp'].min(), full_df['observed_timestamp'].max()
        t_range = max_t - min_t if max_t > min_t else 1.0
        
        all_k = []
        with torch.no_grad():
            for i in range(0, len(df), 512):
                b_df = df.iloc[i:i+512]
                # SAFETY FIX 2: Ensure strictly string lists are passed to the tokenizer
                msg_list = b_df['message'].tolist()
                enc = tokenizer(msg_list, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
                
                t_tensor = torch.tensor(((b_df['observed_timestamp'].values - min_t) / t_range).astype(np.float32)).unsqueeze(1).to(device)
                _, _, b_k = model(enc['input_ids'], enc['attention_mask'], t_tensor)
                all_k.append(b_k.cpu())
        K_mat = torch.cat(all_k, dim=0).to(device)
        
        t_row = df.iloc[local_trig_idx]
        # SAFETY FIX 3: Explicit cast to string for the trigger
        e_trig = tokenizer([str(t_row['message'])], padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
        t_trig = torch.tensor([[(t_row['observed_timestamp'] - min_t) / t_range]], dtype=torch.float32).to(device)
        
        with torch.no_grad(): _, q_trig, _ = model(e_trig['input_ids'], e_trig['attention_mask'], t_trig)
        scores = torch.softmax((q_trig @ K_mat.T) / np.sqrt(q_trig.size(-1)), dim=-1).squeeze(0)
        
        _, top25_idx = torch.topk(scores, 25)
        retrieved_ids = df.iloc[top25_idx.cpu().numpy()]['log_id'].tolist()
        
        hits = [1 if r_id in ground_truth_ids and r_id != trig_id else 0 for r_id in retrieved_ids]
        
        recalls[5].append(1 if sum(hits[:5]) > 0 else 0)
        recalls[15].append(1 if sum(hits[:15]) > 0 else 0)
        recalls[25].append(1 if sum(hits[:25]) > 0 else 0)
        
        try:
            first_hit_rank = hits.index(1) + 1
            mrrs.append(1.0 / first_hit_rank)
        except ValueError:
            mrrs.append(0.0)

    print("\n" + "="*40)
    print("📈 FINAL EVALUATION METRICS (PROPOSED FRAMEWORK)")
    print("="*40)
    print(f"Recall@5:  {np.mean(recalls[5]):.4f}")
    print(f"Recall@15: {np.mean(recalls[15]):.4f}")
    print(f"Recall@25: {np.mean(recalls[25]):.4f}")
    print(f"MRR:       {np.mean(mrrs):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()