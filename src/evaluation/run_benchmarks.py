import os
import torch
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm

# Models
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IR Metrics for Proposed and Baselines")
    parser.add_argument("--corpus", type=str, default="data/processed/universal_corpus.csv")
    parser.add_argument("--golden_key", type=str, default="data/processed/golden_key.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mhsa_best.pt")
    parser.add_argument("--num_tests", type=int, default=50, help="Number of random triggers to test")
    parser.add_argument("--window_size", type=int, default=10000)
    return parser.parse_args()

# ==========================================
# Proposed Model Architecture
# ==========================================
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

# ==========================================
# Helper to calculate Metrics
# ==========================================
def calculate_metrics(retrieved_ids, ground_truth_ids, trig_id):
    hits = [1 if r_id in ground_truth_ids and r_id != trig_id else 0 for r_id in retrieved_ids]
    r5 = 1 if sum(hits[:5]) > 0 else 0
    r15 = 1 if sum(hits[:15]) > 0 else 0
    r25 = 1 if sum(hits[:25]) > 0 else 0
    try:
        mrr = 1.0 / (hits.index(1) + 1)
    except ValueError:
        mrr = 0.0
    return r5, r15, r25, mrr

# ==========================================
# Main Evaluation Loop
# ==========================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📂 Loading Corpus & Golden Key...")
    full_df = pd.read_csv(args.corpus)
    full_df['message'] = full_df['message'].fillna("").astype(str)
    
    with open(args.golden_key, "r") as f:
        golden_key = json.load(f)
        
    # Group by fault incident
    trace_groups = {}
    for entry in golden_key:
        incident_id = f"{entry['system']}_{entry['fault_type']}_{entry['trial']}"
        if incident_id not in trace_groups: trace_groups[incident_id] = []
        trace_groups[incident_id].append(entry['log_id'])
        
    valid_triggers = [log_ids[0] for logs in trace_groups.values() if len(logs) > 1 for log_ids in [logs]]
    test_triggers = np.random.choice(valid_triggers, min(args.num_tests, len(valid_triggers)), replace=False)

    print("🧠 Loading Proposed Model...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = CrossModalLogEncoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    print("🧠 Loading DPR Baseline Model...")
    dpr_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1').to(device)

    results = {
        "BM25": {"r5": [], "r15": [], "r25": [], "mrr": []},
        "DPR": {"r5": [], "r15": [], "r25": [], "mrr": []},
        "Proposed": {"r5": [], "r15": [], "r25": [], "mrr": []}
    }

    print(f"📊 Running Benchmarks on {len(test_triggers)} triggers...")
    
    for trig_id in tqdm(test_triggers):
        global_trig_idx = full_df[full_df['log_id'] == trig_id].index[0]
        incident = next(k for k, v in trace_groups.items() if trig_id in v)
        ground_truth_ids = set(trace_groups[incident])
        
        # Local Window Extraction
        start_idx = max(0, global_trig_idx - (args.window_size // 2))
        end_idx = min(len(full_df), global_trig_idx + (args.window_size // 2))
        df = full_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        local_trig_idx = global_trig_idx - start_idx
        
        trigger_msg = df.iloc[local_trig_idx]['message']
        corpus_msgs = df['message'].tolist()
        corpus_ids = df['log_id'].tolist()

        # ==========================================
        # 1. EVALUATE BM25 (Sparse Lexical)
        # ==========================================
        tokenized_corpus = [doc.split(" ") for doc in corpus_msgs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = trigger_msg.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top25_idx = np.argsort(bm25_scores)[::-1][:25]
        bm25_retrieved_ids = [corpus_ids[i] for i in bm25_top25_idx]
        
        r5, r15, r25, mrr = calculate_metrics(bm25_retrieved_ids, ground_truth_ids, trig_id)
        results["BM25"]["r5"].append(r5); results["BM25"]["r15"].append(r15); results["BM25"]["r25"].append(r25); results["BM25"]["mrr"].append(mrr)

        # ==========================================
        # 2. EVALUATE DPR (Dense Semantic)
        # ==========================================
        with torch.no_grad():
            doc_embeddings = dpr_model.encode(corpus_msgs, convert_to_tensor=True, show_progress_bar=False)
            query_embedding = dpr_model.encode(trigger_msg, convert_to_tensor=True, show_progress_bar=False)
            dpr_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            _, dpr_top25_idx = torch.topk(dpr_scores, 25)
            dpr_retrieved_ids = [corpus_ids[i] for i in dpr_top25_idx.cpu().numpy()]
            
        r5, r15, r25, mrr = calculate_metrics(dpr_retrieved_ids, ground_truth_ids, trig_id)
        results["DPR"]["r5"].append(r5); results["DPR"]["r15"].append(r15); results["DPR"]["r25"].append(r25); results["DPR"]["mrr"].append(mrr)

        # ==========================================
        # 3. EVALUATE PROPOSED (Cross-Modal Attention)
        # ==========================================
        min_t, max_t = full_df['observed_timestamp'].min(), full_df['observed_timestamp'].max()
        t_range = max_t - min_t if max_t > min_t else 1.0
        
        all_k = []
        with torch.no_grad():
            for i in range(0, len(df), 512):
                b_df = df.iloc[i:i+512]
                enc = tokenizer(b_df['message'].tolist(), padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
                t_tensor = torch.tensor(((b_df['observed_timestamp'].values - min_t) / t_range).astype(np.float32)).unsqueeze(1).to(device)
                _, _, b_k = model(enc['input_ids'], enc['attention_mask'], t_tensor)
                all_k.append(b_k.cpu())
        K_mat = torch.cat(all_k, dim=0).to(device)
        
        e_trig = tokenizer([str(trigger_msg)], padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
        t_trig = torch.tensor([[(df.iloc[local_trig_idx]['observed_timestamp'] - min_t) / t_range]], dtype=torch.float32).to(device)
        
        with torch.no_grad(): _, q_trig, _ = model(e_trig['input_ids'], e_trig['attention_mask'], t_trig)
        prop_scores = torch.softmax((q_trig @ K_mat.T) / np.sqrt(q_trig.size(-1)), dim=-1).squeeze(0)
        
        _, prop_top25_idx = torch.topk(prop_scores, 25)
        prop_retrieved_ids = df.iloc[prop_top25_idx.cpu().numpy()]['log_id'].tolist()
        
        r5, r15, r25, mrr = calculate_metrics(prop_retrieved_ids, ground_truth_ids, trig_id)
        results["Proposed"]["r5"].append(r5); results["Proposed"]["r15"].append(r15); results["Proposed"]["r25"].append(r25); results["Proposed"]["mrr"].append(mrr)

    # ==========================================
    # Print Final LaTeX Table Output
    # ==========================================
    print("\n" + "="*50)
    print("📈 FINAL EVALUATION METRICS (TABLE 1)")
    print("="*50)
    print(f"{'Algorithm':<15} | {'R@5':<6} | {'R@15':<6} | {'R@25':<6} | {'MRR':<6}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {np.mean(metrics['r5']):.4f} | {np.mean(metrics['r15']):.4f} | {np.mean(metrics['r25']):.4f} | {np.mean(metrics['mrr']):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()