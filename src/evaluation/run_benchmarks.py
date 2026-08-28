import os
import torch
import pandas as pd
import numpy as np
import json
import argparse
import gc  # Added for memory management
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IR and Generative Metrics")
    parser.add_argument("--corpus", type=str, default="data/processed/universal_corpus.csv")
    parser.add_argument("--golden_key", type=str, default="data/processed/golden_key.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mhsa_best.pt")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--num_tests", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=25)
    return parser.parse_args()

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

def calculate_retrieval_metrics(retrieved_ids, ground_truth_ids, trig_id):
    hits = [1 if r_id in ground_truth_ids and r_id != trig_id else 0 for r_id in retrieved_ids]
    r5 = 1 if sum(hits[:5]) > 0 else 0
    r15 = 1 if sum(hits[:15]) > 0 else 0
    r25 = 1 if sum(hits[:25]) > 0 else 0
    try: mrr = 1.0 / (hits.index(1) + 1)
    except ValueError: mrr = 0.0
    return r5, r15, r25, mrr

def evaluate_llm_ac(llm_model, llm_tokenizer, context_logs, true_fault, trigger_msg):
    real_service = true_fault.rsplit("_", 1)[0].lower()
    prompt = f"""You are an Expert Site Reliability Engineer. Identify the root cause of this anomaly.
TRIGGER LOG: {trigger_msg}
CONTEXT LOGS:
{context_logs}

Provide your top 3 suspected root cause services. 
Format exactly like this:
Hypothesis 1: [Service] - [Reason]
Hypothesis 2: [Service] - [Reason]
Hypothesis 3: [Service] - [Reason]"""

    messages = [{"role": "system", "content": "You are a diagnostic AI."}, {"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        out_ids = llm_model.generate(**inputs, max_new_tokens=150, do_sample=False)
        response = llm_tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lower()

    hypotheses = response.split("hypothesis")
    ac1, ac3 = 0, 0
    svc_clean = real_service.replace("-", "").replace("_", "").replace("service", "")

    if len(hypotheses) > 1:
        h1 = hypotheses[1].replace("-", "").replace("_", "").replace("service", "")
        if svc_clean in h1: ac1, ac3 = 1, 1
            
    if ac3 == 0:
        for h in hypotheses[1:4]:
            h_clean = h.replace("-", "").replace("_", "").replace("service", "")
            if svc_clean in h_clean:
                ac3 = 1
                break
    return ac1, ac3

def free_memory():
    """Forces PyTorch to release VRAM back to the OS."""
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📂 Loading Corpus & Golden Key...")
    full_df = pd.read_csv(args.corpus)
    full_df['message'] = full_df['message'].fillna("").astype(str)
    
    with open(args.golden_key, "r") as f:
        golden_key = json.load(f)
        
    trace_groups = {}
    for entry in golden_key:
        incident_id = f"{entry['system']}||{entry['fault_type']}||{entry['trial']}"
        if incident_id not in trace_groups: trace_groups[incident_id] = []
        trace_groups[incident_id].append(entry['log_id'])
        
    valid_triggers = [log_ids[0] for logs in trace_groups.values() if len(logs) > 1 for log_ids in [logs]]
    test_triggers = np.random.choice(valid_triggers, min(args.num_tests, len(valid_triggers)), replace=False)

    # ==========================================
    # PART 1: RETRIEVAL (Load small models)
    # ==========================================
    print("🧠 Loading Retrieval Models (RoBERTa & DPR)...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = CrossModalLogEncoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    dpr_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1').to(device)

    ret_results = {"BM25": {"r5": [], "r15": [], "r25": [], "mrr": []}, "DPR": {"r5": [], "r15": [], "r25": [], "mrr": []}, "Proposed": {"r5": [], "r15": [], "r25": [], "mrr": []}}
    
    # We will save the retrieved contexts here to feed to the LLM later
    llm_tasks = []

    print(f"\n📊 [STAGE 1] Running Information Retrieval on {len(test_triggers)} triggers...\n")
    
    for trig_id in tqdm(test_triggers):
        global_trig_idx = full_df[full_df['log_id'] == trig_id].index[0]
        incident = next(k for k, v in trace_groups.items() if trig_id in v)
        ground_truth_ids = set(trace_groups[incident])
        
        _, true_fault, _ = incident.split("||")
        
        window = args.window_size * 2
        start_idx = max(0, global_trig_idx - (window // 2))
        end_idx = min(len(full_df), global_trig_idx + (window // 2))
        df = full_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        local_trig_idx = global_trig_idx - start_idx
        
        trigger_msg = df.iloc[local_trig_idx]['message']
        corpus_msgs = df['message'].tolist()
        corpus_ids = df['log_id'].tolist()

        # 1. PURE BM25 
        tokenized_corpus = [doc.split(" ") for doc in corpus_msgs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(trigger_msg.split(" "))
        bm25_topk_idx = np.argsort(bm25_scores)[::-1][:args.top_k]
        
        r5, r15, r25, mrr = calculate_retrieval_metrics([corpus_ids[i] for i in bm25_topk_idx], ground_truth_ids, trig_id)
        ret_results["BM25"]["r5"].append(r5); ret_results["BM25"]["r15"].append(r15); ret_results["BM25"]["r25"].append(r25); ret_results["BM25"]["mrr"].append(mrr)
        bm25_context = "\n".join([f"[{df.iloc[i]['service']}] {df.iloc[i]['message']}" for i in bm25_topk_idx])

        # 2. PURE DPR
        with torch.no_grad():
            doc_embeddings = dpr_model.encode(corpus_msgs, batch_size=128, convert_to_tensor=True, show_progress_bar=False)
            query_embedding = dpr_model.encode(trigger_msg, convert_to_tensor=True, show_progress_bar=False)
            dpr_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
            dpr_topk_idx = np.argsort(dpr_scores)[::-1][:args.top_k]
            
        r5, r15, r25, mrr = calculate_retrieval_metrics([corpus_ids[i] for i in dpr_topk_idx], ground_truth_ids, trig_id)
        ret_results["DPR"]["r5"].append(r5); ret_results["DPR"]["r15"].append(r15); ret_results["DPR"]["r25"].append(r25); ret_results["DPR"]["mrr"].append(mrr)
        dpr_context = "\n".join([f"[{df.iloc[i]['service']}] {df.iloc[i]['message']}" for i in dpr_topk_idx])

        # 3. PROPOSED
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
        
        _, prop_topk_idx = torch.topk(prop_scores, args.top_k)
        r5, r15, r25, mrr = calculate_retrieval_metrics(df.iloc[prop_topk_idx.cpu().numpy()]['log_id'].tolist(), ground_truth_ids, trig_id)
        ret_results["Proposed"]["r5"].append(r5); ret_results["Proposed"]["r15"].append(r15); ret_results["Proposed"]["r25"].append(r25); ret_results["Proposed"]["mrr"].append(mrr)
        prop_context = "\n".join([f"[{df.iloc[i]['service']}] {df.iloc[i]['message']}" for i in prop_topk_idx.cpu().numpy()])

        # 4. NAIVE (Chronological)
        naive_topk_idx = list(range(max(0, local_trig_idx - args.top_k), local_trig_idx))
        naive_context = "\n".join([f"[{df.iloc[i]['service']}] {df.iloc[i]['message']}" for i in naive_topk_idx])

        # Store contexts for Stage 2
        llm_tasks.append({
            "true_fault": true_fault,
            "trigger_msg": trigger_msg,
            "contexts": {
                "Naive": naive_context,
                "BM25": bm25_context,
                "DPR": dpr_context,
                "Proposed": prop_context
            }
        })

    # ==========================================
    # MEMORY CLEANUP
    # ==========================================
    print("\n🧹 Freeing GPU memory before loading Qwen-14B...")
    del model
    del dpr_model
    del tokenizer
    free_memory()

    # ==========================================
    # PART 2: GENERATION (Load massive LLM)
    # ==========================================
    print(f"🧠 Loading Generative LLM ({args.llm_model}) in 4-bit...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, quantization_config=bnb_config, device_map="auto")

    gen_results = {"Naive": {"ac1": [], "ac3": []}, "BM25": {"ac1": [], "ac3": []}, "DPR": {"ac1": [], "ac3": []}, "Proposed": {"ac1": [], "ac3": []}}
    
    print(f"\n📊 [STAGE 2] Running Generative Diagnostics...\n")
    for task in tqdm(llm_tasks):
        for method, context in task["contexts"].items():
            ac1, ac3 = evaluate_llm_ac(llm_model, llm_tokenizer, context, task["true_fault"], task["trigger_msg"])
            gen_results[method]["ac1"].append(ac1)
            gen_results[method]["ac3"].append(ac3)

    # ==========================================
    # PRINT RESULTS
    # ==========================================
    print("\n" + "="*55)
    print("📈 TABLE 1: RETRIEVAL EFFECTIVENESS")
    print("="*55)
    print(f"{'Algorithm':<15} | {'R@5':<6} | {'R@15':<6} | {'R@25':<6} | {'MRR':<6}")
    print("-" * 55)
    for m, v in ret_results.items(): print(f"{m:<15} | {np.mean(v['r5']):.4f} | {np.mean(v['r15']):.4f} | {np.mean(v['r25']):.4f} | {np.mean(v['mrr']):.4f}")
    
    print("\n" + "="*55)
    print("📈 TABLE 2: GENERATIVE DIAGNOSTIC ACCURACY")
    print("="*55)
    print(f"{'Configuration':<35} | {'AC@1':<6} | {'AC@3':<6}")
    print("-" * 55)
    for m, v in gen_results.items(): print(f"{m:<35} | {np.mean(v['ac1']):.4f} | {np.mean(v['ac3']):.4f}")

if __name__ == "__main__":
    main()