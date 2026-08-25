import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generative RCA Diagnosis (Stage 4)")
    parser.add_argument("--context_file", type=str, default="data/results/retrieved_context.txt")
    # Qwen2.5-3B or 7B are fantastic for log analysis and easily fit on an RTX A5500
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.context_file):
        print(f"❌ Error: Context file {args.context_file} not found. Run retrieve.py first.")
        return

    print("📖 Reading extracted context from Stage 3...")
    with open(args.context_file, "r") as f:
        retrieved_context = f.read()

    print(f"🧠 Loading Generative LLM ({args.model_name})... This may take a minute to download weights the first time.")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16, # Uses half-precision to save VRAM and run ultra-fast
        device_map="auto"
    )

    # RAG Prompt Formulation (Matches Eq 5 in your paper)
    prompt = f"""You are an Expert AIOps Site Reliability Engineer.
A network anomaly was triggered. A neural retrieval system has extracted the following most relevant logs from a massive distributed cluster. 

Some of these logs are background noise (e.g., routine transactions), but the true root cause is hidden among them.

{retrieved_context}

TASK:
1. Identify the root cause of the TRIGGER LOG.
2. Ignore irrelevant background chatter.
3. Write a brief, highly professional Root Cause Analysis (RCA) report.

RCA REPORT:"""

    messages = [
        {"role": "system", "content": "You are a senior network diagnostic AI."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("✨ Generating Diagnostic RCA Report...\n")
    print("="*60)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.1, # Low temperature for deterministic, factual extraction
            do_sample=True
        )
        
        # Extract only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    print(response)
    print("="*60)
    print("\n✅ Pipeline Complete! IDEAS TIH Framework has successfully diagnosed the root cause.")

if __name__ == "__main__":
    main()