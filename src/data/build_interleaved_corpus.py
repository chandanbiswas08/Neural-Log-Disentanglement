import os
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build Interleaved Log Corpus (Spaghetti Logs)")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Path to the raw RE3 directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to save processed data")
    parser.add_argument("--clock_drift_variance", type=float, default=50.0, help="NTP Jitter Variance (ms)")
    parser.add_argument("--gumbel_scale", type=float, default=15.0, help="Scale for Gumbel network buffer delay")
    return parser.parse_args()

def process_logs(input_dir, clock_drift_var, gumbel_scale):
    all_logs = []
    golden_key = []
    
    # Dictionary to hold the static NTP clock skew per microservice (Delta_s)
    service_ntp_skew = {}
    global_log_id = 0

    print(f"🔍 Scanning directory: {input_dir}")
    log_files = list(Path(input_dir).rglob("logs.csv"))
    print(f"📄 Found {len(log_files)} logs.csv files. Processing...")

    for file_path in log_files:
        parts = file_path.parts
        system, fault_type, trial = parts[-4], parts[-3], parts[-2]
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Skipping {file_path} due to read error: {e}")
            continue

        # Standardize column names
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # --- DYNAMIC COLUMN MAPPING ---
        # 1. Map Time (PRIORITIZE 'timestamp' because 'time' contains strings like "04:14")
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'time' in df.columns:
            time_col = 'time'
        else:
            print(f"⚠️ Warning: Missing time column in {file_path}. Found: {df.columns.tolist()}")
            continue

        # 2. Map Service / Container Name
        if 'service' in df.columns:
            service_col = 'service'
        elif 'container_name' in df.columns:
            service_col = 'container_name'
        elif 'pod_name' in df.columns:
            service_col = 'pod_name'
        else:
            print(f"⚠️ Warning: Missing service/container column in {file_path}. Found: {df.columns.tolist()}")
            continue

        # 3. Map Message
        if 'message' in df.columns:
            msg_col = 'message'
        elif 'log' in df.columns:
            msg_col = 'log'
        else:
            print(f"⚠️ Warning: Missing message column in {file_path}. Found: {df.columns.tolist()}")
            continue

        dropped_rows = 0

        for _, row in df.iterrows():
            service = str(row[service_col])
            
            # Handle potential non-numeric times safely
            try:
                original_time = float(row[time_col])
            except ValueError:
                dropped_rows += 1
                continue # Skip corrupt time rows
                
            message = str(row[msg_col])
            
            # Step 1: Assign an NTP Clock Skew (\Delta_s) per service
            if service not in service_ntp_skew:
                service_ntp_skew[service] = np.random.normal(0, np.sqrt(clock_drift_var))
            delta_s = service_ntp_skew[service]

            # Step 2: Sample Asynchronous Buffer Delay (\epsilon_i)
            epsilon_i = np.random.gumbel(loc=0.0, scale=gumbel_scale)

            # Step 3: Calculate the new ingestion timestamp (\tilde{t})
            observed_time = original_time + delta_s + epsilon_i

            log_entry = {
                "log_id": global_log_id,
                "observed_timestamp": observed_time,
                "service": service,
                "message": message
            }
            all_logs.append(log_entry)

            # Step 5: Save origin to the Golden Key
            golden_key.append({
                "log_id": global_log_id,
                "original_timestamp": original_time,
                "system": system,
                "fault_type": fault_type,
                "trial": trial,
                "original_trace_id": row.get('trace_id', row.get('traceid', 'UNKNOWN'))
            })
            
            global_log_id += 1

        if dropped_rows > 0:
            print(f"⚠️ Warn: Dropped {dropped_rows} rows in {file_path} due to unparsable timestamps.")

    corpus_df = pd.DataFrame(all_logs)
    
    if corpus_df.empty:
        print("🚨 Error: No logs were successfully processed. Please check the warnings above.")
        return corpus_df, golden_key

    print("⏳ Sorting corpus by simulated observed ingestion times...")
    corpus_df = corpus_df.sort_values(by="observed_timestamp").reset_index(drop=True)

    return corpus_df, golden_key

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    corpus_df, golden_key = process_logs(args.input_dir, args.clock_drift_variance, args.gumbel_scale)

    if not corpus_df.empty:
        print(f"💾 Saving interleaved corpus ({len(corpus_df)} logs) to {args.output_dir}/universal_corpus.csv")
        corpus_df.to_csv(f"{args.output_dir}/universal_corpus.csv", index=False)

        print(f"🔑 Saving Golden Key to {args.output_dir}/golden_key.json")
        with open(f"{args.output_dir}/golden_key.json", "w") as f:
            json.dump(golden_key, f, indent=4)
            
        print("✅ Dataset construction complete! The Spaghetti Log corpus is ready.")

if __name__ == "__main__":
    main()