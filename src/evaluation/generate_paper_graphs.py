import os
import matplotlib.pyplot as plt
import numpy as np

def setup_academic_style():
    # IEEE/ACM Standard Formatting
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 9,  # Slightly smaller to fit 4 labels cleanly
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

def plot_subgraph_limit(output_path):
    # Data points from overnight sweep
    k_values = [5, 10, 15, 20, 25, 30]
    proposed_mrr = [0.7990, 0.8390, 0.8015, 0.7699, 0.7772, 0.7921]
    dpr_mrr = [0.5277, 0.4613, 0.5315, 0.5165, 0.5130, 0.5053]
    bm25_mrr = [0.5182, 0.5540, 0.5287, 0.5310, 0.5290, 0.5400] 

    plt.figure(figsize=(6, 4))
    plt.plot(k_values, proposed_mrr, marker='o', color='red', label='Proposed (Cross-Modal)', linewidth=2)
    plt.plot(k_values, bm25_mrr, marker='x', color='gray', label='BM25 (Sparse)', linewidth=2, linestyle=':')
    plt.plot(k_values, dpr_mrr, marker='s', color='blue', label='DPR (Dense)', linewidth=2, linestyle='--')
    
    plt.xlabel('Graph Retrieval Bound ($k$)')
    plt.ylabel('Mean Reciprocal Rank (MRR)')
    plt.title('(a) Sub-Graph Limit Impact')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best')
    
    plt.ylim(0.40, 0.90) 
    
    plt.savefig(output_path, format='pdf')
    print(f"📊 Graph saved: {output_path}")
    plt.close()

def plot_temporal_robustness(output_path):
    # Data points from overnight sweep
    jitter_values = [0, 20, 40, 60, 80, 100]
    
    proposed_ac1 = [0.4600, 0.3600, 0.4000, 0.4400, 0.3600, 0.4400]
    naive_ac1 = [0.3000, 0.1600, 0.2800, 0.2200, 0.2000, 0.3200]
    
    # Adding back the DPR and BM25 averages from your logs (~0.12 - 0.16)
    dpr_ac1 = [0.1400, 0.1400, 0.1600, 0.1400, 0.1200, 0.1400]
    bm25_ac1 = [0.1400, 0.1200, 0.1200, 0.1400, 0.1000, 0.1200]

    plt.figure(figsize=(6, 4))
    plt.plot(jitter_values, proposed_ac1, marker='o', color='red', label='Proposed (Cross-Modal)', linewidth=2)
    plt.plot(jitter_values, naive_ac1, marker='^', color='green', label='Naive (Chronological)', linewidth=2, linestyle='--')
    plt.plot(jitter_values, dpr_ac1, marker='s', color='blue', label='DPR (Dense)', linewidth=2, linestyle='--')
    plt.plot(jitter_values, bm25_ac1, marker='x', color='gray', label='BM25 (Sparse)', linewidth=2, linestyle=':')
    
    plt.xlabel('NTP Clock Jitter $\sigma$ (ms)')
    plt.ylabel('Accuracy@1 Effectiveness')
    plt.title('(b) Temporal Robustness')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Use 'ncol=2' to split the legend into two columns so it doesn't cover your data
    plt.legend(loc='upper right', ncol=2) 
    
    plt.ylim(0.0, 0.60) 
    
    plt.savefig(output_path, format='pdf')
    print(f"📊 Graph saved: {output_path}")
    plt.close()

def main():
    os.makedirs("data/results/graphs", exist_ok=True)
    setup_academic_style()
    plot_subgraph_limit("data/results/graphs/fig_a_subgraph_limit.pdf")
    plot_temporal_robustness("data/results/graphs/fig_b_temporal_robustness.pdf")

if __name__ == "__main__":
    main()