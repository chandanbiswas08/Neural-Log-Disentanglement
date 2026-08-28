#!/bin/bash

LOG_FILE="data/results/overnight_run.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Error: $LOG_FILE not found!"
    exit 1
fi

echo "=================================================="
echo "📄 EXTRACTING EXACT RESULTS FOR FIRE 2026 PAPER"
echo "=================================================="

# 1. Isolate only the VERY LAST overnight run to ignore old failed tests
START_LINE=$(grep -n "STARTING OVERNIGHT BATCH EXECUTION" "$LOG_FILE" | tail -n 1 | cut -d: -f1)
tail -n +"$START_LINE" "$LOG_FILE" > data/results/tmp_latest_run.log
TMP_LOG="data/results/tmp_latest_run.log"

# 2. Get line numbers for the FIRST occurrence in the latest run (Phase 1)
LINE_T1=$(grep -n "TABLE 1: RETRIEVAL EFFECTIVENESS" "$TMP_LOG" | head -n 1 | cut -d: -f1)
LINE_T2=$(grep -n "TABLE 2: GENERATIVE DIAGNOSTIC ACCURACY" "$TMP_LOG" | head -n 1 | cut -d: -f1)

P2_START=$(grep -n "PHASE 2: Sweeping Graph 1" "$TMP_LOG" | head -n 1 | cut -d: -f1)
P3_START=$(grep -n "PHASE 3: Sweeping Graph 2" "$TMP_LOG" | head -n 1 | cut -d: -f1)
P4_START=$(grep -n "BATCH COMPLETE" "$TMP_LOG" | head -n 1 | cut -d: -f1)

echo -e "\n---> RESULTS FOR OVERLEAF (TABLE 1: Retrieval Effectiveness)"
sed -n "$((LINE_T1 - 1)),$((LINE_T1 + 7))p" "$TMP_LOG"

echo -e "\n---> RESULTS FOR OVERLEAF (TABLE 2: Generative Accuracy)"
sed -n "$((LINE_T2 - 1)),$((LINE_T2 + 8))p" "$TMP_LOG"

echo -e "\n---> PYTHON ARRAYS FOR generate_paper_graphs.py"

# EXTRACT MRR (Phase 2). Table 1 has 5 columns (NF==5).
P_MRR=$(sed -n "${P2_START},${P3_START}p" "$TMP_LOG" | awk -F'|' 'NF==5 && /^Proposed / {print $5}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')
D_MRR=$(sed -n "${P2_START},${P3_START}p" "$TMP_LOG" | awk -F'|' 'NF==5 && /^DPR / {print $5}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')
B_MRR=$(sed -n "${P2_START},${P3_START}p" "$TMP_LOG" | awk -F'|' 'NF==5 && /^BM25 / {print $5}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')

# EXTRACT AC@1 (Phase 3). Table 2 has 3 columns (NF==3).
P_AC1=$(sed -n "${P3_START},${P4_START}p" "$TMP_LOG" | awk -F'|' 'NF==3 && /^Proposed / {print $2}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')
N_AC1=$(sed -n "${P3_START},${P4_START}p" "$TMP_LOG" | awk -F'|' 'NF==3 && /^Naive / {print $2}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')
D_AC1=$(sed -n "${P3_START},${P4_START}p" "$TMP_LOG" | awk -F'|' 'NF==3 && /^DPR / {print $2}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')
B_AC1=$(sed -n "${P3_START},${P4_START}p" "$TMP_LOG" | awk -F'|' 'NF==3 && /^BM25 / {print $2}' | tr -d ' ' | paste -sd, - | sed 's/,/, /g')

echo "# For plot_subgraph_limit() :"
echo "    k_values = [5, 10, 15, 20, 25, 30]"
echo "    proposed_mrr = [$P_MRR]"
echo "    dpr_mrr = [$D_MRR]"
echo "    bm25_mrr = [$B_MRR]"
echo ""
echo "# For plot_temporal_robustness() :"
echo "    jitter_values = [0, 20, 40, 60, 80, 100]"
echo "    proposed_ac1 = [$P_AC1]"
echo "    naive_ac1 = [$N_AC1]"
echo "    dpr_ac1 = [$D_AC1]"
echo "    bm25_ac1 = [$B_AC1]"

echo -e "\n=================================================="
echo "✅ Extraction complete! Copy these values directly into your draft."

# Cleanup temporary file
rm -f "$TMP_LOG"