#!/bin/bash
set -e

# Setup log directory
mkdir -p data/results
LOG_FILE="data/results/overnight_run.log"
GRAPH_ARRAYS="data/results/graph_arrays.txt"

echo "==================================================" | tee -a $LOG_FILE
echo "🚀 STARTING OVERNIGHT BATCH EXECUTION" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE

# Set highly rigorous test conditions & optimized window size
NUM_TESTS=50
WINDOW_SIZE=2500

# ---------------------------------------------------------
# PHASE 1: GENERATE TABLE 1 AND TABLE 2 (Default Params)
# ---------------------------------------------------------
echo -e "\n[$(date +'%H:%M:%S')] 📊 PHASE 1: Generating Paper Tables 1 & 2..." | tee -a $LOG_FILE
echo "Building standard dataset (Jitter = 50ms)..." | tee -a $LOG_FILE
python src/data/build_interleaved_corpus.py --clock_drift_variance 50.0 > /dev/null

echo "Running Full Evaluation (Top-K = 25)..." | tee -a $LOG_FILE
python src/evaluation/run_benchmarks.py --num_tests $NUM_TESTS --window_size $WINDOW_SIZE --top_k 25 | tee -a $LOG_FILE


# ---------------------------------------------------------
# PHASE 2: GRAPH 1 (Sweeping Top-K)
# ---------------------------------------------------------
echo -e "\n[$(date +'%H:%M:%S')] 📈 PHASE 2: Sweeping Graph 1 (Top-K limits)..." | tee -a $LOG_FILE
K_VALUES=(5 10 15 20 25 30)

PROPOSED_MRR=()
DPR_MRR=()

for k in "${K_VALUES[@]}"; do
    echo " -> Running Top-K = $k..." | tee -a $LOG_FILE
    OUTPUT=$(python src/evaluation/run_benchmarks.py --num_tests $NUM_TESTS --window_size $WINDOW_SIZE --top_k $k)
    
    # Save the output to the log file so you can review it tomorrow
    echo "$OUTPUT" >> $LOG_FILE
    
    # Extract MRR (Column 5) STRICTLY from Table 1
    P_MRR=$(echo "$OUTPUT" | sed -n '/TABLE 1/,/TABLE 2/p' | awk -F'|' '/^Proposed / {print $5}' | tr -d ' ')
    D_MRR=$(echo "$OUTPUT" | sed -n '/TABLE 1/,/TABLE 2/p' | awk -F'|' '/^DPR / {print $5}' | tr -d ' ')
    
    PROPOSED_MRR+=($P_MRR)
    DPR_MRR+=($D_MRR)
done


# ---------------------------------------------------------
# PHASE 3: GRAPH 2 (Sweeping NTP Jitter)
# ---------------------------------------------------------
echo -e "\n[$(date +'%H:%M:%S')] ⏱️ PHASE 3: Sweeping Graph 2 (NTP Jitter Variances)..." | tee -a $LOG_FILE
JITTER_VALUES=(0 20 40 60 80 100)

PROPOSED_AC1=()
BASELINE_AC1=()

for jitter in "${JITTER_VALUES[@]}"; do
    echo " -> Building dataset with Jitter = $jitter ms..." | tee -a $LOG_FILE
    python src/data/build_interleaved_corpus.py --clock_drift_variance $jitter > /dev/null
    
    echo " -> Evaluating AC@1 for Jitter = $jitter ms..." | tee -a $LOG_FILE
    OUTPUT=$(python src/evaluation/run_benchmarks.py --num_tests $NUM_TESTS --window_size $WINDOW_SIZE --top_k 25)
    
    # Save the output to the log file
    echo "$OUTPUT" >> $LOG_FILE
    
    # Extract AC@1 (Column 2) STRICTLY from Table 2
    P_AC1=$(echo "$OUTPUT" | sed -n '/TABLE 2/,$p' | awk -F'|' '/^Proposed / {print $2}' | tr -d ' ')
    B_AC1=$(echo "$OUTPUT" | sed -n '/TABLE 2/,$p' | awk -F'|' '/^DPR / {print $2}' | tr -d ' ')
    
    PROPOSED_AC1+=($P_AC1)
    BASELINE_AC1+=($B_AC1)
done

# ---------------------------------------------------------
# PHASE 4: SAVE ARRAYS TO FILE
# ---------------------------------------------------------
echo -e "\n[$(date +'%H:%M:%S')] ✅ BATCH COMPLETE! Saving Python arrays..." | tee -a $LOG_FILE

> $GRAPH_ARRAYS
echo "# For plot_subgraph_limit() :" >> $GRAPH_ARRAYS
echo "k_values = [${K_VALUES[*]}]" | sed 's/ /, /g' >> $GRAPH_ARRAYS
echo "proposed_mrr = [${PROPOSED_MRR[*]}]" | sed 's/ /, /g' >> $GRAPH_ARRAYS
echo "dpr_mrr = [${DPR_MRR[*]}]" | sed 's/ /, /g' >> $GRAPH_ARRAYS
echo "" >> $GRAPH_ARRAYS
echo "# For plot_temporal_robustness() :" >> $GRAPH_ARRAYS
echo "jitter_values = [${JITTER_VALUES[*]}]" | sed 's/ /, /g' >> $GRAPH_ARRAYS
echo "proposed_ac1 = [${PROPOSED_AC1[*]}]" | sed 's/ /, /g' >> $GRAPH_ARRAYS
echo "baseline_ac1 = [${BASELINE_AC1[*]}]" | sed 's/ /, /g' >> $GRAPH_ARRAYS

echo "Results saved to $GRAPH_ARRAYS" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE

# Final cleanup: Rebuild the dataset back to the default 50.0 variance
python src/data/build_interleaved_corpus.py --clock_drift_variance 50.0 > /dev/null