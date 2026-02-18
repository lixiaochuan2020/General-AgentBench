#!/bin/bash
#SBATCH --job-name=tau2_gemini_telecom_22k
#SBATCH --time=47:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ===============================
# Tau2 Benchmark Test Script - 1 Trials per Model
# ===============================

echo "[INFO] Job started at $(date)"
echo "[INFO] Running on node: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

declare -A MODELS=(
    ["Gemini-2.5-Flash"]="gemini/gemini-2.5-flash"
)

USER_LLM="bedrock/openai.gpt-oss-120b-1:0"

NUM_TRIALS=1
MAX_CONCURRENCY=1
TOKEN_BUDGET=22000
DOMAINS=("telecom")

# Base results directory
RESULTS_BASE="results"
mkdir -p "logs"

for MODEL_NAME in "${!MODELS[@]}"; do
    AGENT_LLM="${MODELS[$MODEL_NAME]}"
    
    # Create base results directory for this model
    MODEL_RESULTS_DIR="${RESULTS_BASE}/simulation_${TOKEN_BUDGET}_${MODEL_NAME}"
    
    echo "========================================="
    echo "[INFO] Starting $NUM_TRIALS trials for model: $MODEL_NAME"
    echo "LLM Path: $AGENT_LLM"
    echo "Base results directory: $MODEL_RESULTS_DIR"
    echo "-----------------------------------------"

    # Loop through each trial
    for TRIAL in $(seq 1 $NUM_TRIALS); do
        TRIAL_DIR="${MODEL_RESULTS_DIR}/trial_${TRIAL}"
        mkdir -p "$TRIAL_DIR"
        
        echo "[INFO] Trial $TRIAL/$NUM_TRIALS - Starting at $(date)"
        
        # Loop through each domain
        for DOMAIN in "${DOMAINS[@]}"; do
            OUTPUT_FILE="${TRIAL_DIR}/budget_${TOKEN_BUDGET}_${MODEL_NAME}_trial${TRIAL}_${DOMAIN}_lite.json"
            
            echo "  [DOMAIN] Processing $DOMAIN..."
            echo "  [OUTPUT] $OUTPUT_FILE"
            
            # Run the benchmark for this domain
            tau2 run \
                --domain "$DOMAIN" \
                --task-set-name "$DOMAIN"_lite \
                --agent-llm "$AGENT_LLM" \
                --user-llm "$USER_LLM" \
                --num-trials 1 \
                --max-concurrency "$MAX_CONCURRENCY" \
                --save-to "$OUTPUT_FILE" \
                --user-llm-args '{"temperature": 0.0}' \
                --agent-llm-args '{"temperature": 0.7}' \
                --agent-token-budget "$TOKEN_BUDGET" \
                || { echo "[ERROR] tau2 run failed on $MODEL_NAME / $DOMAIN / Trial $TRIAL"; exit 1; }
            
            echo "  [SUCCESS] $DOMAIN completed"
        done
        
        echo "[INFO] Trial $TRIAL/$NUM_TRIALS - Completed at $(date)"
        echo "-----------------------------------------"
    done
    
    echo "[INFO] All trials for $MODEL_NAME completed!"
    echo "========================================="
    echo
done

echo "[INFO] All models processed!"
echo "[INFO] Job finished at $(date)"
