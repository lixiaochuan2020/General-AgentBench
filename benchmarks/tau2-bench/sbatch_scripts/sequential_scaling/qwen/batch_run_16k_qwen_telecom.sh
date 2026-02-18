#!/bin/bash
#SBATCH --job-name=tau2_qwen_telecom_16k
#SBATCH --time=47:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "[INFO] Job started at $(date)"
echo "[INFO] Running on node: $(hostname)"
echo

declare -A MODELS=(
    ["Qwen3-235B"]="bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
)

USER_LLM="bedrock/openai.gpt-oss-120b-1:0"
NUM_TRIALS=1
MAX_CONCURRENCY=1
TOKEN_BUDGET=16000
DOMAINS=("telecom")
RESULTS_BASE="results"
mkdir -p "logs"

for MODEL_NAME in "${!MODELS[@]}"; do
    AGENT_LLM="${MODELS[$MODEL_NAME]}"
    MODEL_RESULTS_DIR="${RESULTS_BASE}/simulation_${TOKEN_BUDGET}_${MODEL_NAME}"
    
    echo "========================================="
    echo "[INFO] Starting $NUM_TRIALS trials for model: $MODEL_NAME"
    echo "LLM Path: $AGENT_LLM"
    echo "Base results directory: $MODEL_RESULTS_DIR"
    echo "-----------------------------------------"

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        TRIAL_DIR="${MODEL_RESULTS_DIR}/trial_${TRIAL}"
        mkdir -p "$TRIAL_DIR"
        
        echo "[INFO] Trial $TRIAL/$NUM_TRIALS - Starting at $(date)"
        
        for DOMAIN in "${DOMAINS[@]}"; do
            OUTPUT_FILE="${TRIAL_DIR}/budget_${TOKEN_BUDGET}_${MODEL_NAME}_trial${TRIAL}_${DOMAIN}_lite.json"
            
            echo "  [DOMAIN] Processing $DOMAIN..."
            echo "  [OUTPUT] $OUTPUT_FILE"
            
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
