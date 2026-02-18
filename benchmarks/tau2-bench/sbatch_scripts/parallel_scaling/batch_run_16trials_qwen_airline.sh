#!/bin/bash
#SBATCH --job-name=tau2_qwen_airline
#SBATCH --time=47:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ===============================
# Tau2 Benchmark Test Script - 16 Trials per Model
# ===============================

echo "[INFO] Job started at $(date)"
echo "[INFO] Running on node: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

declare -A MODELS=(
    ["Qwen3-235B"]="bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
)

USER_LLM="bedrock/openai.gpt-oss-120b-1:0"

NUM_TRIALS=13
MAX_CONCURRENCY=1
DOMAINS=("airline")

# Base results directory
RESULTS_BASE="results"
mkdir -p "logs"

for MODEL_NAME in "${!MODELS[@]}"; do
    AGENT_LLM="${MODELS[$MODEL_NAME]}"
    
    # Create base results directory for this model
    MODEL_RESULTS_DIR="${RESULTS_BASE}/simulation_16_${MODEL_NAME}"
    
    echo "========================================="
    echo "[INFO] Starting $NUM_TRIALS trials for model: $MODEL_NAME"
    echo "LLM Path: $AGENT_LLM"
    echo "Base results directory: $MODEL_RESULTS_DIR"
    echo "-----------------------------------------"

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        # Create subdirectory for this specific trial
        TRIAL_DIR="${MODEL_RESULTS_DIR}/trial_${TRIAL}"
        
        echo "[INFO] === TRIAL $TRIAL / $NUM_TRIALS ==="
        echo "[INFO] Trial results directory: $TRIAL_DIR"
        
        for DOMAIN in "${DOMAINS[@]}"; do
            echo "[INFO] Running domain: $DOMAIN (Trial $TRIAL)"
            echo "-----------------------------------------"

            # Create trial-domain specific output file
            OUTPUT_FILE="${TRIAL_DIR}/trial_${TRIAL}_${DOMAIN}.json"
            mkdir -p "$TRIAL_DIR"

            # Run tau2 benchmark with trial-specific output directory
            tau2 run \
                --domain "$DOMAIN" \
                --agent-llm "$AGENT_LLM" \
                --user-llm "$USER_LLM" \
                --num-trials 1 \
                --max-concurrency "$MAX_CONCURRENCY" \
                --save-to "$OUTPUT_FILE" \
                --user-llm-args '{"temperature": 0.0}' \
                --agent-llm-args '{"temperature": 0.7}' \
                || { echo "[ERROR] tau2 run failed on $MODEL_NAME / $DOMAIN / Trial $TRIAL"; exit 1; }

            echo "[INFO] Finished domain: $DOMAIN (Trial $TRIAL)"
            echo
        done
        
        echo "[INFO] Completed Trial $TRIAL / $NUM_TRIALS for model: $MODEL_NAME"
        echo
    done

    echo "[INFO] Completed all $NUM_TRIALS trials for model: $MODEL_NAME"
    echo "[INFO] Results saved in: $MODEL_RESULTS_DIR"
    echo
done

echo "[INFO] All models and trials completed at $(date)"
echo "[INFO] Total trials per model: $NUM_TRIALS"
echo "[INFO] Total domains per trial: ${#DOMAINS[@]} (${DOMAINS[*]})"
echo "[INFO] Total simulations per model: $((NUM_TRIALS * ${#DOMAINS[@]}))"
