#!/bin/bash
#SBATCH --job-name=tau2_next_airline_16k
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
    # ["Qwen3-235B"]="bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
    # ["OpenAI-oss-120B"]="bedrock/openai.gpt-oss-120b-1:0"
    # ["Gemini-2.5-Flash"]="gemini/gemini-2.5-flash"
    # ["Llama-3.3-70B"]="bedrock/us.meta.llama3-3-70b-instruct-v1:0"
    # ["Claude-Sonnet-4.5"]="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # ["DeepSeek-R1"]="bedrock/us.deepseek.r1-v1:0"
    ["Qwen3-Next"]="huggingface/Qwen/Qwen3-Next-80B-A3B-Thinking:novita"
)

USER_LLM="bedrock/openai.gpt-oss-120b-1:0"

NUM_TRIALS=1
MAX_CONCURRENCY=1
TOKEN_BUDGET=16000
DOMAINS=("airline")

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

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        # Create subdirectory for this specific trial
        TRIAL_DIR="${MODEL_RESULTS_DIR}/trial_${TRIAL}"
        
        echo "[INFO] === TRIAL $TRIAL / $NUM_TRIALS ==="
        echo "[INFO] Trial results directory: $TRIAL_DIR"
        
        for DOMAIN in "${DOMAINS[@]}"; do
            echo "[INFO] Running domain: $DOMAIN (Trial $TRIAL)"
            echo "-----------------------------------------"

            # Create trial-domain specific output file
            OUTPUT_FILE="${TRIAL_DIR}/budget_${TOKEN_BUDGET}_${AGENT_LLM}_trial${TRIAL}_${DOMAIN}_lite.json"
            mkdir -p "$TRIAL_DIR"

            # Run tau2 benchmark with trial-specific output directory
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
