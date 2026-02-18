#!/bin/bash
#SBATCH --job-name=mcpbench
#SBATCH --time=47:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ===============================
# MCP-Bench All Models × All Tasks Script
# ===============================

echo "[INFO] Job started at $(date)"
echo "[INFO] Running on node: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

echo "[INFO] Loading .env configuration..."
set -a
source .env
set +a
echo "[INFO] .env loaded successfully."
echo

declare -A MODELS=(
    ["Qwen3-235B"]="bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
    ["OpenAI-oss-120B"]="bedrock/openai.gpt-oss-120b-1:0"
    ["Gemini-2.5-Flash"]="gemini/gemini-2.5-flash"
)

JUDGE_MODEL="bedrock/openai.gpt-oss-120b-1:0"
JUDGE_PROVIDER="litellm"
JUDGE_NAME="GPT-OSS-120B"
TASK_FILES=(
    "tasks/mcpbench_tasks_single_runner_format_no_biomcp.json"
    "tasks/mcpbench_tasks_multi_2server_runner_format_no_biomcp.json"
    "tasks/mcpbench_tasks_multi_3server_runner_format_no_biomcp.json"
)

RESULTS_DIR="results/results_parallel_scaling"
mkdir -p "$RESULTS_DIR"
mkdir -p "logs"

for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"

    for TASK_FILE in "${TASK_FILES[@]}"; do
        # Extract task name from filename (strip path and extension)
        TASK_NAME=$(basename "$TASK_FILE" .json)

        # Result filename: model_{model_name}_judge_{judge_name}_task_{task_name}.jsonl
        OUTPUT_FILE="${RESULTS_DIR}/model_${MODEL_NAME}_judge_${JUDGE_NAME}_task_${TASK_NAME}.jsonl"
        OUTPUT_LOG_FILE="${RESULTS_DIR}/log_model_${MODEL_NAME}_judge_${JUDGE_NAME}_task_${TASK_NAME}.json"

        echo "========================================="
        echo "[INFO] Running benchmark:"
        echo "  Model : $MODEL_NAME"
        echo "  Judge : $JUDGE_NAME"
        echo "  Task  : $TASK_NAME"
        echo "-----------------------------------------"

        python run_benchmark.py \
            --models "$MODEL_PATH" \
            --judge-model "$JUDGE_MODEL" \
            --judge-provider "$JUDGE_PROVIDER" \
            --tasks-file "$TASK_FILE" \
            --output "$OUTPUT_FILE" \
            --output-log "$OUTPUT_LOG_FILE" \
            --temperature 0.7 \
            || { echo "[ERROR] Benchmark failed for $MODEL_NAME on $TASK_NAME"; exit 1; }

        echo "[INFO] Finished: $MODEL_NAME / $TASK_NAME"
        echo
    done
done

echo "[INFO] All benchmarks completed successfully at $(date)"
