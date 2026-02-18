#!/bin/bash
# ===============================
# Tau2 Benchmark Debug Script - Observe evaluation workflow
# Run 1 task for each domain
# ===============================

# Activate tau2bench environment
set -e
source ~/miniconda3/bin/activate tau2bench

cd /home/tianshim/agentic-long-bench/tau2-bench

echo "[INFO] Script started at $(date)"
echo "[INFO] Running on node: $(hostname)"
echo

# Model settings
AGENT_LLM="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
USER_LLM="bedrock/openai.gpt-oss-120b-1:0"

# Test parameters
MAX_CONCURRENCY=1

# Save to official data directory
RESULTS_DIR="data/simulations/results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "[INFO] Debug run for understanding evaluation"
echo "Agent LLM: $AGENT_LLM"
echo "User LLM: $USER_LLM"
echo "-----------------------------------------"

# ============ AIRLINE ============
echo
echo "========================================="
echo "[INFO] Running domain: airline"
echo "========================================="
OUTPUT_FILE="${RESULTS_DIR}/debug_airline_$(date +%Y%m%d_%H%M%S).json"
echo "[INFO] Output: $OUTPUT_FILE"

tau2 run \
    --domain airline \
    --task-set-name airline_lite \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 1 \
    --max-concurrency "$MAX_CONCURRENCY" \
    --save-to "$OUTPUT_FILE" \
    --user-llm-args '{"temperature": 0.0}' \
    --agent-llm-args '{"temperature": 0.7}' \
    --log-level INFO \
    --task-ids 0

echo "[INFO] Finished airline at $(date)"

# ============ RETAIL ============
echo
echo "========================================="
echo "[INFO] Running domain: retail"
echo "========================================="
OUTPUT_FILE="${RESULTS_DIR}/debug_retail_$(date +%Y%m%d_%H%M%S).json"
echo "[INFO] Output: $OUTPUT_FILE"

tau2 run \
    --domain retail \
    --task-set-name retail_lite \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 1 \
    --max-concurrency "$MAX_CONCURRENCY" \
    --save-to "$OUTPUT_FILE" \
    --user-llm-args '{"temperature": 0.0}' \
    --agent-llm-args '{"temperature": 0.7}' \
    --log-level INFO \
    --task-ids 0

echo "[INFO] Finished retail at $(date)"

# ============ TELECOM ============
# telecom task ids are complex strings, use --num-tasks 1 to only run the first task
echo
echo "========================================="
echo "[INFO] Running domain: telecom"
echo "========================================="
OUTPUT_FILE="${RESULTS_DIR}/debug_telecom_$(date +%Y%m%d_%H%M%S).json"
echo "[INFO] Output: $OUTPUT_FILE"

tau2 run \
    --domain telecom \
    --task-set-name telecom_lite \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials 1 \
    --max-concurrency "$MAX_CONCURRENCY" \
    --save-to "$OUTPUT_FILE" \
    --user-llm-args '{"temperature": 0.0}' \
    --agent-llm-args '{"temperature": 0.7}' \
    --log-level INFO \
    --num-tasks 1

echo "[INFO] Finished telecom at $(date)"

echo
echo "========================================="
echo "[INFO] All domains completed at $(date)"
echo "========================================="
