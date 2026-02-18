#!/bin/bash
# ============================================================================
# run_baseline.sh — Unified Baseline Benchmark Runner for Omni
#
# Run a single model on one or more benchmarks with sensible defaults.
# All tuneable parameters are exposed as command-line flags.
#
# Usage:
#   ./run_baseline.sh --model <model_path> --benchmark <bench>[,<bench>,...]
#
# Examples:
#   # Run tau2bench with Claude Haiku 4.5
#   ./run_baseline.sh \
#       --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
#       --benchmark tau2bench
#
#   # Run all 6 benchmarks with Gemini 2.5 Flash
#   ./run_baseline.sh \
#       --model gemini/gemini-2.5-flash \
#       --benchmark tau2bench,mcpbench,mathhay,search,swebench,terminalbench
#
#   # Run search with custom output dir and timeout
#   ./run_baseline.sh \
#       --model openai/gpt-5 \
#       --benchmark search \
#       --task-timeout 2400 \
#       --output-dir results/my_run
# ============================================================================

set -euo pipefail

# ========================  Constants  ========================
VALID_BENCHMARKS="tau2bench mcpbench mathhay search swebench terminalbench"

# Default task files (benchmark -> file, relative to AGENT_DIR)
declare -A DEFAULT_TASK_FILES=(
    ["tau2bench"]="data/tau2bench_benchmark.json"
    ["mcpbench"]="data/mcpbench_benchmark.json"
    ["mathhay"]="data/mathhay_benchmark.json"
    ["search"]="data/search_benchmark.json"
    ["swebench"]="data/swebench_test_50.json"
    ["terminalbench"]="data/terminalbench_benchmark.json"
)

# Per-benchmark recommended task timeouts (seconds)
declare -A DEFAULT_TIMEOUTS=(
    ["tau2bench"]="3600"
    ["mcpbench"]="3600"
    ["mathhay"]="600"
    ["search"]="1800"
    ["swebench"]="3600"
    ["terminalbench"]="3600"
)

# Per-benchmark recommended max steps (empty = let run.py decide)
declare -A DEFAULT_MAX_STEPS=(
    ["tau2bench"]=""
    ["mcpbench"]=""
    ["mathhay"]=""
    ["search"]="100"
    ["swebench"]=""
    ["terminalbench"]=""
)

# ========================  Defaults  ========================
MODEL=""
BENCHMARKS=""
MODEL_NAME=""                  # auto-derived if empty
TASK_FILE=""                   # auto-derived per benchmark if empty
TEMPERATURE="0.7"
USER_MODEL="bedrock/openai.gpt-oss-120b-1:0"
USER_TEMPERATURE="0.0"
DISTRACTION="all"
MAX_STEPS=""                   # per-benchmark default or run.py default
TASK_TIMEOUT=""                # per-benchmark default
MAX_TOKENS=""
LOG_LEVEL="INFO"
OUTPUT_DIR=""                  # auto-derived if empty
RESUME="yes"                   # yes | no
USE_LITELLM="yes"
COMPRESS_TOOLS="no"
TOOL_DESC_MAX_LEN=""
SIMULATION_SEED=""
TOOL_SEED=""
ENV_FILE=".env"
DOCKER_CLEANUP="auto"         # auto | yes | no
DRY_RUN="no"

# Resolve AGENT_DIR: the general_agent/ directory that contains run.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ========================  Usage  ========================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --model <path>               LiteLLM model path (no default)
  --benchmark <b1>[,<b2>,...]  Comma-separated benchmarks to run (no default)
                               Available: $VALID_BENCHMARKS

Model options:
  --model-name <name>          Human-readable model name (default: auto-derived)
  --temperature <float>        Sampling temperature (default: $TEMPERATURE)
  --max-tokens <int>           Max tokens per response
  --user-model <path>          User simulator model (default: $USER_MODEL)
  --user-temperature <float>   User simulator temperature (default: $USER_TEMPERATURE)
  --use-litellm                Use LiteLLM client (default: $USE_LITELLM)
  --no-litellm                 Disable LiteLLM client

Task options:
  --task-file <path>           Override task file (default: auto per benchmark)
  --distraction <count|all>    Distraction tool count (default: $DISTRACTION)
  --max-steps <int>            Max steps per task (default: per-benchmark)
  --task-timeout <int>         Task timeout in seconds (default: per-benchmark)
  --compress-tools             Compress tool descriptions
  --tool-desc-max-len <int>    Max tool description length (default: 75)
  --simulation-seed <int>      Simulation random seed
  --tool-seed <int>            Tool selection random seed

Output options:
  --output-dir <path>          Output directory (default: auto-generated)
  --log-level <level>          Log level: DEBUG, INFO, WARNING (default: $LOG_LEVEL)
  --resume / --no-resume       Resume from previous run (default: resume)

Environment:
  --env-file <path>            Path to .env file (default: $ENV_FILE)
  --agent-dir <path>            Path to agent directory (default: $AGENT_DIR)
  --docker-cleanup <mode>      Docker cleanup: auto|yes|no (default: $DOCKER_CLEANUP)
                               auto = clean only for terminalbench/swebench
  --dry-run                    Print the run.py command without executing

  -h, --help                   Show this help

Benchmark-specific defaults:
  tau2bench     timeout=3600s
  mcpbench      timeout=3600s
  mathhay       timeout=600s
  search        timeout=1800s  max-steps=100
  swebench      timeout=3600s
  terminalbench timeout=3600s
EOF
    exit 0
}

# ========================  Parse Args  ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";              shift 2;;
        --benchmark)    BENCHMARKS="$2";         shift 2;;
        --model-name)   MODEL_NAME="$2";         shift 2;;
        --temperature)  TEMPERATURE="$2";        shift 2;;
        --max-tokens)   MAX_TOKENS="$2";         shift 2;;
        --user-model)   USER_MODEL="$2";         shift 2;;
        --user-temperature) USER_TEMPERATURE="$2"; shift 2;;
        --use-litellm)  USE_LITELLM="yes";       shift;;
        --no-litellm)   USE_LITELLM="no";        shift;;
        --task-file)    TASK_FILE="$2";           shift 2;;
        --distraction)  DISTRACTION="$2";        shift 2;;
        --max-steps)    MAX_STEPS="$2";          shift 2;;
        --task-timeout) TASK_TIMEOUT="$2";       shift 2;;
        --compress-tools) COMPRESS_TOOLS="yes";  shift;;
        --tool-desc-max-len) TOOL_DESC_MAX_LEN="$2"; shift 2;;
        --simulation-seed) SIMULATION_SEED="$2"; shift 2;;
        --tool-seed)    TOOL_SEED="$2";          shift 2;;
        --output-dir)   OUTPUT_DIR="$2";         shift 2;;
        --log-level)    LOG_LEVEL="$2";          shift 2;;
        --resume)       RESUME="yes";            shift;;
        --no-resume)    RESUME="no";             shift;;
        --env-file)     ENV_FILE="$2";           shift 2;;
        --agent-dir)     AGENT_DIR="$2";           shift 2;;
        --docker-cleanup) DOCKER_CLEANUP="$2";   shift 2;;
        --dry-run)      DRY_RUN="yes";           shift;;
        -h|--help)      usage;;
        *) echo "[ERROR] Unknown option: $1"; usage;;
    esac
done

# ========================  Validation  ========================
if [[ -z "$MODEL" ]]; then
    echo "[ERROR] --model is required."
    echo "  Example: --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    exit 1
fi

if [[ -z "$BENCHMARKS" ]]; then
    echo "[ERROR] --benchmark is required."
    echo "  Available: $VALID_BENCHMARKS"
    echo "  Use comma-separated list for multiple: --benchmark tau2bench,mcpbench"
    exit 1
fi

# Parse comma-separated benchmarks
BENCHMARK_LIST=$(echo "$BENCHMARKS" | tr ',' ' ')
for bench in $BENCHMARK_LIST; do
    if [[ ! " $VALID_BENCHMARKS " =~ " $bench " ]]; then
        echo "[ERROR] Invalid benchmark: $bench"
        echo "[INFO]  Available: $VALID_BENCHMARKS"
        exit 1
    fi
done

# ========================  Auto-derive model name  ========================
if [[ -z "$MODEL_NAME" ]]; then
    # Extract readable name from model path:
    #   bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 -> claude-haiku-4-5
    #   gemini/gemini-2.5-flash -> gemini-2.5-flash
    #   huggingface/deepseek-ai/DeepSeek-V3.2:novita -> DeepSeek-V3.2_novita
    MODEL_NAME=$(echo "$MODEL" \
        | sed 's|.*/||' \
        | sed 's|-[0-9]\{8\}-v[0-9]*:[0-9]*$||' \
        | sed 's|^us\.anthropic\.||' \
        | tr ':' '_')
fi

# ========================  Environment Setup  ========================
cd "$AGENT_DIR"

echo "=============================================="
echo "[INFO] Omni Baseline Benchmark Runner"
echo "[INFO] Started at $(date)"
echo "=============================================="
echo
echo "[INFO] Configuration:"
echo "  Model:        $MODEL"
echo "  Model Name:   $MODEL_NAME"
echo "  Benchmarks:   $BENCHMARK_LIST"
echo "  Distraction:  $DISTRACTION"
echo "  Temperature:  $TEMPERATURE"
echo "  User Model:   $USER_MODEL"
echo "  Agent Dir:     $AGENT_DIR"
echo

# Load .env for API keys
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[INFO] Loaded environment from $ENV_FILE"
elif [[ "$ENV_FILE" != ".env" ]]; then
    echo "[ERROR] Specified env file not found: $ENV_FILE"
    exit 1
else
    echo "[WARNING] .env file not found; using existing environment variables."
fi

# Create directories
mkdir -p logs

# ========================  Helper Functions  ========================

docker_cleanup() {
    echo "[INFO] Cleaning up Docker resources..."
    # Prune networks
    local nets_before nets_after
    nets_before=$(docker network ls -q 2>/dev/null | wc -l)
    docker network prune -f > /dev/null 2>&1 || true
    nets_after=$(docker network ls -q 2>/dev/null | wc -l)
    echo "[INFO]   Networks: $nets_before -> $nets_after"
    # Remove leftover containers
    docker ps -aq --filter "name=terminalbench" 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    docker ps -aq --filter "name=swebench"      2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    docker ps -aq --filter "name=tb_eval"       2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    echo "[INFO]   Removed leftover containers"
}

# Build the run.py command for a given benchmark
build_command() {
    local bench="$1"

    # Resolve task file
    local tf="${TASK_FILE:-${DEFAULT_TASK_FILES[$bench]}}"

    # Resolve timeout (command-line > per-benchmark default)
    local timeout="${TASK_TIMEOUT:-${DEFAULT_TIMEOUTS[$bench]}}"

    # Resolve max steps (command-line > per-benchmark default)
    local steps="${MAX_STEPS:-${DEFAULT_MAX_STEPS[$bench]:-}}"

    # Resolve output dir
    local odir="${OUTPUT_DIR:-results/${bench}_${MODEL_NAME}_distraction_${DISTRACTION}}"

    # Verify task file exists
    if [[ ! -f "$tf" ]]; then
        echo "[ERROR] Task file not found: $tf"
        return 1
    fi

    # Additional check for mathhay
    if [[ "$bench" == "mathhay" ]] && [[ ! -f "data/mathhay_documents.json" ]]; then
        echo "[ERROR] MathHay documents file not found: data/mathhay_documents.json"
        return 1
    fi

    mkdir -p "$odir"

    # Count tasks
    local num_tasks
    num_tasks=$(python3 -c "import json; print(len(json.load(open('$tf'))))" 2>/dev/null || echo "?")
    echo "[INFO] Tasks in $tf: $num_tasks"

    # ---- Assemble command ----
    local cmd=("python3" "run.py")

    # Task file (tau2bench prefers --benchmark flag)
    if [[ "$bench" == "tau2bench" ]]; then
        cmd+=("--benchmark" "tau2")
    fi
    cmd+=("--task-file" "$tf")

    # Model
    cmd+=("--model" "$MODEL")
    cmd+=("--user-model" "$USER_MODEL")

    # LiteLLM
    [[ "$USE_LITELLM" == "yes" ]] && cmd+=("--use-litellm")

    # Temperatures
    cmd+=("--temperature" "$TEMPERATURE")
    cmd+=("--user-temperature" "$USER_TEMPERATURE")

    # Distraction
    cmd+=("--distraction-count" "$DISTRACTION")

    # Output
    cmd+=("--output-dir" "$odir")

    # Timeout
    [[ -n "$timeout" ]] && cmd+=("--task-timeout" "$timeout")

    # Steps
    [[ -n "$steps" ]] && cmd+=("--max-steps" "$steps")

    # Max tokens
    [[ -n "$MAX_TOKENS" ]] && cmd+=("--max-tokens" "$MAX_TOKENS")

    # Compress tools
    if [[ "$COMPRESS_TOOLS" == "yes" ]]; then
        cmd+=("--compress-tools")
        [[ -n "$TOOL_DESC_MAX_LEN" ]] && cmd+=("--tool-description-max-len" "$TOOL_DESC_MAX_LEN")
    fi

    # Seeds
    [[ -n "$SIMULATION_SEED" ]] && cmd+=("--simulation-seed" "$SIMULATION_SEED")
    [[ -n "$TOOL_SEED" ]] && cmd+=("--tool-seed" "$TOOL_SEED")

    # Log level
    cmd+=("--log-level" "$LOG_LEVEL")

    # Resume
    if [[ "$RESUME" == "no" ]]; then
        cmd+=("--no-resume")
    fi

    # Return command as a single string (stored in global CMD_STR)
    CMD_STR="${cmd[*]}"
    CMD_LOG="$odir/run.log"
}

# ========================  Main Loop  ========================
TOTAL=$(echo "$BENCHMARK_LIST" | wc -w)
CURRENT=0
declare -A RESULTS

for bench in $BENCHMARK_LIST; do
    ((CURRENT++)) || true
    echo
    echo "=============================================="
    echo "[$CURRENT/$TOTAL] Benchmark: $bench"
    echo "=============================================="

    # Docker cleanup for container-based benchmarks
    if [[ "$DOCKER_CLEANUP" == "yes" ]] || \
       { [[ "$DOCKER_CLEANUP" == "auto" ]] && [[ "$bench" == "terminalbench" || "$bench" == "swebench" ]]; }; then
        docker_cleanup
    fi

    # Build command
    CMD_STR=""
    CMD_LOG=""
    if ! build_command "$bench"; then
        RESULTS["$bench"]="SKIPPED"
        continue
    fi

    echo "[INFO] Command:"
    echo "  $CMD_STR"
    echo

    if [[ "$DRY_RUN" == "yes" ]]; then
        RESULTS["$bench"]="DRY_RUN"
        continue
    fi

    # Execute
    eval "$CMD_STR" 2>&1 | tee "$CMD_LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        RESULTS["$bench"]="SUCCESS"
        echo "[INFO] $bench completed successfully."
    else
        RESULTS["$bench"]="FAILED (exit $EXIT_CODE)"
        echo "[ERROR] $bench failed with exit code $EXIT_CODE."
    fi

    # Print summary.json if available
    local_odir="${OUTPUT_DIR:-results/${bench}_${MODEL_NAME}_distraction_${DISTRACTION}}"
    if [[ -f "$local_odir/summary.json" ]]; then
        echo "[INFO] Results summary:"
        python3 -c "
import json, sys
with open('$local_odir/summary.json') as f:
    s = json.load(f)
overall = s.get('overall', s)
for k, v in overall.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')
" 2>/dev/null || true
    fi
done

# ========================  Final Summary  ========================
echo
echo "=============================================="
echo "[INFO] Baseline Run Complete"
echo "[INFO] Finished at $(date)"
echo "=============================================="
echo
echo "  Model:   $MODEL_NAME ($MODEL)"
echo
echo "  Results:"
for bench in $BENCHMARK_LIST; do
    printf "    %-16s %s\n" "$bench" "${RESULTS[$bench]:-NOT_RUN}"
done
echo

# Exit with error if any benchmark failed
for bench in $BENCHMARK_LIST; do
    if [[ "${RESULTS[$bench]:-}" == FAILED* ]]; then
        exit 1
    fi
done
exit 0
