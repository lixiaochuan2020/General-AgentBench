#!/bin/bash
# ============================================================================
# run_sequential_scaling.sh — Unified Sequential Scaling Script for Omni
#
# Run a single model on a benchmark at multiple token budgets to measure
# how performance scales with available context window.
#
# Two modes:
#   sequential-reuse — each budget resumes from the previous budget's
#                      checkpoint (extends the agent's work)
#   independent      — each budget runs from scratch with temperature=0
#                      for deterministic prefix consistency
#
# Usage:
#   ./run_sequential_scaling.sh --model <model_path> --benchmark <bench>
#
# Examples:
#   # Default budgets (64k-128k), independent mode
#   ./run_sequential_scaling.sh \
#       --model gemini/gemini-2.5-flash \
#       --benchmark mcpbench
#
#   # Sequential reuse with checkpoint sharing
#   ./run_sequential_scaling.sh \
#       --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
#       --benchmark tau2bench \
#       --sequential-reuse
#
#   # Custom budgets for mathhay (larger context)
#   ./run_sequential_scaling.sh \
#       --model openai/gpt-5 \
#       --benchmark mathhay \
#       --budgets 128000,144000,160000,176000,192000
#
#   # Preview without executing
#   ./run_sequential_scaling.sh \
#       --model openai/gpt-5 \
#       --benchmark search \
#       --dry-run
# ============================================================================

set -euo pipefail

# ========================  Constants  ========================
VALID_BENCHMARKS="tau2bench mcpbench mathhay search swebench terminalbench"

# Default task files (relative to AGENT_DIR)
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
    ["mathhay"]="3600"
    ["search"]="3600"
    ["swebench"]="3600"
    ["terminalbench"]="3600"
)

# Per-benchmark recommended max steps (empty = let run.py decide / scaling mode ignores)
declare -A DEFAULT_MAX_STEPS=(
    ["tau2bench"]=""
    ["mcpbench"]=""
    ["mathhay"]="10"
    ["search"]=""
    ["swebench"]=""
    ["terminalbench"]=""
)

# Per-benchmark force-max-steps flag
declare -A DEFAULT_FORCE_MAX_STEPS=(
    ["tau2bench"]="no"
    ["mcpbench"]="no"
    ["mathhay"]="yes"
    ["search"]="no"
    ["swebench"]="no"
    ["terminalbench"]="no"
)

# Per-benchmark default compress-tools setting (auto = per-benchmark)
declare -A DEFAULT_COMPRESS_TOOLS=(
    ["tau2bench"]="no"
    ["mcpbench"]="no"
    ["mathhay"]="yes"
    ["search"]="yes"
    ["swebench"]="no"
    ["terminalbench"]="no"
)

# Per-benchmark default tool-description-max-len
declare -A DEFAULT_TOOL_DESC_LEN=(
    ["tau2bench"]=""
    ["mcpbench"]=""
    ["mathhay"]="75"
    ["search"]="20"
    ["swebench"]=""
    ["terminalbench"]=""
)

# Benchmarks requiring Docker cleanup
DOCKER_BENCHMARKS="swebench terminalbench"

# ========================  Defaults  ========================
MODEL=""
BENCHMARK=""
MODEL_NAME=""                  # auto-derived if empty
TASK_FILE=""                   # auto-derived per benchmark if empty
BUDGETS="64000,80000,96000,112000,128000"
DISTRACTION="all"
SCALING_SEED=42
TEMPERATURE=""                 # auto: 0.7 if sequential-reuse, 0.0 otherwise
USER_MODEL="bedrock/openai.gpt-oss-120b-1:0"
USER_TEMPERATURE="0.0"
MAX_STEPS=""                   # per-benchmark default
FORCE_MAX_STEPS=""             # per-benchmark default (auto)
TASK_TIMEOUT=""                # per-benchmark default
MAX_TOKENS=""
LOG_LEVEL="INFO"
OUTPUT_DIR=""                  # auto-derived if empty
RESUME="yes"                   # yes | no
USE_LITELLM="yes"
COMPRESS_TOOLS="auto"         # auto | yes | no
TOOL_DESC_MAX_LEN=""           # per-benchmark default
MATHHAY_HAYSTACK_LEN=""        # only for mathhay
SEQUENTIAL_REUSE="no"
CHECKPOINT_DIR=""              # auto-derived if empty
SKIP_SCORES="no"               # skip score calculation
ENV_FILE=".env"
DOCKER_CLEANUP="auto"         # auto | yes | no
DRY_RUN="no"

# Resolve AGENT_DIR: directory containing run.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to calculate_sequential_scores.py
SCORES_SCRIPT="$AGENT_DIR/scripts/calculate_sequential_scores.py"

# ========================  Usage  ========================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Sequential Scaling: run a benchmark at multiple token budgets, measure scaling.

Required:
  --model <path>               LiteLLM model path (no default)
  --benchmark <name>           Benchmark to run (no default)
                               Available: $VALID_BENCHMARKS

Scaling options:
  --budgets <list>             Comma-separated token budgets (default: $BUDGETS)
                               Budgets are auto-sorted in ascending order.
  --sequential-reuse           Enable checkpoint reuse across budgets
                               (each budget resumes from previous budget's checkpoint)
  --checkpoint-dir <path>      Checkpoint directory (default: <output-dir>/checkpoints)
  --scaling-seed <int>         Scaling seed for deterministic replay (default: $SCALING_SEED)

Model options:
  --model-name <name>          Human-readable model name (default: auto-derived)
  --temperature <float>        Sampling temperature (default: auto)
                               auto = 0.7 with --sequential-reuse, 0.0 without
  --max-tokens <int>           Max tokens per response
  --user-model <path>          User simulator model (default: $USER_MODEL)
  --user-temperature <float>   User simulator temperature (default: $USER_TEMPERATURE)
  --use-litellm                Use LiteLLM client (default: $USE_LITELLM)
  --no-litellm                 Disable LiteLLM client

Task options:
  --task-file <path>           Override task file (default: auto per benchmark)
  --distraction <count|all>    Distraction tool count (default: $DISTRACTION)
  --max-steps <int>            Max steps per task (default: per-benchmark)
  --force-max-steps            Force max_steps even in scaling mode (default: per-benchmark)
  --task-timeout <int>         Task timeout in seconds (default: per-benchmark)
  --compress-tools <mode>      Compress tool descriptions: auto|yes|no (default: $COMPRESS_TOOLS)
                               auto = yes for mathhay/search, no otherwise
  --tool-desc-max-len <int>    Max tool description length (default: per-benchmark)
  --mathhay-haystack-len <int> MathHay haystack document length (default: 80000)

Output options:
  --output-dir <path>          Base output directory (default: auto-generated)
                               Budgets are stored in <output-dir>/budget_<Nk>/
  --log-level <level>          Log level: DEBUG, INFO, WARNING (default: $LOG_LEVEL)
  --resume / --no-resume       Resume from previous run (default: resume)

Post-processing:
  --skip-scores                Skip score calculation after all budgets

Environment:
  --env-file <path>            Path to .env file (default: $ENV_FILE)
  --agent-dir <path>            Path to agent directory (default: auto-detected)
  --docker-cleanup <mode>      Docker cleanup: auto|yes|no (default: $DOCKER_CLEANUP)
                               auto = clean for swebench/terminalbench
  --dry-run                    Print commands without executing

  -h, --help                   Show this help

Benchmark-specific defaults:
  tau2bench     timeout=3600s
  mcpbench      timeout=3600s
  mathhay       timeout=3600s  compress-tools  max-steps=10  force-max-steps
  search        timeout=3600s  compress-tools  tool-desc-max-len=20
  swebench      timeout=3600s  docker-cleanup  task=swebench_test_50.json
  terminalbench timeout=3600s  docker-cleanup

Output structure:
  results/000_sequential_scaling/<model>_<bench>_distraction_<d>_<mode>/
    budget_64k/  budget_80k/  ...  budget_128k/
    checkpoints/               ← shared checkpoint store (sequential-reuse)
    sequential_scores.json     ← aggregated scaling curve
EOF
    exit 0
}

# ========================  Parse Args  ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL="$2";              shift 2;;
        --benchmark)          BENCHMARK="$2";          shift 2;;
        --budgets)            BUDGETS="$2";            shift 2;;
        --sequential-reuse)   SEQUENTIAL_REUSE="yes";  shift;;
        --checkpoint-dir)     CHECKPOINT_DIR="$2";     shift 2;;
        --scaling-seed)       SCALING_SEED="$2";       shift 2;;
        --model-name)         MODEL_NAME="$2";         shift 2;;
        --temperature)        TEMPERATURE="$2";        shift 2;;
        --max-tokens)         MAX_TOKENS="$2";         shift 2;;
        --user-model)         USER_MODEL="$2";         shift 2;;
        --user-temperature)   USER_TEMPERATURE="$2";   shift 2;;
        --use-litellm)        USE_LITELLM="yes";       shift;;
        --no-litellm)         USE_LITELLM="no";        shift;;
        --task-file)          TASK_FILE="$2";           shift 2;;
        --distraction)        DISTRACTION="$2";        shift 2;;
        --max-steps)          MAX_STEPS="$2";          shift 2;;
        --force-max-steps)    FORCE_MAX_STEPS="yes";   shift;;
        --task-timeout)       TASK_TIMEOUT="$2";       shift 2;;
        --compress-tools)     COMPRESS_TOOLS="$2";     shift 2;;
        --tool-desc-max-len)  TOOL_DESC_MAX_LEN="$2";  shift 2;;
        --mathhay-haystack-len) MATHHAY_HAYSTACK_LEN="$2"; shift 2;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2;;
        --log-level)          LOG_LEVEL="$2";          shift 2;;
        --resume)             RESUME="yes";            shift;;
        --no-resume)          RESUME="no";             shift;;
        --skip-scores)        SKIP_SCORES="yes";       shift;;
        --env-file)           ENV_FILE="$2";           shift 2;;
        --agent-dir)           AGENT_DIR="$2";           shift 2;;
        --docker-cleanup)     DOCKER_CLEANUP="$2";     shift 2;;
        --dry-run)            DRY_RUN="yes";           shift;;
        -h|--help)            usage;;
        *) echo "[ERROR] Unknown option: $1"; usage;;
    esac
done

# ========================  Validation  ========================
if [[ -z "$MODEL" ]]; then
    echo "[ERROR] --model is required."
    echo "  Example: --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    exit 1
fi

if [[ -z "$BENCHMARK" ]]; then
    echo "[ERROR] --benchmark is required."
    echo "  Available: $VALID_BENCHMARKS"
    exit 1
fi

if [[ ! " $VALID_BENCHMARKS " =~ " $BENCHMARK " ]]; then
    echo "[ERROR] Invalid benchmark: $BENCHMARK"
    echo "[INFO]  Available: $VALID_BENCHMARKS"
    exit 1
fi

# ========================  Auto-derive model name  ========================
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME=$(echo "$MODEL" \
        | sed 's|.*/||' \
        | sed 's|-[0-9]\{8\}-v[0-9]*:[0-9]*$||' \
        | sed 's|^us\.anthropic\.||' \
        | tr ':' '_')
fi

# ========================  Resolve per-benchmark defaults  ========================
# Task file
TASK_FILE="${TASK_FILE:-${DEFAULT_TASK_FILES[$BENCHMARK]}}"

# Timeout
TASK_TIMEOUT="${TASK_TIMEOUT:-${DEFAULT_TIMEOUTS[$BENCHMARK]}}"

# Max steps
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS[$BENCHMARK]:-}}"

# Force max steps
if [[ -z "$FORCE_MAX_STEPS" ]]; then
    FORCE_MAX_STEPS="${DEFAULT_FORCE_MAX_STEPS[$BENCHMARK]}"
fi

# Compress tools (auto = per-benchmark)
if [[ "$COMPRESS_TOOLS" == "auto" ]]; then
    COMPRESS_TOOLS="${DEFAULT_COMPRESS_TOOLS[$BENCHMARK]}"
fi

# Tool description max len
TOOL_DESC_MAX_LEN="${TOOL_DESC_MAX_LEN:-${DEFAULT_TOOL_DESC_LEN[$BENCHMARK]:-}}"

# Temperature: auto = 0.7 with sequential-reuse, 0.0 without
if [[ -z "$TEMPERATURE" ]]; then
    if [[ "$SEQUENTIAL_REUSE" == "yes" ]]; then
        TEMPERATURE="0.7"
    else
        TEMPERATURE="0.0"
    fi
fi

# MathHay haystack length
if [[ "$BENCHMARK" == "mathhay" && -z "$MATHHAY_HAYSTACK_LEN" ]]; then
    MATHHAY_HAYSTACK_LEN="80000"
fi

# Mode suffix for output dir
if [[ "$SEQUENTIAL_REUSE" == "yes" ]]; then
    MODE_SUFFIX="sequential"
else
    MODE_SUFFIX="independent"
fi

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-results/000_sequential_scaling/${MODEL_NAME}_${BENCHMARK}_distraction_${DISTRACTION}_${MODE_SUFFIX}}"

# Checkpoint directory
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"

# ========================  Sort budgets  ========================
cd "$AGENT_DIR"

SORTED_BUDGETS=$(python3 -c "
budgets = [int(b.strip()) for b in '$BUDGETS'.split(',')]
budgets.sort()
print(','.join(str(b) for b in budgets))
")
IFS=',' read -ra BUDGET_ARRAY <<< "$SORTED_BUDGETS"
NUM_BUDGETS=${#BUDGET_ARRAY[@]}

if [[ $NUM_BUDGETS -lt 1 ]]; then
    echo "[ERROR] No valid budgets provided."
    exit 1
fi

# ========================  Environment Setup  ========================

echo "=============================================="
echo "[INFO] Sequential Scaling Runner"
echo "[INFO] Started at $(date)"
echo "=============================================="
echo
echo "[INFO] Configuration:"
echo "  Model:            $MODEL"
echo "  Model Name:       $MODEL_NAME"
echo "  Benchmark:        $BENCHMARK"
echo "  Task File:        $TASK_FILE"
echo "  Budgets:          ${BUDGET_ARRAY[*]} ($NUM_BUDGETS levels)"
echo "  Sequential Reuse: $SEQUENTIAL_REUSE"
echo "  Temperature:      $TEMPERATURE"
echo "  Scaling Seed:     $SCALING_SEED"
echo "  Distraction:      $DISTRACTION"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Checkpoint Dir:   $CHECKPOINT_DIR"
echo "  User Model:       $USER_MODEL"
echo "  Agent Dir:         $AGENT_DIR"
if [[ "$COMPRESS_TOOLS" == "yes" ]]; then
    echo "  Compress Tools:   yes (max-len: ${TOOL_DESC_MAX_LEN:-75})"
fi
if [[ -n "$MAX_STEPS" ]]; then
    echo "  Max Steps:        $MAX_STEPS (force: $FORCE_MAX_STEPS)"
fi
if [[ -n "$MATHHAY_HAYSTACK_LEN" ]]; then
    echo "  MathHay Haystack: $MATHHAY_HAYSTACK_LEN"
fi
echo

# Verify task file exists
if [[ ! -f "$TASK_FILE" ]]; then
    echo "[ERROR] Task file not found: $TASK_FILE"
    exit 1
fi

# Verify mathhay documents
if [[ "$BENCHMARK" == "mathhay" ]] && [[ ! -f "data/mathhay_documents.json" ]]; then
    echo "[ERROR] MathHay documents file not found: data/mathhay_documents.json"
    exit 1
fi

# Count tasks
NUM_TASKS=$(python3 -c "import json; print(len(json.load(open('$TASK_FILE'))))" 2>/dev/null || echo "?")
echo "[INFO] Tasks in $TASK_FILE: $NUM_TASKS"
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

# ========================  Helper Functions  ========================

docker_cleanup() {
    echo "[INFO] Cleaning up Docker resources..."
    local nets_before nets_after
    nets_before=$(docker network ls -q 2>/dev/null | wc -l)
    docker network prune -f > /dev/null 2>&1 || true
    nets_after=$(docker network ls -q 2>/dev/null | wc -l)
    echo "[INFO]   Networks: $nets_before -> $nets_after"
    docker ps -aq --filter "name=terminalbench" 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    docker ps -aq --filter "name=swebench"      2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    docker ps -aq --filter "name=tb_eval"       2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    echo "[INFO]   Removed leftover containers"
}

needs_docker_cleanup() {
    if [[ "$DOCKER_CLEANUP" == "yes" ]]; then
        return 0
    elif [[ "$DOCKER_CLEANUP" == "auto" ]] && [[ " $DOCKER_BENCHMARKS " =~ " $BENCHMARK " ]]; then
        return 0
    fi
    return 1
}

# Convert budget number to human-readable label (e.g., 64000 -> 64k)
budget_label() {
    python3 -c "print(f'{$1 // 1000}k')"
}

# Build run.py command for a specific budget
build_budget_command() {
    local budget="$1"
    local label
    label=$(budget_label "$budget")
    local budget_dir="${OUTPUT_DIR}/budget_${label}"

    local cmd=("python3" "run.py")

    # Task file
    if [[ "$BENCHMARK" == "tau2bench" ]]; then
        cmd+=("--benchmark" "tau2")
    fi
    cmd+=("--task-file" "$TASK_FILE")

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
    cmd+=("--output-dir" "$budget_dir")

    # Token budget (the core of sequential scaling)
    cmd+=("--token-budget" "$budget")

    # Checkpoint directory
    cmd+=("--checkpoint-dir" "$CHECKPOINT_DIR")

    # Scaling seed
    cmd+=("--scaling-seed" "$SCALING_SEED")

    # Sequential reuse
    [[ "$SEQUENTIAL_REUSE" == "yes" ]] && cmd+=("--sequential-reuse")

    # Timeout
    [[ -n "$TASK_TIMEOUT" ]] && cmd+=("--task-timeout" "$TASK_TIMEOUT")

    # Steps
    [[ -n "$MAX_STEPS" ]] && cmd+=("--max-steps" "$MAX_STEPS")

    # Force max steps
    [[ "$FORCE_MAX_STEPS" == "yes" ]] && cmd+=("--force-max-steps")

    # Max tokens
    [[ -n "$MAX_TOKENS" ]] && cmd+=("--max-tokens" "$MAX_TOKENS")

    # Compress tools
    if [[ "$COMPRESS_TOOLS" == "yes" ]]; then
        cmd+=("--compress-tools")
        [[ -n "$TOOL_DESC_MAX_LEN" ]] && cmd+=("--tool-description-max-len" "$TOOL_DESC_MAX_LEN")
    fi

    # MathHay haystack length
    [[ -n "$MATHHAY_HAYSTACK_LEN" ]] && cmd+=("--mathhay-haystack-len" "$MATHHAY_HAYSTACK_LEN")

    # Log level
    cmd+=("--log-level" "$LOG_LEVEL")

    # Resume
    [[ "$RESUME" == "no" ]] && cmd+=("--no-resume")

    echo "${cmd[*]}"
}

# Build the score calculation command
build_scores_command() {
    echo "python3 $SCORES_SCRIPT --input-dir $OUTPUT_DIR --budgets $SORTED_BUDGETS --output-file $OUTPUT_DIR/sequential_scores.json --update-files"
}

# ========================  Main Execution  ========================

echo "=============================================="
echo "[INFO] Running $NUM_BUDGETS budget levels sequentially"
echo "=============================================="

BUDGET_RESULTS=()

for ((i=0; i<NUM_BUDGETS; i++)); do
    BUDGET=${BUDGET_ARRAY[$i]}
    LABEL=$(budget_label "$BUDGET")
    BUDGET_DIR="${OUTPUT_DIR}/budget_${LABEL}"

    echo
    echo "----------------------------------------------"
    echo "[Budget $((i+1))/$NUM_BUDGETS] ${LABEL} (${BUDGET} tokens)"
    echo "----------------------------------------------"

    # Docker cleanup before each budget if needed
    if needs_docker_cleanup; then
        docker_cleanup
    fi

    BUDGET_CMD=$(build_budget_command "$BUDGET")
    mkdir -p "$BUDGET_DIR"

    echo "[INFO] Command:"
    echo "  $BUDGET_CMD"
    echo

    if [[ "$DRY_RUN" == "yes" ]]; then
        BUDGET_RESULTS+=("DRY_RUN")
        continue
    fi

    LOG_FILE="${BUDGET_DIR}/run.log"
    eval "$BUDGET_CMD" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        BUDGET_RESULTS+=("SUCCESS")
        echo "[INFO] Budget ${LABEL} completed successfully."
    else
        BUDGET_RESULTS+=("FAILED (exit $EXIT_CODE)")
        echo "[WARNING] Budget ${LABEL} failed with exit code $EXIT_CODE."
    fi
done

# ---- Score Calculation ----
if [[ "$SKIP_SCORES" != "yes" && "$DRY_RUN" != "yes" ]]; then
    echo
    echo "=============================================="
    echo "[INFO] Computing sequential scaling scores..."
    echo "=============================================="
    SCORES_CMD=$(build_scores_command)
    echo "[INFO] Command: $SCORES_CMD"
    eval "$SCORES_CMD"
    if [[ -f "$OUTPUT_DIR/sequential_scores.json" ]]; then
        echo "[INFO] Scores saved to $OUTPUT_DIR/sequential_scores.json"
    fi
elif [[ "$DRY_RUN" == "yes" ]]; then
    echo
    echo "[DRY_RUN] Score calculation command:"
    echo "  $(build_scores_command)"
fi

# ---- Summary ----
echo
echo "=============================================="
echo "[INFO] Sequential Scaling Complete"
echo "[INFO] Finished at $(date)"
echo "=============================================="
echo
echo "  Model:           $MODEL_NAME ($MODEL)"
echo "  Benchmark:       $BENCHMARK"
echo "  Budgets:         ${BUDGET_ARRAY[*]}"
echo "  Sequential Reuse: $SEQUENTIAL_REUSE"
echo "  Temperature:     $TEMPERATURE"
echo "  Output:          $OUTPUT_DIR"
echo
echo "  Budget Results:"
for ((i=0; i<NUM_BUDGETS; i++)); do
    LABEL=$(budget_label "${BUDGET_ARRAY[$i]}")
    printf "    %-10s %s\n" "$LABEL" "${BUDGET_RESULTS[$i]:-NOT_RUN}"
done
echo

# Exit with error if any budget failed
for result in "${BUDGET_RESULTS[@]}"; do
    if [[ "$result" == FAILED* ]]; then
        exit 1
    fi
done

exit 0
