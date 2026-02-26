#!/bin/bash
# ============================================================================
# run_parallel_scaling.sh — Unified Parallel Scaling Script for Omni
#
# Run a single model on a benchmark multiple times (passes) with different
# random seeds, then compute Best@K statistics.
#
# Best@K: for each task, take the max reward across K passes, then average.
#
# Supports two execution modes:
#   sequential — runs passes one-by-one in a for loop (simple, reliable)
#   tmux       — launches each pass in a separate tmux session (fast)
#
# Usage:
#   ./run_parallel_scaling.sh --model <model_path> --benchmark <bench>
#
# Examples:
#   # 4 sequential passes on tau2bench (defaults)
#   ./run_parallel_scaling.sh \
#       --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
#       --benchmark tau2bench
#
#   # 8 parallel passes via tmux on swebench
#   ./run_parallel_scaling.sh \
#       --model gemini/gemini-2.5-flash \
#       --benchmark swebench \
#       --num-passes 8 \
#       --mode tmux
#
#   # Preview commands without executing
#   ./run_parallel_scaling.sh \
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

# Benchmarks requiring Docker cleanup before each pass
DOCKER_BENCHMARKS="swebench terminalbench"

# ========================  Defaults  ========================
MODEL=""
BENCHMARK=""
MODEL_NAME=""                  # auto-derived if empty
TASK_FILE=""                   # auto-derived per benchmark if empty
NUM_PASSES=4
BASE_SEED=42
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
COMPRESS_TOOLS="auto"         # auto | yes | no  (auto = yes for mathhay)
TOOL_DESC_MAX_LEN="75"
MODE="sequential"             # sequential | tmux
ENV_FILE=".env"
DOCKER_CLEANUP="auto"         # auto | yes | no
SKIP_BESTK="no"               # skip Best@K calculation
DRY_RUN="no"

# Resolve AGENT_DIR: directory containing run.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to calculate_parallel_scale_scores.py
BESTK_SCRIPT="$AGENT_DIR/scripts/calculate_parallel_scale_scores.py"

# ========================  Usage  ========================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Parallel Scaling: run a benchmark N times with different seeds, compute Best@K.

Required:
  --model <path>               LiteLLM model path (no default)
  --benchmark <name>           Benchmark to run (no default)
                               Available: $VALID_BENCHMARKS

Scaling options:
  --num-passes <int>           Number of passes (default: $NUM_PASSES)
  --base-seed <int>            Base random seed; pass i uses seed+i-1 (default: $BASE_SEED)
  --mode <mode>                Execution mode: sequential|tmux (default: $MODE)
                               sequential = run passes one by one, then compute Best@K
                               tmux       = launch passes in parallel tmux sessions

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
  --compress-tools <mode>      Compress tool descriptions: auto|yes|no (default: $COMPRESS_TOOLS)
                               auto = yes for mathhay, no otherwise
  --tool-desc-max-len <int>    Max tool description length (default: $TOOL_DESC_MAX_LEN)

Output options:
  --output-dir <path>          Base output directory (default: auto-generated)
                               Passes are stored in <output-dir>/pass_1/, pass_2/, ...
  --log-level <level>          Log level: DEBUG, INFO, WARNING (default: $LOG_LEVEL)
  --resume / --no-resume       Resume from previous run (default: resume)

Post-processing:
  --skip-bestk                 Skip Best@K calculation after sequential passes

Environment:
  --env-file <path>            Path to .env file (default: $ENV_FILE)
  --agent-dir <path>            Path to agent directory (default: auto-detected)
  --docker-cleanup <mode>      Docker cleanup: auto|yes|no (default: $DOCKER_CLEANUP)
                               auto = clean for swebench/terminalbench
  --dry-run                    Print commands without executing

  -h, --help                   Show this help

Benchmark-specific defaults:
  tau2bench     timeout=3600s  task=tau2bench_benchmark.json
  mcpbench      timeout=3600s  task=mcpbench_benchmark.json
  mathhay       timeout=600s   task=mathhay_benchmark.json    compress-tools=yes
  search        timeout=1800s  task=search_benchmark.json     max-steps=100
  swebench      timeout=3600s  task=swebench_test_50.json     docker-cleanup
  terminalbench timeout=3600s  task=terminalbench_benchmark.json docker-cleanup

Output structure:
  results/000_parallel_scaling/<model_name>_<benchmark>_distraction_<d>/
    pass_1/  pass_2/  ...  pass_N/
    summary.json  ← Best@K aggregated results
EOF
    exit 0
}

# ========================  Parse Args  ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL="$2";              shift 2;;
        --benchmark)          BENCHMARK="$2";          shift 2;;
        --num-passes)         NUM_PASSES="$2";         shift 2;;
        --base-seed)          BASE_SEED="$2";          shift 2;;
        --mode)               MODE="$2";               shift 2;;
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
        --task-timeout)       TASK_TIMEOUT="$2";       shift 2;;
        --compress-tools)     COMPRESS_TOOLS="$2";     shift 2;;
        --tool-desc-max-len)  TOOL_DESC_MAX_LEN="$2";  shift 2;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2;;
        --log-level)          LOG_LEVEL="$2";          shift 2;;
        --resume)             RESUME="yes";            shift;;
        --no-resume)          RESUME="no";             shift;;
        --skip-bestk)         SKIP_BESTK="yes";        shift;;
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

if [[ "$MODE" != "sequential" && "$MODE" != "tmux" ]]; then
    echo "[ERROR] --mode must be 'sequential' or 'tmux' (got: $MODE)"
    exit 1
fi

if [[ "$NUM_PASSES" -lt 1 ]]; then
    echo "[ERROR] --num-passes must be >= 1 (got: $NUM_PASSES)"
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

# Compress tools (auto = yes for mathhay)
if [[ "$COMPRESS_TOOLS" == "auto" ]]; then
    if [[ "$BENCHMARK" == "mathhay" ]]; then
        COMPRESS_TOOLS="yes"
    else
        COMPRESS_TOOLS="no"
    fi
fi

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-results/000_parallel_scaling/${MODEL_NAME}_${BENCHMARK}_distraction_${DISTRACTION}}"

# ========================  Environment Setup  ========================
cd "$AGENT_DIR"

echo "=============================================="
echo "[INFO] Parallel Scaling Runner"
echo "[INFO] Started at $(date)"
echo "=============================================="
echo
echo "[INFO] Configuration:"
echo "  Model:         $MODEL"
echo "  Model Name:    $MODEL_NAME"
echo "  Benchmark:     $BENCHMARK"
echo "  Task File:     $TASK_FILE"
echo "  Num Passes:    $NUM_PASSES"
echo "  Base Seed:     $BASE_SEED"
echo "  Temperature:   $TEMPERATURE"
echo "  Distraction:   $DISTRACTION"
echo "  Mode:          $MODE"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  User Model:    $USER_MODEL"
echo "  Agent Dir:      $AGENT_DIR"
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

# Build run.py command for a specific pass
build_pass_command() {
    local pass_num="$1"
    local pass_seed=$(( BASE_SEED + pass_num - 1 ))
    local pass_dir="${OUTPUT_DIR}/pass_${pass_num}"

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
    cmd+=("--output-dir" "$pass_dir")

    # Scaling seed
    cmd+=("--scaling-seed" "$pass_seed")

    # Timeout
    [[ -n "$TASK_TIMEOUT" ]] && cmd+=("--task-timeout" "$TASK_TIMEOUT")

    # Steps
    [[ -n "$MAX_STEPS" ]] && cmd+=("--max-steps" "$MAX_STEPS")

    # Max tokens
    [[ -n "$MAX_TOKENS" ]] && cmd+=("--max-tokens" "$MAX_TOKENS")

    # Compress tools
    if [[ "$COMPRESS_TOOLS" == "yes" ]]; then
        cmd+=("--compress-tools")
        [[ -n "$TOOL_DESC_MAX_LEN" ]] && cmd+=("--tool-description-max-len" "$TOOL_DESC_MAX_LEN")
    fi

    # Log level
    cmd+=("--log-level" "$LOG_LEVEL")

    # Resume
    [[ "$RESUME" == "no" ]] && cmd+=("--no-resume")

    echo "${cmd[*]}"
}

# Build the Best@K calculation command
build_bestk_command() {
    echo "python3 $BESTK_SCRIPT --input-dir $OUTPUT_DIR --num-passes $NUM_PASSES --output-file $OUTPUT_DIR/summary.json"
}

# ========================  Main Execution  ========================

if [[ "$MODE" == "sequential" ]]; then
    # ---- Sequential mode: run passes one by one ----
    echo "=============================================="
    echo "[INFO] Sequential mode: running $NUM_PASSES passes"
    echo "=============================================="

    PASS_RESULTS=()

    for pass_num in $(seq 1 "$NUM_PASSES"); do
        echo
        echo "----------------------------------------------"
        echo "[Pass $pass_num/$NUM_PASSES] Seed: $((BASE_SEED + pass_num - 1))"
        echo "----------------------------------------------"

        # Docker cleanup before each pass if needed
        if needs_docker_cleanup; then
            docker_cleanup
        fi

        PASS_CMD=$(build_pass_command "$pass_num")
        PASS_DIR="${OUTPUT_DIR}/pass_${pass_num}"
        mkdir -p "$PASS_DIR"

        echo "[INFO] Command:"
        echo "  $PASS_CMD"
        echo

        if [[ "$DRY_RUN" == "yes" ]]; then
            PASS_RESULTS+=("DRY_RUN")
            continue
        fi

        LOG_FILE="${PASS_DIR}/run.log"
        eval "$PASS_CMD" 2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}

        if [[ $EXIT_CODE -eq 0 ]]; then
            PASS_RESULTS+=("SUCCESS")
            echo "[INFO] Pass $pass_num completed successfully."
        else
            PASS_RESULTS+=("FAILED (exit $EXIT_CODE)")
            echo "[ERROR] Pass $pass_num failed with exit code $EXIT_CODE."
        fi
    done

    # ---- Best@K Calculation ----
    if [[ "$SKIP_BESTK" != "yes" && "$DRY_RUN" != "yes" ]]; then
        echo
        echo "=============================================="
        echo "[INFO] Computing Best@K statistics..."
        echo "=============================================="
        BESTK_CMD=$(build_bestk_command)
        echo "[INFO] Command: $BESTK_CMD"
        eval "$BESTK_CMD"
        if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
            echo "[INFO] Best@K results saved to $OUTPUT_DIR/summary.json"
            echo "[INFO] Summary:"
            python3 -c "
import json
with open('$OUTPUT_DIR/summary.json') as f:
    s = json.load(f)
for k, v in s.items():
    if isinstance(v, dict):
        print(f'  {k}:')
        for k2, v2 in v.items():
            if isinstance(v2, float):
                print(f'    {k2}: {v2:.4f}')
            else:
                print(f'    {k2}: {v2}')
    elif isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')
" 2>/dev/null || true
        fi
    elif [[ "$DRY_RUN" == "yes" ]]; then
        echo
        echo "[DRY_RUN] Best@K command:"
        echo "  $(build_bestk_command)"
    fi

    # ---- Summary ----
    echo
    echo "=============================================="
    echo "[INFO] Parallel Scaling Complete"
    echo "[INFO] Finished at $(date)"
    echo "=============================================="
    echo
    echo "  Model:      $MODEL_NAME ($MODEL)"
    echo "  Benchmark:  $BENCHMARK"
    echo "  Passes:     $NUM_PASSES"
    echo "  Seed Range: $BASE_SEED — $((BASE_SEED + NUM_PASSES - 1))"
    echo "  Output:     $OUTPUT_DIR"
    echo
    echo "  Pass Results:"
    for i in $(seq 1 "$NUM_PASSES"); do
        printf "    pass_%-4d %s\n" "$i" "${PASS_RESULTS[$((i-1))]:-NOT_RUN}"
    done
    echo

    # Exit with error if any pass failed
    for result in "${PASS_RESULTS[@]}"; do
        if [[ "$result" == FAILED* ]]; then
            exit 1
        fi
    done

elif [[ "$MODE" == "tmux" ]]; then
    # ---- Tmux mode: launch each pass in a parallel tmux session ----
    echo "=============================================="
    echo "[INFO] Tmux mode: launching $NUM_PASSES parallel sessions"
    echo "=============================================="

    # Build a short prefix for tmux session names
    SESSION_PREFIX="${MODEL_NAME:0:10}_${BENCHMARK:0:6}"
    # Sanitize for tmux (replace dots and slashes)
    SESSION_PREFIX=$(echo "$SESSION_PREFIX" | tr './' '__')

    echo
    for pass_num in $(seq 1 "$NUM_PASSES"); do
        SESSION_NAME="${SESSION_PREFIX}_p${pass_num}"
        PASS_CMD=$(build_pass_command "$pass_num")
        PASS_DIR="${OUTPUT_DIR}/pass_${pass_num}"

        echo "[Pass $pass_num] Session: $SESSION_NAME"
        echo "  Seed: $((BASE_SEED + pass_num - 1))"
        echo "  Dir:  $PASS_DIR"
        echo "  Cmd:  $PASS_CMD"
        echo

        if [[ "$DRY_RUN" == "yes" ]]; then
            continue
        fi

        mkdir -p "$PASS_DIR"

        # Build the full tmux session command
        # 1. cd to agent dir
        # 2. source .env if available
        # 3. docker cleanup if needed
        # 4. run the pass command
        TMUX_CMD="cd $AGENT_DIR"
        if [[ -f "$ENV_FILE" ]]; then
            TMUX_CMD="$TMUX_CMD && set -a && source $ENV_FILE && set +a"
        fi
        if needs_docker_cleanup; then
            TMUX_CMD="$TMUX_CMD && docker network prune -f 2>/dev/null || true"
            TMUX_CMD="$TMUX_CMD && docker ps -aq --filter name=terminalbench | xargs -r docker rm -f 2>/dev/null || true"
            TMUX_CMD="$TMUX_CMD && docker ps -aq --filter name=swebench | xargs -r docker rm -f 2>/dev/null || true"
        fi
        TMUX_CMD="$TMUX_CMD && $PASS_CMD 2>&1 | tee $PASS_DIR/run.log"

        # Kill existing session with same name (if any)
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

        # Launch
        tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"
        echo "[INFO] Launched tmux session: $SESSION_NAME"

        # Small delay to avoid resource contention
        sleep 0.5
    done

    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "[DRY_RUN] No tmux sessions launched."
    else
        echo
        echo "=============================================="
        echo "[INFO] All $NUM_PASSES tmux sessions launched."
        echo "=============================================="
        echo
        echo "Monitor sessions:"
        echo "  tmux ls                            # list all sessions"
        echo "  tmux attach -t ${SESSION_PREFIX}_p1  # attach to pass 1"
        echo "  tmux kill-session -t <name>        # kill a session"
        echo
        echo "After all passes finish, compute Best@K:"
        echo "  cd $AGENT_DIR"
        echo "  $(build_bestk_command)"
        echo
    fi
fi

exit 0
