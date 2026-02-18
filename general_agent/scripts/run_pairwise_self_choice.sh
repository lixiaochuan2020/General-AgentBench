#!/bin/bash
# ============================================================================
# run_pairwise_self_choice.sh — Unified Pairwise Self-Choice (Bump Sort) Script
#
# Compare agent trajectories pairwise using an LLM judge, then use Bump Sort
# to find the best trajectory across K passes. Computes Best@K incrementally.
#
# Bump Sort: sequential tournament — current champion vs next pass, winner
# becomes new champion. K passes require K−1 comparisons.
#
# Two modes:
#   self-eval  — model judges its own trajectories (default)
#   judge      — an external model judges trajectories
#
# Usage:
#   ./run_pairwise_self_choice.sh --benchmark <bench> --source <dir>
#
# Examples:
#   # Self-evaluation (model auto-detected from source directory name)
#   ./run_pairwise_self_choice.sh \
#       --benchmark tau2bench \
#       --source results/000_parallel_scaling/Qwen3-235B_tau2bench_distraction_all
#
#   # External judge with GPT-5
#   ./run_pairwise_self_choice.sh \
#       --model openai/gpt-5 \
#       --benchmark swebench \
#       --source results/000_parallel_scaling/DeepSeek-V3.2_swebench_distraction_all
#
#   # With minimal tools and dry-run preview
#   ./run_pairwise_self_choice.sh \
#       --benchmark terminalbench \
#       --source results/000_parallel_scaling/Model_terminalbench_distraction_all \
#       --minimal-tools --dry-run
# ============================================================================

set -euo pipefail

# ========================  Constants  ========================
VALID_BENCHMARKS="tau2bench mcpbench mathhay search swebench terminalbench"

# ========================  Defaults  ========================
MODEL=""                       # empty = auto-detect for self-evaluation
BENCHMARK=""
SOURCE_DIR=""
OUTPUT_DIR=""                  # auto-derived if empty
THRESHOLD="0.5"
COMPRESS_TOOLS="no"            # yes | no
MINIMAL_TOOLS="no"             # yes | no
ENV_FILE=".env"
DRY_RUN="no"

# Resolve AGENT_DIR: directory containing run.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ========================  Usage  ========================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Pairwise Self-Choice (Bump Sort): compare trajectories pairwise to find the best.

Bump Sort performs a sequential tournament: current champion vs next pass,
winner becomes new champion. K passes → K−1 LLM comparisons. Best@K is
computed incrementally at each round.

Required:
  --benchmark <name>           Benchmark to evaluate (no default)
                               Available: $VALID_BENCHMARKS
  --source <dir>               Parallel scaling results directory with pass_1/, pass_2/, ...

Model options:
  --model <path>               Judge model (LiteLLM path, no default)
                               If omitted, the model is auto-detected from the source
                               directory name for self-evaluation mode.
                               Example: openai/gpt-5, gemini/gemini-2.5-flash

Evaluation options:
  --threshold <float>          Success threshold for is_correct (default: $THRESHOLD)
  --compress-tools             Compress tool descriptions to reduce tokens
  --minimal-tools              Most aggressive compression (text format, truncated)

Output options:
  --output-dir <path>          Output directory (default: auto-generated)
                               Self-eval: results/002_bump_sort/{name}_selfjudge
                               Judge:     results/002_bump_sort/{name}_judge-{model}

Environment:
  --env-file <path>            Path to .env file (default: $ENV_FILE)
  --agent-dir <path>            Path to agent directory (default: auto-detected)
  --dry-run                    Print commands without executing

  -h, --help                   Show this help

Examples:
  # Self-evaluation with auto-detected model
  $(basename "$0") --benchmark tau2bench \\
      --source results/000_parallel_scaling/Model_tau2bench_distraction_all

  # External judge with minimal tools
  $(basename "$0") --model openai/gpt-5 --benchmark search \\
      --source results/000_parallel_scaling/Model_search_distraction_all \\
      --minimal-tools

  # MCPBench with custom threshold
  $(basename "$0") --benchmark mcpbench \\
      --source results/000_parallel_scaling/Model_mcpbench_distraction_100 \\
      --threshold 0.3

Output structure:
  results/002_bump_sort/{name}_selfjudge/
    per_task/
      {task_id}.json             per-task comparison history + best@k
    bump_sort_results.json       aggregated summary (accuracy, oracle, best@k)
EOF
    exit 0
}

# ========================  Parse Args  ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL="$2";              shift 2;;
        --benchmark)          BENCHMARK="$2";          shift 2;;
        --source)             SOURCE_DIR="$2";         shift 2;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2;;
        --threshold)          THRESHOLD="$2";          shift 2;;
        --compress-tools)     COMPRESS_TOOLS="yes";    shift;;
        --minimal-tools)      MINIMAL_TOOLS="yes";     shift;;
        --env-file)           ENV_FILE="$2";           shift 2;;
        --agent-dir)           AGENT_DIR="$2";           shift 2;;
        --dry-run)            DRY_RUN="yes";           shift;;
        -h|--help)            usage;;
        *) echo "[ERROR] Unknown option: $1"; usage;;
    esac
done

# ========================  Validation  ========================
if [[ -z "$SOURCE_DIR" ]]; then
    echo "[ERROR] --source is required."
    echo "  Example: --source results/000_parallel_scaling/Model_benchmark_distraction_all"
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

# Verify source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "[ERROR] Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Verify at least pass_1 and pass_2 exist (need ≥2 passes for pairwise comparison)
if [[ ! -d "$SOURCE_DIR/pass_1" ]]; then
    echo "[ERROR] No pass_1/ found in source directory: $SOURCE_DIR"
    echo "[INFO]  Source must have pass_1/, pass_2/, ... from parallel scaling"
    exit 1
fi
if [[ ! -d "$SOURCE_DIR/pass_2" ]]; then
    echo "[ERROR] No pass_2/ found in source directory: $SOURCE_DIR"
    echo "[INFO]  Pairwise comparison requires at least 2 passes"
    exit 1
fi

# ========================  Auto-detect Model  ========================
EVAL_MODE="self-evaluation"

if [[ -z "$MODEL" ]]; then
    # Auto-detect from source directory name
    SOURCE_BASENAME=$(basename "$SOURCE_DIR")

    # Try to extract model name from the directory basename
    # Convention: {ModelName}_{benchmark}_distraction_{d}
    DETECTED_MODEL=""
    for bench_candidate in $VALID_BENCHMARKS; do
        if [[ "$SOURCE_BASENAME" == *"_${bench_candidate}_"* ]]; then
            DETECTED_MODEL="${SOURCE_BASENAME%%_${bench_candidate}_*}"
            break
        fi
    done

    if [[ -z "$DETECTED_MODEL" ]]; then
        echo "[ERROR] Cannot auto-detect model from source directory: $SOURCE_DIR"
        echo "[INFO]  Expected format: {ModelName}_{benchmark}_distraction_{d}"
        echo "[INFO]  Please specify model with --model"
        exit 1
    fi

    echo "[INFO] Auto-detected model name from source dir: $DETECTED_MODEL"
    echo "[INFO] Mode: SELF-EVALUATION (model compares its own trajectories pairwise)"
    echo "[INFO]"
    echo "[INFO] The bump_sort Python module will use this identity when calling the LLM."
    echo "[INFO] If auto-detection is wrong, specify --model explicitly."
    echo ""

    # Let the Python module handle model resolution via its own auto-detection
    MODEL=""
    EVAL_MODE="self-evaluation (Python auto-detect)"
else
    EVAL_MODE="external judge"
fi

# ========================  Auto-derive Output Directory  ========================
if [[ -z "$OUTPUT_DIR" ]]; then
    SOURCE_NAME=$(basename "$SOURCE_DIR")
    # Remove _distraction_* suffix
    CLEAN_NAME=$(echo "$SOURCE_NAME" | sed 's/_distraction_[^_]*$//')

    if [[ "$EVAL_MODE" == "external judge" ]]; then
        # Derive a short judge name from the model path
        JUDGE_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr ':' '_')
        OUTPUT_DIR="$AGENT_DIR/results/002_bump_sort/${CLEAN_NAME}_judge-${JUDGE_SHORT}"
    else
        OUTPUT_DIR="$AGENT_DIR/results/002_bump_sort/${CLEAN_NAME}_selfjudge"
    fi
fi

# ========================  Build Python Command  ========================
build_eval_command() {
    local cmd=("python3" "-m" "source.bump_sort.main")

    # Model (only if explicitly specified)
    if [[ -n "$MODEL" ]]; then
        cmd+=("--model" "$MODEL")
    fi

    # Required
    cmd+=("--benchmark" "$BENCHMARK")
    cmd+=("--source" "$SOURCE_DIR")
    cmd+=("--output" "$OUTPUT_DIR")

    # Options
    cmd+=("--threshold" "$THRESHOLD")

    # Tool compression
    [[ "$COMPRESS_TOOLS" == "yes" ]] && cmd+=("--compress-tools")
    [[ "$MINIMAL_TOOLS" == "yes" ]] && cmd+=("--minimal-tools")

    echo "${cmd[*]}"
}

# ========================  Environment Setup  ========================
cd "$AGENT_DIR"

echo "=============================================="
echo "[INFO] Pairwise Self-Choice (Bump Sort) Evaluation"
echo "[INFO] Started at $(date)"
echo "=============================================="
echo
echo "[INFO] Configuration:"
echo "  Mode:            $EVAL_MODE"
if [[ -n "$MODEL" ]]; then
    echo "  Judge Model:     $MODEL"
else
    echo "  Judge Model:     (auto-detect from source directory)"
fi
echo "  Benchmark:       $BENCHMARK"
echo "  Source:           $SOURCE_DIR"
echo "  Output:          $OUTPUT_DIR"
echo "  Threshold:       $THRESHOLD"
echo "  Compress Tools:  $COMPRESS_TOOLS"
echo "  Minimal Tools:   $MINIMAL_TOOLS"
echo "  Agent Dir:        $AGENT_DIR"
echo

# Count available passes
AVAILABLE_PASSES=0
for p in "$SOURCE_DIR"/pass_*; do
    if [[ -d "$p" ]]; then
        AVAILABLE_PASSES=$((AVAILABLE_PASSES + 1))
    fi
done
echo "[INFO] Available passes in source: $AVAILABLE_PASSES"
echo "[INFO] Bump sort will perform $((AVAILABLE_PASSES - 1)) pairwise comparisons per task"
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

# ========================  Main Execution  ========================
EVAL_CMD=$(build_eval_command)

echo
echo "=============================================="
echo "[INFO] Running Pairwise Self-Choice (Bump Sort)"
echo "=============================================="
echo "[INFO] Command:"
echo "  $EVAL_CMD"
echo

if [[ "$DRY_RUN" == "yes" ]]; then
    echo "[DRY_RUN] Skipping execution."
else
    mkdir -p "$OUTPUT_DIR"

    eval "$EVAL_CMD"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] Bump sort evaluation failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi

    echo
    echo "[INFO] Evaluation complete. Results in: $OUTPUT_DIR"
fi

# ========================  Summary  ========================
echo
echo "=============================================="
echo "[INFO] Pairwise Self-Choice (Bump Sort) Complete"
echo "[INFO] Finished at $(date)"
echo "=============================================="
echo
echo "  Mode:       $EVAL_MODE"
if [[ -n "$MODEL" ]]; then
    echo "  Judge:      $MODEL"
fi
echo "  Benchmark:  $BENCHMARK"
echo "  Source:      $SOURCE_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Passes:     $AVAILABLE_PASSES"
echo "  Comparisons per task: $((AVAILABLE_PASSES - 1))"
echo
echo "  Output files:"
echo "    per_task/{task_id}.json       — per-task comparison history + best@k"
echo "    bump_sort_results.json        — aggregated summary (accuracy, oracle, best@k)"
echo

exit 0
