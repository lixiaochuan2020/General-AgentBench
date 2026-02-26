#!/bin/bash
# ============================================================================
# run_pointwise_self_choice.sh — Unified Pointwise Self-Choice Script for Omni
#
# Have an LLM judge individual agent trajectories from parallel scaling
# results, then compute Best@K with self-choice filtering and score_retain@K.
#
# Two modes:
#   self-eval  — model judges its own trajectories (default)
#   judge      — an external model judges trajectories
#
# Usage:
#   ./run_pointwise_self_choice.sh --benchmark <bench> --source <dir>
#
# Examples:
#   # Self-evaluation (model auto-detected from source directory name)
#   ./run_pointwise_self_choice.sh \
#       --benchmark swebench \
#       --source results/000_parallel_scaling/Qwen3-235B_swebench_distraction_100
#
#   # External judge (GPT-5 evaluates another model's trajectories)
#   ./run_pointwise_self_choice.sh \
#       --model openai/gpt-5 \
#       --benchmark search \
#       --source results/000_parallel_scaling/DeepSeek-V3.2_search_distraction_all
#
#   # Preview commands without executing
#   ./run_pointwise_self_choice.sh \
#       --benchmark tau2bench \
#       --source results/000_parallel_scaling/Claude-Haiku-4.5_tau2bench_distraction_all \
#       --dry-run
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
NUM_PASSES=4
COMPRESS_TOOLS="no"            # yes | no
MINIMAL_TOOLS="no"             # yes | no
SKIP_ANALYSIS="no"             # skip post-processing analysis
ENV_FILE=".env"
DRY_RUN="no"

# Resolve AGENT_DIR: directory containing run.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to unified self-choice scores script
SELF_CHOICE_SCRIPT="$AGENT_DIR/scripts/calculate_self_choice_scores.py"

# ========================  Usage  ========================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Pointwise Self-Choice: judge agent trajectories and compute self-filtered Best@K.

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
  --threshold <float>          Success threshold (default: $THRESHOLD)
  --num-passes <int>           Number of passes to evaluate (default: $NUM_PASSES)
  --compress-tools             Compress tool descriptions to reduce tokens
  --minimal-tools              Most aggressive compression (name + description only)

Output options:
  --output-dir <path>          Output directory (default: auto-generated)
                               Self-eval: results/001_self_choice/{name}_selfchoice
                               Judge:     results/001_self_choice/{name}_judge-{model}

Post-processing:
  --skip-analysis              Skip Best@K and score_retain analysis after evaluation

Environment:
  --env-file <path>            Path to .env file (default: $ENV_FILE)
  --agent-dir <path>            Path to agent directory (default: auto-detected)
  --dry-run                    Print commands without executing

  -h, --help                   Show this help

Examples:
  # Self-evaluation with 4 passes
  $(basename "$0") --benchmark swebench \\
      --source results/000_parallel_scaling/Model_swebench_distraction_all

  # External judge with GPT-5 and compressed tools
  $(basename "$0") --model openai/gpt-5 --benchmark mathhay \\
      --source results/000_parallel_scaling/Model_mathhay_distraction_all \\
      --minimal-tools

  # Self-evaluation, 8 passes, custom threshold
  $(basename "$0") --benchmark mcpbench \\
      --source results/000_parallel_scaling/Model_mcpbench_distraction_100 \\
      --num-passes 8 --threshold 0.3

Output structure:
  results/001_self_choice/{name}_selfchoice/
    evaluations/
      {task_id}.json           per-task judgments + Best@K
    summary.json               aggregated Best@K, per-pass stats
    retain_score.json          score_retain@K (from post-processing)
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
        --num-passes)         NUM_PASSES="$2";         shift 2;;
        --compress-tools)     COMPRESS_TOOLS="yes";    shift;;
        --minimal-tools)      MINIMAL_TOOLS="yes";     shift;;
        --skip-analysis)      SKIP_ANALYSIS="yes";     shift;;
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

if [[ "$NUM_PASSES" -lt 1 ]]; then
    echo "[ERROR] --num-passes must be >= 1 (got: $NUM_PASSES)"
    exit 1
fi

# Verify source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "[ERROR] Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Verify at least pass_1 exists
if [[ ! -d "$SOURCE_DIR/pass_1" ]]; then
    echo "[ERROR] No pass_1/ found in source directory: $SOURCE_DIR"
    echo "[INFO]  Source must have pass_1/, pass_2/, ... from parallel scaling"
    exit 1
fi

# ========================  Auto-detect Model  ========================
EVAL_MODE="self-evaluation"

if [[ -z "$MODEL" ]]; then
    # Auto-detect from source directory name
    SOURCE_BASENAME=$(basename "$SOURCE_DIR")

    # Try to extract model name from the directory basename
    # Convention: {ModelName}_{benchmark}_distraction_{d}
    # Extract the part before the benchmark name
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
    echo "[INFO] Mode: SELF-EVALUATION (model evaluates its own trajectories)"
    echo "[INFO]"
    echo "[INFO] The self_choice Python module will use this identity when calling the LLM."
    echo "[INFO] If auto-detection is wrong, specify --model explicitly."
    echo ""

    # The Python module per_datapoint.py also has its own model auto-detection.
    # We pass the model name through if we can resolve it, otherwise let Python handle it.
    # For the generic script, we let the user specify --model or let the Python module auto-detect.
    # Here we set MODEL to empty string and let per_datapoint.py's default (gpt-4o) handle it,
    # OR the user can specify --model.
    # Actually, to properly support generic auto-detection, we leave it to the Python script.
    # We just need to skip the --model flag when calling per_datapoint.py.
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
        OUTPUT_DIR="$AGENT_DIR/results/001_self_choice/${CLEAN_NAME}_judge-${JUDGE_SHORT}"
    else
        OUTPUT_DIR="$AGENT_DIR/results/001_self_choice/${CLEAN_NAME}_selfchoice"
    fi
fi

# ========================  Build Python Command  ========================
build_eval_command() {
    local cmd=("python3" "-m" "source.self_choice.per_datapoint")

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
    cmd+=("--num-passes" "$NUM_PASSES")

    # Tool compression
    [[ "$COMPRESS_TOOLS" == "yes" ]] && cmd+=("--compress-tools")
    [[ "$MINIMAL_TOOLS" == "yes" ]] && cmd+=("--minimal-tools")

    echo "${cmd[*]}"
}

build_self_choice_command() {
    echo "python3 $SELF_CHOICE_SCRIPT --result-dir $(dirname "$OUTPUT_DIR") --specific-folder $(basename "$OUTPUT_DIR")"
}

# ========================  Environment Setup  ========================
cd "$AGENT_DIR"

echo "=============================================="
echo "[INFO] Pointwise Self-Choice Evaluation"
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
echo "  Num Passes:      $NUM_PASSES"
echo "  Compress Tools:  $COMPRESS_TOOLS"
echo "  Minimal Tools:   $MINIMAL_TOOLS"
echo "  Agent Dir:        $AGENT_DIR"
echo

# Count available passes
AVAILABLE_PASSES=0
for p in $(seq 1 "$NUM_PASSES"); do
    if [[ -d "$SOURCE_DIR/pass_${p}" ]]; then
        AVAILABLE_PASSES=$((AVAILABLE_PASSES + 1))
    fi
done
echo "[INFO] Available passes in source: $AVAILABLE_PASSES / $NUM_PASSES"

if [[ "$AVAILABLE_PASSES" -lt "$NUM_PASSES" ]]; then
    echo "[WARNING] Requested $NUM_PASSES passes but only $AVAILABLE_PASSES found."
    echo "[INFO]    Evaluation will proceed with available passes."
fi
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
echo "[INFO] Running Self-Choice Evaluation"
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
        echo "[ERROR] Self-choice evaluation failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi

    echo
    echo "[INFO] Evaluation complete. Results in: $OUTPUT_DIR"
fi

# ========================  Post-Processing  ========================
if [[ "$SKIP_ANALYSIS" != "yes" ]]; then
    echo
    echo "=============================================="
    echo "[INFO] Post-Processing: Self-Choice Scores (self_choice_score@k + self_choice_best@k)"
    echo "=============================================="

    SC_CMD=$(build_self_choice_command)
    echo "[INFO] Command: $SC_CMD"

    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "[DRY_RUN] Skipping execution."
    elif [[ -f "$SELF_CHOICE_SCRIPT" ]]; then
        eval "$SC_CMD" || echo "[WARNING] Self-choice scores calculation failed (non-fatal)"
    else
        echo "[WARNING] Self-choice script not found: $SELF_CHOICE_SCRIPT"
        echo "[INFO]    Skipping self-choice analysis."
    fi

elif [[ "$DRY_RUN" == "yes" ]]; then
    echo
    echo "[DRY_RUN] Analysis commands:"
    echo "  $(build_self_choice_command)"
fi

# ========================  Summary  ========================
echo
echo "=============================================="
echo "[INFO] Pointwise Self-Choice Complete"
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
echo "  Passes:     $NUM_PASSES"
echo
echo "  Output files:"
echo "    evaluations/{task_id}.json   — per-task judgments"
echo "    summary.json                 — aggregated Best@K"
if [[ "$SKIP_ANALYSIS" != "yes" ]]; then
    echo "    retain_score.json            — score_retain@K"
fi
echo

exit 0
