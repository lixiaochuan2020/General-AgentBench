# Experiment Scripts

All scripts live in `general_agent/scripts/` and should be run from `general_agent/`.

## Prerequisites

```bash
cd general_agent
conda activate one
set -a; source .env; set +a        # load API keys
```

Docker is required for `swebench` and `terminalbench`.

---

## 1. Parallel Scaling (`run_parallel_scaling.sh`)

Run a model K times (different seeds) on a benchmark; used for Best@K evaluation.

```bash
# Run 4 passes on search (sequential mode)
./scripts/run_parallel_scaling.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --num-passes 4 \
  --mode sequential \
  --distraction all \
  --compress-tools \
  --tool-desc-max-len 20 \
  --output-dir results/000_parallel_scaling

# Same but launch all 4 passes in parallel tmux sessions
./scripts/run_parallel_scaling.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --num-passes 4 \
  --mode tmux \
  --distraction all \
  --compress-tools \
  --tool-desc-max-len 20 \
  --output-dir results/000_parallel_scaling

# Dry run (print commands without executing)
./scripts/run_parallel_scaling.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --num-passes 4 \
  --dry-run
```

**Output structure:**
```
results/000_parallel_scaling/
  Qwen3-235B_search_distraction_all/
    pass_1/   # seed=42
    pass_2/   # seed=43
    pass_3/   # seed=44
    pass_4/   # seed=45
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | LiteLLM model string |
| `--benchmark` | (required) | One of: search, swebench, terminalbench, mathhay, tau2bench, mcpbench |
| `--num-passes` | 4 | Number of passes (K) |
| `--mode` | sequential | `sequential` or `tmux` |
| `--distraction` | all | Distraction count |
| `--compress-tools` | off | Enable tool description compression |
| `--tool-desc-max-len` | 20 | Max tool description length |
| `--output-dir` | `results/000_parallel_scaling` | Output root |
| `--dry-run` | off | Print commands only |

---

## 2. Sequential Scaling (`run_sequential_scaling.sh`)

Run a model with increasing token budgets; optionally reuse earlier checkpoints as prefixes.

```bash
# Run with 80k and 100k budgets, reusing prefixes
./scripts/run_sequential_scaling.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --budgets 80000,100000 \
  --sequential-reuse \
  --distraction all \
  --compress-tools \
  --tool-desc-max-len 20

# Specify a custom checkpoint directory
./scripts/run_sequential_scaling.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark mathhay \
  --budgets 40000,60000,80000 \
  --sequential-reuse \
  --checkpoint-dir results/my_checkpoints
```

**Output structure:**
```
results/000_sequential_scaling/
  Qwen3-235B_search_distraction_all_sequential/
    budget_80k/
    budget_100k/
```

When `--sequential-reuse` is set, the 100k run will look for the 80k checkpoint and continue from it, saving tokens and time.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | LiteLLM model string |
| `--benchmark` | (required) | Benchmark name |
| `--budgets` | (required) | Comma-separated token budgets (e.g. `80000,100000`) |
| `--sequential-reuse` | off | Reuse prefix checkpoints from smaller budgets |
| `--checkpoint-dir` | auto | Custom checkpoint directory |
| `--scaling-seed` | 42 | Random seed |
| `--distraction` | all | Distraction count |
| `--compress-tools` | off | Enable tool description compression |

---

## 3. Pointwise Self-Choice (`run_pointwise_self_choice.sh`)

Use the model itself as a judge to evaluate each trajectory independently.
Produces `self_choice_best_at_k` and `score_retain_at_k` metrics.

```bash
# Evaluate parallel scaling results
./scripts/run_pointwise_self_choice.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --source results/000_parallel_scaling/Qwen3-235B_search_distraction_all \
  --output-dir results/001_self_choice/Qwen3-235B_search_distraction_all_selfchoice

# With custom threshold and fewer passes
./scripts/run_pointwise_self_choice.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark mcpbench \
  --source results/000_parallel_scaling/Qwen3-235B_mcpbench_distraction_all \
  --num-passes 4 \
  --threshold 0.5
```

**Output structure:**
```
results/001_self_choice/
  Qwen3-235B_search_distraction_all_selfchoice/
    evaluations/       # per-task per-pass judge results
    summary.json       # self_choice_best_at_k for k=1..K
    retain_score.json  # score_retain_at_k for k=1..K
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Judge model (typically the same model) |
| `--benchmark` | (required) | Benchmark name |
| `--source` | (required) | Path to parallel scaling results (with pass_1..pass_K) |
| `--output-dir` | auto | Output directory |
| `--threshold` | 0.5 | Confidence threshold for self-choice |
| `--num-passes` | 4 | Number of passes to evaluate |
| `--skip-analysis` | off | Skip post-processing (judge only) |

---

## 4. Pairwise Self-Choice / Bump Sort (`run_pairwise_self_choice.sh`)

Use pairwise comparisons (Bump Sort) to rank trajectories. Each task goes through K-1 sequential comparisons.

```bash
# Run pairwise comparison on parallel scaling results
./scripts/run_pairwise_self_choice.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --source results/000_parallel_scaling/Qwen3-235B_search_distraction_all \
  --output-dir results/002_bump_sort/Qwen3-235B_search_distraction_all_pairwise

# Dry run
./scripts/run_pairwise_self_choice.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark mcpbench \
  --source results/000_parallel_scaling/Qwen3-235B_mcpbench_distraction_all \
  --dry-run
```

**Output structure:**
```
results/002_bump_sort/
  Qwen3-235B_search_distraction_all_pairwise/
    per_task/              # per-task comparison details
    bump_sort_results.json # bump_sort_accuracy + oracle_accuracy
```

> **Note:** Bump Sort requires ALL passes (pass_1..pass_K) to have complete trace + evaluation data for each task. Tasks with any missing pass are skipped entirely.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Judge model |
| `--benchmark` | (required) | Benchmark name |
| `--source` | (required) | Path to parallel scaling results |
| `--output-dir` | auto | Output directory |
| `--threshold` | 0.5 | Comparison threshold |
| `--dry-run` | off | Print commands only |

---

## 5. Baseline Run (`run_baseline.sh`)

Run a model on one or more benchmarks with sensible defaults. This is the simplest entry point — a thin wrapper around `run.py`.

```bash
# Run a single benchmark
./scripts/run_baseline.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search

# Run multiple benchmarks in one command
./scripts/run_baseline.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark tau2bench,mcpbench,mathhay,search,swebench,terminalbench

# Custom task file, distraction, and compression
./scripts/run_baseline.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --task-file data/_tmp_search_1task.json \
  --distraction all \
  --compress-tools \
  --tool-desc-max-len 20 \
  --output-dir results/my_run

# Dry run (print commands without executing)
./scripts/run_baseline.sh \
  --model bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
  --benchmark search \
  --dry-run
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | LiteLLM model string |
| `--benchmark` | (required) | Comma-separated benchmarks (tau2bench, mcpbench, mathhay, search, swebench, terminalbench) |
| `--task-file` | auto | Override task file (default: full benchmark file) |
| `--distraction` | all | Distraction count |
| `--compress-tools` | off | Enable tool description compression |
| `--tool-desc-max-len` | 75 | Max tool description length |
| `--task-timeout` | per-benchmark | Task timeout in seconds |
| `--max-steps` | per-benchmark | Max steps per task |
| `--output-dir` | auto | Output directory |
| `--resume` / `--no-resume` | resume | Resume from previous run |
| `--docker-cleanup` | auto | Docker cleanup: auto \| yes \| no |
| `--dry-run` | off | Print commands only |

---

## Typical Workflow

```
1. Parallel Scaling  ──►  2. Pointwise Self-Choice
        │                          │
        │                          ▼
        │                  summary.json + retain_score.json
        │
        └──────────────►  3. Pairwise Self-Choice (Bump Sort)
                                   │
                                   ▼
                           bump_sort_results.json

4. Sequential Scaling  (independent, tests token budget scaling)
```

Run parallel scaling first (step 1), then use its output as `--source` for both pointwise (step 2) and pairwise (step 3) self-choice. Sequential scaling (step 4) is independent.
