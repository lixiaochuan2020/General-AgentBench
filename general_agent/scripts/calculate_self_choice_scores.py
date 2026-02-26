#!/usr/bin/env python3
"""
Calculate/Aggregate self-choice metrics for pointwise self-choice results.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def self_choice_score_at_k_binary(passes: List[Dict], k: int) -> float:
    """
    Non-mcpbench (binary scoring).

    self_choice_score@k = sum(original_score × self_choice) / sum(original_score)
    among the first k passes.  Returns 0 when denominator is 0.
    """
    first_k = passes[:k]

    denominator = sum(p.get("original_score", 0) or 0 for p in first_k)
    if denominator == 0:
        return 0.0

    numerator = sum(
        (p.get("original_score", 0) or 0) * (p.get("model_score", 0) or 0)
        for p in first_k
    )
    return numerator / denominator


def self_choice_score_at_k_mcpbench(passes: List[Dict], k: int) -> float:
    """
    mcpbench (continuous scoring).

    self_choice_score@k = sum(original_score × self_choice) / sum(self_choice)
    among the first k passes.  Returns 0 when denominator is 0.
    """
    first_k = passes[:k]

    denominator = sum(p.get("model_score", 0) or 0 for p in first_k)
    if denominator == 0:
        return 0.0

    numerator = sum(
        (p.get("original_score", 0) or 0) * (p.get("model_score", 0) or 0)
        for p in first_k
    )
    return numerator / denominator


# ========================  self_choice_best@k  ========================


def self_choice_best_at_k_binary(passes: List[Dict], k: int) -> float:
    """
    Non-mcpbench (binary scoring).

    self_choice_best@k = 1 if any pass in first k has
    (original_score == 1 AND model_score == 1), else 0.
    """
    first_k = passes[:k]
    for p in first_k:
        if (p.get("original_score", 0) or 0) == 1.0 and (p.get("model_score", 0) or 0) == 1.0:
            return 1.0
    return 0.0


def self_choice_best_at_k_mcpbench(passes: List[Dict], k: int) -> float:
    """
    mcpbench (continuous scoring).

    self_choice_best@k = max(original_score × model_score) over first k passes.
    """
    first_k = passes[:k]
    return max(
        ((p.get("original_score", 0) or 0) * (p.get("model_score", 0) or 0) for p in first_k),
        default=0.0,
    )


def oracle_best_at_k_binary(passes: List[Dict], k: int) -> float:
    """Oracle Best@K for binary benchmarks (ignores self-choice)."""
    first_k = passes[:k]
    for p in first_k:
        if p.get("original_success", False) or (p.get("original_score", 0) or 0) == 1.0:
            return 1.0
    return 0.0


def oracle_best_at_k_mcpbench(passes: List[Dict], k: int) -> float:
    """Oracle Best@K for mcpbench (ignores self-choice)."""
    first_k = passes[:k]
    return max((p.get("original_score", 0) or 0 for p in first_k), default=0.0)

def process_task(eval_data: Dict) -> Optional[Dict[str, Any]]:
    """Compute all metrics for a single task, returning dict with per-k values."""
    benchmark = eval_data.get("benchmark", "")
    passes = eval_data.get("passes", [])
    num_passes = len(passes)

    if num_passes == 0:
        return None

    is_mcpbench = benchmark.lower() == "mcpbench"

    retain_fn = self_choice_score_at_k_mcpbench if is_mcpbench else self_choice_score_at_k_binary
    sc_best_fn = self_choice_best_at_k_mcpbench if is_mcpbench else self_choice_best_at_k_binary
    oracle_fn = oracle_best_at_k_mcpbench if is_mcpbench else oracle_best_at_k_binary

    sc_score = {}
    sc_best = {}
    oracle_best = {}

    for k in range(1, num_passes + 1):
        key = f"k={k}"
        sc_score[key] = round(retain_fn(passes, k), 4)
        sc_best[key] = round(sc_best_fn(passes, k), 4)
        oracle_best[key] = round(oracle_fn(passes, k), 4)

    return {
        "self_choice_score": sc_score,
        "self_choice_best": sc_best,
        "oracle_best": oracle_best,
    }


def _aggregate(task_results: Dict[str, Dict[str, float]], all_k_keys: set) -> Dict[str, float]:
    """Average a metric across all tasks."""
    sorted_k = sorted(all_k_keys, key=lambda x: int(x.split("=")[1]))
    agg = {}
    for k_key in sorted_k:
        values = [task_results[tid].get(k_key, 0.0) for tid in task_results]
        agg[k_key] = round(sum(values) / len(values), 4)
    return agg


def process_folder(folder: Path) -> Optional[Dict[str, Any]]:
    """Process all evaluation JSONs in <folder>/evaluations/."""
    eval_dir = folder / "evaluations"
    if not eval_dir.exists():
        return None

    sc_score_tasks: Dict[str, Dict[str, float]] = {}
    sc_best_tasks: Dict[str, Dict[str, float]] = {}
    oracle_best_tasks: Dict[str, Dict[str, float]] = {}
    all_k_keys: set = set()

    for f in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  [WARNING] Failed to read {f.name}: {e}")
            continue

        task_id = data.get("task_id", f.stem)
        result = process_task(data)
        if result is None:
            continue

        sc_score_tasks[task_id] = result["self_choice_score"]
        sc_best_tasks[task_id] = result["self_choice_best"]
        oracle_best_tasks[task_id] = result["oracle_best"]
        all_k_keys.update(result["self_choice_score"].keys())

    if not sc_score_tasks:
        return None

    return {
        "num_tasks": len(sc_score_tasks),
        "aggregated_self_choice_score": _aggregate(sc_score_tasks, all_k_keys),
        "aggregated_self_choice_best": _aggregate(sc_best_tasks, all_k_keys),
        "aggregated_oracle_best": _aggregate(oracle_best_tasks, all_k_keys),
        "per_task_self_choice_score": sc_score_tasks,
        "per_task_self_choice_best": sc_best_tasks,
        "per_task_oracle_best": oracle_best_tasks,
    }


# ========================  main  ========================


def _print_summary_table(
    summary_sc_score: Dict[str, Dict[str, float]],
    summary_sc_best: Dict[str, Dict[str, float]],
    summary_oracle_best: Dict[str, Dict[str, float]],
) -> None:
    all_k = sorted(
        {k for scores in summary_sc_score.values() for k in scores},
        key=lambda x: int(x.split("=")[1]),
    )
    col_w = 8

    # --- self_choice_score ---
    print("\n" + "=" * 70)
    print("SUMMARY: self_choice_score@k")
    print("=" * 70)
    header = f"{'Experiment':<55} " + " ".join(f"{k:>{col_w}}" for k in all_k)
    print(header)
    print("-" * len(header))
    for name in sorted(summary_sc_score):
        s = summary_sc_score[name]
        row = f"{name:<55} " + " ".join(
            f"{s[k]:>{col_w}.4f}" if k in s else f"{'N/A':>{col_w}}" for k in all_k
        )
        print(row)

    # --- self_choice_best ---
    print("\n" + "=" * 70)
    print("SUMMARY: self_choice_best@k")
    print("=" * 70)
    header = f"{'Experiment':<55} " + " ".join(f"{k:>{col_w}}" for k in all_k)
    print(header)
    print("-" * len(header))
    for name in sorted(summary_sc_best):
        s = summary_sc_best[name]
        row = f"{name:<55} " + " ".join(
            f"{s[k]:>{col_w}.4f}" if k in s else f"{'N/A':>{col_w}}" for k in all_k
        )
        print(row)

    # --- oracle_best ---
    print("\n" + "=" * 70)
    print("SUMMARY: oracle_best@k (reference)")
    print("=" * 70)
    header = f"{'Experiment':<55} " + " ".join(f"{k:>{col_w}}" for k in all_k)
    print(header)
    print("-" * len(header))
    for name in sorted(summary_oracle_best):
        s = summary_oracle_best[name]
        row = f"{name:<55} " + " ".join(
            f"{s[k]:>{col_w}.4f}" if k in s else f"{'N/A':>{col_w}}" for k in all_k
        )
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate self-choice metrics (self_choice_score@k, self_choice_best@k) "
                    "for pointwise self-choice results"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing self-choice result folders",
    )
    parser.add_argument(
        "--specific-folder",
        type=str,
        default=None,
        help="Process only this folder (relative to --result-dir)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="self_choice_scores.json",
        help="Output filename saved inside each result folder "
             "(default: self_choice_scores.json)",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"[ERROR] Result directory not found: {result_dir}")
        return

    # Discover folders
    if args.specific_folder:
        folders = [result_dir / args.specific_folder]
    else:
        folders = sorted(
            d for d in result_dir.iterdir()
            if d.is_dir()
            and ("selfchoice" in d.name.lower() or "judge" in d.name.lower())
        )

    print(f"[INFO] Found {len(folders)} self-choice result folder(s)")
    print("=" * 70)

    summary_sc_score: Dict[str, Dict[str, float]] = {}
    summary_sc_best: Dict[str, Dict[str, float]] = {}
    summary_oracle_best: Dict[str, Dict[str, float]] = {}

    for folder in folders:
        print(f"\n[INFO] Processing: {folder.name}")
        result = process_folder(folder)

        if result is None:
            print("  [WARNING] No valid results found")
            continue

        # Save per-folder result
        out_path = folder / args.output_file
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  Saved to:               {out_path}")
        print(f"  Tasks:                  {result['num_tasks']}")
        print(f"  self_choice_score@k:    {result['aggregated_self_choice_score']}")
        print(f"  self_choice_best@k:     {result['aggregated_self_choice_best']}")
        print(f"  oracle_best@k:          {result['aggregated_oracle_best']}")

        summary_sc_score[folder.name] = result["aggregated_self_choice_score"]
        summary_sc_best[folder.name] = result["aggregated_self_choice_best"]
        summary_oracle_best[folder.name] = result["aggregated_oracle_best"]

    if not summary_sc_score:
        print("\n[WARNING] No results to summarize.")
        return

    _print_summary_table(summary_sc_score, summary_sc_best, summary_oracle_best)

    # Save overall summary
    overall = {
        "self_choice_score": summary_sc_score,
        "self_choice_best": summary_sc_best,
        "oracle_best": summary_oracle_best,
    }
    summary_path = result_dir / "self_choice_scores_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2) + "\n")
    print(f"\n[INFO] Overall summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
