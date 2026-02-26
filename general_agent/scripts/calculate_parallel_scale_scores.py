#!/usr/bin/env python3
"""
Calculate/Aggregate Best@K statistics for Parallel Scaling experiments.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_task_results(pass_dir: Path) -> dict:
    """Load individual task results from summary.json's results array."""
    summary_file = pass_dir / "summary.json"
    if not summary_file.exists():
        return {}
    
    task_rewards = {}
    try:
        with open(summary_file) as f:
            data = json.load(f)
        
        # Read rewards from the results array in summary.json
        results = data.get("results", [])
        for result in results:
            task_id = result.get("task_id")
            reward = result.get("reward", 0)
            if task_id:
                # Handle None rewards (e.g., from errors)
                task_rewards[task_id] = reward if reward is not None else 0
    except Exception as e:
        print(f"[WARNING] Failed to load {summary_file}: {e}")
    
    return task_rewards


def load_pass_summary(pass_dir: Path) -> dict:
    """Load summary.json from a pass directory."""
    summary_file = pass_dir / "summary.json"
    if not summary_file.exists():
        return None
    
    try:
        with open(summary_file) as f:
            data = json.load(f)
        return {
            "average_reward": data.get("average_reward", 0),
            "success_rate": data.get("success_rate", 0),
            "total_tasks": data.get("total_tasks", 0),
            "completed_tasks": data.get("completed_tasks", 0),
        }
    except Exception as e:
        print(f"[WARNING] Failed to load {summary_file}: {e}")
        return None


def calculate_task_level_best_at_k(all_pass_task_rewards: list, k: int) -> dict:
    """
    Calculate task-level Best@K.
    
    Args:
        all_pass_task_rewards: List of dicts, each dict is {task_id: reward} for one pass
        k: Number of passes to consider
        
    Returns:
        Dict with Best@K statistics
    """
    if k > len(all_pass_task_rewards):
        k = len(all_pass_task_rewards)
    
    # Collect all task IDs across all passes
    all_task_ids = set()
    for pass_rewards in all_pass_task_rewards[:k]:
        all_task_ids.update(pass_rewards.keys())
    
    if not all_task_ids:
        return {
            "k": k,
            "best_at_k_reward": 0,
            "best_at_k_success_rate": 0,
            "num_tasks": 0,
            "task_best_rewards": {},
        }
    
    # For each task, find the best reward across K passes
    task_best_rewards = {}
    for task_id in all_task_ids:
        rewards_for_task = []
        for pass_rewards in all_pass_task_rewards[:k]:
            if task_id in pass_rewards:
                rewards_for_task.append(pass_rewards[task_id])
        task_best_rewards[task_id] = max(rewards_for_task) if rewards_for_task else 0
    
    # Calculate average of task-level Best@K
    best_at_k_reward = sum(task_best_rewards.values()) / len(task_best_rewards) if task_best_rewards else 0
    
    # Calculate success rate (reward >= 0.8 threshold)
    success_count = sum(1 for r in task_best_rewards.values() if r >= 0.8)
    best_at_k_success_rate = success_count / len(task_best_rewards) if task_best_rewards else 0
    
    return {
        "k": k,
        "best_at_k_reward": best_at_k_reward,
        "best_at_k_success_rate": best_at_k_success_rate,
        "num_tasks": len(task_best_rewards),
        "task_best_rewards": task_best_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate Best@K statistics (task-level)")
    parser.add_argument("--input-dir", required=True, help="Base directory with pass_1, pass_2, etc.")
    parser.add_argument("--num-passes", type=int, default=8, help="Number of passes")
    parser.add_argument("--output-file", required=True, help="Output JSON file for summary")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    print(f"[INFO] Loading results from {input_dir}")
    print(f"[INFO] Using task-level Best@K calculation")
    
    # Load all pass data (both summary and individual task results)
    pass_summaries = []
    all_pass_task_rewards = []
    
    for pass_num in range(1, args.num_passes + 1):
        pass_dir = input_dir / f"pass_{pass_num}"
        if pass_dir.exists():
            summary = load_pass_summary(pass_dir)
            task_rewards = load_task_results(pass_dir)
            pass_summaries.append(summary)
            all_pass_task_rewards.append(task_rewards)
            
            if summary:
                print(f"  Pass {pass_num}: avg_reward={summary['average_reward']:.4f}, tasks={len(task_rewards)}")
            else:
                print(f"  Pass {pass_num}: summary.json NOT FOUND, tasks={len(task_rewards)}")
        else:
            print(f"  Pass {pass_num}: DIRECTORY NOT FOUND")
            pass_summaries.append(None)
            all_pass_task_rewards.append({})
    
    # Calculate Best@K for K = 1, 2, 3, 4, 8 (and any available)
    available_passes = sum(1 for rewards in all_pass_task_rewards if rewards)
    k_values = [1, 2, 3, 4, 8]
    k_values = [k for k in k_values if k <= available_passes]
    
    # Also add the actual number of passes if not in list
    if args.num_passes not in k_values and args.num_passes <= available_passes:
        k_values.append(args.num_passes)
    k_values = sorted(set(k_values))
    
    summary = {
        "input_dir": str(input_dir),
        "num_passes": args.num_passes,
        "available_passes": available_passes,
        "calculation_method": "task-level Best@K (per-task max, then average)",
        "best_at_k": {},
        "per_pass": {},
    }
    
    # Store per-pass data
    for pass_num, ps in enumerate(pass_summaries, 1):
        if ps is not None:
            summary["per_pass"][f"pass_{pass_num}"] = ps
    
    print(f"\n[INFO] Calculating task-level Best@K statistics...")
    print(f"{'K':<5} {'Best@K Reward':>15} {'Best@K Success':>15} {'Num Tasks':>12}")
    print("-" * 50)
    
    for k in k_values:
        stats = calculate_task_level_best_at_k(all_pass_task_rewards, k)
        summary["best_at_k"][f"k={k}"] = {
            "k": k,
            "best_at_k_reward": stats["best_at_k_reward"],
            "best_at_k_success_rate": stats["best_at_k_success_rate"],
            "num_tasks": stats["num_tasks"],
        }
        print(f"{k:<5} {stats['best_at_k_reward']:>15.4f} {stats['best_at_k_success_rate']:>15.2%} {stats['num_tasks']:>12}")
    
    # Save summary
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[INFO] Summary saved to {output_file}")
    
    # Print per-pass comparison
    print(f"\n[INFO] Per-Pass Statistics (original pass-level averages):")
    print(f"{'Pass':<8} {'Avg Reward':>12} {'Success Rate':>15}")
    print("-" * 40)
    for pass_key, pass_stats in summary["per_pass"].items():
        print(f"{pass_key:<8} {pass_stats['average_reward']:>12.4f} {pass_stats['success_rate']:>15.2%}")
    
    # Print task-level Best@K details for the maximum K
    if k_values:
        max_k = max(k_values)
        final_stats = calculate_task_level_best_at_k(all_pass_task_rewards, max_k)
        print(f"\n[INFO] Task-level Best@{max_k} details:")
        print(f"{'Task ID':<40} {'Best Reward':>12}")
        print("-" * 55)
        for task_id, best_reward in sorted(final_stats["task_best_rewards"].items()):
            print(f"{task_id:<40} {best_reward:>12.4f}")


if __name__ == "__main__":
    main()
