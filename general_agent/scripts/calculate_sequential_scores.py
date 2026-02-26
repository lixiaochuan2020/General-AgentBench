#!/usr/bin/env python3
"""
Calculate/Aggregate Sequential Scaling Scores
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path


def _detect_format(data: dict) -> str:
    """Detect evaluation file format.
    
    Returns one of: 'mcpbench', 'tau2bench', 'mathhay', 'swebench', 'terminalbench', 'search', or 'unknown'.
    """
    if "evaluation" in data:
        return "mcpbench"
    if "reward_info" in data:
        return "tau2bench"
    benchmark = data.get("benchmark", "")
    if benchmark == "swebench":
        return "swebench"
    if benchmark == "terminalbench":
        return "terminalbench"
    if "mathhay" in data.get("task_id", "") and "score" in data:
        return "mathhay"
    if "score" in data and "question" in data:
        return "search"
    return "unknown"


def load_evaluation_files(budget_dir: str) -> list:
    """Load all evaluation JSON files from the evaluations directory.
    
    Supports: MCP-Bench, tau2bench, mathhay, swebench, terminalbench, and search formats.
    """
    eval_dir = os.path.join(budget_dir, "evaluations")
    if not os.path.exists(eval_dir):
        return []
    
    evaluations = []
    for filename in os.listdir(eval_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(eval_dir, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    fmt = _detect_format(data)
                    if fmt != "unknown":
                        data["_filepath"] = filepath
                        data["_format"] = fmt
                        evaluations.append(data)
            except Exception as e:
                print(f"[WARNING] Failed to load {filepath}: {e}")
    
    return evaluations


def calculate_task_score(evaluation: dict) -> dict:
    """
    Calculate score for a single task.  Dispatches by _format tag.
    """
    fmt = evaluation.get("_format", "unknown")
    
    # --- swebench / terminalbench: binary reward ---
    if fmt in ("swebench", "terminalbench"):
        reward = float(evaluation.get("reward", 0.0))
        return {
            "task_id": evaluation.get("task_id", "unknown"),
            "server_name": fmt,
            "s_rule": reward,
            "s_llm": reward,
            "overall_score": reward,
            "fmt": fmt,
            "components": {
                "reward": reward,
                "success": evaluation.get("success", False),
            }
        }
    
    # --- search / deep research: binary score ---
    if fmt == "search":
        score = float(evaluation.get("score", 0.0))
        return {
            "task_id": evaluation.get("question", "unknown")[:80],
            "server_name": "search",
            "s_rule": score,
            "s_llm": score,
            "overall_score": score,
            "fmt": "search",
            "components": {
                "score": score,
            }
        }
    
    # --- mathhay ---
    if fmt == "mathhay":
        score = float(evaluation.get("score", 0.0))
        is_correct = evaluation.get("is_correct", False)
        numerical_match = evaluation.get("numerical_match", False)
        return {
            "task_id": evaluation.get("task_id", "unknown"),
            "server_name": "mathhay",
            "s_rule": score,
            "s_llm": score,
            "overall_score": score,
            "fmt": "mathhay",
            "components": {
                "score": score,
                "is_correct": is_correct,
                "numerical_match": numerical_match,
            }
        }
    
    # --- tau2bench ---
    if fmt == "tau2bench":
        reward_info = evaluation.get("reward_info") or {}
        reward = float(reward_info.get("reward", 0.0)) if reward_info else 0.0
        db_check = reward_info.get("db_check") if reward_info else None
        db_match = db_check.get("db_match", False) if db_check else False
        return {
            "task_id": evaluation.get("task_id", "unknown"),
            "server_name": evaluation.get("server_name", "tau2bench"),
            "s_rule": reward,
            "s_llm": reward,
            "overall_score": reward,
            "fmt": "tau2bench",
            "components": {
                "reward": reward,
                "db_match": db_match,
            }
        }
    
    # --- MCP-Bench (default) ---
    eval_data = evaluation.get("evaluation", {})
    
    def get_float(key, default=0.0):
        val = eval_data.get(key)
        return val if val is not None else default
    
    valid_tool_name_rate = get_float("valid_tool_name_rate", 0.0)
    input_schema_compliance = get_float("input_schema_compliance", 0.0)
    execution_success_rate = get_float("execution_success_rate", 0.0)
    s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3.0
    
    task_fulfillment = get_float("task_fulfillment", 0.0)
    grounding = get_float("grounding", 0.0)
    tool_appropriateness = get_float("tool_appropriateness", 0.0)
    parameter_accuracy = get_float("parameter_accuracy", 0.0)
    dependency_awareness = get_float("dependency_awareness", 0.0)
    s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5.0 / 10.0
    
    overall_score = (s_rule + s_llm) / 2.0
    
    return {
        "task_id": evaluation.get("task_id", "unknown"),
        "server_name": evaluation.get("server_name", "unknown"),
        "s_rule": s_rule,
        "s_llm": s_llm,
        "overall_score": overall_score,
        "fmt": "mcpbench",
        "components": {
            "valid_tool_name_rate": valid_tool_name_rate,
            "input_schema_compliance": input_schema_compliance,
            "execution_success_rate": execution_success_rate,
            "task_fulfillment": task_fulfillment,
            "grounding": grounding,
            "tool_appropriateness": tool_appropriateness,
            "parameter_accuracy": parameter_accuracy,
            "dependency_awareness": dependency_awareness,
        }
    }


def _compute_avg_components(fmt: str, task_scores: list, avg_overall_score: float) -> dict:
    """Compute average component metrics based on benchmark format."""
    n = len(task_scores)
    if n == 0:
        return {}
    
    if fmt == "mathhay":
        num_correct = sum(1 for s in task_scores if s["components"].get("is_correct", False))
        num_numerical_match = sum(1 for s in task_scores if s["components"].get("numerical_match", False))
        return {
            "score": avg_overall_score,
            "accuracy": num_correct / n,
            "numerical_match_rate": num_numerical_match / n,
        }
    elif fmt == "tau2bench":
        return {
            "reward": sum(s["components"].get("reward", 0) for s in task_scores) / n,
            "success_rate": avg_overall_score,
        }
    elif fmt in ("swebench", "terminalbench"):
        num_success = sum(1 for s in task_scores if s["components"].get("success", False))
        return {
            "reward": avg_overall_score,
            "success_rate": num_success / n,
        }
    elif fmt == "search":
        return {
            "score": avg_overall_score,
            "success_rate": avg_overall_score,
        }
    else:
        # MCP-Bench format
        component_keys = ["valid_tool_name_rate", "input_schema_compliance", "execution_success_rate",
                          "task_fulfillment", "grounding", "tool_appropriateness",
                          "parameter_accuracy", "dependency_awareness"]
        return {key: sum(s["components"][key] for s in task_scores) / n for key in component_keys}


def calculate_budget_scores(budget_dir: str, budget_label: str) -> dict:
    """Calculate aggregated scores for a single budget level."""
    evaluations = load_evaluation_files(budget_dir)
    
    if not evaluations:
        return {
            "budget_label": budget_label,
            "num_tasks": 0,
            "status": "no_evaluations",
            "avg_s_rule": None,
            "avg_s_llm": None,
            "avg_overall_score": None,
        }
    
    task_scores = []
    for eval_data in evaluations:
        score = calculate_task_score(eval_data)
        task_scores.append(score)
    
    # Calculate averages
    avg_s_rule = sum(s["s_rule"] for s in task_scores) / len(task_scores)
    avg_s_llm = sum(s["s_llm"] for s in task_scores) / len(task_scores)
    avg_overall_score = sum(s["overall_score"] for s in task_scores) / len(task_scores)
    
    # Detect format from first task
    fmt = task_scores[0].get("fmt", "mcpbench") if task_scores else "mcpbench"
    
    # Component averages by format
    avg_components = _compute_avg_components(fmt, task_scores, avg_overall_score)
    
    return {
        "budget_label": budget_label,
        "num_tasks": len(task_scores),
        "status": "completed",
        "format": fmt,
        "avg_s_rule": avg_s_rule,
        "avg_s_llm": avg_s_llm,
        "avg_overall_score": avg_overall_score,
        "avg_components": avg_components,
        "task_scores": task_scores,
    }


def calculate_all_budgets(base_dir: str, budgets: list) -> dict:
    """Calculate scores for all budget levels."""
    results = {
        "base_dir": base_dir,
        "generated_at": datetime.now().isoformat(),
        "budgets": [],
        "scaling_curve": {},
    }
    
    for budget in budgets:
        budget_label = f"{budget // 1000}k"
        budget_dir = os.path.join(base_dir, f"budget_{budget_label}")
        
        if os.path.exists(budget_dir):
            budget_result = calculate_budget_scores(budget_dir, budget_label)
            budget_result["budget"] = budget
        else:
            budget_result = {
                "budget": budget,
                "budget_label": budget_label,
                "num_tasks": 0,
                "status": "not_found",
                "avg_s_rule": None,
                "avg_s_llm": None,
                "avg_overall_score": None,
            }
        
        results["budgets"].append(budget_result)
        
        if budget_result.get("avg_overall_score") is not None:
            results["scaling_curve"][budget_label] = budget_result["avg_overall_score"]
    
    # Calculate statistics
    completed = [b for b in results["budgets"] if b.get("status") == "completed"]
    if completed:
        scores = [b["avg_overall_score"] for b in completed]
        results["statistics"] = {
            "num_budgets": len(budgets),
            "num_completed": len(completed),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_improvement": max(scores) - min(scores) if len(scores) >= 2 else None,
        }
    
    return results


def print_summary(results: dict):
    """Print a formatted summary table."""
    print()
    print("=" * 80)
    print("Sequential Scaling Scores Summary")
    print("=" * 80)
    print()
    print(f"{'Budget':<10} {'Tasks':<8} {'S_rule':<10} {'S_llm':<10} {'Overall':<10} {'Status'}")
    print("-" * 80)
    
    for budget in results["budgets"]:
        label = budget["budget_label"]
        num_tasks = budget.get("num_tasks", 0)
        s_rule = f"{budget['avg_s_rule']:.4f}" if budget.get("avg_s_rule") is not None else "N/A"
        s_llm = f"{budget['avg_s_llm']:.4f}" if budget.get("avg_s_llm") is not None else "N/A"
        overall = f"{budget['avg_overall_score']:.4f}" if budget.get("avg_overall_score") is not None else "N/A"
        status = budget.get("status", "unknown")
        
        print(f"{label:<10} {num_tasks:<8} {s_rule:<10} {s_llm:<10} {overall:<10} {status}")
    
    if results.get("statistics"):
        stats = results["statistics"]
        print()
        print(f"[INFO] Completed: {stats['num_completed']}/{stats['num_budgets']} budgets")
        if stats.get("score_improvement") is not None:
            print(f"[INFO] Score improvement: {stats['min_score']:.4f} -> {stats['max_score']:.4f} (+{stats['score_improvement']:.4f})")
    
    print()


def update_evaluation_files(budget_dir: str):
    """Update each evaluation file with s_rule, s_llm, and overall_score."""
    evaluations = load_evaluation_files(budget_dir)
    updated_count = 0
    
    for eval_data in evaluations:
        filepath = eval_data.get("_filepath")
        if not filepath:
            continue
        
        # Calculate scores
        score = calculate_task_score(eval_data)
        
        # Add scores to evaluation dict
        eval_section = eval_data.get("evaluation", {})
        eval_section["s_rule"] = score["s_rule"]
        eval_section["s_llm"] = score["s_llm"]
        eval_section["overall_score"] = score["overall_score"]
        
        # Remove internal filepath key before saving
        del eval_data["_filepath"]
        
        # Save back
        try:
            with open(filepath, 'w') as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            updated_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to update {filepath}: {e}")
    
    return updated_count


def update_summary_json(budget_dir: str, budget_scores: dict):
    """Update the summary.json file with overall score statistics."""
    summary_file = os.path.join(budget_dir, "summary.json")
    
    if not os.path.exists(summary_file):
        return False
    
    try:
        with open(summary_file) as f:
            summary = json.load(f)
        
        # Add overall score stats
        summary["avg_s_rule"] = budget_scores.get("avg_s_rule")
        summary["avg_s_llm"] = budget_scores.get("avg_s_llm")
        summary["avg_overall_score"] = budget_scores.get("avg_overall_score")
        summary["avg_components"] = budget_scores.get("avg_components")
        
        # Update each result with overall_score if we have task_scores
        if "results" in summary and "task_scores" in budget_scores:
            task_score_map = {ts["task_id"]: ts for ts in budget_scores.get("task_scores", [])}
            for result in summary["results"]:
                task_id = result.get("task_id")
                if task_id in task_score_map:
                    ts = task_score_map[task_id]
                    result["s_rule"] = ts["s_rule"]
                    result["s_llm"] = ts["s_llm"]
                    result["overall_score"] = ts["overall_score"]
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"[WARNING] Failed to update summary.json: {e}")
        return False


def calculate_and_update_budget(budget_dir: str, budget_label: str, update_files: bool = False) -> dict:
    """Calculate scores for a budget and optionally update files."""
    evaluations = load_evaluation_files(budget_dir)
    
    if not evaluations:
        return {
            "budget_label": budget_label,
            "num_tasks": 0,
            "status": "no_evaluations",
            "avg_s_rule": None,
            "avg_s_llm": None,
            "avg_overall_score": None,
        }
    
    task_scores = []
    for eval_data in evaluations:
        score = calculate_task_score(eval_data)
        score["_filepath"] = eval_data.get("_filepath")
        task_scores.append(score)
    
    # Calculate averages
    avg_s_rule = sum(s["s_rule"] for s in task_scores) / len(task_scores)
    avg_s_llm = sum(s["s_llm"] for s in task_scores) / len(task_scores)
    avg_overall_score = sum(s["overall_score"] for s in task_scores) / len(task_scores)
    
    # Detect format from first task
    fmt = task_scores[0].get("fmt", "mcpbench") if task_scores else "mcpbench"
    
    # Component averages by format
    avg_components = _compute_avg_components(fmt, task_scores, avg_overall_score)
    
    result = {
        "budget_label": budget_label,
        "num_tasks": len(task_scores),
        "status": "completed",
        "avg_s_rule": avg_s_rule,
        "avg_s_llm": avg_s_llm,
        "avg_overall_score": avg_overall_score,
        "avg_components": avg_components,
        "task_scores": task_scores,
    }
    
    # Update files if requested
    if update_files:
        updated = update_evaluation_files(budget_dir)
        print(f"  Updated {updated} evaluation files")
        
        if update_summary_json(budget_dir, result):
            print(f"  Updated summary.json")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate Sequential Scaling Scores")
    parser.add_argument("--input-dir", required=True, help="Base directory containing budget_* subdirectories")
    parser.add_argument("--budgets", default="64000,80000,96000,112000,128000", 
                        help="Comma-separated list of budget values (default: 64000,80000,96000,112000,128000)")
    parser.add_argument("--output-file", help="Output JSON file path (default: <input-dir>/sequential_scores.json)")
    parser.add_argument("--include-task-details", action="store_true", 
                        help="Include per-task score details in output")
    parser.add_argument("--update-files", action="store_true",
                        help="Update evaluation files and summary.json with calculated scores")
    
    args = parser.parse_args()
    
    # Parse budgets
    budgets = [int(b.strip()) for b in args.budgets.split(",")]
    budgets.sort()
    
    # Calculate scores
    print(f"[INFO] Calculating sequential scaling scores from: {args.input_dir}")
    print(f"[INFO] Budgets: {[f'{b//1000}k' for b in budgets]}")
    if args.update_files:
        print(f"[INFO] Will update evaluation files and summary.json")
    
    # Build results
    results = {
        "base_dir": args.input_dir,
        "generated_at": datetime.now().isoformat(),
        "budgets": [],
        "scaling_curve": {},
    }
    
    for budget in budgets:
        budget_label = f"{budget // 1000}k"
        budget_dir = os.path.join(args.input_dir, f"budget_{budget_label}")
        
        if os.path.exists(budget_dir):
            print(f"[INFO] Processing budget {budget_label}...")
            budget_result = calculate_and_update_budget(budget_dir, budget_label, args.update_files)
            budget_result["budget"] = budget
        else:
            budget_result = {
                "budget": budget,
                "budget_label": budget_label,
                "num_tasks": 0,
                "status": "not_found",
                "avg_s_rule": None,
                "avg_s_llm": None,
                "avg_overall_score": None,
            }
        
        results["budgets"].append(budget_result)
        
        if budget_result.get("avg_overall_score") is not None:
            results["scaling_curve"][budget_label] = budget_result["avg_overall_score"]
    
    # Calculate statistics
    completed = [b for b in results["budgets"] if b.get("status") == "completed"]
    if completed:
        scores = [b["avg_overall_score"] for b in completed]
        results["statistics"] = {
            "num_budgets": len(budgets),
            "num_completed": len(completed),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_improvement": max(scores) - min(scores) if len(scores) >= 2 else None,
        }
    
    # Remove task details if not requested (to keep output smaller)
    if not args.include_task_details:
        for budget in results["budgets"]:
            if "task_scores" in budget:
                del budget["task_scores"]
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = args.output_file or os.path.join(args.input_dir, "sequential_scores.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
