#!/usr/bin/env python3
"""
Self-Choice Evaluation - Per-Datapoint Mode

For each task (datapoint), evaluate all K passes together and store results in one JSON file.
This makes it easier to calculate Best@K and track per-task performance.

Output structure:
  results/001_self_choice/Model_benchmark/
    evaluations/
      task_id_1.json  # Contains all K passes for this task
      task_id_2.json
      ...
    summary.json  # Contains best@k statistics
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from source.host import BenchmarkHost
from source.self_choice.runner import SelfChoiceRunner
from source.self_choice.evaluator import SelfChoiceEvaluator


def get_servers_for_benchmark(benchmark: str) -> dict[str, dict]:
    """Get required server configuration based on benchmark type"""
    from run import DEFAULT_SERVERS
    
    if benchmark == "search":
        return {"search": DEFAULT_SERVERS["search"]}
    elif benchmark == "tau2bench":
        return {
            "tau2-airline": DEFAULT_SERVERS["tau2-airline"],
            "tau2-retail": DEFAULT_SERVERS["tau2-retail"],
            "tau2-telecom": DEFAULT_SERVERS["tau2-telecom"],
        }
    elif benchmark == "swebench":
        return {"swebench": DEFAULT_SERVERS["swebench"]}
    elif benchmark == "terminalbench":
        return {"terminalbench": DEFAULT_SERVERS["terminalbench"]}
    elif benchmark == "mathhay":
        return {"mathhay": DEFAULT_SERVERS["mathhay"]}
    elif benchmark == "mcpbench":
        # MCPBench needs to load all mcp-bench servers
        from run import get_all_servers
        all_servers = get_all_servers()
        # Filter out mcp-bench related servers (exclude tau2, search, mathhay, etc.)
        exclude_keys = ["tau2-airline", "tau2-retail", "tau2-telecom", "search", "mathhay", "terminalbench", "swebench"]
        return {k: v for k, v in all_servers.items() if k not in exclude_keys}
    else:
        from run import get_all_servers
        return get_all_servers()


def get_original_score(eval_data: dict, benchmark: str, threshold: float = 0.5) -> float:
    """Get original evaluation score based on benchmark type"""
    if benchmark == "search":
        return float(eval_data.get("score", 0))
    elif benchmark == "tau2bench":
        reward_info = eval_data.get("reward_info", {})
        return reward_info.get("reward", 0.0)
    elif benchmark == "terminalbench":
        return float(eval_data.get("reward", 0))
    elif benchmark == "swebench":
        reward = eval_data.get("reward")
        if reward is not None:
            return float(reward)
        report = eval_data.get("report", {})
        return 1.0 if report.get("resolved", False) else 0.0
    elif benchmark == "mathhay":
        # MathHay uses score field (0.0 or 1.0)
        return float(eval_data.get("score", 0))
    elif benchmark == "mcpbench":
        # MCPBench uses composite scoring formula:
        # S_overall = (S_rule + S_llm) / 2
        # S_rule = avg(valid_tool_name_rate, input_schema_compliance, execution_success_rate)
        # S_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
        evaluation = eval_data.get("evaluation", {})
        
        # Rule-based score (handle None values - use 1.0 when None, consistent with run.py)
        valid_tool_name_rate = evaluation.get("valid_tool_name_rate", 1.0)
        if valid_tool_name_rate is None:
            valid_tool_name_rate = 1.0
        input_schema_compliance = evaluation.get("input_schema_compliance", 1.0)
        if input_schema_compliance is None:
            input_schema_compliance = 1.0
        execution_success_rate = evaluation.get("execution_success_rate", 1.0)
        if execution_success_rate is None:
            execution_success_rate = 1.0
        s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3
        
        # LLM-judge score (5 dimensions, each 0-10, normalized to 0-1) (handle None values)
        task_fulfillment = evaluation.get("task_fulfillment") or 0
        grounding = evaluation.get("grounding") or 0
        tool_appropriateness = evaluation.get("tool_appropriateness") or 0
        parameter_accuracy = evaluation.get("parameter_accuracy") or 0
        dependency_awareness = evaluation.get("dependency_awareness") or 0
        s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
        
        # Overall score
        s_overall = (s_rule + s_llm) / 2
        return s_overall
    else:
        if "score" in eval_data:
            return float(eval_data.get("score", 0))
        if "reward_info" in eval_data:
            return eval_data["reward_info"].get("reward", 0.0)
        return eval_data.get("reward", 0.0)


def collect_task_passes(source_dir: Path, num_passes: int) -> dict:
    """
    Collect information for all tasks across all passes
    
    Returns:
        Dict mapping task_id -> list of (pass_num, trace_file, eval_file)
    """
    task_passes = defaultdict(list)
    
    for pass_num in range(1, num_passes + 1):
        pass_dir = source_dir / f"pass_{pass_num}"
        if not pass_dir.exists():
            logger.warning(f"Pass {pass_num} directory not found: {pass_dir}")
            continue
        
        trace_dir = pass_dir / "traces"
        eval_dir = pass_dir / "evaluations"
        
        if not trace_dir.exists():
            logger.warning(f"No traces directory in pass {pass_num}")
            continue
        
        if not eval_dir.exists():
            logger.warning(f"No evaluations directory in pass {pass_num}")
            continue
        
        # Iterate over all trace files
        for trace_file in sorted(trace_dir.glob("*.json")):
            task_id = trace_file.stem
            
            # Find corresponding evaluation file
            eval_file = eval_dir / f"{task_id}.json"
            if not eval_file.exists():
                # Try other naming formats
                # search benchmark: browsecomp_1 -> result_1.json
                task_num = task_id.split('_')[-1] if '_' in task_id else task_id
                possible_names = [
                    f"result_{task_id}.json",
                    f"{task_id.replace('browsecomp_', '')}.json",
                    f"result_{task_num}.json",  # browsecomp_1 -> result_1.json
                ]
                for name in possible_names:
                    alt_file = eval_dir / name
                    if alt_file.exists():
                        eval_file = alt_file
                        break
            
            if eval_file.exists():
                task_passes[task_id].append({
                    "pass_num": pass_num,
                    "trace_file": trace_file,
                    "eval_file": eval_file,
                })
    
    return task_passes


def calculate_best_at_k(passes_results: list, num_passes: int, benchmark: str = None) -> dict:
    """
    Calculate Best@K for a single task
    
    For binary benchmarks (search, tau2bench, terminalbench, swebench, mathhay):
        Best@K = 1 if any of first K passes has (original=1 AND self_choice=1)
    
    For continuous score benchmarks (mcpbench):
        Best@K = max(self_choice_score) over first K passes
        where self_choice_score = original_score * model_score (0 or 1)
    """
    best_at_k = {}
    
    if benchmark == "mcpbench":
        # MCPBench: Best@K = max(self_choice_score) over first K passes
        for k in range(1, num_passes + 1):
            best = 0.0
            for result in passes_results[:k]:
                sc_score = result.get("self_choice_score", 0.0)
                if sc_score > best:
                    best = sc_score
            best_at_k[f"k={k}"] = best
    else:
        # Binary benchmarks: Best@K = 1 if any pass has both_correct=True
        for k in range(1, num_passes + 1):
            best = 0
            for result in passes_results[:k]:
                if result.get("both_correct", False):
                    best = 1
                    break
            best_at_k[f"k={k}"] = best
    
    return best_at_k


def calculate_score_retain_at_k(passes_results: list, num_passes: int, benchmark: str = None) -> dict:
    """
    Calculate score_retain@k for a single task
    
    score_retain@k measures the proportion of originally correct answers where the model
    correctly identifies that it answered correctly (recall/retention rate).
    
    For binary benchmarks:
        score_retain@k = count(original>=0.5 AND model>=0.5) / count(original>=0.5)
        over the first k passes. Returns 0 when denominator is 0.
    
    For mcpbench (continuous scores):
        score_retain@k = sum(original_score * model_score) / sum(original_score)
        over the first k passes. Returns 0 when denominator is 0.
    """
    score_retain = {}
    is_mcpbench = benchmark == "mcpbench"
    
    for k in range(1, num_passes + 1):
        first_k = passes_results[:k]
        
        if is_mcpbench:
            count_model_correct = sum(1 for p in first_k if (p.get("model_score", 0) or 0) >= 0.5)
            if count_model_correct == 0:
                score_retain[f"k={k}"] = 0.0
            else:
                weighted_sum = sum(
                    (p.get("original_score", 0) or 0) * (p.get("model_score", 0) or 0)
                    for p in first_k
                )
                score_retain[f"k={k}"] = round(weighted_sum / count_model_correct, 4)
        else:
            count_orig_correct = sum(
                1 for p in first_k if (p.get("original_score", 0) or 0) >= 0.5
            )
            if count_orig_correct == 0:
                score_retain[f"k={k}"] = 0.0
            else:
                count_both = sum(
                    1 for p in first_k
                    if (p.get("original_score", 0) or 0) >= 0.5
                    and (p.get("model_score", 0) or 0) >= 0.5
                )
                score_retain[f"k={k}"] = round(count_both / count_orig_correct, 4)
    
    return score_retain


async def evaluate_single_task(
    task_id: str,
    pass_infos: list,
    runner: SelfChoiceRunner,
    benchmark: str,
    threshold: float,
    num_passes: int,
    required_servers: list[str] = None,
) -> dict:
    """
    Evaluate all passes for a single task
    
    Returns:
        Dict with task evaluation results across all passes
    """
    passes_results = []
    
    # Sort by pass_num
    pass_infos = sorted(pass_infos, key=lambda x: x["pass_num"])
    
    for info in pass_infos:
        pass_num = info["pass_num"]
        trace_file = info["trace_file"]
        eval_file = info["eval_file"]
        
        try:
            # Read original evaluation results
            with open(eval_file) as f:
                eval_data = json.load(f)
            original_score = get_original_score(eval_data, benchmark, threshold)
            original_success = original_score >= threshold
            
            # Run self-choice evaluation (load all tools including distraction)
            result = await runner.evaluate_trajectory(
                trace_file,
                required_servers=required_servers,
                distraction_count=-1,  # -1 = load all distraction tools
                eval_file=eval_file,  # For mathhay question extraction
                benchmark=benchmark,  # Benchmark name
            )
            model_judgment = result.model_judgment
            model_score = 1.0 if model_judgment.lower() == "correct" else 0.0
            
            # Calculate both_correct and self_choice_score
            both_correct = original_success and model_score >= 0.5
            
            # For mcpbench, self_choice_score = original_score * model_score
            # For other benchmarks, self_choice_score is just the binary both_correct
            if benchmark == "mcpbench":
                self_choice_score = original_score * model_score
            else:
                self_choice_score = 1.0 if both_correct else 0.0
            
            passes_results.append({
                "pass": pass_num,
                "original_score": original_score,
                "original_success": original_success,
                "model_judgment": model_judgment,
                "model_score": model_score,
                "both_correct": both_correct,
                "self_choice_score": self_choice_score,
                "tokens": result.total_tokens,
            })
            
            if benchmark == "mcpbench":
                logger.info(f"  Pass {pass_num}: original={original_score:.3f}, "
                           f"judgment={model_judgment}, sc_score={self_choice_score:.3f}")
            else:
                logger.info(f"  Pass {pass_num}: original={original_score:.0f}, "
                           f"judgment={model_judgment}, both={both_correct}")
            
        except Exception as e:
            logger.error(f"  Pass {pass_num}: ERROR - {e}")
            passes_results.append({
                "pass": pass_num,
                "original_score": 0,
                "original_success": False,
                "model_judgment": "Error",
                "model_score": 0,
                "both_correct": False,
                "self_choice_score": 0.0,
                "error": str(e),
            })
    
    # Calculate Best@K
    best_at_k = calculate_best_at_k(passes_results, num_passes, benchmark)
    
    # Calculate score_retain@k
    score_retain_at_k = calculate_score_retain_at_k(passes_results, num_passes, benchmark)
    
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "num_passes": len(passes_results),
        "passes": passes_results,
        "best_at_k": best_at_k,
        "score_retain_at_k": score_retain_at_k,
    }


async def run_per_datapoint_evaluation(
    source_dir: Path,
    output_dir: Path,
    model: str,
    benchmark: str,
    threshold: float,
    num_passes: int,
    compress_tools: bool = False,
    minimal_tools: bool = False,
):
    """
    Run per-datapoint Self-Choice evaluation
    """
    # Collect pass information for all tasks
    logger.info(f"Collecting tasks from {source_dir}...")
    task_passes = collect_task_passes(source_dir, num_passes)
    
    if not task_passes:
        logger.error("No tasks found!")
        return
    
    logger.info(f"Found {len(task_passes)} tasks across {num_passes} passes")
    
    # Create output directory
    eval_output_dir = output_dir / "evaluations"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get server configuration
    servers = get_servers_for_benchmark(benchmark)
    logger.info(f"Loading servers for {benchmark}: {list(servers.keys())}")
    
    # Initialize Host
    async with BenchmarkHost() as host:
        # Load servers
        for name, config in servers.items():
            try:
                logger.info(f"Loading server: {name}")
                await host.create_clients_from_config({name: config})
            except Exception as e:
                logger.warning(f"Failed to load server '{name}': {e}")
        
        # Display tool information
        tools = host.get_tools_schema()
        logger.info(f"Total tools available: {len(tools)}")
        
        # Initialize runner
        runner = SelfChoiceRunner(
            host=host,
            llm_model=model,
            temperature=0.0,
            max_tokens=8192,
            compress_tools=compress_tools,
            minimal_tools=minimal_tools,
        )
        
        # Evaluate each task
        all_results = []
        skipped = 0
        total = len(task_passes)
        
        for idx, (task_id, pass_infos) in enumerate(sorted(task_passes.items()), 1):
            # Check if evaluation result already exists (support resume)
            task_output_file = eval_output_dir / f"{task_id}.json"
            if task_output_file.exists():
                try:
                    with open(task_output_file) as f:
                        existing_result = json.load(f)
                    # Backfill score_retain_at_k for old results that don't have it
                    if "score_retain_at_k" not in existing_result and "passes" in existing_result:
                        existing_result["score_retain_at_k"] = calculate_score_retain_at_k(
                            existing_result["passes"], 
                            existing_result.get("num_passes", num_passes),
                            benchmark
                        )
                        with open(task_output_file, "w") as f:
                            json.dump(existing_result, f, indent=2, ensure_ascii=False)
                    all_results.append(existing_result)
                    skipped += 1
                    logger.info(f"[{idx}/{total}] Skipping {task_id} (already exists)")
                    continue
                except Exception:
                    pass  # File corrupted, re-evaluate
            
            logger.info(f"[{idx}/{total}] Evaluating {task_id} ({len(pass_infos)} passes)...")
            
            try:
                result = await evaluate_single_task(
                    task_id=task_id,
                    pass_infos=pass_infos,
                    runner=runner,
                    benchmark=benchmark,
                    threshold=threshold,
                    num_passes=num_passes,
                    required_servers=list(servers.keys()),
                )
                
                # Save individual task result
                with open(task_output_file, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                all_results.append(result)
                
                # Print Best@K summary
                best_at_k_str = ", ".join([f"@{k.split('=')[1]}={v}" 
                                           for k, v in result["best_at_k"].items()])
                logger.info(f"  -> Best@K: {best_at_k_str}")
                
            except Exception as e:
                logger.error(f"Error evaluating {task_id}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"Evaluation complete: {len(all_results)} total, {skipped} skipped (resumed)")
        
        # Calculate overall statistics
        if all_results:
            summary = calculate_overall_summary(all_results, num_passes, benchmark)
            summary["model"] = model
            summary["benchmark"] = benchmark
            summary["source_dir"] = str(source_dir)
            summary["threshold"] = threshold
            
            # Save summary
            summary_file = output_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n{'='*60}")
            logger.info("Self-Choice Best@K Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Total tasks: {summary['total_tasks']}")
            
            print(f"\n{'K':<5} {'Original Best@K':>18} {'Self-Choice Best@K':>20} {'Difference':>12} {'Score Retain':>14}")
            print("-" * 74)
            for k in range(1, num_passes + 1):
                orig = summary["original_best_at_k"].get(f"k={k}", 0)
                sc = summary["self_choice_best_at_k"].get(f"k={k}", 0)
                diff = sc - orig
                sr = summary["score_retain_at_k"].get(f"k={k}", 0)
                print(f"{k:<5} {orig:>18.2%} {sc:>20.2%} {diff:>+12.2%} {sr:>14.4f}")
            
            logger.info(f"\nResults saved to {output_dir}")
        else:
            logger.error("No results to summarize")


def calculate_overall_summary(all_results: list, num_passes: int, benchmark: str = None) -> dict:
    """Calculate overall Best@K statistics"""
    total_tasks = len(all_results)
    is_mcpbench = benchmark == "mcpbench"
    
    # Calculate Self-Choice Best@K
    self_choice_best_at_k = {}
    for k in range(1, num_passes + 1):
        k_key = f"k={k}"
        if is_mcpbench:
            # MCPBench: Best@K is the average of best@k values across all tasks
            total_score = sum(r["best_at_k"].get(k_key, 0) for r in all_results)
            self_choice_best_at_k[k_key] = total_score / total_tasks if total_tasks > 0 else 0
        else:
            # Binary benchmarks: Best@K is the success rate
            success_count = sum(1 for r in all_results if r["best_at_k"].get(k_key, 0) == 1)
            self_choice_best_at_k[k_key] = success_count / total_tasks if total_tasks > 0 else 0
    
    # Calculate Original Best@K (only original score, for comparison)
    original_best_at_k = {}
    for k in range(1, num_passes + 1):
        k_key = f"k={k}"
        if is_mcpbench:
            # MCPBench: Original Best@K is the average of max(original_score) across all tasks
            total_score = 0.0
            for result in all_results:
                passes = result.get("passes", [])[:k]
                max_orig = max((p.get("original_score", 0) for p in passes), default=0)
                total_score += max_orig
            original_best_at_k[k_key] = total_score / total_tasks if total_tasks > 0 else 0
        else:
            # Binary benchmarks: Original Best@K is the success rate
            success_count = 0
            for result in all_results:
                passes = result.get("passes", [])[:k]
                if any(p.get("original_success", False) for p in passes):
                    success_count += 1
            original_best_at_k[k_key] = success_count / total_tasks if total_tasks > 0 else 0
    
    # Per-pass statistics
    per_pass_stats = {}
    for pass_num in range(1, num_passes + 1):
        original_correct = 0
        self_choice_correct = 0
        both_correct = 0
        total_with_pass = 0
        original_score_sum = 0.0
        self_choice_score_sum = 0.0
        
        for result in all_results:
            passes = result.get("passes", [])
            for p in passes:
                if p.get("pass") == pass_num:
                    total_with_pass += 1
                    if p.get("original_success", False):
                        original_correct += 1
                    if p.get("model_score", 0) >= 0.5:
                        self_choice_correct += 1
                    if p.get("both_correct", False):
                        both_correct += 1
                    # For mcpbench: track average scores
                    original_score_sum += p.get("original_score", 0)
                    self_choice_score_sum += p.get("self_choice_score", 0)
                    break
        
        stats = {
            "total_tasks": total_with_pass,
            "original_correct": original_correct,
            "self_choice_correct": self_choice_correct,
            "both_correct": both_correct,
            "original_rate": original_correct / total_with_pass if total_with_pass > 0 else 0,
            "self_choice_rate": self_choice_correct / total_with_pass if total_with_pass > 0 else 0,
            "both_rate": both_correct / total_with_pass if total_with_pass > 0 else 0,
        }
        
        # Add average scores for mcpbench
        if is_mcpbench and total_with_pass > 0:
            stats["avg_original_score"] = original_score_sum / total_with_pass
            stats["avg_self_choice_score"] = self_choice_score_sum / total_with_pass
        
        per_pass_stats[f"pass_{pass_num}"] = stats
    
    # Calculate score_retain@k (average across all tasks)
    score_retain_at_k = {}
    for k in range(1, num_passes + 1):
        k_key = f"k={k}"
        task_retains = []
        for result in all_results:
            sr = result.get("score_retain_at_k", {})
            if k_key in sr:
                task_retains.append(sr[k_key])
        if task_retains:
            score_retain_at_k[k_key] = round(sum(task_retains) / len(task_retains), 4)
        else:
            score_retain_at_k[k_key] = 0.0
    
    return {
        "total_tasks": total_tasks,
        "num_passes": num_passes,
        "self_choice_best_at_k": self_choice_best_at_k,
        "original_best_at_k": original_best_at_k,
        "score_retain_at_k": score_retain_at_k,
        "per_pass_stats": per_pass_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Choice Evaluation (Per-Datapoint Mode)")
    parser.add_argument("--model", default="gpt-4o", help="Judge model")
    parser.add_argument("--benchmark", required=True, 
                       choices=["search", "tau2bench", "swebench", "terminalbench", "mathhay", "mcpbench"],
                       help="Benchmark type")
    parser.add_argument("--source", required=True, help="Source results directory with pass_1, pass_2, etc.")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold")
    parser.add_argument("--num-passes", type=int, default=4, help="Number of passes to evaluate")
    parser.add_argument("--compress-tools", action="store_true", help="Compress tool descriptions to reduce token count")
    parser.add_argument("--minimal-tools", action="store_true", help="Minimal mode: only name, description, param names (most aggressive)")
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    logger.info("=" * 60)
    logger.info("Self-Choice Evaluation (Per-Datapoint Mode)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Num Passes: {args.num_passes}")
    logger.info(f"Compress Tools: {args.compress_tools}")
    logger.info(f"Minimal Tools: {args.minimal_tools}")
    logger.info("=" * 60)
    
    asyncio.run(run_per_datapoint_evaluation(
        source_dir=source_dir,
        output_dir=output_dir,
        model=args.model,
        benchmark=args.benchmark,
        threshold=args.threshold,
        num_passes=args.num_passes,
        compress_tools=args.compress_tools,
        minimal_tools=args.minimal_tools,
    ))


if __name__ == "__main__":
    main()
