"""Bump Sort Evaluation Entry Point"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from dataclasses import asdict
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from source.host import BenchmarkHost
from source.bump_sort.runner import BumpSortRunner
from source.bump_sort.evaluator import BumpSortEvaluator, BumpSortEvalResult


def get_servers_for_benchmark(benchmark: str) -> dict:
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
    else:
        from run import get_all_servers
        return get_all_servers()


def get_all_servers_for_bump_sort() -> dict:
    """Get all server configurations (for bump sort to load the full tool set)"""
    from run import get_all_servers
    return get_all_servers()


def collect_passes(source_dir: Path) -> list[str]:
    """Collect all pass directories"""
    passes = []
    for d in sorted(source_dir.iterdir()):
        if d.is_dir() and d.name.startswith("pass_"):
            passes.append(d.name)
    return passes


def get_eval_file_for_trace(trace_file: Path, eval_dir: Path, benchmark: str) -> Path | None:
    """
    Find the corresponding evaluation file for a trace file
    
    Data structure differences:
    - tau2bench: trace and eval filenames are the same (tau2_airline_0_cfcd2084.json)
    - search: trace filename browsecomp_N.json, eval filename result_N.json
    
    Args:
        trace_file: Trajectory file path
        eval_dir: Evaluations directory
        benchmark: Benchmark type
        
    Returns:
        Path to the corresponding evaluation file, or None if not found
    """
    # tau2bench: same filename
    if benchmark == "tau2bench":
        eval_file = eval_dir / trace_file.name
        return eval_file if eval_file.exists() else None
    
    # search benchmark: browsecomp_N.json / webvoyager_N.json -> result_N.json
    if benchmark == "search":
        # Extract numeric ID from browsecomp_N or webvoyager_N
        match = re.search(r'_(\d+)\.json$', trace_file.name)
        if match:
            num_id = match.group(1)
            eval_file = eval_dir / f"result_{num_id}.json"
            if eval_file.exists():
                return eval_file
    
    # Default: try the same filename
    eval_file = eval_dir / trace_file.name
    return eval_file if eval_file.exists() else None


def find_matching_files(
    passes: list[str],
    source_dir: Path,
    benchmark: str,
) -> dict[str, list[tuple[str, Path, Path]]]:
    """
    Find matching files for all tasks across all passes
    
    Data structures:
    - tau2bench: traces/tau2_xxx.json <-> evaluations/tau2_xxx.json (same filename)
    - search: traces/browsecomp_N.json <-> evaluations/result_N.json (requires mapping)
    
    Returns:
        {task_id: [(pass_id, trace_file, eval_file), ...]}
    """
    task_files = {}
    
    # Collect all tasks from the first pass
    first_pass = passes[0]
    trace_dir = source_dir / first_pass / "traces"
    eval_dir = source_dir / first_pass / "evaluations"
    
    for trace_file in sorted(trace_dir.glob("*.json")):
        # Find the corresponding eval file
        eval_file = get_eval_file_for_trace(trace_file, eval_dir, benchmark)
        if eval_file:
            task_id = trace_file.stem  # Use trace filename as task_id
            task_files[task_id] = []
    
    # Collect files from all passes for each task
    for task_id in list(task_files.keys()):
        for pass_id in passes:
            trace_file = source_dir / pass_id / "traces" / f"{task_id}.json"
            eval_dir = source_dir / pass_id / "evaluations"
            
            if trace_file.exists():
                eval_file = get_eval_file_for_trace(trace_file, eval_dir, benchmark)
                if eval_file:
                    task_files[task_id].append((pass_id, trace_file, eval_file))
    
    # Filter out incomplete tasks
    complete_tasks = {
        task_id: files 
        for task_id, files in task_files.items() 
        if len(files) == len(passes)
    }
    
    return complete_tasks


def get_score_from_eval(eval_data: dict, benchmark: str) -> float:
    """Get evaluation score based on benchmark type"""
    if benchmark == "tau2bench":
        # tau2bench: reward_info.reward
        if "reward_info" in eval_data:
            return eval_data.get("reward_info", {}).get("reward", 0.0)
        return eval_data.get("reward", 0.0)
    elif benchmark in ("swebench", "terminalbench"):
        # swebench, terminalbench: reward field
        return float(eval_data.get("reward", 0))
    elif benchmark == "mcpbench":
        # mcpbench: compute from evaluation components S_overall = (S_rule + S_llm) / 2
        evaluation = eval_data.get("evaluation", {})
        
        # Rule-based score (0-1), handle None values
        valid_tool_name_rate = evaluation.get("valid_tool_name_rate") or 0.0
        input_schema_compliance = evaluation.get("input_schema_compliance") or 0.0
        execution_success_rate = evaluation.get("execution_success_rate") or 0.0
        s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3.0
        
        # LLM-judge score (normalized to 0-1), handle None values
        task_fulfillment = evaluation.get("task_fulfillment") or 0.0
        grounding = evaluation.get("grounding") or 0.0
        tool_appropriateness = evaluation.get("tool_appropriateness") or 0.0
        parameter_accuracy = evaluation.get("parameter_accuracy") or 0.0
        dependency_awareness = evaluation.get("dependency_awareness") or 0.0
        s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5.0 / 10.0
        
        return (s_rule + s_llm) / 2.0
    else:
        # search, mathhay: score field
        return float(eval_data.get("score", 0))


async def run_bump_sort_evaluation(
    source_dir: Path,
    output_dir: Path,
    model: str,
    benchmark: str,
    threshold: float,
    compress_tools: bool = False,
    minimal_tools: bool = False,
):
    """Run Bump Sort evaluation"""
    
    # Collect passes
    passes = collect_passes(source_dir)
    logger.info(f"Found {len(passes)} passes: {passes}")
    
    if len(passes) < 2:
        logger.error("Need at least 2 passes for bump sort comparison")
        return
    
    # Find matching files for all tasks
    task_files = find_matching_files(passes, source_dir, benchmark)
    logger.info(f"Found {len(task_files)} complete tasks across all passes")
    
    if not task_files:
        logger.error("No complete tasks found")
        return
    
    # Get server configuration
    required_servers_config = get_servers_for_benchmark(benchmark)
    required_servers = list(required_servers_config.keys())
    logger.info(f"Required servers for {benchmark}: {required_servers}")
    
    # Get all servers (including distraction)
    all_servers = get_all_servers_for_bump_sort()
    logger.info(f"Loading ALL {len(all_servers)} servers for bump sort (including distraction)")
    
    # Initialize Host
    async with BenchmarkHost() as host:
        for name, config in all_servers.items():
            try:
                logger.info(f"Loading server: {name}")
                await host.create_clients_from_config({name: config})
            except Exception as e:
                logger.warning(f"Failed to load server '{name}': {e}")
        
        # Display tool info
        tools = host.get_tools_schema()
        logger.info(f"Total tools available: {len(tools)}")
        
        # Initialize runner and evaluator (load all tools including distraction)
        runner = BumpSortRunner(
            host=host,
            llm_model=model,
            temperature=0.0,
            max_tokens=8192,
            required_servers=required_servers,
            distraction_count=-1,  # -1 = load all distraction tools
            compress_tools=compress_tools,
            minimal_tools=minimal_tools,
        )
        evaluator = BumpSortEvaluator(success_threshold=threshold)
        
        # Run evaluation
        results = []
        total = len(task_files)
        
        # Create per-task results directory
        per_task_dir = output_dir / "per_task"
        per_task_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (task_id, files) in enumerate(sorted(task_files.items()), 1):
            # Check if already completed
            task_output_file = per_task_dir / f"{task_id}.json"
            if task_output_file.exists():
                logger.info(f"[{idx}/{total}] Skipping {task_id} (already completed)")
                # Load existing result
                with open(task_output_file) as f:
                    existing_result = json.load(f)
                # Rebuild BumpSortEvalResult object
                from dataclasses import fields
                eval_result = BumpSortEvalResult(**existing_result)
                results.append(eval_result)
                continue
            
            logger.info(f"[{idx}/{total}] Processing {task_id}...")
            
            # Load data for all passes
            passes_data = []
            all_pass_scores = {}
            domain = "unknown"
            
            for pass_id, trace_file, eval_file in files:
                with open(trace_file) as f:
                    trace_data = json.load(f)
                with open(eval_file) as f:
                    eval_data = json.load(f)
                
                passes_data.append((pass_id, trace_data, eval_data))
                
                # Get score
                score = get_score_from_eval(eval_data, benchmark)
                all_pass_scores[pass_id] = score
                
                # Get domain
                domain = trace_data.get("domain", eval_data.get("domain", "unknown"))
            
            try:
                # Execute Bump Sort
                bump_result = await runner.bump_sort(task_id, passes_data, benchmark)
                
                # Evaluate
                eval_result = evaluator.evaluate(
                    task_id=task_id,
                    bump_sort_result=bump_result,
                    all_pass_scores=all_pass_scores,
                    benchmark=benchmark,
                    domain=domain or "unknown",
                )
                results.append(eval_result)
                
                # Save individual task result immediately
                with open(task_output_file, 'w') as f:
                    json.dump(asdict(eval_result), f, indent=2, ensure_ascii=False)
                
                status = "✓" if eval_result.is_correct else "✗"
                skip_info = " (skipped)" if bump_result.skipped else ""
                logger.info(f"  -> Winner: {bump_result.final_winner_pass}, Score: {bump_result.final_winner_score} {status}{skip_info}")
                logger.info(f"  -> Saved to: {task_output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {task_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        if results:
            output_file = output_dir / "bump_sort_results.json"
            summary = evaluator.batch_evaluate(results, output_file)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.error("No results to evaluate")


def main():
    parser = argparse.ArgumentParser(description="Bump Sort Evaluation")
    parser.add_argument("--model", default="gpt-4o", help="Judge model (e.g., gpt-4o, gemini/gemini-2.5-pro)")
    parser.add_argument("--benchmark", required=True, choices=["search", "tau2bench", "swebench", "terminalbench", "mcpbench", "mathhay"], help="Benchmark type")
    parser.add_argument("--source", required=True, help="Parallel scaling results directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold")
    parser.add_argument("--compress-tools", action="store_true", default=False, help="Compress tool descriptions to reduce token count")
    parser.add_argument("--minimal-tools", action="store_true", default=False, help="Minimal mode: only name, description, param names")
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("Bump Sort Evaluation")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Compress tools: {args.compress_tools}")
    logger.info(f"Minimal tools: {args.minimal_tools}")
    logger.info("=" * 50)
    
    # Run evaluation
    asyncio.run(run_bump_sort_evaluation(
        source_dir=source_dir,
        output_dir=output_dir,
        model=args.model,
        benchmark=args.benchmark,
        threshold=args.threshold,
        compress_tools=args.compress_tools,
        minimal_tools=args.minimal_tools,
    ))


if __name__ == "__main__":
    main()
