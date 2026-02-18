"""Self-Choice Evaluation Entry Point"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from source.host import BenchmarkHost
from source.self_choice.runner import SelfChoiceRunner
from source.self_choice.evaluator import SelfChoiceEvaluator


def get_eval_file_for_trace(trace_file: Path, eval_dir: Path, trace_data: dict) -> Path | None:
    """
    Find the corresponding evaluation file for a trace file
    
    Supported naming formats:
    - Same filename: trace_file.stem == eval_file.stem
    - result_N format: browsecomp_N.json -> result_N.json
    - task_id match: lookup by task_id
    
    Args:
        trace_file: Trajectory file path
        eval_dir: Evaluations directory
        trace_data: Trajectory data (for extracting task_id)
        
    Returns:
        Corresponding evaluation file path, or None if not found
    """
    task_id = trace_data.get("task_id", trace_file.stem)
    
    # 1. Try direct filename match
    eval_file = eval_dir / f"{trace_file.stem}.json"
    if eval_file.exists():
        return eval_file
    
    # 2. Try result_N format
    eval_file = eval_dir / f"result_{task_id}.json"
    if eval_file.exists():
        return eval_file
    
    # 3. Try extracting numeric ID from trace_file
    import re
    match = re.search(r'_(\d+)\.json$', trace_file.name)
    if match:
        num_id = match.group(1)
        eval_file = eval_dir / f"result_{num_id}.json"
        if eval_file.exists():
            return eval_file
    
    return None


def get_servers_for_benchmark(benchmark: str) -> dict[str, dict]:
    """
    Get required server configuration based on benchmark type
    
    Args:
        benchmark: Benchmark type ("search" or "tau2bench")
        
    Returns:
        Server configuration dictionary
    """
    from run import DEFAULT_SERVERS, get_mcpbench_server_configs, convert_to_host_config
    
    if benchmark == "search":
        # search benchmark only needs search server
        return {
            "search": DEFAULT_SERVERS["search"]
        }
    elif benchmark == "tau2bench":
        # tau2bench needs tau2-related servers
        return {
            "tau2-airline": DEFAULT_SERVERS["tau2-airline"],
            "tau2-retail": DEFAULT_SERVERS["tau2-retail"],
            "tau2-telecom": DEFAULT_SERVERS["tau2-telecom"],
        }
    elif benchmark == "swebench":
        # swebench uses swebench server
        return {
            "swebench": DEFAULT_SERVERS["swebench"]
        }
    elif benchmark == "terminalbench":
        # terminalbench uses terminalbench server
        return {
            "terminalbench": DEFAULT_SERVERS["terminalbench"]
        }
    else:
        # Default: load all servers
        from run import get_all_servers
        return get_all_servers()


async def run_self_choice_evaluation(
    source_dir: Path,
    output_dir: Path,
    model: str,
    benchmark: str,
    threshold: float,
):
    """
    Run Self-Choice evaluation
    
    Args:
        source_dir: Source results directory (containing traces/ and evaluations/)
        output_dir: Output directory
        model: Model for judgment
        benchmark: Benchmark type
        threshold: Success threshold
    """
    # Check directories
    trace_dir = source_dir / "traces"
    eval_dir = source_dir / "evaluations"
    
    if not trace_dir.exists():
        logger.error(f"Traces directory not found: {trace_dir}")
        return
    
    if not eval_dir.exists():
        logger.error(f"Evaluations directory not found: {eval_dir}")
        return
    
    # Collect trajectory files
    trace_files = sorted(trace_dir.glob("*.json"))
    logger.info(f"Found {len(trace_files)} trajectory files")
    
    if not trace_files:
        logger.error("No trajectory files found")
        return
    
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
        # Initialize runner and evaluator
        runner = SelfChoiceRunner(
            host=host,
            llm_model=model,
            temperature=0.0,
            max_tokens=8192,
        )
        evaluator = SelfChoiceEvaluator(
            success_threshold=threshold,
        )
        
        # Run evaluation
        results = []
        total = len(trace_files)
        
        for idx, trace_file in enumerate(trace_files, 1):
            task_id = trace_file.stem
            
            # First load trace data to get task_id
            with open(trace_file) as f:
                trace_data = json.load(f)
            
            # Find corresponding evaluation file
            eval_file = get_eval_file_for_trace(trace_file, eval_dir, trace_data)
            
            if not eval_file:
                logger.warning(f"[{idx}/{total}] Evaluation file not found for: {trace_file.name}")
                continue
            
            logger.info(f"[{idx}/{total}] Evaluating {task_id}...")
            
            try:
                # Run Self-Choice
                result = await runner.evaluate_trajectory(trace_file)
                
                results.append((
                    task_id,
                    result.model_judgment,
                    eval_file,
                    trace_file,
                    benchmark,
                ))
                
                logger.info(f"  -> Judgment: {result.model_judgment} (tokens: {result.total_tokens})")
                
            except Exception as e:
                logger.error(f"Error evaluating {task_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Batch evaluate
        if results:
            output_file = output_dir / "self_choice_results.json"
            summary = evaluator.batch_evaluate(results, output_file)
            
            logger.info(f"Results saved to {output_file}")
            logger.info(f"Overall Accuracy: {summary['accuracy']:.2%}")
        else:
            logger.error("No results to evaluate")


def main():
    parser = argparse.ArgumentParser(description="Self-Choice Evaluation")
    parser.add_argument("--model", default="gpt-4o", help="Judge model (e.g., gpt-4o, claude-sonnet-4-5)")
    parser.add_argument("--benchmark", default="search", choices=["search", "tau2bench", "swebench", "terminalbench"], help="Benchmark type")
    parser.add_argument("--source", required=True, help="Source results directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold")
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("Self-Choice Evaluation")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 50)
    
    # Run evaluation
    asyncio.run(run_self_choice_evaluation(
        source_dir=source_dir,
        output_dir=output_dir,
        model=args.model,
        benchmark=args.benchmark,
        threshold=args.threshold,
    ))


if __name__ == "__main__":
    main()
