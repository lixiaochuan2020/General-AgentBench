"""Bump Sort Evaluator"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class BumpSortEvalResult:
    """Bump Sort evaluation result"""
    task_id: str
    benchmark: str
    domain: str
    
    # Selection result
    final_winner_pass: str       # Final winning pass
    final_winner_score: float    # Ground truth score of the winning trajectory
    is_correct: bool             # score >= threshold
    
    # best@k evaluation results
    best_at_k: dict              # {"k=1": 0/1, "k=2": 0/1, ...}
    
    # Comparison statistics
    num_passes: int              # Total number of passes
    num_comparisons: int         # Number of comparisons
    skipped: bool                # Whether skipped
    
    # Detailed information
    all_pass_scores: dict        # Scores for all passes
    comparison_history: list     # Comparison history


class BumpSortEvaluator:
    """
    Bump Sort Evaluator
    
    Evaluates the model's ability to select the optimal answer through bump sorting.
    """
    
    def __init__(self, success_threshold: float = 0.5):
        self.success_threshold = success_threshold
    
    def evaluate(
        self,
        task_id: str,
        bump_sort_result,  # BumpSortResult
        all_pass_scores: dict,
        benchmark: str,
        domain: str,
    ) -> BumpSortEvalResult:
        """Evaluate the Bump Sort result for a single task"""
        
        is_correct = bump_sort_result.final_winner_score >= self.success_threshold
        
        return BumpSortEvalResult(
            task_id=task_id,
            benchmark=benchmark,
            domain=domain,
            final_winner_pass=bump_sort_result.final_winner_pass,
            final_winner_score=bump_sort_result.final_winner_score,
            is_correct=is_correct,
            best_at_k=bump_sort_result.best_at_k,
            num_passes=len(all_pass_scores),
            num_comparisons=bump_sort_result.num_comparisons,
            skipped=bump_sort_result.skipped,
            all_pass_scores=all_pass_scores,
            comparison_history=bump_sort_result.comparison_history,
        )
    
    def batch_evaluate(
        self,
        results: list[BumpSortEvalResult],
        output_file: Path,
    ) -> dict:
        """Batch evaluate and save results"""
        
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        skipped = sum(1 for r in results if r.skipped)
        
        # Compute oracle (correct if any pass is correct)
        oracle_correct = 0
        for r in results:
            if any(score >= self.success_threshold for score in r.all_pass_scores.values()):
                oracle_correct += 1
        
        # Compute pass@1 (only look at the first pass)
        pass1_correct = 0
        for r in results:
            first_pass = min(r.all_pass_scores.keys())  # pass_1
            if r.all_pass_scores[first_pass] >= self.success_threshold:
                pass1_correct += 1
        
        # Compute best@k summary
        best_at_k_summary = {}
        if results and results[0].best_at_k:
            # Get all k values
            k_values = list(results[0].best_at_k.keys())
            for k_key in k_values:
                # Compute total score for each k (best@k is the score directly; non-mcpbench is 0 or 1, mcpbench is the raw score)
                k_total = sum(r.best_at_k.get(k_key, 0.0) for r in results)
                best_at_k_summary[k_key] = {
                    "total_score": k_total,
                    "accuracy": k_total / total if total > 0 else 0.0,
                }
        
        summary = {
            "total_tasks": total,
            "bump_sort_correct": correct,
            "bump_sort_accuracy": correct / total if total > 0 else 0.0,
            "oracle_correct": oracle_correct,
            "oracle_accuracy": oracle_correct / total if total > 0 else 0.0,
            "pass1_correct": pass1_correct,
            "pass1_accuracy": pass1_correct / total if total > 0 else 0.0,
            "best_at_k": best_at_k_summary,
            "skipped_count": skipped,
            "total_comparisons": sum(r.num_comparisons for r in results),
        }
        
        output_data = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Bump Sort] Accuracy:  {summary['bump_sort_accuracy']:.2%} ({correct}/{total})")
        logger.info(f"[Bump Sort] Oracle:    {summary['oracle_accuracy']:.2%} ({oracle_correct}/{total})")
        logger.info(f"[Bump Sort] Pass@1:    {summary['pass1_accuracy']:.2%} ({pass1_correct}/{total})")
        
        # Output best@k results
        for k_key, k_stats in best_at_k_summary.items():
            logger.info(f"[Bump Sort] Best@{k_key}: {k_stats['accuracy']:.2%} (score: {k_stats['total_score']:.1f}/{total})")
        
        logger.info(f"[Bump Sort] Skipped:   {skipped} tasks")
        
        return summary
