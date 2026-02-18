"""Self-Choice Evaluator"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class SelfChoiceEvalResult:
    """Self-Choice comprehensive evaluation result"""
    task_id: str
    benchmark: str
    domain: str
    
    # Self-Choice judgment
    model_judgment: str         # "Correct" or "Wrong"
    model_score: float          # 1.0 if Correct, 0.0 if Wrong
    
    # Original evaluation result
    original_score: float       # search: score (0/1), tau2bench: reward (0.0/1.0)
    original_success: bool      # Whether original evaluation succeeded
    
    # Ground Truth (based on original evaluation)
    ground_truth: str           # "Correct" if score >= threshold else "Wrong"
    
    # Comparison result
    match: bool                 # model_score == ground_truth_score
    
    # Details
    details: dict = None


class SelfChoiceEvaluator:
    """
    Self-Choice Evaluator
    
    Compares model's judgment with original evaluation results
    to calculate the accuracy of the model's autonomous judgment.
    
    Ground Truth sources:
    - search benchmark: uses "score" field (0 or 1)
    - tau2bench: uses "reward" field (0.0 or 1.0)
    
    Model judgment conversion:
    - "Correct" -> 1.0
    - "Wrong" -> 0.0
    """
    
    def __init__(
        self,
        success_threshold: float = 0.5,
    ):
        """
        Initialize evaluator
        
        Args:
            success_threshold: Original score/reward >= threshold is considered Correct
        """
        self.success_threshold = success_threshold
    
    def _get_original_score(self, eval_data: dict, benchmark: str) -> float:
        """
        Get original evaluation score based on benchmark type
        
        Args:
            eval_data: Evaluation data
            benchmark: Benchmark type
            
        Returns:
            Original evaluation score (0.0 or 1.0)
        """
        if benchmark == "search":
            # search benchmark uses "score" field, value is 0 or 1
            return float(eval_data.get("score", 0))
        elif benchmark == "tau2bench":
            # tau2bench uses "reward_info.reward" field, value is 0.0 or 1.0
            reward_info = eval_data.get("reward_info", {})
            return reward_info.get("reward", 0.0)
        elif benchmark == "terminalbench":
            # terminalbench uses "reward" field, value is 0 or 1
            return float(eval_data.get("reward", 0))
        elif benchmark == "swebench":
            # swebench: prefer "reward" field, otherwise use "report.resolved"
            reward = eval_data.get("reward")
            if reward is not None:
                return float(reward)
            # fallback to report.resolved
            report = eval_data.get("report", {})
            return 1.0 if report.get("resolved", False) else 0.0
        else:
            # Default: try both
            if "score" in eval_data:
                return float(eval_data.get("score", 0))
            # Try reward_info.reward
            if "reward_info" in eval_data:
                return eval_data["reward_info"].get("reward", 0.0)
            return eval_data.get("reward", 0.0)
    
    def _judgment_to_score(self, judgment: str) -> float:
        """
        Convert model judgment to score
        
        Args:
            judgment: Model judgment ("Correct" or "Wrong")
            
        Returns:
            1.0 if Correct, 0.0 if Wrong
        """
        return 1.0 if judgment.lower() == "correct" else 0.0
    
    def evaluate(
        self,
        task_id: str,
        model_judgment: str,
        evaluation_file: Path,
        trajectory_file: Optional[Path] = None,
        benchmark: str = "unknown",
    ) -> SelfChoiceEvalResult:
        """
        Evaluate Self-Choice result for a single task
        
        Args:
            task_id: Task ID
            model_judgment: Model's judgment ("Correct" or "Wrong")
            evaluation_file: Original evaluation file path
            trajectory_file: Trajectory file (for obtaining benchmark/domain)
            benchmark: Benchmark type ("search" or "tau2bench")
            
        Returns:
            SelfChoiceEvalResult: Comprehensive evaluation result
        """
        # Load original evaluation results
        with open(evaluation_file) as f:
            eval_data = json.load(f)
        
        domain = eval_data.get("domain", "unknown")
        
        # If benchmark is unknown, get it from trajectory
        if trajectory_file and benchmark == "unknown":
            with open(trajectory_file) as f:
                trace_data = json.load(f)
            benchmark = trace_data.get("benchmark", benchmark)
            domain = trace_data.get("domain", domain)
        
        # Get original evaluation score (select field based on benchmark type)
        original_score = self._get_original_score(eval_data, benchmark)
        original_success = original_score >= self.success_threshold
        
        # Convert model judgment to score
        model_score = self._judgment_to_score(model_judgment)
        
        # Calculate ground truth
        ground_truth = "Correct" if original_score >= self.success_threshold else "Wrong"
        
        # Compare: whether the two scores are consistent
        match = model_score == (1.0 if original_score >= self.success_threshold else 0.0)
        
        return SelfChoiceEvalResult(
            task_id=task_id,
            benchmark=benchmark,
            domain=domain,
            model_judgment=model_judgment,
            model_score=model_score,
            original_score=original_score,
            original_success=original_success,
            ground_truth=ground_truth,
            match=match,
            details={
                "success_threshold": self.success_threshold,
                "evaluation_file": str(evaluation_file),
            }
        )
    
    def batch_evaluate(
        self,
        results: list[tuple],  # [(task_id, model_judgment, eval_file, trace_file, benchmark), ...]
        output_file: Path,
    ) -> dict:
        """
        Batch evaluate and generate report
        
        Metrics:
        - Accuracy: Proportion where judge agrees with original eval
        - Recall: Proportion where judge is correct given original eval is correct
          = (judge correct AND original correct) / original correct
        
        Args:
            results: List of evaluation results
            output_file: Output file path
            
        Returns:
            Summary statistics
        """
        all_results = []
        
        # Statistics counters
        match_count = 0           # Number of judge-original eval agreements (Accuracy numerator)
        original_correct = 0      # Number of correct original evals (Recall denominator)
        both_correct = 0          # Number where both judge and original eval are correct (Recall numerator)
        
        for task_id, model_judgment, eval_file, trace_file, benchmark in results:
            result = self.evaluate(task_id, model_judgment, eval_file, trace_file, benchmark)
            all_results.append(asdict(result))
            
            # Accuracy: count matches
            if result.match:
                match_count += 1
            
            # Recall: count original correct, and both correct
            if result.original_success:  # Original eval correct
                original_correct += 1
                if result.model_score == 1.0:  # Judge also judged as Correct
                    both_correct += 1
        
        # Summary statistics
        total = len(all_results)
        accuracy = match_count / total if total > 0 else 0.0
        recall = both_correct / original_correct if original_correct > 0 else 0.0
        
        summary = {
            "total_tasks": total,
            # Accuracy related
            "correct_judgments": match_count,
            "accuracy": accuracy,
            # Recall related (core metric)
            "original_correct_count": original_correct,
            "both_correct_count": both_correct,
            "recall": recall,
            # Detailed results
            "results": all_results,
        }
        
        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Self-Choice] Accuracy: {accuracy:.2%} ({match_count}/{total})")
        logger.info(f"[Self-Choice] Recall: {recall:.2%} ({both_correct}/{original_correct}) - Judge correct AND original correct / original correct")
        
        return summary
