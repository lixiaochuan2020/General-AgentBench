#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from prompt import *
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.token_calculator import tokenize
import concurrent.futures
import threading
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types
import google.generativeai as generativeai


MODEL_ID = "gemini-2.5-flash"
# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)


def call_llm(prompt, question_id=None):
    """
    Call Gemini API to generate critique
    
    Args:
        prompt: Input prompt
        question_id: Question ID for logging
    
    Returns:
        str: LLM generated critique content, returns empty string on failure
    """    
    max_try_times = 3
    for attempt in range(max_try_times):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            if "context" in str(e).lower() or "length" in str(e).lower():
                raise ValueError(f"Context length error for question {question_id}: {e}")
            
            if attempt == max_try_times - 1:
                return ""  # Return empty string on failure
            else:
                time.sleep(random.randint(1, 3))
    
    return ""


def call_llm_batch(critique_data_list, max_workers=5):
    """
    Concurrently call LLM to generate multiple critiques
    
    Args:
        critique_data_list: List of critique data, each element contains prompt and related information
        max_workers: Maximum number of concurrent workers
    
    Returns:
        list: List containing critique results
    """
    def process_single_critique(critique_data):
        """Process single critique"""
        question_id = critique_data['question_id']
        prompt = critique_data['critique_prompt']
        critique_response = call_llm(prompt, question_id)
        print(f"generated critique_response for question {question_id}")
        critique_data['critique'] = critique_response
        return critique_data
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect results
        futures = [executor.submit(process_single_critique, critique_data) 
                  for critique_data in critique_data_list]
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Throw serious errors directly
                raise e
    
    return results

def critique(results_dir, logs_dir, suffix, n, questions, output_dir="critique_results", use_llm=False, add_thinking=False, max_workers=32, use_ground_truth=False):
    """
    Generate critique report, analyze multiple attempt trajectories and summarize lessons learned
    
    Args:
        results_dir: Results directory path
        logs_dir: Logs directory path  
        suffix: Model suffix name
        n: Number of attempts
        questions: List of questions
        output_dir: Critique output directory
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure critique_input and critique_output directories exist
    critique_input_path = Path("critique/critique_input")
    critique_input_path.mkdir(parents=True, exist_ok=True)
    
    critique_output_path = Path("critique/critique_output")
    critique_output_path.mkdir(parents=True, exist_ok=True)
    
    critique_data_list = []
    
    for question in questions:
        question_text = question['question']
        ground_truth = question['answer']
        question_id = question['id']
        
        if use_ground_truth:
            formatted_prompt_1 = critique_prompt_1.format(question=question_text, ground_truth=ground_truth)
        else:
            formatted_prompt_1 = critique_prompt_1_no_ground_truth.format(question=question_text)
        
        trajectories_and_results = []
        
        for i in range(1, n+1):
            result_dir = Path(results_dir) / f"{suffix}_{i}"
            log_dir = Path(logs_dir) / f"{suffix}_{i}"
            trajectory_file = log_dir / f"trajectory_{question_id}.jsonl"
            evaluation_file = result_dir / f"result_{question_id}.json"
            
            # Check if files exist
            if not trajectory_file.exists() or not evaluation_file.exists():
                continue
            
            try:
                # Read trajectory file (JSONL format)
                trajectory_content = ""
                with open(trajectory_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for step_idx, line in enumerate(lines):
                        line_data = json.loads(line.strip())
                        agent_response = line_data.get('response', '')
                        if agent_response is not None and (not add_thinking) and ("</think>" in agent_response):
                            agent_response = agent_response.split("</think>")[1]
                        environment_feedback = line_data.get('next_obs', '')
                        trajectory_content += f"### Step {step_idx + 1}:\n\n#### Agent output: {agent_response}\n\n#### Environment Feedback: {environment_feedback}\n\n"

                # Read evaluation results
                with open(evaluation_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    score = eval_data.get('score', 'N/A')
                    if score == 1:
                        score = "Correct"
                    elif score == 0:
                        score = "Incorrect"
                    else:
                        score = "N/A"
                    
                # Format second part of prompt
                formatted_prompt_2 = critique_prompt_2.format(i=i, trajectory=trajectory_content, evaluation_results=score)
                trajectories_and_results.append(formatted_prompt_2)
                
            except Exception as e:
                # Throw serious errors directly
                raise e
        
        if trajectories_and_results:
            # Combine complete critique prompt
            full_critique_prompt = formatted_prompt_1  + "\n\n".join(trajectories_and_results)
            
            # Save critique prompt to separate file
            if use_ground_truth:
                critique_input_dir = critique_input_path / f"{suffix}"
            else:
                critique_input_dir = critique_input_path / f"{suffix}_wo_ground_truth"
            critique_input_dir.mkdir(parents=True, exist_ok=True)
            critique_prompt_file = critique_input_dir / f"{question_id}.txt"
            with open(critique_prompt_file, 'w', encoding='utf-8') as f:
                f.write(full_critique_prompt)
            
            critique_input_length = tokenize(full_critique_prompt)
            
            critique_data = {
                "question_id": question_id,
                "question": question_text,
                "ground_truth": ground_truth,
                "model_suffix": suffix,
                "num_attempts": len(trajectories_and_results),
                "critique_prompt_length": critique_input_length,
                "critique_prompt": full_critique_prompt,
                "critique": ""  # Default empty, filled later by LLM
            }
            
            critique_data_list.append(critique_data)
    
    # Concurrently call LLM to generate critique
    if use_llm and critique_data_list:
        critique_data_list = call_llm_batch(critique_data_list, max_workers)
   
    # Save all individual question critique files
    critiques = []
    for critique_data in critique_data_list:
        question_id = critique_data['question_id']
        if use_ground_truth:
            output_dir = output_path / f"{suffix}"
        else:
            output_dir = output_path / f"{suffix}_wo_ground_truth"
        # Save complete critique data to original directory
        
        output_dir.mkdir(parents=True, exist_ok=True)
        question_output_file = output_dir / f"{question_id}.json"
        with open(question_output_file, 'w', encoding='utf-8') as f:
            json.dump(critique_data, f, ensure_ascii=False, indent=2)
        
        # Save individual critique results to critique_output directory
        if critique_data['critique']:  # Only save when critique is not empty
            if use_ground_truth:
                critique_output_dir = critique_output_path / f"{suffix}"
            else:
                critique_output_dir = critique_output_path / f"{suffix}_wo_ground_truth"
            critique_output_dir.mkdir(parents=True, exist_ok=True)
            critique_result_file = critique_output_dir / f"{question_id}.txt"
            with open(critique_result_file, 'w', encoding='utf-8') as f:
                f.write(critique_data['critique'])
        
        critiques.append(critique_data)
    
    # Calculate statistics
    if critiques:
        critique_lengths = [c['critique_prompt_length'] for c in critiques]
        avg_length = sum(critique_lengths) / len(critique_lengths)
        max_length = max(critique_lengths)
        min_length = min(critique_lengths)
        
        length_stats = {
            "total_questions": len(critiques),
            "avg_length": round(avg_length, 2),
            "max_length": max_length,
            "min_length": min_length,
            "length_range": max_length - min_length
        }
    else:
        length_stats = {
            "total_questions": 0,
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
            "length_range": 0
        }
    
    # Save summary file
    summary_data = {
        "critiques": critiques,
        "length_statistics": length_stats
    }
    
    summary_file = output_path / f"critique_summary_{suffix}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    return critiques


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate critique report to analyze agent trajectories")
    parser.add_argument("--suffix", default="qwen3_1.7b_grpo_1_step216", help="Model suffix name")
    parser.add_argument("--results_dir", default="results/webwalkerqa/best_of_n", help="Results directory path")
    parser.add_argument("--logs_dir", default="logs/webwalkerqa/best_of_n", help="Logs directory path")
    parser.add_argument("--question_file", default="evaluation/short/webwalkerqa/webwalkerqa_200.json", help="Question file path")
    parser.add_argument("--n", type=int, default=8, help="Number of attempts")
    parser.add_argument("--use_ground_truth", action="store_true", help="Whether to use ground truth answers")
    parser.add_argument("--add_thinking", action="store_true", help="Whether to include thinking process in trajectory")
    parser.add_argument("--output_dir", default="critique/critique_results", help="Critique output directory")
    parser.add_argument("--use_llm", action="store_true", help="Whether to use real LLM to generate critique")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of threads for concurrent LLM calls")
    parser.add_argument("--max_questions", type=int, help="Maximum number of questions to process (for testing)")
    
    args = parser.parse_args()
    
    # Use parsed arguments
    suffix = args.suffix
    results_dir = args.results_dir
    logs_dir = args.logs_dir
    add_thinking = args.add_thinking
    question_file = args.question_file
    n = args.n
    use_ground_truth = args.use_ground_truth
    # Check if question file exists
    if not os.path.exists(question_file):
        print(f"Error: Question file does not exist: {question_file}")
        exit(1)
    
    # Read question data
    try:
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # If max questions specified, truncate
        if args.max_questions:
            questions = questions[:args.max_questions]
            
    except Exception as e:
        raise Exception(f"Failed to read question file: {e}")
    
    # Run critique analysis
    critiques = critique(
        results_dir=results_dir, 
        logs_dir=logs_dir, 
        suffix=suffix, 
        n=n, 
        questions=questions,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        add_thinking=add_thinking,
        max_workers=args.max_workers,
        use_ground_truth=use_ground_truth
    )
    
    print(f"Processing completed: {len(critiques)} questions")
    
    # Display statistics
    if critiques:
        critique_lengths = [c['critique_prompt_length'] for c in critiques]
        avg_length = sum(critique_lengths) / len(critique_lengths)
        max_length = max(critique_lengths)
        min_length = min(critique_lengths)
        
        print(f"Average length: {avg_length:.2f} tokens")
        print(f"Length range: {min_length} - {max_length} tokens")
    
