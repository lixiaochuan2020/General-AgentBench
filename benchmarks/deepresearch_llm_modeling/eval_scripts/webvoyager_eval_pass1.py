import os
import json
from collections import defaultdict
import openai
import argparse

def load_questions(test_file, question_ids_filter=None):
    """Load questions from test.json"""
    with open(test_file, 'r') as f:
        all_questions = {q['id']: q['question'] for q in json.load(f)}
    
    # Filter by question IDs if provided
    if question_ids_filter:
        return {qid: q for qid, q in all_questions.items() if qid in question_ids_filter}
    return all_questions

def convert_result_to_webvoyager_format(result_data, question_text):
    """
    Convert your result_x.json format to WebVoyager evaluation format
    """
    context_lengths = result_data.get('context lengths', [])
    
    # --- THIS IS THE FIX ---
    max_context = 0
    if context_lengths:
        # Check if the list contains dictionaries
        if isinstance(context_lengths[0], dict):
            # Use a lambda function to get the 'total' value from each dict
            max_context = max(item.get('total', 0) for item in context_lengths)
        else:
            # If it's a list of numbers, it will work as before
            max_context = max(context_lengths)
    # --- END OF FIX ---

    return {
        'task': question_text,
        'final_answer': result_data.get('answer', ''),
        'model': result_data.get('model', ''),
        'turns': result_data.get('turns', 0),
        'context_lengths': context_lengths,
        'max_context': max_context  # Use the correctly calculated max_context
    }

def evaluate_single_result(task, answer, api_key, model="gpt-4o"):
    """
    Evaluates a single result, returning a tuple of (is_success, reasoning_text).
    """
    
    evaluation_prompt = f"""
You are an expert evaluator for web navigation tasks. Your job is to determine if an AI agent successfully completed a given task based on its final answer.

**Task**: {task}

**Agent's Final Answer**:
{answer}

**Instructions**:
1.  **Reasoning**: In a single sentence, briefly explain your reasoning for why the agent's answer is a success or failure.
2.  **Verdict**: On a new line, respond with exactly one word: "SUCCESS" or "FAILURE".

**Example**:
Reasoning: The agent correctly found the price of the item but failed to find the shipping cost.
FAILURE
"""
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0,
            max_tokens=150
        )
        
        raw_output = response.choices[0].message.content.strip()
        lines = raw_output.splitlines()

        if not lines:
            return False, "Evaluation Error: LLM returned an empty response."

        # Robustly parse the output
        verdict_line = lines[-1].strip().upper()
        reasoning = "\n".join(lines[:-1]).strip().replace("Reasoning: ", "") or "No reasoning provided."
        
        is_success = verdict_line == "SUCCESS"
        return is_success, reasoning
        
    except Exception as e:
        error_msg = f"Evaluation error: {e}"
        print(error_msg)
        return False, error_msg

def load_results(result_dir, questions):
    """
    Load results from a single directory
    """
    all_results = defaultdict(list)
    
    print(f"Loading results from: {result_dir}")
    
    if not os.path.exists(result_dir):
        print(f"Warning: {result_dir} not found!")
        return all_results
        
    for filename in os.listdir(result_dir):
        if filename.startswith('result_') and filename.endswith('.json'):
            question_id = int(filename.replace('result_', '').replace('.json', ''))
            
            if question_id not in questions:
                continue
                
            with open(os.path.join(result_dir, filename), 'r') as f:
                result_data = json.load(f)
            
            formatted_result = convert_result_to_webvoyager_format(
                result_data, questions[question_id]
            )
            all_results[question_id].append(formatted_result)
    
    return all_results

def calculate_pass_at_k_with_webvoyager_eval(all_results, api_key, k=1, min_context_tokens=0, model="gpt-4o"):
    """
    Calculate pass@k using WebVoyager-style evaluation.
    MODIFIED: Captures and displays reasoning.
    """
    
    filtered_results = {}
    for qid, results in all_results.items():
        has_sufficient_context = any(r['max_context'] >= min_context_tokens for r in results)
        if has_sufficient_context:
            filtered_results[qid] = results[:k]
    
    print(f"Evaluating {len(filtered_results)} questions...")
    
    detailed_results = {}
    successful_questions = 0
    
    for qid, results in filtered_results.items():
        print(f"\nEvaluating question {qid}...")
        
        successes = []
        reasonings = []
        for i, result in enumerate(results):
            is_success, reasoning = evaluate_single_result(
                result['task'], 
                result['final_answer'], 
                api_key,
                model
            )
            successes.append(is_success)
            reasonings.append(reasoning)
            
            status = "✅ SUCCESS" if is_success else "❌ FAILURE"
            print(f"  Attempt {i+1}: {status}")
            print(f"    Reasoning: {reasoning}")
        
        question_passed = any(successes)
        if question_passed:
            successful_questions += 1
        
        detailed_results[qid] = {
            'passed': question_passed,
            'attempts': len(results),
            'successes_per_attempt': successes,
            'reasonings_per_attempt': reasonings,
            'success_count': sum(successes),
            'task': results[0]['task'][:100] + "..."
        }
    
    pass_at_k_rate = successful_questions / len(filtered_results) if filtered_results else 0
    
    return pass_at_k_rate, detailed_results

# Main execution
def main():
    parser = argparse.ArgumentParser(description='WebVoyager Pass@1 Evaluation')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to questions JSON file')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result files')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output_file', type=str, default='webvoyager_results.json', help='Output file path')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM judge model')
    parser.add_argument('--question_ids', type=int, nargs='*', default=None, 
                        help='Optional: specific question IDs to evaluate')
    
    args = parser.parse_args()
    
    print("=== WebVoyager-Style Pass@1 Evaluation ===")

    # Load questions with filtering
    question_ids_filter = set(args.question_ids) if args.question_ids else None
    questions = load_questions(args.questions_file, question_ids_filter)
    print(f"Loaded {len(questions)} questions")

    # Load results
    all_results = load_results(args.result_dir, questions)
    print(f"Loaded results for {len(all_results)} questions")

    # Calculate pass@1
    pass_at_1, detailed = calculate_pass_at_k_with_webvoyager_eval(
        all_results, args.api_key, k=1, model=args.model
    )

    # Results
    print(f"\n=== FINAL RESULTS ===")
    print(f"Pass@1 Rate: {pass_at_1:.3f} ({pass_at_1*100:.1f}%)")
    print(f"Questions evaluated: {len(detailed)}")
    print(f"Questions that passed: {sum(1 for r in detailed.values() if r['passed'])}")

    with open(args.output_file, 'w') as f:
        json.dump({
            'benchmark': 'WebVoyager',
            'pass_at_1_rate': pass_at_1,
            'evaluation_method': 'webvoyager_task_completion',
            'total_questions': len(detailed),
            'successful_questions': sum(1 for r in detailed.values() if r['passed']),
            'detailed_results': detailed,
            'methodology': {
                'evaluation_type': 'task_success',
                'judge_model': args.model
            }
        }, f, indent=2)

    print(f"\nResults saved to {args.output_file}")
    print("✅ WebVoyager evaluation complete!")


if __name__ == "__main__":
    main()