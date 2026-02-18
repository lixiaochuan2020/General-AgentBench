import os
import json
from collections import defaultdict
import openai
import argparse


def load_browsecomp_ground_truth(ground_truth_file):
    """Load BrowseComp ground truth with answers"""
    questions = {}
    with open(ground_truth_file, 'r') as f:
        data = json.load(f)
    
    for item in data:
        question_id = item['id']
        questions[question_id] = {
            'question': item.get('question', ''),
            'correct_answer': item.get('answer', '')
        }
    
    return questions


def convert_result_to_browsecomp_format(result_data, question_data):
    context_lengths = result_data.get('context_lengths', result_data.get('context lengths', []))
    if isinstance(context_lengths, list) and len(context_lengths) > 0:
        if isinstance(context_lengths[0], dict):
            max_context = max(cl.get('total', 0) for cl in context_lengths)
        else:
            max_context = max(context_lengths) if context_lengths else 0
    else:
        max_context = 0
    
    return {
        'question': question_data['question'],
        'correct_answer': question_data['correct_answer'],
        'agent_answer': result_data.get('answer', ''),
        'model': result_data.get('model', ''),
        'turns': result_data.get('turns', 0),
        'max_context': max_context
    }


def evaluate_browsecomp_answer(question, correct_answer, agent_answer, api_key, model="gpt-4o"):
    # 1. FIX: Expanded the list with even more failure synonyms.
    failure_indicators = [
        'did not find', 'could not find', 'unable to find', 'unable to identify',
        'cannot be identified', 'cannot be definitively identified', 'not found', 
        'no information', 'insufficient information', 'could not be determined', 
        'cannot determine', 'i don\'t know', 'i do not know', 'no answer', 
        'answer not available', 'information not available', 'not publicly available'
    ]
    agent_answer_lower = agent_answer.lower().strip()
    if any(indicator in agent_answer_lower for indicator in failure_indicators):
        # This will now correctly catch Question 75 and Question 122
        return False
    if len(agent_answer.strip()) < 3:
        return False

    # The prompt from before is good. No changes needed here.
    grader_prompt = f"""
You are a strict, impartial expert evaluator for the BrowseComp benchmark. Your task is to determine if the agent's answer is correct based on the ground truth.

**EVALUATION RUBRIC:**
1.  **Correct:** The agent's answer must be a semantically equivalent match to the ground truth answer. Exact wording is not required, but the core facts must be identical.
2.  **Incorrect:** The agent's answer is incorrect if it contains any factual errors, is a different entity/value, is only partially correct, or is related but not the same as the ground truth. Be very strict about this.

**TASK:**
**Question**: {question}
**Ground Truth Answer**: {correct_answer}
**Agent's Answer**: {agent_answer}

Carefully compare the Agent's Answer to the Ground Truth Answer based on the rubric.

First, provide a one-sentence reasoning for your decision. Then, on a new line, output your final verdict: "CORRECT" or "INCORRECT".

**EXAMPLE 1:**
Ground Truth Answer: 1986, Portugal
Agent's Answer: 1983, Spain
Reasoning: The agent provided the wrong year and the wrong country.
INCORRECT

**EXAMPLE 2:**
Ground Truth Answer: DN AGRAR Group
Agent's Answer: Astarta Holding
Reasoning: The agent provided the name of a different company.
INCORRECT

**Your Evaluation:**
"""
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": grader_prompt}],
            temperature=0,
            max_tokens=100 
        )
        raw_output = response.choices[0].message.content
        
        # DEBUG: Print the raw output from the LLM to see what it's thinking.
        # You can comment this out once you confirm it's working.
        print(f"--- LLM Judge Raw Output ---\n{raw_output}\n--------------------------")
        
        result = raw_output.strip().upper()
        last_line = result.splitlines()[-1].strip() # Get the last line and strip whitespace

        # 2. FIX: Change the check from 'in' to an exact match '=='.
        # This is the critical fix for the main bug.
        return last_line == "CORRECT"

    except Exception as e:
        print(f"Evaluation error: {e}")
        return False


def load_browsecomp_results(result_dir, questions):
    all_results = defaultdict(list)
    print(f"Loading BrowseComp results from: {result_dir}")
    if not os.path.exists(result_dir):
        print(f"Warning: {result_dir} not found, skipping...")
        return all_results

    for filename in os.listdir(result_dir):
        if filename.startswith('result_') and filename.endswith('.json'):
            question_id = int(filename.replace('result_', '').replace('.json', ''))
            if question_id not in questions:
                continue
            with open(os.path.join(result_dir, filename), 'r') as f:
                result_data = json.load(f)
            formatted_result = convert_result_to_browsecomp_format(
                result_data, questions[question_id]
            )
            all_results[question_id].append(formatted_result)
    return all_results


def calculate_browsecomp_pass_at_k(all_results, api_key, k=1, min_context_tokens=0, model="gpt-4o"):
    filtered_results = {}
    for question_id, results in all_results.items():
        has_sufficient_context = any(r['max_context'] >= min_context_tokens for r in results)
        if has_sufficient_context:
            filtered_results[question_id] = results[:k]

    print(f"Evaluating {len(filtered_results)} BrowseComp questions...")

    detailed_results = {}
    successful_questions = 0
    for question_id, results in filtered_results.items():
        print(f"\nEvaluating question {question_id}...")
        successes = []
        attempt_details = []
        for i, result in enumerate(results):
            is_correct = evaluate_browsecomp_answer(
                result['question'],
                result['correct_answer'],
                result['agent_answer'],
                api_key,
                model
            )
            successes.append(is_correct)
            attempt_details.append({
                'correct': is_correct,
                'answer_length': len(result['agent_answer']),
                'turns': result['turns']
            })
            print(f"  Attempt {i+1}: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            print(f"    Ground truth: {result['correct_answer'][:100]}...")
            print(f"    Agent answer: {result['agent_answer'][:100]}...")
        question_passed = any(successes)
        if question_passed:
            successful_questions += 1
        detailed_results[question_id] = {
            'passed': question_passed,
            'attempts': len(results),
            'successes_per_attempt': successes,
            'success_count': sum(successes),
            'attempt_details': attempt_details,
            'question': results[0]['question'][:200] + "...",
            'correct_answer': results[0]['correct_answer']
        }

    pass_at_k_rate = successful_questions / len(filtered_results) if filtered_results else 0
    return pass_at_k_rate, detailed_results


def main():
    parser = argparse.ArgumentParser(description='BrowseComp Pass@1 Evaluation')
    parser.add_argument('--ground_truth_file', type=str, required=True, help='Path to BrowseComp ground truth file with answers')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result files')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output_file', type=str, default='browsecomp_results.json', help='Output file path')
    parser.add_argument('--min_context_tokens', type=int, default=0, help='Minimum context tokens threshold')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM judge model')
    
    args = parser.parse_args()
    
    print("=== BrowseComp Pass@1 Evaluation ===")
    print("Using LLM-as-a-Judge for answer correctness\n")

    questions = load_browsecomp_ground_truth(args.ground_truth_file)
    print(f"Loaded {len(questions)} BrowseComp questions with ground truth")

    all_results = load_browsecomp_results(args.result_dir, questions)
    print(f"Loaded results for {len(all_results)} questions")

    pass_at_1, detailed = calculate_browsecomp_pass_at_k(
        all_results, args.api_key, k=1, min_context_tokens=args.min_context_tokens, model=args.model
    )

    print(f"\n" + "="*50)
    print(f"BROWSECOMP PASS@1 RESULTS")
    print(f"="*50)
    print(f"Pass@1 Rate: {pass_at_1:.3f} ({pass_at_1*100:.1f}%)")
    print(f"Questions evaluated: {len(detailed)}")
    print(f"Questions that passed: {sum(1 for r in detailed.values() if r['passed'])}")

    if detailed:
        first_attempt_success = sum(1 for r in detailed.values()
                                    if r['successes_per_attempt'][0]) / len(detailed)
        print(f"\nFIRST ATTEMPT ANALYSIS:")
        print(f"First attempt success rate: {first_attempt_success:.3f} ({first_attempt_success*100:.1f}%)")
        print(f"(Note: pass@1 equals first attempt success rate)")

    print(f"\nSAMPLE RESULTS:")
    for i, (q_id, result) in enumerate(list(detailed.items())[:3]):
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"\nQuestion {q_id}: {status} ({result['success_count']}/{result['attempts']} correct)")
        print(f"  Q: {result['question']}")
        print(f"  Correct answer: {result['correct_answer']}")

    with open(args.output_file, 'w') as f:
        json.dump({
            'benchmark': 'BrowseComp',
            'evaluation_method': 'llm_as_judge_answer_correctness',
            'pass_at_1_rate': pass_at_1,
            'total_questions': len(detailed),
            'successful_questions': sum(1 for r in detailed.values() if r['passed']),
            'first_attempt_success_rate': first_attempt_success if detailed else 0,
            'detailed_results': detailed,
            'methodology': {
                'context_filter': f'>= {args.min_context_tokens} tokens',
                'evaluation_type': 'binary_correctness',
                'judge_model': args.model
            }
        }, f, indent=2)

    print(f"\n📊 Results saved to: {args.output_file}")
    print(f"🎯 BrowseComp evaluation complete!")


if __name__ == "__main__":
    main()