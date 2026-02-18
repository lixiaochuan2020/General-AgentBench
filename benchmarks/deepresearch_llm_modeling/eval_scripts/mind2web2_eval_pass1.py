import os
import json
import re
import argparse
from collections import defaultdict
import openai


def load_rubrics(rubrics_file):
    """Load task-specific rubrics from file
    
    Expected format:
    {
        "125": {
            "task_name": "example_task",
            "rubric": {
                "type": "and",  # or "or"
                "children": [
                    {
                        "type": "correctness",
                        "description": "Does answer contain X?",
                        "criteria": "..."
                    },
                    {
                        "type": "attribution", 
                        "description": "Is claim Y attributed to source?",
                        "criteria": "..."
                    }
                ]
            }
        }
    }
    """
    with open(rubrics_file, 'r') as f:
        return json.load(f)


def parse_trajectory_jsonl(trajectory_file):
    """Parse trajectory JSONL file"""
    turns = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            turns.append(json.loads(line.strip()))
    
    # Handle both dict and int formats for context_length
    max_context = 0
    if turns:
        for turn in turns:
            ctx_len = turn.get('context_length', 0)
            if isinstance(ctx_len, dict):
                ctx_len = ctx_len.get('total', 0)
            max_context = max(max_context, ctx_len)
    
    return {
        'turns': turns,
        'total_turns': len(turns),
        'max_context_length': max_context
    }


def extract_final_answer(trajectory):
    """Extract final answer from <answer> tags"""
    for turn in reversed(trajectory['turns']):
        response = turn.get('response', '')
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
    return ""


def extract_sources_from_trajectory(trajectory):
    """Extract all sources found during trajectory"""
    sources = []
    source_contents = {}
    
    for turn in trajectory['turns']:
        next_obs = turn.get('next_obs', '')
        
        # Extract information blocks
        info_matches = re.findall(r'<information>(.*?)</information>', next_obs, re.DOTALL)
        for info_block in info_matches:
            # Extract URLs
            url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+[^\s<>"\'`|(){}[\].,;:]'
            urls = re.findall(url_pattern, info_block)
            
            for url in urls:
                if url not in source_contents:
                    sources.append(url)
                    # Store the content associated with this URL
                    source_contents[url] = info_block
    
    return sources, source_contents


def extract_citations_from_answer(answer):
    """Extract cited URLs from answer"""
    url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+[^\s<>"\'`|(){}[\].,;:]'
    return re.findall(url_pattern, answer)


def evaluate_rubric_node(node, answer, sources, source_contents, question, api_key, model="gpt-4o"):
    """
    Recursively evaluate a rubric node.
    FIXED: Properly handles recursive formatting of reasoning strings.
    """
    node_type = node.get('type')
    
    if node_type == 'and' or node_type == 'or':
        children = node.get('children', [])
        if not children:
            return True, f"No criteria under this '{node_type}' node."

        all_reasons = []
        # For 'and', start by assuming success. For 'or', assume failure.
        final_pass = (node_type == 'and') 

        for i, child in enumerate(children):
            # Recursively evaluate the child node
            is_pass, reason = evaluate_rubric_node(child, answer, sources, source_contents, question, api_key, model)
            
            status = 'PASS' if is_pass else 'FAIL'
            child_type = child.get('type')

            # --- THIS IS THE CRITICAL FIX ---
            # Check if the child was a leaf or another branch to format correctly.
            if child_type in ['correctness', 'attribution']:
                # Child was a LEAF. The 'reason' is a simple string. Format it on one line.
                formatted_reason = f"  - Criteria #{i+1} ({status}): {reason}"
            else:
                # Child was a BRANCH. The 'reason' is already a formatted, multi-line block.
                # Give it a header and indent the entire block for nice hierarchy.
                indented_sub_reason = "    " + reason.replace("\n", "\n    ")
                formatted_reason = f"  - Criteria #{i+1} ({status}):\n{indented_sub_reason}"
            
            all_reasons.append(formatted_reason)

            # Update final pass status based on the logic ('and' or 'or')
            if node_type == 'and' and not is_pass:
                final_pass = False
            if node_type == 'or' and is_pass:
                final_pass = True
        
        return final_pass, "\n".join(all_reasons)
    
    elif node_type == 'correctness':
        # This is a leaf node, it returns a raw (bool, reason) tuple
        return evaluate_correctness(node, answer, question, api_key, model)
    
    elif node_type == 'attribution':
        # This is also a leaf node
        return evaluate_attribution(node, answer, sources, source_contents, api_key, model)
    
    else:
        reason = f"Warning: Unknown node type '{node_type}', defaulting to FAIL."
        print(reason)
        return False, reason

def evaluate_correctness(node, answer, question, api_key, model="gpt-4o"):
    """Evaluate if answer meets correctness criteria"""
    
    description = node.get('description', '')
    criteria = node.get('criteria', '')
    
    # --- MODIFICATION START ---
    # New, more detailed prompt
    prompt = f"""You are a strict evaluator for an AI agent.
Your task is to determine if the agent's answer satisfies the given criteria for the question.

**Task/Question**: {question}

**Agent's Answer**:
{answer}

**Evaluation Criteria**: {description}
{criteria}

**Instructions**:
1.  **Reasoning**: Briefly explain why the agent's answer passes or fails the criteria.
2.  **Verdict**: On a new line, respond with a single word: "PASS" or "FAIL".

**Example**:
Reasoning: The agent correctly identified the capital city as requested.
PASS
"""
    # --- MODIFICATION END ---
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # Increased max_tokens for reasoning
            max_tokens=150
        )
        
        raw_output = response.choices[0].message.content.strip()
        lines = raw_output.splitlines()
        
        if not lines:
            return False, "Evaluation Error: LLM returned empty response."
        
        # Robustly parse the output
        verdict_line = lines[-1].strip().upper()
        reasoning = "\n".join(lines[:-1]).strip() or "No reasoning provided."
        
        is_pass = "PASS" in verdict_line and "FAIL" not in verdict_line
        return is_pass, reasoning
        
    except Exception as e:
        error_msg = f"Correctness evaluation error: {e}"
        print(error_msg)
        return False, error_msg

def evaluate_attribution(node, answer, sources, source_contents, api_key, model="gpt-4o"):
    """Evaluate if claims in answer are attributed to sources"""
    
    description = node.get('description', '')
    criteria = node.get('criteria', '')
    
    cited_urls = extract_citations_from_answer(answer)
    source_context = ""
    for url in cited_urls[:5]:
        content = source_contents.get(url, "")
        source_context += f"\n\n**Source: {url}**\n{content[:500]}..."
    
    if not source_context:
        # If no sources were cited, we can fail early.
        return False, "Attribution Failure: The agent's answer did not cite any sources."
    
    # --- MODIFICATION START ---
    # New, more detailed prompt
    prompt = f"""You are a strict evaluator for an AI agent.
Your task is to determine if the claims in the agent's answer are properly supported by the sources it cited.

**Agent's Answer**:
{answer}

**Cited Sources Content (Partial)**:
{source_context}

**Evaluation Criteria**: {description}
{criteria}

**Instructions**:
1.  **Reasoning**: Briefly explain why the answer is or is not well-supported by the provided sources.
2.  **Verdict**: On a new line, respond with a single word: "PASS" or "FAIL".
"""
    # --- MODIFICATION END ---

    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # Increased max_tokens for reasoning
            max_tokens=150
        )
        
        raw_output = response.choices[0].message.content.strip()
        lines = raw_output.splitlines()

        if not lines:
            return False, "Evaluation Error: LLM returned empty response."

        # Robustly parse the output
        verdict_line = lines[-1].strip().upper()
        reasoning = "\n".join(lines[:-1]).strip() or "No reasoning provided."

        is_pass = "PASS" in verdict_line and "FAIL" not in verdict_line
        return is_pass, reasoning
        
    except Exception as e:
        error_msg = f"Attribution evaluation error: {e}"
        print(error_msg)
        return False, error_msg


def load_questions(questions_file):
    """Load questions from JSON file"""
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    questions = {}
    for item in data:
        question_id = item['id']
        question_text = item.get('question', '')
        
        # Try to parse if it's a stringified dict
        if isinstance(question_text, str) and question_text.startswith('{'):
            try:
                import ast
                question_data = ast.literal_eval(question_text)
                question_text = question_data.get('task_description', question_data.get('question', question_text))
            except:
                pass
        
        questions[question_id] = question_text
    
    return questions


def load_trajectory_results(trajectory_dir, questions, question_ids):
    """Load and parse trajectory files for specific question IDs"""
    all_results = {}
    
    print(f"Loading Mind2Web-2 trajectories from: {trajectory_dir}")
    
    if not os.path.exists(trajectory_dir):
        print(f"Error: {trajectory_dir} not found!")
        return all_results
    
    for question_id in question_ids:
        filename = f'trajectory_{question_id}.jsonl'
        trajectory_path = os.path.join(trajectory_dir, filename)
        
        if not os.path.exists(trajectory_path):
            print(f"Warning: {filename} not found")
            continue
        
        try:
            trajectory = parse_trajectory_jsonl(trajectory_path)
            final_answer = extract_final_answer(trajectory)
            sources, source_contents = extract_sources_from_trajectory(trajectory)
            
            all_results[question_id] = {
                'question': questions.get(question_id, ''),
                'final_answer': final_answer,
                'sources': sources,
                'source_contents': source_contents,
                'max_context_length': trajectory['max_context_length'],
                'total_turns': trajectory['total_turns']
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    
    return all_results


def evaluate_mind2web2(all_results, rubrics, api_key, min_context_tokens=0):
    """Evaluate using Mind2Web-2's Agent-as-a-Judge with rubrics"""
    
    filtered_results = {
        qid: result for qid, result in all_results.items()
        if result['max_context_length'] >= min_context_tokens
    }
    
    print(f"\nEvaluating {len(filtered_results)} Mind2Web-2 questions with rubrics...")
    
    detailed_results = {}
    successful_questions = 0
    
    for qid, result in filtered_results.items():
        print(f"\nEvaluating question {qid}...")
        
        rubric = rubrics.get(str(qid))
        
        if not rubric:
            print(f"  ⚠️  No rubric found for question {qid}, skipping")
            continue
        
        # --- MODIFICATION START ---
        # Call the modified function and capture the reasoning
        is_success, reasoning = evaluate_rubric_node(
            rubric['rubric'],
            result['final_answer'],
            result['sources'],
            result['source_contents'],
            result['question'],
            api_key
        )
        
        if is_success:
            successful_questions += 1
        
        status = "✅ SUCCESS" if is_success else "❌ FAILURE"
        print(f"  {status}")
        # Print the detailed reasoning
        print(f"  Judge's Reasoning:\n{reasoning}")
        print(f"  Sources found: {len(result['sources'])}, Turns: {result['total_turns']}")
        
        detailed_results[qid] = {
            'passed': is_success,
            # Store the reasoning in the results
            'reasoning': reasoning,
            'question': result['question'][:200] + "..." if len(result['question']) > 200 else result['question'],
            'answer_length': len(result['final_answer']),
            'num_sources': len(result['sources']),
            'total_turns': result['total_turns'],
            'task_name': rubric.get('task_name', f'task_{qid}')
        }
        # --- MODIFICATION END ---
    
    pass_rate = successful_questions / len(filtered_results) if filtered_results else 0
    
    return pass_rate, detailed_results


def main():
    parser = argparse.ArgumentParser(description='Mind2Web-2 evaluation with Agent-as-a-Judge rubrics')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to questions JSON file')
    parser.add_argument('--trajectory_dir', type=str, required=True, help='Directory containing trajectory files')
    parser.add_argument('--rubrics_file', type=str, required=True, help='Path to rubrics JSON file')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output_file', type=str, default='mind2web2_evaluation_results.json', help='Output file path')
    parser.add_argument('--min_context_tokens', type=int, default=0, help='Minimum context tokens threshold')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM judge model')
    parser.add_argument('--question_ids', type=int, nargs='+', default=list(range(125, 135)), 
                        help='Question IDs to evaluate (default: 125-134)')
    
    args = parser.parse_args()
    
    print("=== Mind2Web-2 Evaluation (Agent-as-a-Judge) ===")
    print(f"Judge model: {args.model}")
    print(f"Question IDs: {args.question_ids}")
    print(f"Min context tokens: {args.min_context_tokens}\n")
    
    # Load rubrics
    rubrics = load_rubrics(args.rubrics_file)
    print(f"Loaded rubrics for {len(rubrics)} tasks")
    
    # Load questions
    questions = load_questions(args.questions_file)
    print(f"Loaded {len(questions)} total questions")
    
    # Load trajectories for Mind2Web-2 questions only
    all_results = load_trajectory_results(args.trajectory_dir, questions, args.question_ids)
    print(f"Loaded {len(all_results)} Mind2Web-2 trajectory results")
    
    # Evaluate with rubrics
    pass_rate, detailed = evaluate_mind2web2(
        all_results, 
        rubrics,
        args.api_key, 
        args.min_context_tokens
    )
    
    # Calculate aggregate statistics
    avg_sources = sum(r['num_sources'] for r in detailed.values()) / len(detailed) if detailed else 0
    avg_turns = sum(r['total_turns'] for r in detailed.values()) / len(detailed) if detailed else 0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MIND2WEB-2 RESULTS (Agent-as-a-Judge)")
    print(f"{'='*60}")
    print(f"Pass Rate: {pass_rate:.3f} ({pass_rate*100:.1f}%)")
    print(f"Questions evaluated: {len(detailed)}")
    print(f"Questions passed: {sum(1 for r in detailed.values() if r['passed'])}")
    print(f"\nAGGREGATE STATISTICS:")
    print(f"Average sources found: {avg_sources:.1f}")
    print(f"Average turns per question: {avg_turns:.1f}")
    
    # Show sample results
    print(f"\nSAMPLE RESULTS:")
    for i, (qid, result) in enumerate(list(detailed.items())[:5]):
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"\nQuestion {qid} ({result['task_name']}): {status}")
        print(f"  Q: {result['question']}")
        print(f"  Sources: {result['num_sources']}, Turns: {result['total_turns']}")
    
    # Save results
    output_data = {
        'benchmark': 'Mind2Web-2',
        'evaluation_method': 'agent_as_judge_with_rubrics',
        'judge_model': args.model,
        'pass_rate': pass_rate,
        'total_questions': len(detailed),
        'successful_questions': sum(1 for r in detailed.values() if r['passed']),
        'aggregate_statistics': {
            'avg_sources': avg_sources,
            'avg_turns': avg_turns
        },
        'detailed_results': detailed,
        'evaluation_framework': {
            'correctness': 'Does answer satisfy all task requirements?',
            'attribution': 'Are claims backed by sources found?',
            'rubric_based': True,
            'min_context_tokens': args.min_context_tokens
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📊 Results saved to: {args.output_file}")
    print(f"✅ Mind2Web-2 evaluation complete!")


if __name__ == "__main__":
    main()