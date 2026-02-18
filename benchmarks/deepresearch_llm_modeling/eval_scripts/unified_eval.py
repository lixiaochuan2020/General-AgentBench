import os
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor


def load_questions(questions_file):
    """Load all questions to determine routing"""
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Separate by benchmark based on ID ranges
    browsecomp_ids = []
    mind2web2_ids = []
    webvoyager_ids = []
    
    for item in data:
        qid = item['id']
        if 1 <= qid <= 124:
            browsecomp_ids.append(qid)
        elif 125 <= qid <= 134:
            mind2web2_ids.append(qid)
        elif 135 <= qid <= 199:
            webvoyager_ids.append(qid)
    
    return {
        'browsecomp': browsecomp_ids,
        'mind2web2': mind2web2_ids,
        'webvoyager': webvoyager_ids
    }


def load_existing_result(output_dir, benchmark):
    """Load existing result file if it exists"""
    result_file = os.path.join(output_dir, f'{benchmark}_results.json')
    if os.path.exists(result_file):
        print(f"\n📂 Loading existing {benchmark} results from {result_file}")
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def run_browsecomp_eval(browsecomp_ground_truth, result_dir, api_key, output_dir):
    """Run BrowseComp evaluation"""
    # Check if results already exist
    existing = load_existing_result(output_dir, 'browsecomp')
    if existing:
        return existing
    
    print("\n" + "="*60)
    print("RUNNING BROWSECOMP EVALUATION")
    print("="*60 + "\n")
    
    cmd = [
        'python', 'browsecomp_eval_pass1.py',
        '--ground_truth_file', browsecomp_ground_truth,
        '--result_dir', result_dir,
        '--api_key', api_key,
        '--output_file', os.path.join(output_dir, 'browsecomp_results.json')
    ]
    
    subprocess.run(cmd, check=True)
    
    # Load results
    with open(os.path.join(output_dir, 'browsecomp_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def run_mind2web2_eval(questions_file, trajectory_dir, rubrics_file, api_key, output_dir, question_ids):
    """Run Mind2Web-2 evaluation"""
    # Check if results already exist
    existing = load_existing_result(output_dir, 'mind2web2')
    if existing:
        return existing
    
    print("\n" + "="*60)
    print("RUNNING MIND2WEB-2 EVALUATION")
    print("="*60 + "\n")
    
    cmd = [
        'python', 'mind2web2_eval_pass1.py',
        '--questions_file', questions_file,
        '--trajectory_dir', trajectory_dir,
        '--rubrics_file', rubrics_file,
        '--api_key', api_key,
        '--output_file', os.path.join(output_dir, 'mind2web2_results.json'),
        '--question_ids', *[str(qid) for qid in question_ids]
    ]
    
    subprocess.run(cmd, check=True)
    
    # Load results
    with open(os.path.join(output_dir, 'mind2web2_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def run_webvoyager_eval(questions_file, result_dir, api_key, output_dir, question_ids):
    """Run WebVoyager evaluation"""
    # Check if results already exist
    existing = load_existing_result(output_dir, 'webvoyager')
    if existing:
        return existing
    
    print("\n" + "="*60)
    print("RUNNING WEBVOYAGER EVALUATION")
    print("="*60 + "\n")
    
    cmd = [
        'python', 'webvoyager_eval_pass1.py',
        '--questions_file', questions_file,
        '--result_dir', result_dir,
        '--api_key', api_key,
        '--output_file', os.path.join(output_dir, 'webvoyager_results.json'),
        '--question_ids', *[str(qid) for qid in question_ids]
    ]
    
    subprocess.run(cmd, check=True)
    
    # Load results
    with open(os.path.join(output_dir, 'webvoyager_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_results(browsecomp_results, mind2web2_results, webvoyager_results, question_counts):
    """Aggregate results across all benchmarks"""
    
    total_questions = sum(len(v) for v in question_counts.values())
    
    # Weighted average pass rate
    browsecomp_pass = browsecomp_results.get('pass_at_1_rate', 0) * len(question_counts['browsecomp'])
    mind2web2_pass = mind2web2_results.get('pass_rate', 0) * len(question_counts['mind2web2'])
    webvoyager_pass = webvoyager_results.get('pass_at_1_rate', 0) * len(question_counts['webvoyager'])
    
    overall_pass_rate = (browsecomp_pass + mind2web2_pass + webvoyager_pass) / total_questions if total_questions > 0 else 0
    
    aggregate = {
        'total_questions': total_questions,
        'overall_pass_rate': overall_pass_rate,
        'by_benchmark': {
            'browsecomp': {
                'count': len(question_counts['browsecomp']),
                'pass_rate': browsecomp_results.get('pass_at_1_rate', 0),
                'passed': browsecomp_results.get('successful_questions', 0)
            },
            'mind2web2': {
                'count': len(question_counts['mind2web2']),
                'pass_rate': mind2web2_results.get('pass_rate', 0),
                'passed': mind2web2_results.get('successful_questions', 0)
            },
            'webvoyager': {
                'count': len(question_counts['webvoyager']),
                'pass_rate': webvoyager_results.get('pass_at_1_rate', 0),
                'passed': webvoyager_results.get('successful_questions', 0)
            }
        }
    }
    
    return aggregate


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation router for consolidated benchmark')
    parser.add_argument('--questions_file', type=str, required=True, 
                        help='Path to consolidated questions JSON file (id + question only)')
    parser.add_argument('--browsecomp_ground_truth', type=str, required=True,
                        help='Path to BrowseComp ground truth file (id + question + answer)')
    parser.add_argument('--rubrics_file', type=str, required=True, 
                        help='Path to Mind2Web-2 rubrics JSON file')
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='Directory containing result_X.json files')
    parser.add_argument('--trajectory_dir', type=str, required=True, 
                        help='Directory containing trajectory_X.jsonl files')
    parser.add_argument('--api_key', type=str, required=True, 
                        help='OpenAI API key')
    parser.add_argument('--output_dir', type=str, default='unified_evaluation_outputs', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("UNIFIED EVALUATION ROUTER (PARALLELIZED)")
    print("="*60)
    print(f"Consolidated questions: {args.questions_file}")
    print(f"BrowseComp ground truth: {args.browsecomp_ground_truth}")
    print(f"Mind2Web-2 rubrics: {args.rubrics_file}")
    print(f"Results directory: {args.result_dir}")
    print(f"Trajectory directory: {args.trajectory_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load and route questions
    question_counts = load_questions(args.questions_file)
    
    print(f"Question distribution:")
    print(f"  BrowseComp (IDs 1-124): {len(question_counts['browsecomp'])} questions")
    print(f"  Mind2Web-2 (IDs 125-134): {len(question_counts['mind2web2'])} questions")
    print(f"  WebVoyager (IDs 135-199): {len(question_counts['webvoyager'])} questions")
    print(f"  Total: {sum(len(v) for v in question_counts.values())} questions\n")
    
    # Run individual evaluations in parallel
    results = {}
    futures = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        if question_counts['browsecomp']:
            futures['browsecomp'] = executor.submit(
                run_browsecomp_eval,
                args.browsecomp_ground_truth,
                args.result_dir,
                args.api_key,
                args.output_dir
            )
        
        if question_counts['mind2web2']:
            futures['mind2web2'] = executor.submit(
                run_mind2web2_eval,
                args.questions_file,
                args.trajectory_dir,
                args.rubrics_file,
                args.api_key,
                args.output_dir,
                question_counts['mind2web2']
            )
        
        if question_counts['webvoyager']:
            futures['webvoyager'] = executor.submit(
                run_webvoyager_eval,
                args.questions_file,
                args.result_dir,
                args.api_key,
                args.output_dir,
                question_counts['webvoyager']
            )
        
        # Collect results as they complete
        for benchmark, future in futures.items():
            results[benchmark] = future.result()
    
    # Aggregate results
    aggregate = aggregate_results(
        results.get('browsecomp', {}),
        results.get('mind2web2', {}),
        results.get('webvoyager', {}),
        question_counts
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL AGGREGATE RESULTS")
    print("="*60)
    print(f"Overall Pass Rate: {aggregate['overall_pass_rate']:.3f} ({aggregate['overall_pass_rate']*100:.1f}%)")
    print(f"Total Questions: {aggregate['total_questions']}\n")
    
    print("By Benchmark:")
    for benchmark, stats in aggregate['by_benchmark'].items():
        print(f"  {benchmark.upper()}:")
        print(f"    Questions: {stats['count']}")
        print(f"    Passed: {stats['passed']}")
        print(f"    Pass Rate: {stats['pass_rate']:.3f} ({stats['pass_rate']*100:.1f}%)\n")
    
    # Save aggregate results
    output_file = os.path.join(args.output_dir, 'unified_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'aggregate': aggregate,
            'individual_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Unified results saved to: {output_file}")
    print("✅ Unified evaluation complete!")


if __name__ == "__main__":
    main()