import os
import json
import argparse


def load_results_from_dirs(result_dirs):
    """Load all results from multiple directories"""
    all_runs = []
    
    for dir_path in result_dirs:
        run_results = {}
        
        # Load each benchmark's results
        browsecomp_path = os.path.join(dir_path, 'browsecomp_results.json')
        mind2web2_path = os.path.join(dir_path, 'mind2web2_results.json')
        webvoyager_path = os.path.join(dir_path, 'webvoyager_results.json')
        
        if os.path.exists(browsecomp_path):
            with open(browsecomp_path, 'r') as f:
                run_results['browsecomp'] = json.load(f)
        
        if os.path.exists(mind2web2_path):
            with open(mind2web2_path, 'r') as f:
                run_results['mind2web2'] = json.load(f)
        
        if os.path.exists(webvoyager_path):
            with open(webvoyager_path, 'r') as f:
                run_results['webvoyager'] = json.load(f)
        
        all_runs.append(run_results)
    
    return all_runs


def get_valid_question_ids(benchmark):
    """Get valid question ID ranges for each benchmark"""
    if benchmark == 'browsecomp':
        return set(range(1, 125))  # 1-124
    elif benchmark == 'mind2web2':
        return set(range(125, 135))  # 125-134
    elif benchmark == 'webvoyager':
        return set(range(135, 200))  # 135-199
    return set()


def aggregate_by_question(all_runs, benchmark):
    """Aggregate results by question ID for a benchmark"""
    valid_ids = get_valid_question_ids(benchmark)
    question_results = {}
    
    for run in all_runs:
        if benchmark not in run:
            continue
        
        detailed = run[benchmark].get('detailed_results', {})
        for qid, result in detailed.items():
            # Convert qid to int and filter by valid range
            qid_int = int(qid)
            if qid_int not in valid_ids:
                continue
                
            if qid not in question_results:
                question_results[qid] = []
            question_results[qid].append(result['passed'])
    
    return question_results


def calculate_pass_at_k(question_results, k):
    """Calculate pass@k rate"""
    total = len(question_results)
    if total == 0:
        return 0.0, 0
    
    passed = 0
    for qid, attempts in question_results.items():
        # Check if any of the first k attempts passed
        if any(attempts[:k]):
            passed += 1
    
    return passed / total, passed


def main():
    parser = argparse.ArgumentParser(description='Calculate Pass@K metrics from multiple runs')
    parser.add_argument('--result_dirs', nargs='+', required=True,
                        help='List of result directories in order')
    parser.add_argument('--output_file', default='pass_at_k_results.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PASS@K EVALUATION")
    print("="*60)
    print(f"Loading results from {len(args.result_dirs)} directories\n")
    
    # Load all results
    all_runs = load_results_from_dirs(args.result_dirs)
    
    # Calculate pass@k for each benchmark
    benchmarks = ['browsecomp', 'mind2web2', 'webvoyager']
    k_values = [1, 2, 4, 8]
    
    results = {}
    
    for benchmark in benchmarks:
        print(f"\nProcessing {benchmark.upper()}...")
        question_results = aggregate_by_question(all_runs, benchmark)
        print(f"  Total questions: {len(question_results)}")
        
        results[benchmark] = {
            'total_questions': len(question_results),
            'pass_at_k': {}
        }
        
        for k in k_values:
            if k > len(args.result_dirs):
                continue
            rate, passed = calculate_pass_at_k(question_results, k)
            results[benchmark]['pass_at_k'][f'pass@{k}'] = {
                'rate': rate,
                'passed': passed
            }
            print(f"  Pass@{k}: {rate:.3f} ({rate*100:.1f}%) - {passed}/{len(question_results)} questions")
    
    # Calculate consolidated metrics
    print("\n" + "="*60)
    print("CONSOLIDATED RESULTS")
    print("="*60)
    
    consolidated = {}
    for k in k_values:
        if k > len(args.result_dirs):
            continue
        
        total_questions = sum(results[b]['total_questions'] for b in benchmarks if b in results)
        total_passed = sum(results[b]['pass_at_k'][f'pass@{k}']['passed'] for b in benchmarks if b in results)
        
        overall_rate = total_passed / total_questions if total_questions > 0 else 0
        
        consolidated[f'pass@{k}'] = {
            'rate': overall_rate,
            'passed': total_passed,
            'total': total_questions
        }
        
        print(f"\nPass@{k}: {overall_rate:.3f} ({overall_rate*100:.1f}%)")
        print(f"  Total: {total_passed}/{total_questions} questions")
    
    # Save results
    output = {
        'num_runs': len(args.result_dirs),
        'by_benchmark': results,
        'consolidated': consolidated
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📊 Results saved to: {args.output_file}")
    print("✅ Pass@K evaluation complete!")


if __name__ == "__main__":
    main()