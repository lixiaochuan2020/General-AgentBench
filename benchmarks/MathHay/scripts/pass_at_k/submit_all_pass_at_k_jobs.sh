#!/bin/bash
#
# Submit all 5 pass@K (parallel scaling) evaluation jobs
# They will run in parallel if resources are available
#
# Models for Parallel Scaling:
#   - gpt-oss-120B
#   - Qwen3-Next
#   - Qwen3-235B
#   - DeepSeek-V3.2
#   - Gemini-2.5-Flash
#

cd /home/apaladug/MathHay/scripts/pass_at_k

echo "========================================="
echo "Submitting Pass@K (Parallel Scaling) Jobs"
echo "========================================="
echo "Models: 5"
echo "  - Gemini-2.5-Flash"
echo "  - GPT-OSS-120B"
echo "  - Qwen3-Next"
echo "  - Qwen3-235B"
echo "  - DeepSeek-V3.2"
echo ""
echo "Task: 3s3d (75 questions, 128K context)"
echo "Attempts per question: 8"
echo "Total API calls: 600 per model = 3000 total"
echo "========================================="
echo ""

# Submit each job
echo "Submitting jobs..."
echo ""

JOB1=$(sbatch sbatch_pass_at_k_gemini.sh | awk '{print $4}')
echo "Submitted Job $JOB1: Gemini 2.5 Flash"

JOB2=$(sbatch sbatch_pass_at_k_gpt_oss.sh | awk '{print $4}')
echo "Submitted Job $JOB2: GPT-OSS 120B"

JOB3=$(sbatch sbatch_pass_at_k_qwen3_next.sh | awk '{print $4}')
echo "Submitted Job $JOB3: Qwen3 Next 80B"

JOB4=$(sbatch sbatch_pass_at_k_qwen3_235b.sh | awk '{print $4}')
echo "Submitted Job $JOB4: Qwen3 235B"

JOB5=$(sbatch sbatch_pass_at_k_deepseek_v32.sh | awk '{print $4}')
echo "Submitted Job $JOB5: DeepSeek V3.2"

echo ""
echo "========================================="
echo "All 5 jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f /home/apaladug/MathHay/logs/passatk_*_out.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5"
echo ""
