#!/bin/bash
#
# Submit self-correcting (retry) evaluation for ALL 5 models
#
# Models for Retry Mode:
#   - gpt-oss-120B
#   - Qwen3-Next
#   - Qwen3-235B
#   - DeepSeek-V3.2
#   - Gemini-2.5-Flash
#

cd /home/apaladug/MathHay/agentic_multiturn

echo "========================================="
echo "Submitting All Self-Correcting Evaluations"
echo "========================================="
echo "Models: 5"
echo "  - Gemini-2.5-Flash"
echo "  - GPT-OSS-120B"
echo "  - Qwen3-Next"
echo "  - Qwen3-235B"
echo "  - DeepSeek-V3.2"
echo ""
echo "Task: 3s3d (75 questions)"
echo "Max Retries: 8 (up to 9 attempts per question)"
echo "========================================="
echo ""

# Submit each job
echo "Submitting jobs..."
echo ""

JOB1=$(sbatch sbatch_selfcorrect_gemini.sh | awk '{print $4}')
echo "Submitted Job $JOB1: Gemini 2.5 Flash"

JOB2=$(sbatch sbatch_selfcorrect_gpt_oss.sh | awk '{print $4}')
echo "Submitted Job $JOB2: GPT-OSS 120B"

JOB3=$(sbatch sbatch_selfcorrect_qwen3_next.sh | awk '{print $4}')
echo "Submitted Job $JOB3: Qwen3 Next 80B"

JOB4=$(sbatch sbatch_selfcorrect_qwen3_235b.sh | awk '{print $4}')
echo "Submitted Job $JOB4: Qwen3 235B"

JOB5=$(sbatch sbatch_selfcorrect_deepseek_v32.sh | awk '{print $4}')
echo "Submitted Job $JOB5: DeepSeek V3.2"

echo ""
echo "========================================="
echo "All 5 jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ../logs/selfcorrect_*_out.out"
echo ""
echo "Cancel all:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5"
echo ""
