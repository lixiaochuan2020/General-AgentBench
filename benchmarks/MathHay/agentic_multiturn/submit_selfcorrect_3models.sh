#!/bin/bash
#
# Submit self-correcting evaluation for the 3 models that work with 128K context
# (Gemini, Qwen3-Next, DeepSeek-V3.2)
#
# NOTE: gpt-oss-120B and Qwen3-235B (Bedrock) will fail with "request body too large"
# error when using 128K context, so they're excluded for now.
#

cd /home/apaladug/MathHay/agentic_multiturn

echo "========================================="
echo "Submitting Self-Correcting Evaluation"
echo "========================================="
echo "Models: Gemini-2.5-Flash, Qwen3-Next, DeepSeek-V3.2"
echo "Task: 3s3d (75 questions)"
echo "Max Retries: 2 (up to 3 attempts per question)"
echo ""
echo "NOTE: GPT-OSS-120B and Qwen3-235B excluded due to"
echo "Bedrock request body size limit with 128K context"
echo "========================================="
echo ""

# Submit each job
echo "Submitting jobs..."
echo ""

JOB1=$(sbatch sbatch_selfcorrect_gemini.sh | awk '{print $4}')
echo "Submitted Job $JOB1: Gemini 2.5 Flash"

JOB2=$(sbatch sbatch_selfcorrect_qwen3_next.sh | awk '{print $4}')
echo "Submitted Job $JOB2: Qwen3 Next 80B"

JOB3=$(sbatch sbatch_selfcorrect_deepseek_v32.sh | awk '{print $4}')
echo "Submitted Job $JOB3: DeepSeek V3.2"

echo ""
echo "========================================="
echo "3 jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ../logs/selfcorrect_*_out.out"
echo ""
echo "Cancel all:"
echo "  scancel $JOB1 $JOB2 $JOB3"
echo ""



