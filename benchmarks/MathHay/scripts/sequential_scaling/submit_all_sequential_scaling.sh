#!/bin/bash
#
# Submit all 5 sequential scaling evaluation jobs
# Evaluates each model across context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K
#
# Models for Sequential Scaling:
#   - Qwen3-Next
#   - DeepSeek-V3.2
#   - gpt-oss-120B
#   - Qwen3-235B
#   - Gemini-2.5-Flash
#

cd /home/apaladug/MathHay/scripts/sequential_scaling

echo "========================================="
echo "Submitting Sequential Scaling Jobs"
echo "========================================="
echo "Models: 5"
echo "  - Qwen3-Next"
echo "  - DeepSeek-V3.2"
echo "  - GPT-OSS-120B"
echo "  - Qwen3-235B"
echo "  - Gemini-2.5-Flash"
echo ""
echo "Task: 3s3d"
echo "Context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K"
echo "Total: 75 questions × 8 lengths × 5 models = 3000 evaluations"
echo "========================================="
echo ""

# Submit each job
echo "Submitting jobs..."
echo ""

JOB1=$(sbatch sbatch_seq_qwen3_next.sh | awk '{print $4}')
echo "Submitted Job $JOB1: Qwen3-Next"

JOB2=$(sbatch sbatch_seq_deepseek_v32.sh | awk '{print $4}')
echo "Submitted Job $JOB2: DeepSeek-V3.2"

JOB3=$(sbatch sbatch_seq_gpt_oss_120b.sh | awk '{print $4}')
echo "Submitted Job $JOB3: GPT-OSS-120B"

JOB4=$(sbatch sbatch_seq_qwen3_235b.sh | awk '{print $4}')
echo "Submitted Job $JOB4: Qwen3-235B"

JOB5=$(sbatch sbatch_seq_gemini_flash.sh | awk '{print $4}')
echo "Submitted Job $JOB5: Gemini 2.5 Flash"

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
echo "  tail -f /home/apaladug/MathHay/logs/seqscale_*_out.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5"
echo ""

