#!/bin/bash
#
# Submit all 10 main table evaluation jobs for MathHay 3s3d
# Models are run in parallel if resources are available
#

cd /home/apaladug/MathHay/scripts/main_table

echo "========================================="
echo "Submitting Main Table Evaluation Jobs"
echo "========================================="
echo "Models: 10"
echo "  1. GPT-OSS-120B (AWS Bedrock)"
echo "  2. Qwen3-Next (HuggingFace)"
echo "  3. Qwen3-235B (AWS Bedrock)"
echo "  4. DeepSeek-V3.2 (HuggingFace)"
echo "  5. Gemini-2.5-Flash (Google API)"
echo "  6. Gemini-2.5-Pro (Google API)"
echo "  7. DeepSeek-R1 (AWS Bedrock)"
echo "  8. Claude Sonnet 4.5 (AWS Bedrock)"
echo "  9. GPT-5 (OpenAI API)"
echo "  10. Claude Haiku 4.5 (AWS Bedrock)"
echo ""
echo "Task: 3s3d (75 questions, 128K context)"
echo "========================================="
echo ""

# Submit each job
echo "Submitting jobs..."
echo ""

JOB1=$(sbatch sbatch_gpt_oss_120b.sh | awk '{print $4}')
echo "Submitted Job $JOB1: GPT-OSS-120B"

JOB2=$(sbatch sbatch_qwen3_next.sh | awk '{print $4}')
echo "Submitted Job $JOB2: Qwen3-Next"

JOB3=$(sbatch sbatch_qwen3_235b.sh | awk '{print $4}')
echo "Submitted Job $JOB3: Qwen3-235B"

JOB4=$(sbatch sbatch_deepseek_v32.sh | awk '{print $4}')
echo "Submitted Job $JOB4: DeepSeek-V3.2"

JOB5=$(sbatch sbatch_gemini_flash.sh | awk '{print $4}')
echo "Submitted Job $JOB5: Gemini-2.5-Flash"

JOB6=$(sbatch sbatch_gemini_pro.sh | awk '{print $4}')
echo "Submitted Job $JOB6: Gemini-2.5-Pro"

JOB7=$(sbatch sbatch_deepseek_r1.sh | awk '{print $4}')
echo "Submitted Job $JOB7: DeepSeek-R1"

JOB8=$(sbatch sbatch_claude_sonnet_4_5.sh | awk '{print $4}')
echo "Submitted Job $JOB8: Claude Sonnet 4.5"

JOB9=$(sbatch sbatch_gpt5.sh | awk '{print $4}')
echo "Submitted Job $JOB9: GPT-5"

JOB10=$(sbatch sbatch_claude_haiku_4_5.sh | awk '{print $4}')
echo "Submitted Job $JOB10: Claude Haiku 4.5"

echo ""
echo "========================================="
echo "All 10 jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5, $JOB6, $JOB7, $JOB8, $JOB9, $JOB10"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f /home/apaladug/MathHay/logs/*_3s3d_out.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7 $JOB8 $JOB9 $JOB10"
echo ""

