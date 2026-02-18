#!/bin/bash
#
#SBATCH --job-name=selfcorrect_qwen3_next
#SBATCH --output=/home/apaladug/MathHay/logs/selfcorrect_qwen3_next_8retry_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/selfcorrect_qwen3_next_8retry_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# Self-Correcting (Retry) Evaluation - Qwen3 Next 80B
# 75 questions with up to 8 retries (max 9 attempts per question)

echo "========================================="
echo "Self-Correcting: Qwen3 Next 80B"
echo "========================================="
echo "Task: 3s3d (75 questions)"
echo "Max Retries: 8 (up to 9 attempts per question)"
echo "========================================="
echo ""
echo "Start time: $(date)"
echo ""

# Source API keys from secure location
if [ -f ~/.mathhay_keys ]; then
    source ~/.mathhay_keys
fi

# Verify HuggingFace API key is set
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "ERROR: HUGGINGFACE_API_KEY environment variable is not set!"
    echo "Set it in ~/.mathhay_keys or export it before running"
    exit 1
fi

echo "API keys: Configured"
echo ""

# Activate conda environment
source /home/apaladug/miniconda3/etc/profile.d/conda.sh
conda activate capstone_env

# Change to MathHay directory
cd /home/apaladug/MathHay

# Run evaluation
echo "Starting Qwen3 Next 80B self-correcting evaluation..."
python -u agentic_multiturn/run_multiturn_agentic_mathhay.py \
    --models qwen3-next \
    --tasks 3s3d \
    --max_retries 8 \
    --input_dir ./outputs/data/March-2024-to-September-2024/ \
    --output_dir ./outputs/results/March-2024-to-September-2024/

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "Qwen3 Next 80B - Complete!"
    echo "========================================="
else
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    exit 1
fi
