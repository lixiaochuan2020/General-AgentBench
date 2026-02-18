#!/bin/bash
#
#SBATCH --job-name=selfcorrect_qwen3_235b
#SBATCH --output=/home/apaladug/MathHay/logs/selfcorrect_qwen3_235b_8retry_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/selfcorrect_qwen3_235b_8retry_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# Self-Correcting (Retry) Evaluation - Qwen3 235B
# 75 questions with up to 8 retries (max 9 attempts per question)
# NOTE: May hit "request body too large" error with 128K context on Bedrock

echo "========================================="
echo "Self-Correcting: Qwen3 235B"
echo "========================================="
echo "Task: 3s3d (75 questions)"
echo "Max Retries: 8 (up to 9 attempts per question)"
echo "WARNING: May fail with Bedrock body size limit"
echo "========================================="
echo ""
echo "Start time: $(date)"
echo ""

# Source API keys from secure location
if [ -f ~/.mathhay_keys ]; then
    source ~/.mathhay_keys
fi

# Increase boto3 timeouts
export AWS_MAX_ATTEMPTS=10
export AWS_RETRY_MODE=adaptive

# Verify AWS credentials are set
if [ -z "$AWS_BEARER_TOKEN_BEDROCK" ] && [ -z "$AWS_PROFILE" ]; then
    echo "WARNING: Neither AWS_BEARER_TOKEN_BEDROCK nor AWS_PROFILE is set!"
    echo "Using default AWS profile: bedrock-account"
    export AWS_PROFILE=bedrock-account
fi

echo "API keys: Configured"
echo ""

# Activate conda environment
source /home/apaladug/miniconda3/etc/profile.d/conda.sh
conda activate capstone_env

# Change to MathHay directory
cd /home/apaladug/MathHay

# Run evaluation
echo "Starting Qwen3 235B self-correcting evaluation..."
python -u agentic_multiturn/run_multiturn_agentic_mathhay.py \
    --models qwen3-235b \
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
    echo "Qwen3 235B - Complete!"
    echo "========================================="
else
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    echo "Check logs for 'request body too large' errors"
    exit 1
fi
