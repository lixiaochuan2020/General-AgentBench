#!/bin/bash
#
#SBATCH --job-name=selfcorrect_test
#SBATCH --output=/home/apaladug/MathHay/logs/selfcorrect_test_%j.out
#SBATCH --error=/home/apaladug/MathHay/logs/selfcorrect_test_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# Quick test of self-correcting evaluation with one cheap model
# Test with qwen3-32b (cheapest model) and 2 retries

echo "========================================="
echo "Self-Correcting Test - Single Model"
echo "========================================="
echo "Model: qwen3-32b"
echo "Task: 3s3d (75 questions)"
echo "Max Retries: 2"
echo "========================================="
echo ""
echo "Start time: $(date)"
echo ""

# Source API keys from secure location
if [ -f ~/.mathhay_keys ]; then
    source ~/.mathhay_keys
fi

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

# Run test evaluation with one model
echo "Starting test run with qwen3-32b..."
echo ""

python -u agentic_multiturn/run_multiturn_agentic_mathhay.py \
    --models qwen3-32b \
    --tasks 3s3d \
    --max_retries 2 \
    --input_dir ./outputs/data/March-2024-to-September-2024/ \
    --output_dir ./outputs/results/March-2024-to-September-2024/

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "Test Run - Complete!"
    echo "========================================="
    echo ""
    echo "Results saved to: ./outputs/results/March-2024-to-September-2024/"
    echo ""
    echo "If this looks good, submit the full evaluation with:"
    echo "  cd agentic_multiturn && ./submit_all_selfcorrect.sh"
else
    echo "ERROR: Test evaluation failed with exit code $EXIT_CODE"
    exit 1
fi
