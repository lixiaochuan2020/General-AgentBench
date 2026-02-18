#!/bin/bash
#
#SBATCH --job-name=passatk_gpt_oss
#SBATCH --output=/home/apaladug/MathHay/logs/passatk_gpt_oss_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/passatk_gpt_oss_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# Pass@K (Parallel Scaling) Evaluation - GPT-OSS 120B
# 75 questions × 8 attempts = 600 API calls
# Pricing: $0.15/1M input, $0.60/1M output

echo "========================================="
echo "Pass@K Evaluation: GPT-OSS 120B"
echo "========================================="
echo "Task: 3s3d (75 questions)"
echo "Attempts per question: 8"
echo "Context: 128K tokens"
echo "========================================="
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

# Run evaluation
echo "Starting GPT-OSS 120B evaluation..."
python -u run_context_length_evaluation.py \
    --models gpt-oss-120b \
    --context-lengths 128000 \
    --placement middle \
    --pass-at-k \
    --output-dir outputs/results/context-length-evaluation

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "GPT-OSS 120B - Complete!"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi
