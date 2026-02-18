#!/bin/bash
#
#SBATCH --job-name=mathhay_claude_sonnet_4_5_3s3d
#SBATCH --output=/home/apaladug/MathHay/logs/claude_sonnet_4_5_3s3d_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/claude_sonnet_4_5_3s3d_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# MathHay Claude Sonnet 4.5 Inference on 3s3d Dataset
# Model: us.anthropic.claude-sonnet-4-5-20250929-v1:0 via AWS Bedrock
# Pricing: $3.00/1M input, $15.00/1M output

echo "========================================="
echo "MathHay Claude Sonnet 4.5 Inference on 3s3d"
echo "========================================="
echo "Model: us.anthropic.claude-sonnet-4-5-20250929-v1:0"
echo "Provider: AWS Bedrock (Inference Profile)"
echo "Region: us-east-1"
echo "Task: 3s3d (3-step, 3-doc)"
echo "Questions: 75"
echo "Context: 128k tokens"
echo "========================================="
echo ""

# Create log directory if it doesn't exist
mkdir -p /home/apaladug/MathHay/logs

# Activate conda environment
echo "Activating conda environment: capstone_env"
source /home/apaladug/miniconda3/etc/profile.d/conda.sh
conda activate capstone_env

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

echo "AWS credentials: Configured"
echo ""

# Change to MathHay directory
cd /home/apaladug/MathHay

# Run evaluation
echo "Starting Claude Sonnet 4.5 inference..."
python -u run_multimodel_evaluation.py \
    --models claude-sonnet-4.5 \
    --tasks 3s3d \
    --haystack_len 128000 \
    --placement middle \
    --input_dir ./outputs/data/March-2024-to-September-2024/ \
    --output_dir ./outputs/results/March-2024-to-September-2024/

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Claude Sonnet 4.5 Inference Complete!"
    echo "========================================="
else
    echo "ERROR: Inference failed with exit code $?"
    exit 1
fi

