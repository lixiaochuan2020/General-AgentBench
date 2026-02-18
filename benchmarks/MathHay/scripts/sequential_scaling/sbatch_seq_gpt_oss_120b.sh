#!/bin/bash
#
#SBATCH --job-name=seqscale_gpt_oss_120b
#SBATCH --output=/home/apaladug/MathHay/logs/seqscale_gpt_oss_120b_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/seqscale_gpt_oss_120b_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 2-23:55:00

# Sequential Scaling Evaluation - GPT-OSS-120B
# Evaluates across context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K
# Pricing: $0.15/1M input, $0.60/1M output

echo "========================================="
echo "Sequential Scaling: GPT-OSS-120B"
echo "========================================="
echo "Model: openai.gpt-oss-120b-1:0"
echo "Provider: AWS Bedrock"
echo "Region: us-east-1"
echo "Context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K"
echo "Task: 3s3d (75 questions per length)"
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

# Create log directory
mkdir -p /home/apaladug/MathHay/logs

# Activate conda environment
source /home/apaladug/miniconda3/etc/profile.d/conda.sh
conda activate capstone_env

# Change to MathHay directory
cd /home/apaladug/MathHay

# Run evaluation
echo "Starting GPT-OSS-120B sequential scaling evaluation..."
python -u run_context_length_evaluation.py \
    --models gpt-oss-120b \
    --context-lengths 4000 8000 12000 16000 22000 32000 64000 128000 \
    --placement middle \
    --output-dir outputs/results/context-length-evaluation

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "GPT-OSS-120B Sequential Scaling - Complete!"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi

