#!/bin/bash
# Sequential Scaling Evaluation - Multiple Context Lengths
# Evaluates models across context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K
#
# Models for Sequential Scaling:
#   - Qwen3-Next
#   - DeepSeek-V3.2
#   - gpt-oss-120B
#   - Qwen3-235B
#   - Gemini-2.5-Flash

# Source API keys from secure location
if [ -f ~/.mathhay_keys ]; then
    source ~/.mathhay_keys
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate capstone_env

# Fallback to AWS profile if no API key
if [ -z "$AWS_BEARER_TOKEN_BEDROCK" ] && [ -z "$AWS_PROFILE" ]; then
    export AWS_PROFILE=bedrock-account
fi

# Change to MathHay directory
cd /home/apaladug/MathHay

echo "========================================="
echo "Sequential Scaling Evaluation"
echo "========================================="
echo "Models: qwen3-next, deepseek-v3.2, gpt-oss-120b, qwen3-235b, gemini-2.5-flash"
echo "Context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K"
echo "Task: 3s3d (75 questions per context length)"
echo "========================================="
echo ""

# Run evaluation
python run_context_length_evaluation.py \
    --models qwen3-next deepseek-v3.2 gpt-oss-120b qwen3-235b gemini-2.5-flash \
    --context-lengths 4000 8000 12000 16000 22000 32000 64000 128000 \
    --placement middle \
    --output-dir outputs/results/context-length-evaluation

echo ""
echo "========================================="
echo "Sequential Scaling Evaluation Complete!"
echo "========================================="
