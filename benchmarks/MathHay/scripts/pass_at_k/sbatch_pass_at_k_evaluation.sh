#!/bin/bash
#
#SBATCH --job-name=mathhay_pass_at_k
#SBATCH --output=/home/apaladug/MathHay/logs/pass_at_k_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/pass_at_k_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 3-23:55:00

# MathHay Pass@K (Parallel Scaling) Evaluation
# This script evaluates 5 models with pass@1,2,4,8 on 3s3d task (128K context)
# Models: gpt-oss-120B, Qwen3-Next, Qwen3-235B, DeepSeek-V3.2, Gemini-2.5-Flash
# 
# For each question, runs 8 attempts and calculates:
# - pass@1: accuracy with 1 attempt (standard accuracy)
# - pass@2: % of questions where at least 1 of 2 attempts is correct
# - pass@4: % of questions where at least 1 of 4 attempts is correct
# - pass@8: % of questions where at least 1 of 8 attempts is correct

echo "========================================="
echo "MathHay Pass@K Evaluation"
echo "========================================="
echo "Models: gpt-oss-120B, Qwen3-Next, Qwen3-235B, DeepSeek-V3.2, Gemini-2.5-Flash"
echo "Task: 3s3d (3-step, 3-doc, 75 questions)"
echo "Context: 128K tokens"
echo "Attempts per question: 8"
echo "Metrics: pass@1, pass@2, pass@4, pass@8"
echo "========================================="
echo ""

# Source API keys from secure location
if [ -f ~/.mathhay_keys ]; then
    source ~/.mathhay_keys
fi

# Create log directory if it doesn't exist
mkdir -p /home/apaladug/MathHay/logs

# Activate conda environment
echo "Activating conda environment: capstone_env"
source /home/apaladug/miniconda3/etc/profile.d/conda.sh
conda activate capstone_env

# Change to MathHay directory
cd /home/apaladug/MathHay

# Verify API keys are set
echo "Checking API keys..."
if [ ! -z "$AWS_BEARER_TOKEN_BEDROCK" ] || [ ! -z "$AWS_PROFILE" ]; then
    echo "  AWS Bedrock: Configured"
fi
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "  OpenAI API key: Configured"
fi
if [ ! -z "$HUGGINGFACE_API_KEY" ]; then
    echo "  HuggingFace API key: Configured"
fi
if [ ! -z "$GEMINI_API_KEY" ]; then
    echo "  Gemini API key: Configured"
fi
echo ""

# Run pass@K evaluation
echo "Starting pass@K evaluation..."
echo "This will run 8 attempts per question for 75 questions = 600 total API calls per model"
echo "Estimated time: 2-3 hours per model"
echo ""

python -u run_context_length_evaluation.py \
    --models gemini-2.5-flash gpt-oss-120b deepseek-v3.2 qwen3-235b qwen3-next \
    --context-lengths 128000 \
    --placement middle \
    --pass-at-k \
    --output-dir outputs/results/context-length-evaluation

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Pass@K Evaluation Complete!"
    echo "========================================="
    echo ""
    echo "Results saved to: outputs/results/context-length-evaluation/"
    echo ""
    echo "To view results:"
    echo "  cd /home/apaladug/MathHay/outputs/results/context-length-evaluation"
    echo "  ls -lh *_128k_passatk_*.json"
    echo ""
    echo "To extract pass@K metrics:"
    echo "  jq '.\"pass@2\".pass_at_k, .\"pass@4\".pass_at_k, .\"pass@8\".pass_at_k' *_passatk_*.json"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Evaluation failed with exit code $?"
    echo "========================================="
    exit 1
fi
