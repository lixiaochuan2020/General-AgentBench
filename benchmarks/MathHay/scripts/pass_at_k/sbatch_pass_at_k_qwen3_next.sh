#!/bin/bash
#
#SBATCH --job-name=passatk_qwen3_next
#SBATCH --output=/home/apaladug/MathHay/logs/passatk_qwen3_next_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/passatk_qwen3_next_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# Pass@K (Parallel Scaling) Evaluation - Qwen3 Next 80B
# 75 questions × 8 attempts = 600 API calls

echo "========================================="
echo "Pass@K Evaluation: Qwen3 Next 80B"
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
echo "Starting Qwen3 Next 80B evaluation..."
python -u run_context_length_evaluation.py \
    --models qwen3-next \
    --context-lengths 128000 \
    --placement middle \
    --pass-at-k \
    --output-dir outputs/results/context-length-evaluation

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Qwen3 Next 80B - Complete!"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi
