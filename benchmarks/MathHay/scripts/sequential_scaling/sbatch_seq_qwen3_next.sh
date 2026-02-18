#!/bin/bash
#
#SBATCH --job-name=seqscale_qwen3_next
#SBATCH --output=/home/apaladug/MathHay/logs/seqscale_qwen3_next_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/seqscale_qwen3_next_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 2-23:55:00

# Sequential Scaling Evaluation - Qwen3-Next
# Evaluates across context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K

echo "========================================="
echo "Sequential Scaling: Qwen3-Next"
echo "========================================="
echo "Model: Qwen/Qwen3-Next-80B-A3B-Instruct"
echo "Provider: HuggingFace (Together)"
echo "Context lengths: 4K, 8K, 12K, 16K, 22K, 32K, 64K, 128K"
echo "Task: 3s3d (75 questions per length)"
echo "========================================="
echo ""

# Source API keys from secure location
if [ -f ~/.mathhay_keys ]; then
    source ~/.mathhay_keys
fi

# Verify HuggingFace API key is set
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "ERROR: HUGGINGFACE_API_KEY environment variable is not set!"
    exit 1
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
echo "Starting Qwen3-Next sequential scaling evaluation..."
python -u run_context_length_evaluation.py \
    --models qwen3-next \
    --context-lengths 4000 8000 12000 16000 22000 32000 64000 128000 \
    --placement middle \
    --output-dir outputs/results/context-length-evaluation

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Qwen3-Next Sequential Scaling - Complete!"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi

