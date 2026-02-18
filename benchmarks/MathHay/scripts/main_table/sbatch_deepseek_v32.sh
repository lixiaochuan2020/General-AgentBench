#!/bin/bash
#
#SBATCH --job-name=mathhay_deepseek_v32_3s3d
#SBATCH --output=/home/apaladug/MathHay/logs/deepseek_v32_3s3d_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/deepseek_v32_3s3d_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# MathHay DeepSeek-V3.2 Inference on 3s3d Dataset
# Model: deepseek-ai/DeepSeek-V3.2-Exp via HuggingFace (Novita)
# Pricing: $0.28/1M input, $0.42/1M output

echo "========================================="
echo "MathHay DeepSeek-V3.2 Inference on 3s3d"
echo "========================================="
echo "Model: deepseek-ai/DeepSeek-V3.2-Exp"
echo "Provider: HuggingFace (Novita)"
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

# Verify HuggingFace API key is set
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "ERROR: HUGGINGFACE_API_KEY environment variable is not set!"
    echo "Set it in ~/.mathhay_keys or export it before running"
    exit 1
fi

echo "HuggingFace API key: Configured"
echo ""

# Change to MathHay directory
cd /home/apaladug/MathHay

# Run evaluation
echo "Starting DeepSeek-V3.2 inference..."
python -u run_multimodel_evaluation.py \
    --models deepseek-v3.2 \
    --tasks 3s3d \
    --haystack_len 128000 \
    --placement middle \
    --input_dir ./outputs/data/March-2024-to-September-2024/ \
    --output_dir ./outputs/results/March-2024-to-September-2024/

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "DeepSeek-V3.2 Inference Complete!"
    echo "========================================="
else
    echo "ERROR: Inference failed with exit code $?"
    exit 1
fi

