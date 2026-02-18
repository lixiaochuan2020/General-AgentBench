#!/bin/bash
#
#SBATCH --job-name=mathhay_gemini_pro_3s3d
#SBATCH --output=/home/apaladug/MathHay/logs/gemini_pro_3s3d_out.out
#SBATCH --error=/home/apaladug/MathHay/logs/gemini_pro_3s3d_err.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-23:55:00

# MathHay Gemini 2.5 Pro Inference on 3s3d Dataset
# Model: gemini-2.5-pro via Google API
# Pricing: $1.25/1M input, $10.00/1M output

echo "========================================="
echo "MathHay Gemini 2.5 Pro Inference on 3s3d"
echo "========================================="
echo "Model: gemini-2.5-pro"
echo "Provider: Google API"
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

# Verify Gemini API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY environment variable is not set!"
    echo "Set it in ~/.mathhay_keys or export it before running"
    exit 1
fi

echo "Gemini API key: Configured"
echo ""

# Change to MathHay directory
cd /home/apaladug/MathHay

# Run evaluation
echo "Starting Gemini 2.5 Pro inference..."
python -u run_multimodel_evaluation.py \
    --models gemini-2.5-pro \
    --tasks 3s3d \
    --haystack_len 128000 \
    --placement middle \
    --input_dir ./outputs/data/March-2024-to-September-2024/ \
    --output_dir ./outputs/results/March-2024-to-September-2024/

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Gemini 2.5 Pro Inference Complete!"
    echo "========================================="
else
    echo "ERROR: Inference failed with exit code $?"
    exit 1
fi

