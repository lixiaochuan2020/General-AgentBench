#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    # "batch_run_16trials_qwen.sh"
    # "batch_run_16trials_gpt.sh"
    # "batch_run_16trials_gemini.sh"
    # "batch_run_16trials_deepseek_v3.2.sh"
    "deepseek3.2/batch_run_16trials_deepseek_v3.2_1server.sh"
    "deepseek3.2/batch_run_16trials_deepseek_v3.2_2server.sh"
    "deepseek3.2/batch_run_16trials_deepseek_v3.2_3server.sh"
    "next/batch_run_16trials_next_1server.sh"
    "next/batch_run_16trials_next_2server.sh"
    "next/batch_run_16trials_next_3server.sh"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    sbatch "${SCRIPT_DIR}/${SCRIPT}"
    sleep 0.5
done