#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    # "batch_run_gpt_5.sh"
    # "batch_run_gemini_2.5_pro.sh"
    "batch_run_claude_haiku_4.5.sh"
    "batch_run_claude_sonnet_4.5.sh"
    "batch_run_deepseek_r1.sh"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    sbatch "${SCRIPT_DIR}/${SCRIPT}"
    sleep 0.5
done