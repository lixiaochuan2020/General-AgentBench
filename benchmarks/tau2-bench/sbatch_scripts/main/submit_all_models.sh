#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    # # claude-haiku-4.5
    # claude-haiku-4.5/batch_run_claude_haiku_4.5_airline.sh
    # claude-haiku-4.5/batch_run_claude_haiku_4.5_retail.sh
    # claude-haiku-4.5/batch_run_claude_haiku_4.5_telecom.sh

    # # deepseek-r1
    # deepseek-r1/batch_run_deepseek_r1_airline.sh
    # deepseek-r1/batch_run_deepseek_r1_retail.sh
    # deepseek-r1/batch_run_deepseek_r1_telecom.sh

    # claude-sonnet-4.5
    # claude-sonnet-4.5/batch_run_claude_sonnet_4.5_airline.sh
    # claude-sonnet-4.5/batch_run_claude_sonnet_4.5_retail.sh
    # claude-sonnet-4.5/batch_run_claude_sonnet_4.5_telecom.sh

    # gemini_2.5_pro
    gemini_2.5_pro/batch_run_gemini_2.5_pro_airline.sh
    gemini_2.5_pro/batch_run_gemini_2.5_pro_retail.sh
    gemini_2.5_pro/batch_run_gemini_2.5_pro_telecom.sh

    # # gpt-5
    # gpt-5/batch_run_gpt_5_airline.sh
    # gpt-5/batch_run_gpt_5_retail.sh
    # gpt-5/batch_run_gpt_5_telecom.sh
)

for SCRIPT in "${SCRIPTS[@]}"; do
    sbatch "${SCRIPT_DIR}/${SCRIPT}"
    sleep 0.5
done


