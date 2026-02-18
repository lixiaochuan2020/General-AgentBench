#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    # "deepseek/batch_run_8trials_deepseek_airline_trial1_2.sh"
    # "deepseek/batch_run_8trials_deepseek_airline_trial3_4.sh"
    # "deepseek/batch_run_8trials_deepseek_airline_trial5_6.sh"
    # "deepseek/batch_run_8trials_deepseek_airline_trial7_8.sh"
    # "deepseek/batch_run_8trials_deepseek_retail_trial1_2.sh"
    # "deepseek/batch_run_8trials_deepseek_retail_trial3_4.sh"
    # "deepseek/batch_run_8trials_deepseek_retail_trial5_6.sh"
    # "deepseek/batch_run_8trials_deepseek_retail_trial7_8.sh"
    # "deepseek/batch_run_8trials_deepseek_telecom_trial1_2.sh" #
    # "deepseek/batch_run_8trials_deepseek_telecom_trial3_4.sh"
    # "deepseek/batch_run_8trials_deepseek_telecom_trial5_6.sh"
    # "deepseek/batch_run_8trials_deepseek_telecom_trial7_8.sh"

    "next/batch_run_8trials_next_airline_trial1_2.sh"
    "next/batch_run_8trials_next_airline_trial3_4.sh"
    "next/batch_run_8trials_next_airline_trial5_6.sh"
    "next/batch_run_8trials_next_airline_trial7_8.sh"  #
    "next/batch_run_8trials_next_retail_trial1_2.sh"
    "next/batch_run_8trials_next_retail_trial3_4.sh"
    "next/batch_run_8trials_next_retail_trial5_6.sh"
    "next/batch_run_8trials_next_retail_trial7_8.sh"  #
    "next/batch_run_8trials_next_telecom_trial1_2.sh"
    "next/batch_run_8trials_next_telecom_trial3_4.sh"
    "next/batch_run_8trials_next_telecom_trial5_6.sh"
    "next/batch_run_8trials_next_telecom_trial7_8.sh"  #
)

for SCRIPT in "${SCRIPTS[@]}"; do
    sbatch "${SCRIPT_DIR}/${SCRIPT}"
    sleep 0.5
done

