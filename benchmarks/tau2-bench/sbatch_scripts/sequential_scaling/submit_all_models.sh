#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    # OSS scripts
    # "oss/batch_run_4k_gpt_airline.sh"
    # "oss/batch_run_4k_gpt_retail.sh"
    # "oss/batch_run_4k_gpt_telecom.sh"
    # "oss/batch_run_8k_gpt_airline.sh"
    # "oss/batch_run_8k_gpt_retail.sh"
    # "oss/batch_run_8k_gpt_telecom.sh"
    # "oss/batch_run_12k_gpt_airline.sh"
    # "oss/batch_run_12k_gpt_retail.sh"
    # "oss/batch_run_12k_gpt_telecom.sh"
    # "oss/batch_run_16k_gpt_airline.sh"
    # "oss/batch_run_16k_gpt_retail.sh"
    # "oss/batch_run_16k_gpt_telecom.sh"
    # "oss/batch_run_22k_gpt_airline.sh"
    # "oss/batch_run_22k_gpt_retail.sh"
    # "oss/batch_run_22k_gpt_telecom.sh"
    # "oss/batch_run_32k_gpt_airline.sh"
    # "oss/batch_run_32k_gpt_retail.sh"
    # "oss/batch_run_32k_gpt_telecom.sh"
    # "oss/batch_run_64k_gpt_airline.sh"
    # "oss/batch_run_64k_gpt_retail.sh"
    # "oss/batch_run_64k_gpt_telecom.sh"
    
    # Qwen scripts
    # "qwen/batch_run_4k_qwen_airline.sh"
    # "qwen/batch_run_4k_qwen_retail.sh"
    # "qwen/batch_run_4k_qwen_telecom.sh"
    # "qwen/batch_run_8k_qwen_airline.sh"
    # "qwen/batch_run_8k_qwen_retail.sh"
    # "qwen/batch_run_8k_qwen_telecom.sh"
    # "qwen/batch_run_12k_qwen_airline.sh"
    # "qwen/batch_run_12k_qwen_retail.sh"
    # "qwen/batch_run_12k_qwen_telecom.sh"
    # "qwen/batch_run_16k_qwen_airline.sh"
    # "qwen/batch_run_16k_qwen_retail.sh"
    # "qwen/batch_run_16k_qwen_telecom.sh"
    # "qwen/batch_run_22k_qwen_airline.sh"
    # "qwen/batch_run_22k_qwen_retail.sh"
    # "qwen/batch_run_22k_qwen_telecom.sh"
    # "qwen/batch_run_32k_qwen_airline.sh"
    # "qwen/batch_run_32k_qwen_retail.sh"
    # "qwen/batch_run_32k_qwen_telecom.sh"
    # "qwen/batch_run_64k_qwen_airline.sh"
    # "qwen/batch_run_64k_qwen_retail.sh"
    # "qwen/batch_run_64k_qwen_telecom.sh"
    
    # Gemini scripts
    # "gemini/batch_run_4k_gemini_airline.sh"
    # "gemini/batch_run_4k_gemini_retail.sh"
    # "gemini/batch_run_4k_gemini_telecom.sh"
    # "gemini/batch_run_8k_gemini_airline.sh"
    # "gemini/batch_run_8k_gemini_retail.sh"
    # "gemini/batch_run_8k_gemini_telecom.sh"
    # "gemini/batch_run_12k_gemini_airline.sh"
    # "gemini/batch_run_12k_gemini_retail.sh"
    # "gemini/batch_run_12k_gemini_telecom.sh"
    # "gemini/batch_run_16k_gemini_airline.sh"
    # "gemini/batch_run_16k_gemini_retail.sh"
    # "gemini/batch_run_16k_gemini_telecom.sh"
    # "gemini/batch_run_22k_gemini_airline.sh"
    # "gemini/batch_run_22k_gemini_retail.sh"
    # "gemini/batch_run_22k_gemini_telecom.sh"
    # "gemini/batch_run_32k_gemini_airline.sh"
    # "gemini/batch_run_32k_gemini_retail.sh"
    # "gemini/batch_run_32k_gemini_telecom.sh"
    # "gemini/batch_run_64k_gemini_airline.sh"
    # "gemini/batch_run_64k_gemini_retail.sh"
    # "gemini/batch_run_64k_gemini_telecom.sh"
    
    # DeepSeek scripts
    # "deepseek/batch_run_4k_deepseek_airline.sh"
    # "deepseek/batch_run_4k_deepseek_retail.sh"
    # "deepseek/batch_run_4k_deepseek_telecom.sh"
    # "deepseek/batch_run_8k_deepseek_airline.sh"
    # "deepseek/batch_run_8k_deepseek_retail.sh"
    # "deepseek/batch_run_8k_deepseek_telecom.sh"
    # "deepseek/batch_run_12k_deepseek_airline.sh"
    # "deepseek/batch_run_12k_deepseek_retail.sh"
    # "deepseek/batch_run_12k_deepseek_telecom.sh"
    # "deepseek/batch_run_16k_deepseek_airline.sh"
    # "deepseek/batch_run_16k_deepseek_retail.sh"
    # "deepseek/batch_run_16k_deepseek_telecom.sh"
    # "deepseek/batch_run_22k_deepseek_airline.sh"
    # "deepseek/batch_run_22k_deepseek_retail.sh"
    # "deepseek/batch_run_22k_deepseek_telecom.sh"
    # "deepseek/batch_run_32k_deepseek_airline.sh"
    # "deepseek/batch_run_32k_deepseek_retail.sh"
    # "deepseek/batch_run_32k_deepseek_telecom.sh"
    
    # Next scripts
    # "next/batch_run_4k_next_airline.sh"
    # "next/batch_run_4k_next_retail.sh"
    # "next/batch_run_4k_next_telecom.sh"
    # "next/batch_run_8k_next_airline.sh"
    # "next/batch_run_8k_next_retail.sh"
    # "next/batch_run_8k_next_telecom.sh"
    # "next/batch_run_12k_next_airline.sh"
    # "next/batch_run_12k_next_retail.sh"
    # "next/batch_run_12k_next_telecom.sh"
    # "next/batch_run_16k_next_airline.sh"
    # "next/batch_run_16k_next_retail.sh"
    # "next/batch_run_16k_next_telecom.sh"
    "next/batch_run_22k_next_airline.sh"
    "next/batch_run_22k_next_retail.sh"
    "next/batch_run_22k_next_telecom.sh"
    "next/batch_run_32k_next_airline.sh"
    "next/batch_run_32k_next_retail.sh"
    "next/batch_run_32k_next_telecom.sh"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    sbatch "${SCRIPT_DIR}/${SCRIPT}"
    sleep 0.5
done
# tau2_deepseek_airline_16k_5804565.err
# tau2_deepseek_airline_22k_5804539.err
# tau2_deepseek_airline_32k_5804542.err
# tau2_deepseek_airline_8k_5804559.err
# tau2_deepseek_retail_12k_5804563.err
# tau2_deepseek_retail_16k_5804566.err
# tau2_deepseek_retail_22k_5804540.err
# tau2_deepseek_retail_4k_5804557.err
# tau2_deepseek_retail_8k_5804560.err
# tau2_deepseek_telecom_16k_5804567.err
# tau2_deepseek_telecom_8k_5804561.err
# tau2_qwen_retail_8k_5792020.err





# 5806222        cpu                tau2_deepseek_airline_32k   your name PD       0:00     1 (QOSMaxJobsPerUserLi
# 5806221        cpu                 tau2_deepseek_retail_22k   your name PD       0:00     1 (QOSMaxJobsPerUserLi
# 5806220        cpu                tau2_deepseek_airline_22k   your name PD       0:00     1 (QOSMaxJobsPerUserLi
# 5804541        cpu                tau2_deepseek_telecom_22k   your name  R    6:57:46     1          your node namev
# 5804543        cpu                 tau2_deepseek_retail_32k   your name  R    6:57:46     1          your node namev
# 5804544        cpu                tau2_deepseek_telecom_32k   your name  R    6:57:46     1          your node namex
# 5806211        cpu                      tau2_qwen_retail_8k   your name  R       0:26     1          your node namex
# 5806189        cpu        8trials_deepseek_airline_trial5_6   your name  R       4:28     1          your node namev
# 5806179        cpu         8trials_deepseek_retail_trial1_2   your name  R       5:28     1          your node namen
# 5806180        cpu         8trials_deepseek_retail_trial3_4   your name  R       5:28     1          your node namev
# 5806181        cpu         8trials_deepseek_retail_trial5_6   your name  R       5:28     1          your node namew
# 5806182        cpu         8trials_deepseek_retail_trial7_8   your name  R       5:28     1          your node namex
# 5806183        cpu        8trials_deepseek_telecom_trial1_2   your name  R       5:28     1          your node namev
# 5804184      debug                                     bash   your name  R    8:20:43     1          your node namep
# 5806212    general                  tau2_deepseek_retail_4k   your name  R       0:26     1          your node nameo
# 5806213    general                 tau2_deepseek_airline_8k   your name  R       0:26     1          your node namep
# 5806214    general                  tau2_deepseek_retail_8k   your name  R       0:26     1          your node namep
# 5806215    general                 tau2_deepseek_telecom_8k   your name  R       0:26     1          your node nameq
# 5806216    general                 tau2_deepseek_retail_12k   your name  R       0:26     1          your node names
# 5806217    general                tau2_deepseek_airline_16k   your name  R       0:26     1          your node namet
# 5806218    general                 tau2_deepseek_retail_16k   your name  R       0:26     1          your node namen
# 5806219    general                tau2_deepseek_telecom_16k   your name  R       0:26     1          your node namen
