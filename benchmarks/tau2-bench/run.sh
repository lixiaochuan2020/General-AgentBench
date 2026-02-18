
tau2 run --domain airline \
    --task-set-name airline_lite \
    --agent-llm huggingface/deepseek-ai/DeepSeek-V3.2-Exp:novita \
    --user-llm bedrock/openai.gpt-oss-120b-1:0 \
    --num-trials 1 \
    --max-concurrency 1 \
    --user-llm-args '{"temperature": 0.0}' \
    --agent-llm-args '{"temperature": 0.7}' \
    --save-to results/deepseek_airline_32000_debug.json \
    --agent-token-budget 32000

