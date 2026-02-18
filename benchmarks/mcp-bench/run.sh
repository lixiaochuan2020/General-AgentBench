
# .env configuration
set -a
source .env
# export UV_NO_BUILD=1
# export UV_PYTHON_PREFERENCE=system
# export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
set +a

# Create logs directory if it doesn't exist
mkdir -p logs_test
mkdir -p ./caches/cache_sequential

python run_benchmark.py \
    --models bedrock/openai.gpt-oss-120b-1:0 \
    --judge-model bedrock/openai.gpt-oss-120b-1:0 \
    --judge-provider litellm \
    --tasks-file tasks/test/2server_test.json \
    --output results/test_sequential.jsonl \
    --output-log results/log_test_sequential_stop.json \
    --cache-dir ./caches/cache_sequential_1 \
    --temperature 0.7 \
    --context-budget 24000
    2>&1 | tee logs_test/run_sequential.log

# ["Qwen3-235B"]="bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
# ["OpenAI-oss-120B"]="bedrock/openai.gpt-oss-120b-1:0"
# ["Gemini-2.5-Flash"]="gemini/gemini-2.5-flash"
# ["Llama-3.3-70B"]="bedrock/us.meta.llama3-3-70b-instruct-v1:0"
# ["Claude-Haiku-4.5"]="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
# ["DeepSeek-R1"]="bedrock/us.deepseek.r1-v1:0"
# ["DeepSeek-V3.2"]="huggingface/deepseek-ai/DeepSeek-V3.2-Exp:novita"