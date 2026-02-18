Follow `./mcp-bench/README.md` for environment setup & MCP API key generation. 

Update configurations in `./mcp-bench/.env` and `./mcp-bench/mcp_servers/api_key`

Quick Start (`run.sh`):

```bash
source .env
python run_benchmark.py \
    --models bedrock/us.deepseek.r1-v1:0 \
    --judge-model bedrock/us.deepseek.r1-v1:0 \
    --judge-provider litellm \  # fixed for bedrock
    --tasks-file tasks/mcpbench_tasks_test_single.json
```

---

For $\tau^2$-Bench: 
airlines(50 samples):
- input: 10728.6 tokens
- output: 4429 tokens
retail(114 samples):
- input: 14418.70 tokens
- output: 6974.3 tokens
telecom(2285 samples):
- input: 67781.5 tokens
- output: 13198.1 tokens


For MCP-Bench(104 samples):
- input: 112703 tokens
- output: 9079 tokens


Prices for models:
- Gemini-2.5-Flash
    - input: $0.0003 /1k token
    - output: $0.0025 /1k token
- Gemini-2.5-Pro
    - input: $0.00125 /1k token
    - output:$0.01 /1k token
- GPT-5
    - input: $0.00125 /1k token
    - output:$0.01 /1k token
- Claude-Sonnet 4
    - input: $0.003 /1k token
    - output: $0.015 /1k token
- Claude-Opus 4.1
    - input: $0.015 /1k token
    - output:$0.075 /1k token
- Claude-Sonnet 4.5
    - input: $0.0033 /1k token
    - output:$0.0165 /1k token
- Deepseek-R1: 
    - input: $0.00135/1k token
    - output: $0.0054/1k token
- Deepseek V3.2
    - input: $0.00028 /1k token
    - output:$0.00042 /1k token
- Llama-3.1-405B
    - input: $0.0024 /1k token
    - output:$0.0024 /1k token
- Llama-3.3-70B
    - input: $0.00072 /1k token
    - output:$0.00072 /1k token
- Qwen3-235B
    - input: $0.00022 /1k token
    - output:$0.00088 /1k token
- Qwen3-Next
    - input: 0.00015 /1k token
    - output: 0.00150 /1k token