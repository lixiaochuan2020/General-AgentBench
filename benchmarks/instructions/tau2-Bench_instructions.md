
## Quick Overview
$\tau^2$-bench implements a simulation framework for evaluating customer service agents across various domains (updated version of $\tau$-bench).

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance
- Optionally: A set of tools that the user simulator can use

Domains are:
- `mock`, 9 instances
- `airline`, 50 instances
- `retail`, 114 instances
- `telecom`, 2285 instances(full) + 114(medium) + 20(small)

You can check policy files in `./tau2-bench/data/tau2/domains/`.

## Environment setup
Follow instructions `README.md` in `tau2-bench/`.

Add package `boto3`:

```bash
pip install boto3
```

> Or you can check `./tau2-bench/requirements.txt` for package information.


## AWS Bedrock API Configuration

Update AWS Bedrock credentials in `./tau2-bench/.env` or setup Anthropic key/Openai keys in this file.

```bash
# AWS Bedrock credentials
AWS_ACCESS_KEY_ID=<KEY_ID>
AWS_SECRET_ACCESS_KEY=<ACCESS_KEY>
AWS_REGION_NAME=<REGION_NAME>
```

By default, we use `DeepSeek-R1` for user simulator, customer service agent and results evaluator via AWS Bedrock API. You can change default model setting in `./tau2-bench/src/tau2/config.py`.


## Quick Start

```bash
tau2 run --domain airline \  # [airline, retail, telecom]
--agent-llm bedrock/us.deepseek.r1-v1:0 \
--user-llm bedrock/us.deepseek.r1-v1:0 \
--num-trials 1 \
--num-tasks 10
```

Models:
- DeepSeek-R1: `bedrock/us.deepseek.r1-v1:0`
- Claude Sonnet 4.5: `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0`, please set argument `--max-concurrency` less than 2
- Llama 3.1 405B Instruct: `bedrock/us.meta.llama3-1-405b-instruct-v1:0`
- Llama 3.3 70B Instruct: `bedrock/us.meta.llama3-3-70b-instruct-v1:0`
- Qwen3 235B A22B 2507: `bedrock/qwen.qwen3-235b-a22b-2507-v1:0`


## Customer Service Agent Prompts
Modify agent(customer service) prompt at `../tau2-bench/src/tau2/agent/llm_agent.py`


## Test Experiments

Test experiments on airline subset, 10 examples:

- Deepseek-R1: 
	- cost: $0.0246 per conversation
	- runtime: avg.36s 
- Claude Sonnet 4.5
	- cost: $0.2838 per conversation
	- runtime: avg  71.3s 
- Llama 3.1 405B Instruct
	- $0.4235
	- 136s
- Llama 3.3 70B Instruct
	- $0.0243
	- 3.7s
	- not successful: transfer to human; error action(not allowed -> allowed)
- Qwen3 235B A22B 2507
	- $0.0493
	- 40.2s

For detailed info, please check `./tau2-bench/data/simulations/` or use `tau2 view` command.