# SWE-Bench Instructions

> **Note**: Docker is not supported by Babel, so this assumes using AWS or a local computer.

## Overview

This guide provides instructions for setting up and running the SWE-Bench benchmark using the modified OpenHands repository with custom enhancements for improved reasoning capabilities.

## Custom Modifications

This repository includes several key enhancements to improve reasoning capabilities:

- **Enhanced Reasoning Effort**: Added `super_high` mode with 16,384 reasoning tokens for more thorough analysis
- **Improved Tool Interaction**: Injects user messages after each tool result to fix bug where Gemini doesn't think after tool result message
- **Target Context Length Control**: Configure agents to reach specific context lengths by continuing when finished early or stopping at target (see [Target Context Length Feature](#target-context-length-feature))

## Installation

The OpenHands repository with custom edits is available at:
https://github.com/cxcscmu/directed_research/blob/main/agentic_long_sequence_bench/OpenHands

## Requirements

Follow the installation instructions in the [OpenHands Development Guide](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md):

### System Requirements
- **Docker** (For MacOS users: ensure the default Docker socket is enabled in advanced settings)
- **Python** = 3.12
- **NodeJS** >= 22.x
- **Poetry** >= 1.8

### Additional commands:

sudo apt-get install build-essential python3.12-dev
poetry install --with evaluation

### Verification
After installation, verify everything works correctly:
```bash
make build
```

## LLM Configuration

Create a `config.toml` file to configure your LLM API keys and models:

```toml
[core]
workspace_base="./workspace"

[llm]
model="gpt-4o-mini"
api_key="your_api_key"

[llm.llm]
model="gpt-5"
api_key="your_api_key"

[llm.llm_low]
model="gpt-5"
api_key="your_api_key"
reasoning_effort="low"

[llm.gemini]
model="gemini/gemini-2.5-pro"
api_key="your_api_key"

[llm.gemini_high]
model="gemini/gemini-2.5-pro"
api_key="your_api_key"
reasoning_effort="super_high"

[llm.gemini_flash]
model="gemini/gemini-2.5-flash"
api_key="your_api_key"
native_tool_calling="false"

[llm.deepseek-r1]
model="bedrock/converse/us.deepseek.r1-v1:0"
aws_access_key_id="your_access_key"
aws_region_name="us-east-1"
aws_secret_access_key="your_secret_access_key"

[llm."claude-sonnet-4-5"]
model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
aws_access_key_id="your_access_key"
aws_region_name="us-east-1"
aws_secret_access_key="your_secret_access_key"

[llm."claude-opus-4-1"]
model="bedrock/converse/us.anthropic.claude-opus-4-1-20250805-v1:0"
aws_access_key_id="AKIA6D6JBMNAULFRSNF7"
aws_region_name="us-east-1"
aws_secret_access_key="FySVtMl8S69JaEdXOQtk7hM2H6NArmTX99cJh20o"

[llm."llama3-1-405b-instruct"]
model="bedrock/converse/meta.llama3-1-405b-instruct-v1:0"
aws_access_key_id="your_access_key"
aws_region_name="us-west-2"
aws_secret_access_key="your_secret_access_key"

[llm."llama3-3-70b-instruct"]
model="bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0"
aws_access_key_id="your_access_key"
aws_region_name="us-east-1"
aws_secret_access_key="your_secret_access_key"

[llm."qwen3-235b"]
model="bedrock/converse/qwen.qwen3-235b-a22b-2507-v1:0"
aws_access_key_id="your_access_key"
aws_region_name="us-west-2"
aws_secret_access_key="your_secret_access_key"

[llm."hf-gpt-oss-120b"]
model="huggingface/novita/openai/gpt-oss-120b"
api_key="your_hf_token"
max_input_tokens=131072
num_retries=10

[llm."hf-qwen3-next"]
model="huggingface/together/Qwen/Qwen3-Next-80B-A3B-Thinking"
api_key="your_hf_token"
max_input_tokens=262144
max_output_tokens=81920
num_retries=25

[llm."hf-deepseek-v3.2"]
model="huggingface/novita/deepseek-ai/DeepSeek-V3.2-Exp"
api_key="your_hf_token"
num_retries=20

[llm."qwen3-coder-30b"]
model="hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct"
base_url="http://localhost:8002/v1"
api_key="EMPTY"
native_tool_calling=true
```

### AWS Bedrock Models

**Bedrock API Types:**
- `bedrock/` - Old API
- `bedrock/converse/` - Converse API: Standardized interface, required for cross-region inference profiles

**Cross-Region Inference Profiles:**
Some models require cross-region inference profiles (prefix model ID with `us.`):
- DeepSeek R1: `us.deepseek.r1-v1:0`
- Claude models: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Llama models: `us.meta.llama3-3-70b-instruct-v1:0`
- Qwen models: `us.qwen.qwen3-235b-a22b-2507-v1:0`

**Important Notes:**
- Some Claude models cannot accept both `temperature` and `top_p` parameters - the code automatically removes `top_p`
- Qwen models don't support stop sequences - these are automatically removed

### Using vLLM for Local Models

To run local models with vLLM, you need to:
1. Start a vLLM server on a GPU cluster (e.g., Babel)
2. Create an SSH tunnel from your AWS instance to the vLLM server
3. Configure OpenHands to use the vLLM endpoint

#### Step 1: Start vLLM Server on Babel

Create a SLURM batch script to start the vLLM server. The example below uses the Qwen3-Coder-30B model with 8 L40S GPUs:

```bash
#!/bin/bash
#SBATCH --job-name=vllm_server_30b
#SBATCH --output=logs/vllm_server_%j.out
#SBATCH --error=logs/vllm_server_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=general
#SBATCH --mem=256G
#SBATCH --time=20:00:00

# Activate the vLLM environment
export PATH="/home/andyt/.local/bin:$PATH"
source /home/andyt/vllm-env/bin/activate

# Configure HuggingFace cache directory (use a location with sufficient disk space)
export HF_HOME="/data/user_data/andyt/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/user_data/andyt/hf_cache"
mkdir -p "$HF_HOME"

# Allow extending the model's max context length beyond default
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Server configuration
HOSTNAME=$(hostname)
PORT=8001
HOST="0.0.0.0"
SERVER_URL="http://${HOSTNAME}:${PORT}"

# Save server URL to file for easy reference
echo "$SERVER_URL" > /home/andyt/vllm/vllm_server_url.txt
echo "vLLM server will be running at: $SERVER_URL"

# Verify GPU availability
nvidia-smi

# Model configuration
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

echo "Starting vLLM server with model: $MODEL"
echo "Server URL: $SERVER_URL"

# Start the vLLM OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 8 \
    --max-model-len 128000 \
    --max-num-seqs 32 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

**Key Parameters:**
- `--tensor-parallel-size 8`: Distributes the model across 8 GPUs
- `--max-model-len 128000`: Sets maximum context length to 128K tokens
- `--max-num-seqs 32`: Maximum number of sequences to process in parallel
- `--enable-auto-tool-choice`: Enables automatic tool/function calling
- `--tool-call-parser qwen3_coder`: Uses Qwen3-Coder specific tool call format

Submit the job with: `sbatch vllm_server.sh`

#### Step 2: Set Up SSH Tunnel from AWS to Babel

Once the vLLM server is running, check the output file to find the node name (e.g., `babel-p9-24`), then create an SSH tunnel from your AWS instance:

```bash
ssh -L 8002:babel-p9-24:8001 andyt@login.babel.cs.cmu.edu
```

This command forwards:
- Local port `8002` (on AWS) → `babel-p9-24:8001` (vLLM server on Babel)

Keep this SSH connection open while running evaluations.

#### Step 3: Configure OpenHands

In your `config.toml`, add the vLLM model configuration:

```toml
[llm."qwen3-coder-30b"]
model="hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct"
base_url="http://localhost:8002/v1"
api_key="EMPTY"
native_tool_calling=true
```

**Note:** Native tool calling is supported when using vLLM with appropriate models and parsers.

### Reasoning Effort Configuration

For Gemini models, you can configure the `reasoning_effort` parameter:
- **Default**: 128 reasoning tokens
- **low**: 1024 tokens
- **medium**: 2048 tokens
- **high**: 4096 tokens
- **super_high**: 16384 tokens (custom setting)

### Additional Configuration Options

**Disabling Native Tool Calling**: You may disable the `native_tool_calling` mode for `gemini-2.5-flash` and `Deepseek-V-3.2` as it may be buggy. Set `native_tool_calling="false"` in your config.toml (as shown in the example above).

**Disabling Planning Tool**: You can disable the planning tool by setting `enable_plan_mode = False` in `run_infer.py` for `AgentConfig` on line 246.

When running commands, specify which LLM configuration to use.

## Running the Benchmark

For detailed instructions, refer to the [SWE-Bench README](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/benchmarks/swe_bench/README.md).

### Command Parameters

The `run_infer.sh` script accepts the following parameters:
```bash
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh [MODEL_CONFIG] [COMMIT_HASH] [AGENT] [EVAL_LIMIT] [MAX_ITER] [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS] [MODE] [MAX_BUDGET_PER_TASK] [TARGET_CONTEXT_LENGTH]
```

- **MODEL_CONFIG**: LLM configuration to use (e.g., `gemini_super_high`)
- **COMMIT_HASH**: Git commit hash (e.g., `HEAD`)
- **AGENT**: Agent to use (e.g., `CodeActAgent`)
- **EVAL_LIMIT**: Number of examples to evaluate
- **MAX_ITER**: Maximum iterations per task (default: 100)
- **NUM_WORKERS**: Number of parallel workers (default: 1)
- **DATASET**: Dataset to use (default: `princeton-nlp/SWE-bench_Lite`)
- **SPLIT**: Dataset split (default: `test`)
- **N_RUNS**: Number of runs (default: 1)
- **MODE**: Mode to run in (default: `swe`)
- **MAX_BUDGET_PER_TASK**: Maximum API cost budget per task in dollars (default: 0.0 for no limit)
- **TARGET_CONTEXT_LENGTH**: Target context length in tokens (optional, e.g., `32000`). When set, the agent will be prompted to continue if it finishes early or will receive a final turn message when the target is reached

### Example Commands

**Basic example:**
```bash
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh gemini_super_high HEAD CodeActAgent 1 100 1 princeton-nlp/SWE-bench_Verified test 1 swe 5.0
```

This command runs 1 example using the `gemini_super_high` configuration with a maximum budget of $5.00 per task.

**With target context length:**
```bash
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh gemini_super_high HEAD CodeActAgent 10 100 4 princeton-nlp/SWE-bench_Lite test 1 swe 0.0 32000
```

This command runs 10 examples with a target context length of 32,000 tokens.

### Selecting Specific Tasks

To run inference on a specific subset of tasks, create a `config.toml` file in the `./evaluation/benchmarks/swe_bench/` directory:

```toml
selected_ids = [
    "django__django-10097",
    "django__django-10880",
    "django__django-10914"
]
```

When this file exists, only the tasks whose `instance_id` matches the IDs in the `selected_ids` list will be evaluated. The `EVAL_LIMIT` parameter will apply only to tasks within this list.

## Target Context Length Feature

This feature allows you to set a target context length for SWE-Bench evaluations. When configured, the agent will automatically adjust its behavior to reach the specified context length.

### How It Works

When a target context length is set:

1. **If the agent reaches the target**: The agent receives a message notifying it has reached the target, then gets one final turn to complete any remaining edits before automatically stopping.

2. **If the agent finishes early**: The agent receives a message prompting it to continue reasoning, verifying, and searching for missing information until the target is reached.

### Implementation Details

**Modified Files:**
- `openhands/controller/agent_controller.py` - Core logic for context tracking and message injection
- `openhands/core/setup.py` - Parameter propagation to controller
- `openhands/core/main.py` - Parameter handling in main entry point
- `evaluation/benchmarks/swe_bench/run_infer.py` - Command-line argument support
- `evaluation/benchmarks/swe_bench/scripts/run_infer.sh` - Shell script parameter

**Key Features:**
- Context length is measured using the **maximum prompt token count** from all LLM calls
- Uses a **10% safety margin** to prevent overshooting (e.g., 16k target → 14.4k threshold)
- Works with all agent types (CodeActAgent, etc.)
- Propagates to delegate agents if agent delegation is used

### Usage

#### Using run_infer.sh

```bash
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
  MODEL_CONFIG \
  COMMIT_HASH \
  AGENT \
  EVAL_LIMIT \
  MAX_ITER \
  NUM_WORKERS \
  DATASET \
  SPLIT \
  N_RUNS \
  MODE \
  MAX_BUDGET_PER_TASK \
  TARGET_CONTEXT_LENGTH
```

**Example with 32k target context:**
```bash
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
  gemini_super_high \
  HEAD \
  CodeActAgent \
  10 \
  100 \
  4 \
  princeton-nlp/SWE-bench_Lite \
  test \
  1 \
  swe \
  0.0 \
  32000
```

#### Using run_infer.py directly

```bash
poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config gemini_super_high \
  --max-iterations 100 \
  --eval-num-workers 4 \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --mode swe \
  --target-context-length 32000
```

#### Setting via environment variable

```bash
export TARGET_CONTEXT_LENGTH=32000
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh ...
```

### Behavior Details

#### When Target Context Length is Reached

When the agent's conversation reaches the target context length (measured in tokens):

1. The agent receives this message:
   ```
   You have reached the target context length. You have one final turn to complete any remaining edits, or if you are done, you can finish.
   ```

2. The agent takes one more step (can perform any actions)

3. After this final turn completes, the agent automatically stops with `FINISHED` state

#### When Agent Finishes Early

If the agent sends an `AgentFinishAction` before reaching the target context length:

1. The agent receives this message:
   ```
   Before finalizing your answer, take additional time to verify your reasoning, consider alternative approaches, and search for any missing information that could strengthen your response.
   ```

2. The agent continues working until either:
   - It reaches the target context length (then enters final turn)
   - It finishes again (follows normal completion or target-reached flow)

#### Token Counting

- Context length is measured using the **maximum prompt token count** from all LLM calls so far
- This represents the largest context window used in the conversation
- The prompt tokens are tracked automatically in the metrics after each LLM call
- This approach accounts for:
  - All messages in the current conversation history
  - Any condensation that may have reduced the context
  - The actual tokens sent to the LLM (not just an estimate)
  - Context growth from observations added between LLM calls

#### Safety Margin

To prevent overshooting the target, the system uses a **10% safety margin**:

- **Target specified**: 16,000 tokens
- **Actual threshold**: 14,400 tokens (90% of target)
- **Reason**: Observations get added between LLM calls, causing context to grow

**Example Flow:**
1. Agent makes LLM call with 13k tokens → completes
2. Observation added to history (adds ~1k tokens worth of content)
3. **Check happens (before next LLM call)**: 13k < 14.4k threshold → continue
4. Agent makes LLM call with 14k tokens → completes
5. Observation added (adds ~1k tokens)
6. **Check happens**: 14k < 14.4k threshold → continue
7. Agent makes LLM call with 15k tokens → completes
8. Observation added
9. **Check happens**: 15k >= 14.4k threshold → **INJECT FINAL TURN MESSAGE**
10. Agent receives final turn message and takes one more action
11. Agent finishes or is stopped

This approach ensures the agent stays near the target without significantly exceeding it.

### Output Organization

When using target context length, the evaluation results will be stored in a directory that includes the target in the name:

```
outputs/.../v0.56.0-no-hint-target_ctx_32000-run_1/
```

This makes it easy to organize and compare results with different target context lengths.

### Example Scenarios

**Scenario 1: Agent reaches 32k context naturally**
- Agent works normally until context reaches 32,000 tokens
- System injects: "You have reached the target context length. You have one final turn..."
- Agent takes one more step and stops

**Scenario 2: Agent finishes at 15k context (target: 32k)**
- Agent sends `AgentFinishAction` at ~15,000 tokens
- System injects: "Before finalizing your answer, take additional time to verify..."
- Agent continues working
- Eventually reaches 32k or finishes again

**Scenario 3: Agent finishes at 31k context (target: 32k)**
- Agent sends `AgentFinishAction` at ~31,000 tokens (close to target)
- System injects: "Before finalizing your answer..."
- Agent takes a few more steps
- Reaches 32k context
- System injects: "You have reached the target context length. You have one final turn..."
- Agent takes final turn and stops

### Testing

To test the feature without running a full evaluation:

```bash
# Test with a small target context (e.g., 1000 tokens)
poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config gemini_super_high \
  --max-iterations 10 \
  --eval-num-workers 1 \
  --eval-n-limit 1 \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --mode swe \
  --target-context-length 1000
```

Check the logs for messages like:
- `Target context length reached. Injecting final turn message.`
- `Agent finished early. Injecting continue message to reach target context length.`
- `Current context length: X, Target: Y`

### Notes

- If `target_context_length` is not set (or set to `None`), the feature is disabled and the agent behaves normally
- The feature works with all agent types (CodeActAgent, etc.)
- Context length is checked before each agent step for efficiency
- Log messages at INFO level indicate when messages are injected

## Evaluating Results

To evaluate the diffs submitted by the agent, use the following command format:

```bash
./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gemini-2.5-pro_maxiter_100_N_v0.56.0-no-hint-run_1/output.jsonl "" princeton-nlp/SWE-bench_Verified test
```

Replace the `output.jsonl` path with your actual output file path.
