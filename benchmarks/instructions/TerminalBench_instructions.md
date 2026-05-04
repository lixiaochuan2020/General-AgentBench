# TerminalBench Instructions

Instructions for running TerminalBench with custom Terminus agent.

## API Key Setup

Before running TerminalBench, you need to set up your LLM API key:

```bash
export LLM_API_KEY="your_api_key"
```

Replace `"your_api_key"` with your actual LLM API key.

For Bedrock models this is not needed.

## Installation

Terminal-Bench is distributed as a pip package and can be run using the Terminal-Bench CLI: `tb`.

### Standard Installation
Simply run one of the following commands:

```bash
uv tool install terminal-bench
```

or

```bash
pip install terminal-bench
```

to install the command line tool.

### Editable Installation (for Custom Changes)
To use custom changes in development mode, additionally run:

```bash
uv tool install --force --editable .
```

## Running TerminalBench

### Configuration

You may set additional OpenHands configurations by prepending OPENHANDS, for instance:

```bash
export OPENHANDS_MAX_ITERATIONS=50
export OPENHANDS_AGENT_ENABLE_PLAN_MODE=false
export OPENHANDS_TARGET_CONTEXT_LENGTH=64000
export OPENHANDS_LLM_NUM_RETRIES=15
export OPENHANDS_LLM_MAX_INPUT_TOKENS=262144
export OPENHANDS_LLM_MAX_OUTPUT_TOKENS=81920
```

For Bedrock models, you must set the following environment variables:

```bash
export OPENHANDS_LLM_AWS_ACCESS_KEY_ID=your_aws_access_key_id
export OPENHANDS_LLM_AWS_REGION_NAME=aws_region
export OPENHANDS_LLM_AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

For running with vLLM, set the following environment variables:
```bash
export OPENHANDS_LLM_BASE_URL=http://localhost:8002/v1
```

For SSH tunneling, add the following lines to the end of openhands-setup.sh.j2:
```bash
# Set up SSH tunnel with password authentication
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Configure SSH to disable strict host key checking
cat > ~/.ssh/config <<EOF
Host your address
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
chmod 600 ~/.ssh/config

# Start SSH tunnel in the background using sshpass with SSH_PASSWORD
echo "Starting SSH tunnel with password authentication"
nohup sshpass -p "$SSH_PASSWORD" ssh -f -N -L 8002:your server name:8001 your email \
    -o ExitOnForwardFailure=yes > /tmp/ssh-tunnel.log 2>&1 || true

# Give the tunnel a moment to establish
sleep 2

# Log tunnel status
if pgrep -f "ssh.*8002:your address:8001" > /dev/null; then
    echo "SSH tunnel established successfully on port 8002"
else
    echo "Warning: SSH tunnel may not have started. Check /tmp/ssh-tunnel.log for details"
    cat /tmp/ssh-tunnel.log 2>/dev/null || true
fi
```

Also add the following lines to the end of the _env function in openhands_agent.py:
```python
# Handle SSH password for tunnel authentication
if "SSH_PASSWORD" in os.environ:
    self._logger.info("SSH_PASSWORD found, will use for tunnel authentication")
    env["SSH_PASSWORD"] = os.environ["SSH_PASSWORD"]
```

Make sure to set the environment variable SSH_PASSWORD before you run.

### Basic Usage

#### Standard Example
To run with a standard model, use:

```bash
tb run --agent openhands --model gemini/gemini-2.5-pro --dataset terminal-bench-core --n-concurrent 1 --task-id hello-world --local-registry-path ./registry.json
```

#### Bedrock Example

```bash
tb run --agent openhands --model bedrock/converse/us.deepseek.r1-v1:0 --dataset terminal-bench-core --n-concurrent 1 --task-id hello-world --local-registry-path ./registry.json
```

#### Using 0.1.1 Folder Example

```bash
tb run --agent openhands --model huggingface/novita/deepseek-ai/DeepSeek-V3.2-Exp --n-concurrent 20 --dataset-path terminal-bench-core-0.1.1-tasks --global-timeout-multiplier 3.0
```

**Note:** Please specify local registry path as it fixes the dataset to an old commit.

## Additional Information

### OpenHands Version

This fixes a bug where it doesn't output the trajectories when there is an error.

### Custom Features
The modified version allows you to specify `thinking_tokens` for the terminus 2 agent.
