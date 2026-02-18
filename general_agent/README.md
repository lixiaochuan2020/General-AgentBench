# General Agent Benchmark Framework

A unified agentic evaluation framework built on MCP (Model Context Protocol). A single universal agent connects to all benchmark tools simultaneously and is assessed by each benchmark's native evaluator.

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **tau2bench** | Customer service dialogue (airline, retail, telecom) |
| **mcpbench** | MCP tool-calling (30+ servers) |
| **swebench** | Software engineering code repair (Docker) |
| **terminalbench** | Terminal command execution (Docker) |
| **mathhay** | Long-context mathematical reasoning |
| **search** | Web search via Serper API |

## Setup

1. `cd general_agent && pip install -e .`
2. `cd ../benchmarks/tau2-bench && pip install -e .`
3. `cp .env-example .env` and fill in API keys.
4. Docker is required for **swebench** and **terminalbench**.

## Experiment Scripts

All scripts are located in **`general_agent/scripts/`** and should be run from the `general_agent/` directory. See `scripts/README.md` for full usage and flags.

| Script | Purpose |
|--------|---------|
| `run_baseline.sh` | Run one or more benchmarks with default settings |
| `run_parallel_scaling.sh` | Run K passes (different seeds) for Best@K evaluation |
| `run_sequential_scaling.sh` | Run with increasing token budgets; supports checkpoint reuse |
| `run_pointwise_self_choice.sh` | Pointwise LLM-as-judge trajectory evaluation |
| `run_pairwise_self_choice.sh` | Pairwise Bump Sort trajectory ranking |

### Typical Workflow

```
Parallel Scaling ──► Pointwise Self-Choice ──► summary.json
       │
       └──────────► Pairwise Self-Choice  ──► bump_sort_results.json

Sequential Scaling   (independent, token budget analysis)
```

## Directory Structure

```
general_agent/
├── run.py              # Entry point
├── source/
│   ├── agent.py        # UniversalAgent
│   ├── host.py         # MCP client manager
│   ├── config.py       # Configuration
│   ├── native_evaluators.py
│   ├── llm_api/        # LiteLLM / OpenAI wrappers
│   ├── servers/        # One MCP server per benchmark
│   ├── evaluators/     # SWEBench & TerminalBench evaluators
│   ├── scaling/        # Sequential scaling controller
│   └── self_choice/    # Self-choice evaluation logic
├── scripts/            # Experiment shell scripts
├── data/               # Task data files
└── tests/              # Unit / integration tests
```
