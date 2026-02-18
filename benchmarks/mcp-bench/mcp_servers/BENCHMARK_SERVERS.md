# SWEBench & TerminalBench MCP Servers

本目录包含两个新的 MCP Server，用于将 SWE-Bench 和 TerminalBench 的工具集成到 all_in_one benchmark 系统中。

## 概述

这些是**模拟（mock）实现**，提供了正确的工具 schema，但不执行实际操作。目的是：
1. 让 Universal Agent 知道这些 benchmark 有哪些工具可用
2. 用于测试工具选择和 distraction 逻辑
3. 为将来实际集成提供框架

## SWEBench MCP Server

位置: `mcp-bench/mcp_servers/swebench-mcp/server.py`

提供 OpenHands CodeAct Agent 使用的 6 个工具：

| 工具名 | 描述 |
|--------|------|
| `execute_bash` | 在终端执行 bash 命令 |
| `str_replace_editor` | 文件编辑（view, create, str_replace, insert, undo_edit） |
| `execute_ipython_cell` | 执行 IPython 代码 |
| `browser` | 浏览器交互（goto, click, fill 等） |
| `think` | 记录思考过程 |
| `finish` | 标记任务完成 |

## TerminalBench MCP Server

位置: `mcp-bench/mcp_servers/terminalbench-mcp/server.py`

提供 Terminus Agent 使用的 5 个工具：

| 工具名 | 描述 |
|--------|------|
| `send_keystrokes` | 发送按键到终端（核心交互方式） |
| `get_terminal_state` | 获取当前终端状态 |
| `wait_for_output` | 等待终端输出 |
| `mark_task_complete` | 标记任务完成 |
| `analyze_task` | 记录分析和计划 |

## 配置

这些 server 已添加到 `mcp-bench/mcp_servers/commands.json`：

```json
{
  "SWEBench": {
    "cmd": "python server.py",
    "env": [],
    "cwd": "../swebench-mcp"
  },
  "TerminalBench": {
    "cmd": "python server.py",
    "env": [],
    "cwd": "../terminalbench-mcp"
  }
}
```

并在 `all_in_one/mcp_benchmark/servers/mcpbench_server.py` 的 `SERVER_NAME_MAP` 中注册：
- `SWEBench` -> `mcpbench-swebench`
- `TerminalBench` -> `mcpbench-terminalbench`

## 使用

在 all_in_one 系统中，这些 server 会被自动加载。工具名称会添加到可用工具池中：

```python
# 在任务文件中指定需要这些 server
{
    "benchmark": "swebench",
    "domain": "python",
    "servers": ["SWEBench"],
    "task": { ... }
}
```

## 未来工作

要实现真正的 SWE-Bench 和 TerminalBench 评估，需要：

1. **SWE-Bench**:
   - Docker 运行时支持
   - OpenHands sandbox 集成
   - 工具实际执行（bash, file edit 等）

2. **TerminalBench**:
   - tmux session 管理
   - 终端状态捕获
   - keystroke 发送实现

## 验证

```bash
cd /home/tianshim/agentic-long-bench/all_in_one
source ~/miniconda3/etc/profile.d/conda.sh && conda activate one

# 测试 server 加载
python -c "
import asyncio
from mcp_benchmark.host import BenchmarkHost
from mcp_benchmark.servers.mcpbench_server import get_mcpbench_server_configs, convert_to_host_config

async def test():
    host = BenchmarkHost()
    configs = get_mcpbench_server_configs()
    
    for name in ['SWEBench', 'TerminalBench']:
        raw_config = configs[name]
        host_config = convert_to_host_config(name, raw_config)
        await host.create_client(
            name=f'mcpbench-{name}',
            command=host_config['command'],
            args=host_config['args'],
            env=host_config.get('env'),
            cwd=host_config.get('cwd'),
        )
        client = host.clients[f'mcpbench-{name}']
        print(f'{name}: {[t.name for t in client.tools]}')

asyncio.run(test())
"
```

预期输出：
```
SWEBench: ['execute_bash', 'str_replace_editor', 'execute_ipython_cell', 'browser', 'think', 'finish']
TerminalBench: ['send_keystrokes', 'get_terminal_state', 'wait_for_output', 'mark_task_complete', 'analyze_task']
```
