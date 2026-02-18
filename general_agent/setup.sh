#!/bin/bash
# 安装脚本：设置 omni 环境并集成 tau2-bench

set -e

echo "=== All-in-One Benchmark Setup ==="

# 1. 创建 conda 环境
echo "Step 1: Creating conda environment..."
if conda env list | grep -q "^one "; then
    echo "Environment 'one' already exists. Updating..."
    conda env update -f environment.yml
else
    echo "Creating new environment 'one'..."
    conda env create -f environment.yml
fi

# 2. 激活环境
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate one

# 3. 安装 omni 项目
echo "Step 3: Installing omni package..."
pip install -e .

# 4. 检查 tau2-bench 是否存在
TAU2_BENCH_PATH="../tau2-bench"
if [ -d "$TAU2_BENCH_PATH" ]; then
    echo "Step 4: Found tau2-bench at $TAU2_BENCH_PATH"
    
    # 安装 tau2-bench
    echo "Installing tau2-bench..."
    cd "$TAU2_BENCH_PATH"
    pip install -e .
    cd -
    
    # 设置 TAU2_DATA_DIR 环境变量
    TAU2_DATA_DIR="$(cd $TAU2_BENCH_PATH && pwd)/data"
    echo "export TAU2_DATA_DIR=$TAU2_DATA_DIR" >> ~/.bashrc
    export TAU2_DATA_DIR="$TAU2_DATA_DIR"
    
    echo "✓ Tau2-bench integrated successfully"
    echo "  TAU2_DATA_DIR set to: $TAU2_DATA_DIR"
else
    echo "Warning: tau2-bench not found at $TAU2_BENCH_PATH"
    echo "  Tau2 server will have limited functionality"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To use the benchmark:"
echo "  1. conda activate one"
echo "  2. allinone --config config_tau2.json --model gpt-4"
echo ""
echo "To test tau2 server individually:"
echo "  python -m source.servers.tau2_server"
