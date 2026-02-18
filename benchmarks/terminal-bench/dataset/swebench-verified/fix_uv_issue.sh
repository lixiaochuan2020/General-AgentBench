#!/bin/bash
# Fix uv command not found issue in all run-tests.sh files
# Replace "uv run parser.py" with "pip install -q swebench==4.0.3 datasets==2.16.1 && python parser.py"

cd /home/ubuntu/agentic-long-bench/terminal-bench/dataset/swebench-verified

# Count files to fix
TOTAL=$(grep -l "uv run parser.py" */run-tests.sh | wc -l)
echo "Found $TOTAL files to fix..."

# Apply fix
find . -name "run-tests.sh" -type f -exec sed -i 's|uv run parser.py|pip install -q swebench==4.0.3 datasets==2.16.1 \&\& python parser.py|g' {} \;

# Verify
REMAINING=$(grep -l "uv run parser.py" */run-tests.sh 2>/dev/null | wc -l)
FIXED=$((TOTAL - REMAINING))

echo "Fixed: $FIXED files"
echo "Remaining: $REMAINING files"
