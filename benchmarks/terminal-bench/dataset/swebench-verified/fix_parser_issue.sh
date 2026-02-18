#!/bin/bash
# Fix run-tests.sh to work without swebench in container
# Instead of running parser.py with swebench, just output the required markers

cd /home/ubuntu/agentic-long-bench/terminal-bench/dataset/swebench-verified

# Count files to fix
TOTAL=$(ls -d */ | wc -l)
echo "Found $TOTAL task directories..."

# For each run-tests.sh, we need to:
# 1. Add START_TEST_OUTPUT and END_TEST_OUTPUT markers around test output
# 2. Skip the parser.py part that requires swebench

for dir in */; do
    if [ -f "${dir}run-tests.sh" ]; then
        # Check if already patched
        if grep -q "START_TEST_OUTPUT_MARKER" "${dir}run-tests.sh" 2>/dev/null; then
            continue
        fi
        
        # Replace the pip install && python parser.py with a simple echo
        sed -i 's|pip install -q swebench.*&& python parser.py|echo "SWEBench results starts here"\necho "PARSER_SKIPPED_USE_HOST_SWEBENCH"\necho "SWEBench results ends here"|g' "${dir}run-tests.sh"
    fi
done

echo "Done. Checking sample..."
tail -10 django__django-15278/run-tests.sh
