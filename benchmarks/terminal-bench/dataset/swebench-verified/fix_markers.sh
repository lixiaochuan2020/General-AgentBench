#!/bin/bash
# Fix run-tests.sh to add START/END markers without requiring swebench in container
# The host-side swebench library will parse these markers

cd /home/ubuntu/agentic-long-bench/terminal-bench/dataset/swebench-verified

echo "Fixing run-tests.sh files to add test output markers..."

for dir in */; do
    runtest="${dir}run-tests.sh"
    if [ ! -f "$runtest" ]; then
        continue
    fi
    
    # Check if already has our fix marker
    if grep -q "FIXED_BY_ADD_MARKERS" "$runtest" 2>/dev/null; then
        continue
    fi
    
    # Create a temp file
    tmpfile=$(mktemp)
    
    # Read and modify the script
    # Replace the pip install line with marker addition
    sed 's|pip install -q swebench.*&& python parser.py|# FIXED_BY_ADD_MARKERS: Add markers and skip swebench requirement\nif [ -f "$LOG_FILE" ]; then\n    # Add START/END markers to log file for swebench parsing\n    tmplog=$(mktemp)\n    echo ">>>>> Start Test Output" > "$tmplog"\n    cat "$LOG_FILE" >> "$tmplog"\n    echo ">>>>> End Test Output" >> "$tmplog"\n    mv "$tmplog" "$LOG_FILE"\n    # Output the log content with markers to stdout\n    cat "$LOG_FILE"\nfi\necho "SWEBench results starts here"\necho "MARKER_ADDED"\necho "SWEBench results ends here"|g' "$runtest" > "$tmpfile"
    
    mv "$tmpfile" "$runtest"
done

echo "Done. Verifying..."
tail -20 django__django-15278/run-tests.sh
