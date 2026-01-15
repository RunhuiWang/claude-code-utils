#!/bin/bash
set -e

# Navigate to workspace
cd /app/workspace

# Create logs directory
mkdir -p /app/logs

# Install any additional dependencies if needed
pip install pytest pytest-json-report -q

# Run pytest with JSON output
python -m pytest /app/tests/test_outputs.py \
    --json-report \
    --json-report-file=/app/logs/test_results.json \
    -v \
    2>&1 | tee /app/logs/test_output.log

# Exit with pytest's exit code
exit ${PIPESTATUS[0]}
