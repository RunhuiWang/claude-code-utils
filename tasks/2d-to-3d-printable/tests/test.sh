#!/bin/bash
set -e

echo "=== Installing test dependencies ==="
pip install pytest pytest-json-report trimesh==4.11.1 numpy==2.2.6 Pillow==12.1.0 opencv-python-headless==4.12.0.88 scipy==1.17.0 shapely==2.1.2

echo "=== Running tests ==="
cd /app

# Create logs directory if it doesn't exist
mkdir -p /logs/verifier

# Run pytest with JSON report
pytest tests/test_outputs.py \
    --json-report \
    --json-report-file=/logs/verifier/test_results.json \
    -v \
    || true

# Check test results and write reward
if [ -f /logs/verifier/test_results.json ]; then
    # Extract test results using Python
    python3 << 'EOF'
import json
import sys

with open('/logs/verifier/test_results.json', 'r') as f:
    results = json.load(f)

# Check if all tests passed
summary = results.get('summary', {})
passed = summary.get('passed', 0)
failed = summary.get('failed', 0)
error = summary.get('error', 0)
total = summary.get('total', 0)

print(f"Test Results: {passed}/{total} passed, {failed} failed, {error} errors")

# Write reward: 1 if all passed, 0 otherwise
reward = 1 if (failed == 0 and error == 0 and passed > 0) else 0
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(str(reward))

print(f"Reward: {reward}")
sys.exit(0 if reward == 1 else 1)
EOF
else
    echo "No test results found"
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi
