#!/bin/bash
# Test runner script for Spring Boot 2 to 3 Migration Task

# Don't use set -e so we can always write the reward file
# set -e

# Create logs directory if it doesn't exist
mkdir -p /logs/verifier

# Install pytest and required plugins
echo "Installing test dependencies..."
pip3 install --quiet pytest pytest-json-report 2>/dev/null || {
    echo "Warning: Failed to install pytest, trying with pip..."
    pip install --quiet pytest pytest-json-report 2>/dev/null || {
        echo "Error: Could not install pytest"
        echo "0" > /logs/verifier/reward.txt
        exit 1
    }
}

# Navigate to workspace
cd /workspace

# Source SDKMAN for Java
if [ -f /root/.sdkman/bin/sdkman-init.sh ]; then
    source /root/.sdkman/bin/sdkman-init.sh
    sdk use java 21.0.2-tem 2>/dev/null || true
fi

# Run the tests
echo "Running migration verification tests..."
python3 -m pytest /tests/test_outputs.py -v --tb=short --json-report --json-report-file=/logs/verifier/test_results.json
TEST_RESULT=$?

# Always write the reward file
if [ $TEST_RESULT -eq 0 ]; then
    echo "1" > /logs/verifier/reward.txt
    echo "All tests passed!"
else
    echo "0" > /logs/verifier/reward.txt
    echo "Some tests failed."
fi

exit $TEST_RESULT
