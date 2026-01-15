FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir pytest pytest-json-report

# Create workspace directory
WORKDIR /app

# Copy workspace files (sequential baseline and document generator)
COPY workspace/ /app/workspace/

# Copy skills to multiple agent paths
COPY skills/ /app/.claude/skills/ 2>/dev/null || true
COPY skills/ /app/.codex/skills/ 2>/dev/null || true
COPY skills/ /app/.opencode/skill/ 2>/dev/null || true
COPY skills/ /app/.goose/skills/ 2>/dev/null || true
COPY skills/ /app/.factory/skills/ 2>/dev/null || true
COPY skills/ /app/.agents/skills/ 2>/dev/null || true

# Set working directory to workspace
WORKDIR /app/workspace

# Default command
CMD ["python", "--version"]
