FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Playwright MCP server)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies
RUN uv pip install --system \
    anthropic==0.52.0 \
    httpx==0.28.1 \
    pydantic==2.11.4 \
    playwright==1.52.0

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Install Playwright MCP server globally
RUN npm install -g @anthropic-ai/mcp-playwright

# Set working directory
WORKDIR /root

# Copy workspace files
COPY workspace/ /root/workspace/

# Copy skills to all agent paths
COPY skills/ /root/.claude/skills/
COPY skills/ /root/.codex/skills/
COPY skills/ /root/.opencode/skill/
COPY skills/ /root/.goose/skills/
COPY skills/ /root/.factory/skills/
COPY skills/ /root/.agents/skills/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Create logs directory
RUN mkdir -p /logs/verifier

CMD ["/bin/bash"]
