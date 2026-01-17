FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    blender \
    openscad \
    imagemagick \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for 3D modeling and image processing
RUN pip install --no-cache-dir \
    numpy \
    numpy-stl \
    trimesh \
    Pillow \
    opencv-python-headless \
    scipy \
    scikit-image \
    solidpython \
    matplotlib \
    networkx \
    pyglet \
    shapely \
    rtree

# Set working directory
WORKDIR /app

# Copy workspace files
COPY workspace/ /app/workspace/

# Copy skills to all agent paths
COPY skills/ /app/.claude/skills/
COPY skills/ /app/.codex/skills/
COPY skills/ /app/.opencode/skill/
COPY skills/ /app/.goose/skills/
COPY skills/ /app/.factory/skills/
COPY skills/ /app/.agents/skills/

# Set environment variables
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1

# Create logs directory
RUN mkdir -p /logs/verifier

CMD ["/bin/bash"]
