FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and OpenPose
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p logs \
    openpose/models/pose/coco \
    services \
    websocket

# Copy OpenPose model files (200MB+ - these should exist locally)
COPY openpose/models/pose/coco/pose_iter_440000.caffemodel openpose/models/pose/coco/
COPY openpose/models/pose/coco/pose_deploy_linevec.prototxt openpose/models/pose/coco/

# Copy application code
COPY body_science.py .
COPY emotion_detector.py .
COPY logger.py .
COPY main.py .
COPY pose_detector.py .
COPY posture_analyzer.py .
COPY visualization.py .

# Copy service modules
COPY services/ services/

# Copy websocket modules
COPY websocket/ websocket/

# Expose port
EXPOSE 9005

# Run the application
CMD ["python", "main.py"]