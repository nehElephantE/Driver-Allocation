# Dockerfile for Ride-Hailing Matching Prediction Pipeline
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/interim \
    /app/artifacts/models \
    /app/artifacts/features \
    /app/artifacts/metrics \
    /app/artifacts/predictions \
    /app/logs \
    /app/notebooks

# Set up volume mount points
VOLUME ["/app/data", "/app/artifacts", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["python", "main.py", "--mode", "all"]

# Labels
LABEL maintainer="Your Name <your.email@example.com>" \
      version="1.0.0" \
      description="Ride-Hailing Matching Prediction Pipeline"