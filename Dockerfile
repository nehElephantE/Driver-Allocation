# Dockerfile for Ride-Hailing ML System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (streamlit already in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (only what's needed for Streamlit Cloud demo)
COPY app.py .
COPY config.yaml .
COPY driver_allocation.png .

# Note: src/, scripts/, data/, artifacts/ are NOT copied for Streamlit Cloud demo
# The demo app.py has a built-in model and doesn't need these folders

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check (curl is now installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]