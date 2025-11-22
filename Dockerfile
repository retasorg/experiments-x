# UNSW-NB15 ML Pipeline Docker Image
# Base image with CUDA support for GPU-accelerated XGBoost training

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for Docker layer caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python scripts
COPY 1_preprocess_data_FIXED.py /app/
COPY 2_train_isolation_forest.py /app/
COPY 3_train_xgboost.py /app/
COPY 4_export_to_onnx.py /app/

# Copy pipeline runner script
COPY run_pipeline.sh /app/
RUN chmod +x /app/run_pipeline.sh

# Create output directories
RUN mkdir -p /app/data /app/models

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run the pipeline
CMD ["/app/run_pipeline.sh"]
