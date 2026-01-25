# Use official Python image for HuggingFace Space
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for HuggingFace Space
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user:user requirements.txt .

# Copy the local nano-vllm package
COPY --chown=user:user acestep/third_parts/nano-vllm $HOME/app/acestep/third_parts/nano-vllm

# Install all dependencies from requirements.txt first
# This includes torch and pre-built flash-attn wheel
RUN pip install --no-cache-dir --user -r requirements.txt

# Install nano-vllm with --no-deps since all dependencies are already installed
RUN pip install --no-cache-dir --user --no-deps $HOME/app/acestep/third_parts/nano-vllm

# Copy the rest of the application
COPY --chown=user:user . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
