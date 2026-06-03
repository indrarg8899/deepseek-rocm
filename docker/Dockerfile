# DeepSeek ROCm Docker Image
# Base: ROCm 6.2 + PyTorch on Ubuntu 22.04
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release-2.3.0

LABEL maintainer="indrarg8899"
LABEL description="DeepSeek V2/V3/R1 inference and fine-tuning on AMD ROCm"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY benchmarks/ benchmarks/
COPY tests/ tests/

# ROCm env
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV HIP_VISIBLE_DEVICES=all

# Default: run inference server
EXPOSE 8000
CMD ["python", "-m", "src.server", "--config", "configs/deepseek-v3.yml", "--port", "8000"]
