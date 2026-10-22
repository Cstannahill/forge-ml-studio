# ==============================================================================
# ForgeML Pipeline Studio — Full-Stack GPU Workstation & Cloud Container
# Includes:
#   1. NVIDIA CUDA 12.4 + PyTorch 2.5 + Unsloth + HuggingFace TRL / PEFT / Mergekit
#   2. Node.js 22 LTS + ForgeML Web Studio & Subprocess Orchestrator on Port 3000
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production \
    HF_HOME=/workspace/.cache/huggingface \
    PORT=3000

# 1. Install system dependencies, Python 3.11, and Node.js 22 LTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential cmake ninja-build \
    python3 python3-pip python3-venv python3-dev \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Install Python Fine-Tuning Engine dependencies (Unsloth, Transformers, TRL, PEFT, bitsandbytes)
COPY engine/requirements.txt ./engine/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r ./engine/requirements.txt

# 3. Install Node.js dependencies & build ForgeML Studio frontend
COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

# Create persistent directories for materialized training runs, datasets & checkpoints
RUN mkdir -p /workspace/runs /workspace/checkpoints /workspace/exports /workspace/.cache/huggingface

EXPOSE 3000 8080 8000

CMD ["npm", "run", "start"]
