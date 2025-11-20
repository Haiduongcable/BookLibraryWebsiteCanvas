# ============================================================
# Base image: CUDA 12.2 on Ubuntu 22.04
# ============================================================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Install essential packages (retry apt-get update to avoid transient repo issues)
# ------------------------------------------------------------

RUN rm -f /etc/apt/sources.list.d/cuda*.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        gnupg \
        git \
        git-lfs \
        build-essential \
        curl \
        nano \
        ffmpeg \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2
# ------------------------------------------------------------
# Create working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy requirements first (better caching)
# ------------------------------------------------------------
COPY requirements.txt /app/

# ------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy application code
# ------------------------------------------------------------

COPY code /code
COPY data /data

# Ensure run.sh is executable
RUN chmod +x /code/predict.sh
RUN chmod +x /code/start_jupyter.sh

# ------------------------------------------------------------
# Set entrypoint
# ------------------------------------------------------------
# ENTRYPOINT ["/code/predict.sh"]
CMD ["/code/predict.sh"]
