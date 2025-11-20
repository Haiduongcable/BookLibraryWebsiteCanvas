# ZaloAI2025

Utilities for serving a fine-tuned **Qwen3-VL** model with [vLLM](https://github.com/vllm-project/vllm) and running the asynchronous `call_inference.py` evaluation script over dashcam QA data.

## Requirements

- Python 3.10+ with CUDA-capable GPUs (the Docker image is based on CUDA 12.2 / Ubuntu 22.04).
- FFmpeg libraries (already installed in the Docker image; install via your package manager for local runs).
- The model checkpoint directory (`checkpoint_store` by default) containing the fine-tuned Qwen3-VL weights.

Install Python dependencies once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Launching the vLLM server locally

The `run.sh` script wraps the OpenAI-compatible vLLM API server. It reads the following environment variables (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `./checkpoint_store` | Qwen3-VL checkpoint directory. |
| `SERVED_MODEL_NAME` | `qwen3-vl-sft` | Name exposed via the API. |
| `PORT` | `2782` | REST server port. |
| `HOST` | `0.0.0.0` | Bind host. |
| `DTYPE` | `bfloat16` | Inference dtype. |
| `GPU_MEMORY_UTILIZATION` | `0.95` | vLLM GPU util target. |
| `VLLM_LOG_FILE` | *(unset)* | When set, logs are appended to this file. |

Example:

```bash
MODEL_PATH=/models/qwen3-vl-sft PORT=8727 ./run.sh
# For more flags, pass them after "--", e.g.:
./run.sh --max-num-seqs 8
```

## Batched inference client

After the server is alive, call the asynchronous inference script:

```bash
python call_inference.py \
  --output-csv outputs/submission.csv \
  --json-path /data/public_test_label.json \
  --video-dir /data \
  --video-http-base http://localhost:8392 \
  --vllm-url http://localhost:8727/v1/chat/completions \
  --max-concurrent 12
```

Use `python call_inference.py --help` to inspect all tunable parameters (max tokens, retries, temperature, FPS, etc.). The script writes a CSV report (including per-question predictions, attempts, and accuracy) to the path specified by `--output-csv`.

## Docker workflow

The repository ships with a CUDA-enabled Dockerfile with the Python dependencies pre-installed. Two helper scripts simplify the workflow:

1. `./build.sh` – builds the container image (`zaloai2025:latest` by default).
2. `./run.sh` – used as the image entrypoint to launch the vLLM API server.

Build and run:

```bash
./build.sh                      # docker build -t zaloai2025:latest .
docker run --gpus all \
  -p 8727:2782 \
  -v /path/to/checkpoint_store:/app/checkpoint_store \
  --name zaloai2025 \
  zaloai2025:latest
```

You can override environment variables at `docker run` time (`-e MODEL_PATH=/app/checkpoint_store`). The container emits server logs to STDOUT unless `VLLM_LOG_FILE` is defined.
