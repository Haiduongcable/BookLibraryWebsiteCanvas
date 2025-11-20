#!/usr/bin/env bash
set -e

PORT=9777    # default port 8888 if not provided
WORKDIR=${2:-/code}

echo "🚀 Starting Jupyter Lab on port $PORT"
echo "📂 Workdir: $WORKDIR"
echo "🌐 Network: host"
echo "➡ Access: http://<your_machine_ip>:$PORT"

jupyter lab \
    --ip=0.0.0.0 \
    --port="$PORT" \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir="$WORKDIR"
