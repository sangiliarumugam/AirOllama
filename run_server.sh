#!/bin/bash
# AirOllama macOS Launcher Script
# Starts LLM server on port 11211 with on-demand layer loading

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export HF_HOME="$SCRIPT_DIR/.hf_cache"
PORT=${1:-11211}


echo "=========================================================="
echo "🚀 AirOllama Server (Layer Streaming Engine)"
echo "📡 Listening on http://0.0.0.0:${PORT}"
echo "💻 Web Dashboard: http://localhost:${PORT}"
echo "=========================================================="

python3 -m airollama.cli serve --port "$PORT"
