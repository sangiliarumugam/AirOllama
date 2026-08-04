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
PORT=11211
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        PORT="$arg"
    elif [[ "$arg" == "--api-only" || "$arg" == "--no-ui" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

echo "=========================================================="
echo "🚀 AirOllama Server (Layer Streaming Engine)"
echo "📡 Listening on http://0.0.0.0:${PORT}"
if [[ "$EXTRA_ARGS" == *"--api-only"* || "$EXTRA_ARGS" == *"--no-ui"* ]]; then
    echo "🔌 Mode: API Server Only (OpenCode Agent Ready)"
else
    echo "💻 Web Dashboard: http://localhost:${PORT}"
fi
echo "=========================================================="

python3 -m airollama.cli serve --port "$PORT" $EXTRA_ARGS
