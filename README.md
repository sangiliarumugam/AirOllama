# 🚀 AirOllama

**AirOllama** is an open-source, memory-efficient LLM inference engine, native macOS application, and REST API server that combines **Ollama & OpenAI API compatibility** with **AirLLM-style layer streaming and disk offloading**. Built specifically for macOS (Apple Silicon Metal GPU) and CUDA devices, AirOllama allows developers and researchers to run high-parameter Large Language Models (such as 27B–70B+ models) on consumer hardware with strict RAM limits.

---

## 💡 Purpose & Core Architecture

Running modern 27B+ parameter models normally requires 32GB to 64GB+ of dedicated Unified Memory or VRAM. **AirOllama** bridges this gap by:

1. **AirLLM-Style Layer Streaming**: Dynamically loading and unloading transformer decoder layer blocks between RAM, Metal VRAM, and disk memory-mapped cache files (`.offload`) via PyTorch `accelerate`, keeping RAM consumption below strict user-configured caps.
2. **Native Apple MLX Execution**: Native support for `mlx-community` 4-bit and 8-bit quantized models running natively on Apple Silicon Metal Unified Memory.
3. **Drop-in Ollama & OpenAI API Ecosystem**: Exposing identical REST API endpoints (`/api/generate`, `/api/chat`, `/api/chat/completions`, `/v1/chat/completions`, `/api/embeddings`, `/v1/embeddings`) so existing Ollama frontends, OpenCode Agents, and OpenAI SDKs work out-of-the-box.
4. **Native macOS Application (`AirOllama.app`)**: Native Swift/AppKit menu bar app with embedded WebKit dashboard and auto-spawns the backend engine server.
5. **Modular Web Dashboard & Multi-Threaded Downloader**: Clean multi-page Web UI with live VRAM/RAM gauges, layer visualizers, GFM Markdown chat playground, multi-worker thread LED download matrix, and **Cancel & Retry** download management.

---

## 🙏 Credits & Acknowledgments

AirOllama stands on the shoulders of incredible open-source innovations:

- **[Ollama](https://github.com/ollama/ollama)**: For designing the developer-friendly local LLM API standard and containerized workflow that inspired AirOllama's server architecture and endpoint compatibility.
- **[AirLLM](https://github.com/lyTheoretical/AirLLM)**: For pioneering layer-by-layer inference streaming and disk memory-mapping strategies, proving 70B+ parameter models can execute on limited hardware.
- **[Apple MLX](https://github.com/ml-explore/mlx)**: For Apple Silicon's high-performance machine learning framework, powering AirOllama's native Metal GPU integration.
- **[Hugging Face Accelerate](https://github.com/huggingface/accelerate)** & **[Transformers](https://github.com/huggingface/transformers)**: For robust weight mapping, layer device assignment, and tokenizer infrastructure.

---

## 📖 How to Use Guide

### 1. Prerequisites
- **Operating System**: macOS (Apple Silicon M1/M2/M3/M4 recommended) or Linux with CUDA GPU.
- **Python**: Version 3.10 or higher.
- **Disk Space**: High-speed SSD recommended for optimal offload layer streaming.

---

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/sangiliarumugam/AirOllama.git
cd AirOllama

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 3. Native macOS Application (`AirOllama.app`)

AirOllama includes a native macOS application built with Swift and AppKit.

#### Building the macOS App:
```bash
./build_mac_app.sh
```
This compiles the native Swift application to `./dist/AirOllama.app` and creates a root `./AirOllama.app` shortcut.

#### Launching:
Double-click `AirOllama.app` or run:
```bash
open dist/AirOllama.app
```
When launched, `AirOllama.app` displays a status bar item (`⚡ AirOllama`) and automatically launches the backend server on port `11211`.

---

### 4. Running the Backend Server via CLI

You can also run the backend server standalone or in API-only headless mode (ideal for IDE coding assistants like OpenCode Agent):

```bash
./run_server.sh 11211
```

Or using the Python CLI:

```bash
# Standard mode (Web UI + REST API)
python -m airollama.cli serve --port 11211

# Headless API-only mode (No UI overhead)
python -m airollama.cli serve --port 11211 --api-only
```

Once started, open your browser and navigate to:
👉 **`http://localhost:11211`**

---

### 5. Web Dashboard Architecture

The Web Dashboard is modularly structured into separate page templates (`airollama/static/*.html`):

#### 📊 Dashboard (`dashboard.html`)
- Live system metrics: **Active Model**, **Active Layer**, **Model RAM**, **Available System RAM**, **VRAM / Metal Memory**, and **Disk Offload Status**.
- **Layer-by-Layer Visualizer**: Real-time animated decoder layer blocks glowing as model weights stream into GPU memory.

#### 💬 Chat Playground (`playground.html`)
- **Interactive Multi-Turn Chat**: Full conversational UI with streaming token generation and live `tok/s` speed meter.
- **Markdown & Code Blocks**: Renders GitHub Flavored Markdown with language-tagged dark code blocks and a **1-click Copy Code** button.
- **RAM & Speed Limit Slider**: Adjust layer RAM allocation on the fly with presets (**Min RAM**, **Balanced**, **Max Speed**). Adjusting the slider automatically triggers an `Unload -> Preload` pipeline to re-allocate layers in memory.
- **Web Search & Location Context**: Live web search integration and system location prompt customization.

#### 📦 Models & Multi-Threaded Pull Manager (`models.html`)
- Pull models from **Ollama Registry** (`registry.ollama.ai`) or **Hugging Face Hub** (`huggingface.co`).
- **Worker Thread LED Matrix**: Real-time visual LED indicator panel tracking parallel download worker streams.
- **Cancel & Retry Downloads**: Cancel active downloads (`🛑 Cancel Download`) with automatic partial file cleanup, or restart interrupted pulls with one click (`🔄 Retry Download`).

#### ⚙️ Settings (`settings.html`)
- **Model Storage Directory**: Configure local cache directory path.
- **Disk Offload Directory**: Custom path for layer streaming weight cache files.
- **Hugging Face Token (`HF_TOKEN`)**: Manage HF API tokens for gated models (Meta Llama 3, Gemma) and maximum CDN download speeds.

#### 📖 API Docs (`apidocs.html`)
- Interactive cURL and OpenAI Python SDK code integration guides.

---

### 6. Ollama & OpenAI REST API Compatibility

AirOllama provides full compatibility for Ollama and OpenAI REST API clients on port `11211`:

#### List Local Models (`GET /api/tags`)
```bash
curl http://localhost:11211/api/tags
```

#### Generate Text Completion (`POST /api/generate`)
```bash
curl http://localhost:11211/api/generate -d '{
  "model": "gemma4:e4b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

#### Ollama Chat Completion (`POST /api/chat`)
```bash
curl http://localhost:11211/api/chat -d '{
  "model": "gemma4:e4b",
  "messages": [
    { "role": "user", "content": "Explain quantum computing in one sentence." }
  ],
  "stream": false
}'
```

#### OpenAI Compatible Chat Completion (`POST /v1/chat/completions` or `POST /api/chat/completions`)
```bash
curl http://localhost:11211/v1/chat/completions -d '{
  "model": "gemma4:e4b",
  "messages": [
    { "role": "user", "content": "Hello!" }
  ]
}'
```

#### OpenAI Python SDK Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11211/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma4:e4b",
    messages=[{"role": "user", "content": "Write a Python function to sort a list."}]
)
print(response.choices[0].message.content)
```

#### Embeddings Endpoint (`POST /v1/embeddings` & `POST /api/embeddings`)
```bash
curl http://localhost:11211/v1/embeddings -d '{
  "model": "gemma4:e4b",
  "input": "Sample text for vector embedding"
}'
```

---

## 🛠️ Project Structure

```text
AirOllama/
├── airollama/
│   ├── cli.py               # CLI entrypoint for running server & commands
│   ├── config.py            # Persistent JSON configuration manager (config.json)
│   ├── database.py          # SQLite database manager (~/.airollama/airollama.db)
│   ├── engine.py            # Core Layer-Streaming & Apple MLX Engine
│   ├── ollama_downloader.py # Hugging Face & Ollama manifest multi-threaded downloader
│   ├── ollama_registry.py   # Ollama model name resolution mapping
│   ├── server.py            # FastAPI & Uvicorn REST API server
│   ├── web_search.py        # Web search helper integration
│   └── static/
│       ├── index.html       # Main Web Dashboard layout & async template loader
│       ├── dashboard.html   # Memory stats & layer visualizer template
│       ├── playground.html  # Chat playground & RAM slider controls template
│       ├── models.html      # Multi-threaded pull manager & LED matrix template
│       ├── settings.html    # Directories & HF token settings template
│       └── apidocs.html     # API documentation & SDK code samples template
├── mac_app/
│   └── main.swift           # Native macOS AppKit + WebKit Swift application
├── build_mac_app.sh         # Script to compile native AirOllama.app bundle
├── run_server.sh            # Server launcher script
├── requirements.txt         # Python dependencies
├── config.json.example      # Sample configuration file
└── README.md                # Documentation & User Guide
```

---

## 📄 License

Distributed under the MIT License. Developed with ❤️ by **Sangili Arumugam**.
