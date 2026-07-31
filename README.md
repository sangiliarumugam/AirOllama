# 🚀 AirOllama

**AirOllama** is an open-source, memory-efficient LLM inference engine and server that combines **Ollama API compatibility** with **AirLLM-style layer streaming and disk offloading**. Built specifically for macOS (Apple Silicon Metal GPU) and CUDA devices, AirOllama allows developers and researchers to run high-parameter Large Language Models (such as 27B–70B models) on consumer hardware with strict RAM limits.

---

## 💡 Purpose & Core Architecture

Running modern 27B+ parameter models normally requires 32GB to 64GB+ of dedicated Unified Memory or VRAM. **AirOllama** bridges this gap by:

1. **AirLLM-Style Layer Streaming**: Dynamically loading and unloading transformer layer blocks between RAM, Metal VRAM, and disk memory-mapped cache files (`.offload`) via PyTorch `accelerate`, keeping RAM consumption below strict user-configured caps.
2. **Native Apple MLX Execution**: Native support for `mlx-community` 4-bit and 8-bit quantized models running natively on Apple Silicon Metal Unified Memory.
3. **Drop-in Ollama API Ecosystem**: Exposing identical REST API endpoints (`/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`) so existing Ollama frontends, SDKs, and toolchains work out-of-the-box.
4. **Rich Web Dashboard**: A visual interface with live VRAM/RAM gauges, layer visualizers, GFM Markdown chat playground, and directory configuration management.

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

### 3. Starting the Server

Launch the AirOllama server (default port `11211`):

```bash
./run_server.sh 11211
```

Or using the Python CLI:

```bash
python -m airollama.cli serve --port 11211
```

Once started, open your browser and navigate to:
👉 **`http://localhost:11211`**

---

### 4. Using the Web Dashboard

The Web Dashboard is divided into intuitive control tabs:

#### 💬 Playground (Chat & Single Prompt)
- **Interactive Chat**: Full multi-turn conversational interface with streaming tokens.
- **Markdown & Code Support**: Renders GitHub Flavored Markdown (headings, lists, blockquotes, tables) with language-tagged dark code blocks and a **1-click Copy Code** button.
- **RAM Cap Control**: Adjust the **RAM Safety Slider** or use presets (**Min RAM**, **Balanced**, **Max Speed**). When an MLX model is active, the slider auto-adjusts to `Metal VRAM (Native)`.

#### 📊 Live Memory & Layer Visualizer
- Track real-time **System RAM**, **Process Memory**, **Metal VRAM**, and **Disk Offload Cache** sizes.
- View real-time color-coded **Layer Blocks** showing which transformer decoder layers reside in RAM/VRAM versus Disk Offload.

#### 📥 Model Repository & Hugging Face Pulls
- Pull models directly from Ollama or Hugging Face (e.g. `unsloth/gemma-2-2b-it`, `mlx-community/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-mlx-8Bit`).
- Monitor live download progress bars with speed, size, and ETA metrics.

#### ⚙️ Settings Tab
- **Model Storage Directory**: Configure where downloaded models are stored on disk.
- **Disk Offload Cache Directory**: Custom directory path for memory-mapped weight offloading.
- **Hugging Face Token (`HF_TOKEN`)**: Configure HF API User Access Tokens to unlock high-speed CDN bandwidth and gated repositories (Meta Llama, Gemma).
- Settings automatically persist across restarts in `config.json`.

---

### 5. Using the Ollama REST API

AirOllama serves standard HTTP REST API endpoints for developers and external UI clients:

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

#### Chat Completion (`POST /api/chat`)
```bash
curl http://localhost:11211/api/chat -d '{
  "model": "gemma4:e4b",
  "messages": [
    { "role": "user", "content": "Explain quantum computing in one sentence." }
  ],
  "stream": false
}'
```

#### System Status & Telemetry (`GET /api/status`)
```bash
curl http://localhost:11211/api/status
```

#### Connecting Third-Party Ollama Clients
You can connect frontends like **Open WebUI**, **Chatbox**, or **AnythingLLM** by setting their Ollama Base URL to:
```text
http://localhost:11211
```

---

## 🛠️ Project Structure

```text
AirOllama/
├── airollama/
│   ├── cli.py               # CLI entrypoint for running server & commands
│   ├── config.py            # Persistent JSON configuration manager (config.json)
│   ├── engine.py            # Core Layer-Streaming & MLX Inference Engine
│   ├── ollama_downloader.py # Hugging Face & Ollama manifest model downloader
│   ├── ollama_registry.py   # Ollama model name resolution mapping
│   ├── server.py            # FastAPI & Uvicorn REST API server
│   ├── web_search.py        # Web search helper integration
│   └── static/
│       └── index.html       # Single-page Web Dashboard UI (CSS, JS, Marked.js)
├── run_server.sh            # Server launcher script
├── requirements.txt         # Python dependencies
├── config.json.example      # Sample configuration file
└── README.md                # Documentation & User Guide
```

---

## 📄 License

Distributed under the MIT License. Developed with ❤️ by **Sangili Arumugam**.
