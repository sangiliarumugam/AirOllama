import os
import json
import logging
import requests
from typing import Generator, Dict, Any, Tuple

logger = logging.getLogger("AirOllama.Registry")

OLLAMA_REGISTRY_BASE = "https://registry.ollama.ai/v2"

# Standard mapping of Ollama tags to Hugging Face safetensors repositories for layer streaming
OLLAMA_HF_MAP = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama:latest": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2:latest": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2:1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:latest": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral:latest": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma": "unsloth/gemma-2-2b-it",
    "gemma2": "unsloth/gemma-2-2b-it",
    "gemma2:latest": "unsloth/gemma-2-2b-it",
    "gemma2:2b": "unsloth/gemma-2-2b-it",
    "gemma:2b": "unsloth/gemma-2-2b-it",
    "gemma:7b": "unsloth/gemma-2-9b-it",
    "gemma:27b": "unsloth/gemma-2-27b-it",
    "gemma2:27b": "unsloth/gemma-2-27b-it",
    "gemma:latest": "unsloth/gemma-2-2b-it",
    "gemma4": "unsloth/gemma-2-2b-it",
    "gemma4:latest": "unsloth/gemma-2-2b-it",
    "gemma4:31b-mlx": "unsloth/gemma-2-27b-it",
    "gemma4:31b": "unsloth/gemma-2-27b-it",
    "gemma:31b-mlx": "unsloth/gemma-2-27b-it",
    "gemma2:4b": "unsloth/gemma-2-2b-it",



    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi3:latest": "microsoft/Phi-3-mini-4k-instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "smollm": "HuggingFaceTB/SmolLM-135M-Instruct",
    "smollm:135m": "HuggingFaceTB/SmolLM-135M-Instruct",
    "smollm:360m": "HuggingFaceTB/SmolLM-360M-Instruct",
    "smollm:1.7b": "HuggingFaceTB/SmolLM-1.7B-Instruct"
}


def parse_ollama_name(name: str) -> Tuple[str, str]:
    """Parse 'model' or 'model:tag' or full URLs into (clean_repo, tag)."""
    raw = name.strip()
    if "ollama.com/" in raw:
        raw = raw.split("ollama.com/")[-1].strip("/")
        if raw.startswith("library/"):
            raw = raw.replace("library/", "")
    elif "huggingface.co/" in raw:
        raw = raw.split("huggingface.co/")[-1].strip("/")

    tag = "latest"
    if ":" in raw:
        parts = raw.split(":", 1)
        raw = parts[0]
        tag = parts[1]

    return raw.strip(), tag.strip()

def resolve_ollama_to_hf(name: str) -> str:
    """Resolve an Ollama tag or model string to HF repo ID."""
    if not name:
        return ""
    
    clean_name = name.strip().lower()
    if clean_name in OLLAMA_HF_MAP:
        return OLLAMA_HF_MAP[clean_name]

    clean_repo, tag = parse_ollama_name(name)
    clean_key = clean_repo.lower().replace(" ", "")

    # Check with tag combined e.g. "gemma4:31b-mlx" or "llama3.2:3b" FIRST
    tag_key = f"{clean_key}:{tag}".lower()
    if tag_key in OLLAMA_HF_MAP:
        return OLLAMA_HF_MAP[tag_key]

    # Check base model key
    if clean_key in OLLAMA_HF_MAP:
        return OLLAMA_HF_MAP[clean_key]

    return name




def fetch_ollama_manifest(model_name: str, tag: str = "latest") -> Tuple[bool, Dict[str, Any]]:
    """Fetch OCI manifest from official registry.ollama.ai."""
    url = f"{OLLAMA_REGISTRY_BASE}/library/{model_name}/manifests/{tag}"
    headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return True, res.json()
        return False, {"error": f"Ollama registry returned status {res.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def download_ollama_blob(model_name: str, digest: str, target_path: str) -> Generator[Dict[str, Any], None, None]:
    """Stream download blob file from registry.ollama.ai."""
    url = f"{OLLAMA_REGISTRY_BASE}/library/{model_name}/blobs/{digest}"
    try:
        res = requests.get(url, stream=True, timeout=30)
        total_size = int(res.headers.get("content-length", 0))
        downloaded = 0

        with open(target_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = round((downloaded / total_size) * 100, 1) if total_size > 0 else 0
                    yield {
                        "status": f"downloading blob {digest[:12]}...",
                        "completed": downloaded,
                        "total": total_size,
                        "percent": percent
                    }
    except Exception as e:
        yield {"error": f"Failed to download blob {digest}: {e}"}
