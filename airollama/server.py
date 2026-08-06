import time
import json
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download

from airollama import __version__
from airollama.engine import AirEngine

logger = logging.getLogger("AirOllama.Server")

app = FastAPI(
    title="AirOllama Server",
    description="Layer-by-Layer On-Demand LLM Serving for macOS on Port 11211",
    version=__version__
)

# Enable CORS for web apps & local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = AirEngine()

@app.on_event("startup")
async def startup_clean_slate():
    """Ensure server starts with a 100% clean slate (0.0 GB VRAM/RAM, no models preloaded)."""
    logger.info("🧹 Startup clean slate: purging all memory and resetting layer counters...")
    engine.unload_model()

# Mount Web UI dashboard
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

def is_api_only() -> bool:
    """Check if server is running in API-only / Headless mode."""
    import sys
    import os
    if "--api-only" in sys.argv or "--no-ui" in sys.argv:
        return True
    if os.environ.get("AIROLLAMA_API_ONLY", "").lower() in ["1", "true", "yes"]:
        return True
    return False

@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/dashboard", methods=["GET", "HEAD"])
@app.api_route("/playground", methods=["GET", "HEAD"])
@app.api_route("/models", methods=["GET", "HEAD"])
@app.api_route("/agentic", methods=["GET", "HEAD"])
@app.api_route("/agent", methods=["GET", "HEAD"])
@app.api_route("/settings", methods=["GET", "HEAD"])
@app.api_route("/apidocs", methods=["GET", "HEAD"])
async def get_root(request: Request):
    if is_api_only():
        return HTMLResponse(content="Ollama is running", status_code=200)
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            content = f.read()
            headers = {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
            return HTMLResponse(content=content, headers=headers)
    return HTMLResponse(content="<h1>AirOllama Server is running on port 11211</h1>")




# --- Pydantic Data Models ---
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None
    max_ram_gb: Optional[float] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None
    max_ram_gb: Optional[float] = None
    web_search: Optional[bool] = True
    location: Optional[str] = None


class ShowRequest(BaseModel):
    name: str

class PullRequest(BaseModel):
    name: str
    stream: Optional[bool] = True
    source: Optional[str] = "auto"  # "auto", "ollama", or "huggingface"

class CancelPullRequest(BaseModel):
    name: str


class ConfigRequest(BaseModel):
    models_dir: Optional[str] = None
    hf_token: Optional[str] = None
    offload_dir: Optional[str] = None


class DeleteRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

# --- Helper Functions ---
def get_iso_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"

from airollama.config import apply_config_environment, get_config_path, save_config

@app.on_event("startup")
async def startup_event():
    apply_config_environment()

@app.get("/api/version")
async def get_version():
    return {"version": __version__}

@app.get("/api/config")
async def get_config():
    """Get active server configuration including model storage path, offload path, HF token status, and config file path."""
    token = os.environ.get("HF_TOKEN", "")
    masked_token = (token[:4] + "*" * (len(token) - 8) + token[-4:]) if len(token) > 8 else ("*" * len(token) if token else "")
    return {
        "models_dir": engine.models_dir,
        "offload_dir": engine.get_offload_dir(),
        "device": str(engine.device),
        "hf_token_set": bool(token),
        "hf_token_masked": masked_token,
        "config_path": get_config_path(),
        "version": __version__
    }

@app.post("/api/config")
async def update_config(req: ConfigRequest):
    """Update active model storage path, offload cache path, and Hugging Face API token, persisting to config file."""
    res = {"status": "success"}
    if req.models_dir and req.models_dir.strip():
        new_dir = engine.set_models_dir(req.models_dir.strip())
        res["models_dir"] = new_dir
    
    if req.offload_dir and req.offload_dir.strip():
        new_offload = engine.set_offload_dir(req.offload_dir.strip())
        res["offload_dir"] = new_offload

    if req.hf_token is not None:
        token_val = req.hf_token.strip()
        if token_val:
            os.environ["HF_TOKEN"] = token_val
            try:
                from huggingface_hub import login
                login(token=token_val, add_to_git_credential=False)
            except Exception:
                pass
            save_config({"hf_token": token_val})
            res["hf_token_set"] = True
            logger.info("Hugging Face API Token configured and saved to config file.")
        else:
            os.environ.pop("HF_TOKEN", None)
            save_config({"hf_token": ""})
            res["hf_token_set"] = False
            logger.info("Hugging Face API Token cleared.")

    return res



class LoadRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    max_ram_gb: Optional[float] = None

@app.post("/api/load")
async def load_model_endpoint(req: LoadRequest):
    """Preload model architecture into memory for instant prompt execution."""
    target_name = req.name or req.model
    if not target_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    ram_reqs = engine.get_model_ram_requirements(target_name)
    ram_cap = req.max_ram_gb if req.max_ram_gb is not None else ram_reqs["recommended_ram_gb"]
    
    success = engine.load_model(target_name, max_ram_gb=ram_cap)
    if success:
        return {
            "status": "success",
            "model": target_name,
            "memory": engine.memory_tracker.get_system_memory()
        }
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{target_name}'. Please ensure it is a valid PyTorch/HF format model.")

@app.post("/api/unload")
async def unload_active_model():
    """Unload active model from RAM/VRAM."""
    model_name = engine.current_model_name
    if not model_name:
        return {"status": "info", "message": "No model is currently loaded in memory", "unloaded": None}
    
    engine.unload_model()
    return {"status": "success", "message": f"Successfully unloaded model '{model_name}' from memory", "unloaded": model_name}



@app.post("/api/chat/cancel")
@app.post("/api/generate/cancel")
async def cancel_active_generation():
    """Cancel ongoing prompt generation."""
    engine.stop_requested = True
    logger.info("🛑 Cancel request received for active prompt generation.")
    return {"status": "success", "message": "Generation cancellation requested"}


@app.post("/api/pull/cancel")
async def cancel_pull(req: CancelPullRequest):

    """Cancel an active model pull stream and delete all partial downloads from disk."""
    import shutil
    from airollama.engine import ACTIVE_PULL_CANCELLATIONS
    raw_name = req.name.strip()
    ACTIVE_PULL_CANCELLATIONS.add(raw_name)
    
    # Remove partial download directory from disk
    safe_folder = raw_name.replace("/", "---")
    target_dir = os.path.join(engine.models_dir, safe_folder)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
            logger.info(f"Deleted partial download directory: {target_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove partial download directory {target_dir}: {e}")

    logger.info(f"Cancellation registered for pull: {raw_name}")
    return {"status": "cancelled", "name": raw_name}


@app.get("/api/tags")
async def list_models():
    """List locally cached models (Ollama format)."""
    models = engine.list_local_models()
    result = []
    for m in models:
        ram_reqs = engine.get_model_ram_requirements(m["name"])
        result.append({
            "name": m["name"],
            "model": m["name"],
            "modified_at": get_iso_timestamp(),
            "size": int(m["size_mb"] * 1024 * 1024),
            "digest": f"sha256:{m['id']}",
            "ram_requirements": ram_reqs,
            "details": {
                "parent_model": "",
                "format": "safetensors",
                "family": "transformer",
                "parameter_size": f"Min {ram_reqs['min_ram_gb']} GB RAM",
                "quantization_level": "float16/32"
            }
        })
    return {"models": result}

@app.get("/api/ram_requirements")
async def get_ram_requirements(model: str):
    """Get calculated bare minimum and recommended RAM requirements for a model."""
    if not model:
        raise HTTPException(status_code=400, detail="Model parameter is required")
    return engine.get_model_ram_requirements(model)

@app.get("/api/ps")
@app.get("/api/status")

async def get_status():
    """Get system RAM, VRAM, and active layer execution state."""
    return engine.memory_tracker.get_system_memory()

@app.post("/api/show")
async def show_model(req: ShowRequest):
    """Show details for a specific model."""
    models = engine.list_local_models()
    found = next((m for m in models if m["name"] == req.name), None)
    if not found:
        # Check if it's currently loaded
        if engine.current_model_name == req.name:
            found = {"name": req.name, "path": "active"}
        else:
            raise HTTPException(status_code=404, detail=f"Model '{req.name}' not found")
    
    return {
        "modelfile": f"# AirOllama Modelfile for {req.name}\nFROM {req.name}\nSYSTEM You are a helpful assistant.",
        "parameters": "stop \"<|im_end|>\"",
        "template": "{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT: ",
        "details": {
            "format": "safetensors",
            "family": "transformer",
            "parameter_size": "Layer-by-Layer On-Demand"
        }
    }

@app.post("/api/pull")
async def pull_model(req: PullRequest, request: Request, background_tasks: BackgroundTasks):
    """Pull model from Ollama library tag or Hugging Face hub into local AirOllama model cache."""
    from airollama.ollama_registry import resolve_ollama_to_hf, parse_ollama_name
    from airollama.ollama_downloader import fetch_manifest, download_ollama_model, download_hf_model_in_parallel, DEFAULT_MAX_WORKERS
    from airollama.engine import ACTIVE_PULL_CANCELLATIONS

    raw_input = req.name.strip()
    source = (req.source or "auto").lower().strip()
    ACTIVE_PULL_CANCELLATIONS.discard(raw_input)
    clean_repo, tag = parse_ollama_name(raw_input)
    resolved_hf = resolve_ollama_to_hf(raw_input)
    
    safe_folder = raw_input.replace("/", "---")
    target_dir = f"{engine.models_dir}/{safe_folder}"

    async def generate_pull_stream() -> AsyncGenerator[str, None]:
        try:
            # Mode A: Explicit Ollama Registry
            if source == "ollama":
                success, manifest, found_path = fetch_manifest(clean_repo, tag)
                if success:
                    yield json.dumps({"status": f"⚡ Pulling directly from Ollama Registry ({clean_repo}:{tag})..."}) + "\n"
                    for chunk in download_ollama_model(raw_input, target_dir, max_workers=DEFAULT_MAX_WORKERS):
                        if await request.is_disconnected():
                            logger.info("Client disconnected during model pull.")
                            break
                        yield json.dumps(chunk) + "\n"
                    return
                else:
                    yield json.dumps({"error": f"Model '{raw_input}' not found on Ollama Registry (registry.ollama.ai)"}) + "\n"
                    return

            # Mode B: Explicit Hugging Face Hub
            elif source == "huggingface":
                repo_target = resolved_hf if "/" in resolved_hf else clean_repo
                yield json.dumps({"status": f"⚡ Pulling directly from Hugging Face Hub ({repo_target})..."}) + "\n"
                for chunk in download_hf_model_in_parallel(raw_input, repo_target, target_dir, max_workers=DEFAULT_MAX_WORKERS):
                    if await request.is_disconnected():
                        logger.info("Client disconnected during model pull.")
                        break
                    yield json.dumps(chunk) + "\n"
                return

            # Mode C: Auto Mode (Smart Dual-Source Fallback)
            else:
                if resolved_hf != clean_repo and "/" in resolved_hf:
                    yield json.dumps({"status": f"⚡ Launching multi-core parallel download for {resolved_hf}..."}) + "\n"
                    try:
                        for chunk in download_hf_model_in_parallel(raw_input, resolved_hf, target_dir, max_workers=DEFAULT_MAX_WORKERS):
                            if await request.is_disconnected():
                                logger.info("Client disconnected during model pull.")
                                break
                            yield json.dumps(chunk) + "\n"
                        return
                    except Exception as e:
                        logger.warning(f"HF download for {resolved_hf} failed ({e}), falling back to Ollama registry...")

                success, manifest, found_path = fetch_manifest(clean_repo, tag)
                if success:
                    yield json.dumps({"status": f"⚡ Launching {DEFAULT_MAX_WORKERS} SIMULTANEOUS parallel range streams for ({clean_repo}:{tag})..."}) + "\n"
                    for chunk in download_ollama_model(raw_input, target_dir, max_workers=DEFAULT_MAX_WORKERS):
                        if await request.is_disconnected():
                            logger.info("Client disconnected during model pull.")
                            break
                        yield json.dumps(chunk) + "\n"
                    return

                yield json.dumps({"status": f"⚡ Launching multi-core parallel download for {clean_repo}..."}) + "\n"
                for chunk in download_hf_model_in_parallel(raw_input, clean_repo, target_dir, max_workers=DEFAULT_MAX_WORKERS):
                    if await request.is_disconnected():
                        logger.info("Client disconnected during model pull.")
                        break
                    yield json.dumps(chunk) + "\n"
                return



        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            logger.info("Client stream connection closed.")

    if req.stream:
        return StreamingResponse(generate_pull_stream(), media_type="application/x-ndjson")
    else:
        # Non-streaming implementation
        if source == "ollama":
            success, manifest, found_path = fetch_manifest(clean_repo, tag)
            if success:
                last_chunk = {}
                for chunk in download_ollama_model(raw_input, target_dir, max_workers=32):
                    last_chunk = chunk
                if "error" in last_chunk:
                    raise HTTPException(status_code=500, detail=last_chunk["error"])
                return {"status": "success"}
            else:
                raise HTTPException(status_code=404, detail=f"Model '{raw_input}' not found on Ollama Registry")
        else:
            last_chunk = {}
            for chunk in download_hf_model_in_parallel(raw_input, resolved_hf, target_dir, max_workers=32):
                last_chunk = chunk
            if "error" in last_chunk:
                raise HTTPException(status_code=500, detail=last_chunk["error"])
            return {"status": "success"}

            repo_target = resolved_hf if "/" in resolved_hf else clean_repo
            try:
                snapshot_download(
                    repo_id=repo_target,
                    local_dir=target_dir,
                    max_workers=16,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pt", "*.pth", "onnx/*"]
                )
                return {"status": "success"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))





@app.delete("/api/delete")
async def delete_model_endpoint(req: DeleteRequest):
    """Delete model completely from local cache (Ollama spec)."""
    target_name = req.name or req.model
    if not target_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    success = engine.delete_model(target_name)
    if success:
        return {"status": "success", "message": f"Successfully deleted model '{target_name}'"}
    else:
        raise HTTPException(status_code=404, detail=f"Model '{target_name}' not found or already deleted")



@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    """Generate completion (Ollama spec)."""
    options = req.options or {}
    max_tokens = options.get("num_predict", 2048)
    temp = options.get("temperature", 0.7)
    top_p = options.get("top_p", 0.9)
    stop = options.get("stop", [])
    max_ram_gb = options.get("max_ram_gb", options.get("ram_cap_gb", None))
    if max_ram_gb is not None:
        try:
            max_ram_gb = float(max_ram_gb)
        except (ValueError, TypeError):
            max_ram_gb = None

    prompt = req.prompt
    if req.system:
        prompt = f"System: {req.system}\n\nUser: {prompt}\nAssistant:"

    def generate_iterator():
        try:
            for chunk in engine.generate_stream(
                prompt=prompt,
                model_name=req.model,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop_sequences=stop,
                max_ram_gb=max_ram_gb
            ):

                if "error" in chunk:
                    yield json.dumps(chunk) + "\n"
                    break
                
                payload = {
                    "model": req.model,
                    "created_at": get_iso_timestamp(),
                    "response": chunk.get("response", ""),
                    "done": chunk.get("done", False)
                }
                if chunk.get("done"):
                    payload["eval_count"] = chunk.get("eval_count", 0)
                    payload["eval_duration"] = chunk.get("eval_duration_sec", 0)
                    payload["memory"] = chunk.get("memory", {})

                yield json.dumps(payload) + "\n"
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            logger.info("Client disconnected during generate stream.")

    if req.stream:
        return StreamingResponse(generate_iterator(), media_type="application/x-ndjson")
    else:
        full_text = ""
        last_chunk = {}
        for chunk in engine.generate_stream(
            prompt=prompt,
            model_name=req.model,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p
        ):
            if "error" in chunk:
                raise HTTPException(status_code=500, detail=chunk["error"])
            full_text += chunk.get("response", "")
            last_chunk = chunk

        return {
            "model": req.model,
            "created_at": get_iso_timestamp(),
            "response": full_text,
            "done": True,
            "eval_count": last_chunk.get("eval_count", 0),
            "eval_duration": last_chunk.get("eval_duration_sec", 0),
            "memory": last_chunk.get("memory", {})
        }

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Multi-turn chat completion (Ollama spec with Live Web Search & System Location)."""
    from airollama.web_search import perform_web_search, format_search_context, get_live_weather, get_server_ip_location

    options = req.options or {}
    max_tokens = options.get("num_predict", 2048)
    temp = options.get("temperature", 0.7)
    top_p = options.get("top_p", 0.9)
    max_ram_gb = req.max_ram_gb if req.max_ram_gb is not None else options.get("max_ram_gb", options.get("ram_cap_gb", None))
    if max_ram_gb is not None:
        try:
            max_ram_gb = float(max_ram_gb)
        except (ValueError, TypeError):
            max_ram_gb = None
    target_model = req.model
    if not target_model or not target_model.strip():
        if engine.current_model_name:
            target_model = engine.current_model_name
        else:
            local_models = engine.list_local_models()
            if local_models:
                target_model = local_models[0]["name"]
            else:
                target_model = "gemma4:e4b"
    req.model = target_model

    messages = [m.dict() for m in req.messages]

    # Resolve location (from payload or automatic server IP fallback)
    effective_location = req.location.strip() if (req.location and req.location.strip() and req.location != "Location unavailable") else get_server_ip_location()

    # Prepend System Location
    loc_str = f"[📍 User System Location: {effective_location}]"
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{loc_str}\n{messages[0]['content']}"
    else:
        messages.insert(0, {"role": "system", "content": loc_str})

    # Perform Live Web Search if enabled
    user_prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_prompt = m.get("content", "")
            break

    simple_greetings = {"hi", "hello", "hey", "howdy", "sup", "yo", "good morning", "good evening", "test", "hi there"}
    clean_user_prompt = user_prompt.strip().lower()
    if req.web_search and user_prompt.strip() and clean_user_prompt not in simple_greetings and len(clean_user_prompt) > 3:

        weather_info = ""
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "how hot", "how cold", "climate"]
        if any(k in user_prompt.lower() for k in weather_keywords):
            weather_info = get_live_weather(effective_location)

        search_results = perform_web_search(user_prompt.strip(), location_str=effective_location, max_results=4)
        if search_results or weather_info:
            search_context = format_search_context(user_prompt.strip(), search_results, weather_info=weather_info)
            if messages:
                messages[-1]["content"] = f"{messages[-1]['content']}\n{search_context}"

    prompt = engine.format_chat_prompt(messages)




    def chat_iterator():
        try:
            for chunk in engine.generate_stream(
                prompt=prompt,
                model_name=req.model,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                max_ram_gb=max_ram_gb
            ):

                if "error" in chunk:
                    yield json.dumps(chunk) + "\n"
                    break
                
                text = chunk.get("response", "")
                if text.startswith("⌛ Preparing model"):
                    continue

                payload = {
                    "model": req.model,
                    "created_at": get_iso_timestamp(),
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "done": chunk.get("done", False)
                }
                if chunk.get("done"):
                    payload["eval_count"] = chunk.get("eval_count", 0)
                    payload["memory"] = chunk.get("memory", {})

                yield json.dumps(payload) + "\n"
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            logger.info("Client disconnected during chat stream.")

    if req.stream:
        return StreamingResponse(chat_iterator(), media_type="application/x-ndjson")
    else:
        full_content = ""
        last_chunk = {}
        for chunk in engine.generate_stream(
            prompt=prompt,
            model_name=req.model,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            max_ram_gb=max_ram_gb
        ):

            if "error" in chunk:
                raise HTTPException(status_code=500, detail=chunk["error"])
            full_content += chunk.get("response", "")
            last_chunk = chunk

        return {
            "model": req.model,
            "created_at": get_iso_timestamp(),
            "message": {
                "role": "assistant",
                "content": full_content
            },
            "done": True,
            "eval_count": last_chunk.get("eval_count", 0),
            "memory": last_chunk.get("memory", {})
        }


# --- OpenAI Compatibility Endpoints ---

@app.get("/v1/models")
@app.get("/api/v1/models")
async def openai_list_models():
    models = engine.list_local_models()
    data = []
    for m in models:
        data.append({
            "id": m["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "air-ollama"
        })
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
@app.post("/api/v1/chat/completions")
@app.post("/v1/chat")
async def openai_chat_completions(req: OpenAIChatCompletionRequest):
    messages = [m.dict() for m in req.messages]
    if not engine.current_model_name or engine.current_model_name != req.model:
        success = engine.load_model(req.model)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to load model {req.model}")

    prompt = engine.format_chat_prompt(messages)

    if req.stream:
        def stream_generator():
            created = int(time.time())
            for chunk in engine.generate_stream(
                prompt=prompt,
                model_name=req.model,
                max_new_tokens=req.max_tokens or 256,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9
            ):
                payload = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.get("response", "")},
                        "finish_reason": "stop" if chunk.get("done") else None
                    }]
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        full_content = ""
        for chunk in engine.generate_stream(
            prompt=prompt,
            model_name=req.model,
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9
        ):
            full_content += chunk.get("response", "")

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
        }


class EmbeddingRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    input: Optional[Any] = None


@app.post("/api/embeddings")
@app.post("/api/embed")
async def api_embeddings(req: EmbeddingRequest):
    """Ollama-compatible embeddings endpoint for OpenCode Agent and AI tools."""
    text = req.prompt or (req.input[0] if isinstance(req.input, list) and req.input else str(req.input or ""))
    try:
        vec = engine.generate_embeddings(req.model, text)
        return {"embedding": vec, "embeddings": [vec]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
@app.post("/api/v1/embeddings")
async def openai_embeddings(req: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    text = req.prompt or (req.input[0] if isinstance(req.input, list) and req.input else str(req.input or ""))
    try:
        vec = engine.generate_embeddings(req.model, text)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": vec,
                    "index": 0
                }
            ],
            "model": req.model,
            "usage": {
                "prompt_tokens": len(text.split()),
                "total_tokens": len(text.split())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Coding Agent Workspace APIs ---

class AgentFileSaveRequest(BaseModel):
    path: str
    content: str

class AgentExecRequest(BaseModel):
    command: str

from airollama import database as db

def resolve_base_dir(path: Optional[str] = None, project_id: Optional[int] = None) -> str:
    import os
    if project_id is not None:
        proj = db.get_project(project_id)
        if proj and os.path.exists(proj["path"]):
            return proj["path"]
    if path and os.path.isabs(path) and os.path.exists(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@app.get("/api/agent/tree")
async def get_workspace_tree(path: Optional[str] = None, project_id: Optional[int] = None):
    """List project workspace directory tree for Coding Agent IDE."""
    import os
    base_dir = resolve_base_dir(path, project_id)
    target_dir = os.path.abspath(path) if (path and os.path.exists(path)) else base_dir
    
    if not target_dir.startswith(base_dir) and not os.path.exists(target_dir):
        target_dir = base_dir

    items = []
    try:
        for entry in os.listdir(target_dir):
            if entry.startswith(".") or entry in ["venv", "__pycache__", "dist", "build", "models", "node_modules"]:
                continue
            full_p = os.path.join(target_dir, entry)
            is_dir = os.path.isdir(full_p)
            rel_p = os.path.relpath(full_p, base_dir)
            size = os.path.getsize(full_p) if not is_dir else 0
            items.append({
                "name": entry,
                "path": rel_p,
                "abs_path": full_p,
                "is_dir": is_dir,
                "size": size
            })
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return {"root": base_dir, "current": target_dir, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agent/file")
async def get_workspace_file(path: str, project_id: Optional[int] = None):
    """Read a workspace file content for Coding Agent IDE."""
    import os
    base_dir = resolve_base_dir(None, project_id)
    file_path = os.path.abspath(os.path.join(base_dir, path)) if not os.path.isabs(path) else path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"path": path, "abs_path": file_path, "content": content, "size": len(content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/save")
async def save_workspace_file(req: AgentFileSaveRequest, project_id: Optional[int] = None):
    """Save updated file content back to workspace."""
    import os
    base_dir = resolve_base_dir(None, project_id)
    file_path = os.path.abspath(os.path.join(base_dir, req.path)) if not os.path.isabs(req.path) else req.path
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.content)
        return {"success": True, "path": req.path, "bytes_written": len(req.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/exec")
async def exec_workspace_command(req: AgentExecRequest, project_id: Optional[int] = None):
    """Execute command in workspace terminal context for Coding Agent."""
    import subprocess
    import os
    base_dir = resolve_base_dir(None, project_id)
    try:
        proc = subprocess.run(
            req.command,
            shell=True,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "command": req.command,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except subprocess.TimeoutExpired:
        return {"command": req.command, "exit_code": -1, "stdout": "", "stderr": "Command timed out after 30 seconds"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/utils/select_folder")
async def select_native_folder():
    """Trigger native OS folder picker dialog (macOS osascript / zenity) and return selected path."""
    import sys, subprocess, os
    selected_path = None
    if sys.platform == "darwin":
        cmd = 'osascript -e \'posix path of (choose folder with prompt "Select Project Directory")\''
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and res.stdout.strip():
                selected_path = res.stdout.strip().rstrip('/')
        except Exception as e:
            pass
    elif sys.platform.startswith("linux"):
        cmd = 'zenity --file-selection --directory --title="Select Project Directory"'
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and res.stdout.strip():
                selected_path = res.stdout.strip().rstrip('/')
        except Exception:
            pass
    
    if selected_path:
        folder_name = os.path.basename(selected_path) or "New Project"
        return {"cancelled": False, "path": selected_path, "name": folder_name}
    return {"cancelled": True, "path": "", "name": ""}


# --- Project & Conversation Database Endpoints ---

class CreateProjectRequest(BaseModel):
    name: str
    path: str

class CreateConversationRequest(BaseModel):
    project_id: Optional[int] = None
    title: Optional[str] = "New Coding Thread"
    model: Optional[str] = ""
    role: Optional[str] = ""

class SaveMessageRequest(BaseModel):
    role: str
    content: str
    thought: Optional[str] = None

@app.get("/api/projects")
async def api_list_projects():
    return {"projects": db.list_projects()}

@app.post("/api/projects")
async def api_create_project(req: CreateProjectRequest):
    if not req.name or not req.path:
        raise HTTPException(status_code=400, detail="Name and path are required")
    proj = db.create_project(req.name, req.path)
    return proj

@app.delete("/api/projects/{project_id}")
async def api_delete_project(project_id: int):
    db.delete_project(project_id)
    return {"success": True}

@app.get("/api/conversations")
async def api_list_conversations(project_id: Optional[int] = None):
    return {"conversations": db.list_conversations(project_id)}

@app.post("/api/conversations")
async def api_create_conversation(req: CreateConversationRequest):
    conv = db.create_conversation(req.project_id, req.title or "New Coding Thread", req.model or "", req.role or "")
    return conv

class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    role: Optional[str] = None

@app.put("/api/conversations/{conversation_id}")
async def api_update_conversation(conversation_id: str, req: UpdateConversationRequest):
    updated = db.update_conversation(conversation_id, req.title, req.model, req.role)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return updated

@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    db.delete_conversation(conversation_id)
    return {"success": True}

@app.get("/api/conversations/{conversation_id}/messages")
async def api_get_messages(conversation_id: str):
    return {"messages": db.get_conversation_messages(conversation_id)}

@app.post("/api/conversations/{conversation_id}/messages")
async def api_save_message(conversation_id: str, req: SaveMessageRequest):
    msg = db.add_message(conversation_id, req.role, req.content, req.thought)
    return msg
