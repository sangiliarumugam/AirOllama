import os
import gc
import time
import json
import logging
from typing import Generator, Dict, Any, List, Optional, Callable

# Ensure HF cache stays within workspace
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(base_dir, ".hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import psutil


try:
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
except ImportError:
    AutoTokenizer, AutoConfig, AutoModelForCausalLM = None, None, None

logger = logging.getLogger("AirOllama.Engine")
logging.basicConfig(level=logging.INFO)

import concurrent.futures

class AsyncLayerPrefetcher:
    """
    Asynchronous layer prefetcher for Accelerate disk-offloaded models (Solution 4 & 5).
    Pre-loads OS kernel page cache for Layer N+1 / N+2 from SSD while Layer N executes on GPU/CPU.
    """
    def __init__(self, offload_dir: str):
        self.offload_dir = offload_dir
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="airollama_prefetch")
        self.prefetched_layers = set()

    def prefetch_layer(self, layer_idx: int):
        """Asynchronously pre-read offload files for upcoming layer."""
        if not self.offload_dir or not os.path.exists(self.offload_dir):
            return
        if layer_idx in self.prefetched_layers:
            return
        self.prefetched_layers.add(layer_idx)
        self.executor.submit(self._do_prefetch, layer_idx)

    def _do_prefetch(self, layer_idx: int):
        try:
            target_str = f"layers.{layer_idx}"
            for root, _, files in os.walk(self.offload_dir):
                for f in files:
                    if target_str in f:
                        fp = os.path.join(root, f)
                        if os.path.exists(fp):
                            with open(fp, "rb") as fh:
                                # Warm OS page cache by reading chunks asynchronously into kernel memory
                                while fh.read(1024 * 1024 * 4):  # 4MB chunks
                                    pass
        except Exception:
            pass

    def shutdown(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

class LayerMemoryTracker:
    """Utility to track memory and current layer status during AirLLM inference."""
    def __init__(self):
        self.current_layer: int = -1
        self.total_layers: int = 0
        self.active_model: str = ""
        self.peak_ram_mb: float = 0.0
        self.offload_active: bool = False
        self.disk_layers_count: int = 0
        self.ram_layers_count: int = 0
        self.base_ram_count: int = 0
        self.model_size_bytes: int = 0
        # Snapshot baseline process RAM (engine + Python overhead, before any model load)
        try:
            self.baseline_process_ram_mb: float = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            self.baseline_process_ram_mb = 0.0

    def get_system_memory(self) -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        proc_mem = process.memory_info().rss / (1024 * 1024)
        
        gpu_mem_mb = 0.0
        if torch.backends.mps.is_available():
            try:
                gpu_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                if gpu_mem_mb < 1.0 and hasattr(torch.mps, "driver_allocated_memory"):
                    gpu_mem_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
        elif torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        offload_bytes = 0
        from airollama.config import get_offload_dir
        offload_dir = get_offload_dir()
        if os.path.exists(offload_dir):
            for root, _, files in os.walk(offload_dir):
                for f in files:
                    try:
                        offload_bytes += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass

        has_active = bool(self.active_model)
        if not has_active:
            ram_count = 0
            disk_count = 0
            total_l = 0
            is_paging = False
            base_ram = 0
        else:
            total_l = self.total_layers
            is_paging = self.offload_active and (self.current_layer >= self.base_ram_count)
            if not self.offload_active:
                disk_count = 0
                ram_count = total_l
                base_ram = total_l
            else:
                base_ram = self.base_ram_count
                if is_paging:
                    # Active layer is temporarily loaded into RAM from Disk for execution
                    ram_count = min(total_l, base_ram + 1)
                    disk_count = max(0, total_l - ram_count)
                else:
                    disk_count = self.disk_layers_count
                    ram_count = self.ram_layers_count if self.ram_layers_count > 0 else max(0, total_l - disk_count)

        baseline = self.baseline_process_ram_mb
        model_ram_mb = max(0.0, proc_mem - baseline)

        if has_active and self.offload_active and total_l > 0 and disk_count > 0 and self.model_size_bytes > 0:
            calc_offload_bytes = int((disk_count / total_l) * self.model_size_bytes)
            offload_bytes = max(offload_bytes, calc_offload_bytes)

        return {
            "total_ram_gb": round(mem.total / (1024**3), 2),
            "available_ram_gb": round(mem.available / (1024**3), 2),
            "used_ram_percent": mem.percent,
            "process_ram_mb": round(proc_mem, 2),
            "process_ram_gb": round(proc_mem / 1024, 2),
            "model_ram_mb": round(model_ram_mb, 2),
            "model_ram_gb": round(model_ram_mb / 1024, 2),
            "engine_ram_mb": round(baseline, 2),
            "engine_ram_gb": round(baseline / 1024, 2),
            "vram_mb": round(gpu_mem_mb, 2),
            "vram_gb": round(gpu_mem_mb / 1024, 2),
            "current_layer": self.current_layer,
            "total_layers": total_l,
            "active_model": self.active_model,
            "offload_active": self.offload_active if has_active else False,
            "disk_layers_count": disk_count,
            "ram_layers_count": ram_count,
            "base_ram_count": base_ram,
            "is_paging": is_paging,
            "offload_size_mb": round(offload_bytes / (1024 * 1024), 1),
            "offload_size_gb": round(offload_bytes / (1024**3), 2)
        }





# Global set for active pull cancellation signals
ACTIVE_PULL_CANCELLATIONS = set()

class AirEngine:
    """
    On-Demand Layer-by-Layer LLM Inference Engine.
    Executes model layers sequentially, unloading weights after each layer's forward pass
    to drastically reduce peak RAM/VRAM footprint on macOS.
    """
    def __init__(self, models_dir: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from airollama.config import load_config
        cfg = load_config()
        saved_dir = cfg.get("models_dir", "")
        self.models_dir = models_dir or (saved_dir if saved_dir else os.path.join(base_dir, "models"))
        os.makedirs(self.models_dir, exist_ok=True)

        self.device = self._select_device()
        self.memory_tracker = LayerMemoryTracker()
        
        self.current_model_name: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.config = None
        self.is_mlx_model: bool = False
        self.mlx_model = None
        self.mlx_tokenizer = None
        self.layer_callback: Optional[Callable[[int, int], None]] = None
        self.unload_model()

        logger.info(f"Initialized AirEngine on device: {self.device} (models_dir: {self.models_dir})")

    def set_models_dir(self, new_dir: str) -> str:
        """Update active model storage directory and persist in config file."""
        clean_path = os.path.abspath(os.path.expanduser(new_dir))
        os.makedirs(clean_path, exist_ok=True)
        self.models_dir = clean_path
        from airollama.config import save_config
        save_config({"models_dir": self.models_dir})
        logger.info(f"Updated AirEngine models_dir to: {self.models_dir}")
        return self.models_dir

    def get_offload_dir(self) -> str:
        """Returns active persistent offload cache directory."""
        from airollama.config import get_offload_dir
        return get_offload_dir()

    def set_offload_dir(self, new_dir: str) -> str:
        """Update active offload cache directory and persist in config file."""
        clean_path = os.path.abspath(os.path.expanduser(new_dir.strip()))
        os.makedirs(clean_path, exist_ok=True)
        from airollama.config import save_config
        save_config({"offload_dir": clean_path})
        logger.info(f"Updated AirEngine offload_dir to: {clean_path}")
        return clean_path



    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def list_local_models(self) -> List[Dict[str, Any]]:
        """List all valid and complete models cached or registered locally."""
        models = []
        if os.path.exists(self.models_dir):
            for entry in os.listdir(self.models_dir):
                if entry.startswith("."):
                    continue
                full_path = os.path.join(self.models_dir, entry)
                if os.path.isdir(full_path):
                    has_config = os.path.exists(os.path.join(full_path, "config.json"))
                    has_manifest = os.path.exists(os.path.join(full_path, "manifest.json"))
                    if not has_config and not has_manifest:
                        continue  # Skip incomplete or failed download directories
                    models.append({
                        "name": entry.replace("---", "/"),
                        "id": entry,
                        "path": full_path,
                        "size_mb": round(self._get_dir_size(full_path) / (1024 * 1024), 2)
                    })
        return models


    def _get_dir_size(self, path: str) -> int:
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Load tokenizer and prepare model architecture for layer-streaming."""
        from airollama.ollama_registry import resolve_ollama_to_hf
    def get_model_ram_requirements(self, model_name: str) -> Dict[str, float]:
        """
        Calculate the bare minimum and recommended RAM required for a model to function cleanly.
        """
        from airollama.ollama_registry import resolve_ollama_to_hf
        resolved_name = resolve_ollama_to_hf(model_name)
        safe_folder = model_name.replace("/", "---")
        safe_resolved = resolved_name.replace("/", "---")
        local_path = os.path.join(self.models_dir, safe_folder)
        resolved_local_path = os.path.join(self.models_dir, safe_resolved)

        target_path = local_path if os.path.exists(local_path) else resolved_local_path
        size_bytes = self._get_dir_size(target_path) if os.path.exists(target_path) else 0
        size_gb = size_bytes / (1024 ** 3)

        if size_gb <= 0:
            size_gb = 4.0

        # Bare minimum RAM to function without thrashing disk I/O (embeddings + KV cache + active layers)
        min_ram_gb = round(max(2.5, size_gb * 0.45), 1)

        # Recommended RAM for smooth performance
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        recommended_ram_gb = round(min(size_gb + 1.0, sys_ram_gb * 0.8), 1)

        return {
            "model_size_gb": round(size_gb, 2),
            "min_ram_gb": min_ram_gb,
            "recommended_ram_gb": recommended_ram_gb
        }

    def load_model(self, model_name: str, max_ram_gb: Optional[float] = None) -> bool:
        """
        Load tokenizer and model for layer-by-layer inference.
        """
        self.stop_requested = False
        if not model_name:
            return False

        from airollama.ollama_registry import resolve_ollama_to_hf

        reqs = self.get_model_ram_requirements(model_name)
        min_ram_required = reqs["min_ram_gb"]

        if max_ram_gb is None:
            max_ram_gb = reqs["recommended_ram_gb"]
            logger.info(f"⚡ RAM Cap defaulted to recommended high-speed size: {max_ram_gb} GB")
        elif max_ram_gb < min_ram_required:
            logger.info(f"⚡ RAM Cap ({max_ram_gb} GB) is below minimum ({min_ram_required} GB). Auto-adjusting to minimum required: {min_ram_required} GB")
            max_ram_gb = min_ram_required


        if self.current_model_name == model_name and self.model is not None and getattr(self, "current_ram_cap", None) == max_ram_gb:
            logger.info(f"⚡ Model '{model_name}' is already loaded in memory [RAM Cap: {max_ram_gb} GB]. Skipping reload.")
            return True


        resolved_name = resolve_ollama_to_hf(model_name)
        logger.info(f"Preparing model for layer-by-layer inference: '{model_name}' (Resolved: '{resolved_name}')")

        # Automatically purge previous model weights from RAM/GPU before loading/reloading model with new RAM limit
        self.unload_model()


        safe_folder = model_name.replace("/", "---")
        safe_resolved = resolved_name.replace("/", "---")

        local_path = os.path.join(self.models_dir, safe_folder)
        resolved_local_path = os.path.join(self.models_dir, safe_resolved)

        if os.path.exists(local_path):
            target_path = local_path
        elif os.path.exists(resolved_local_path):
            target_path = resolved_local_path
        else:
            target_path = resolved_name

        # Detect Apple MLX affine quantized models and load via mlx_lm natively
        is_mlx = False
        config_path = os.path.join(target_path, "config.json") if os.path.exists(target_path) else ""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as cf:
                    cfg_json = json.load(cf)
                q_info = cfg_json.get("quantization") or cfg_json.get("quantization_config") or {}
                if isinstance(q_info, dict) and (q_info.get("mode") == "affine" or "mlx" in model_name.lower()):
                    is_mlx = True
            except Exception:
                pass

        if is_mlx or "mlx" in model_name.lower():
            try:
                from mlx_lm import load as mlx_load
                logger.info(f"⚡ Loading Apple MLX native model: '{model_name}' from {target_path}...")
                self.unload_model()
                self.mlx_model, self.mlx_tokenizer = mlx_load(target_path)
                self.is_mlx_model = True
                self.current_model_name = model_name
                self.current_ram_cap = max_ram_gb

                num_layers = getattr(getattr(self.mlx_model, "args", None), "num_hidden_layers", 64)
                self.memory_tracker.total_layers = num_layers
                self.memory_tracker.active_model = model_name
                self.memory_tracker.ram_layers_count = num_layers
                self.memory_tracker.disk_layers_count = 0
                self.memory_tracker.offload_active = False
                logger.info(f"✅ Successfully loaded Apple MLX model '{model_name}' on Metal GPU!")
                return True
            except Exception as mlx_err:
                logger.error(f"Failed to load MLX model '{model_name}': {mlx_err}")
                return False

        # 1. Tokenizer Loading with Fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        except Exception as tok_err:
            logger.warning(f"Local tokenizer load for {target_path} failed ({tok_err}), trying resolved model '{resolved_name}'...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_name, trust_remote_code=True)
            except Exception as tok_err2:
                logger.warning(f"Resolved tokenizer load failed ({tok_err2}), using default fallback tokenizer 'Qwen/Qwen2.5-0.5B-Instruct'...")
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Config Loading with Fallback
        try:
            self.config = AutoConfig.from_pretrained(target_path, trust_remote_code=True)
        except Exception as cfg_err:
            logger.warning(f"Local config load for {target_path} failed ({cfg_err}), trying resolved model '{resolved_name}'...")
            try:
                self.config = AutoConfig.from_pretrained(resolved_name, trust_remote_code=True)
            except Exception:
                self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

        # Count layers in config
        num_layers = getattr(self.config, "num_hidden_layers", 
                     getattr(self.config, "n_layer", 32))
        self.memory_tracker.total_layers = num_layers
        self.memory_tracker.active_model = model_name

        # Check system RAM safety & user RAM cap
        available_ram_bytes = psutil.virtual_memory().available
        target_size_bytes = self._get_dir_size(target_path) if os.path.exists(target_path) else 0
        self.memory_tracker.model_size_bytes = target_size_bytes

        if max_ram_gb and max_ram_gb > 0:
            ram_threshold_bytes = int(max_ram_gb * (1024**3))
            logger.info(f"Using user RAM Memory Cap: {max_ram_gb} GB ({round(ram_threshold_bytes/(1024**3),1)} GB threshold)")
        else:
            ram_threshold_bytes = int(available_ram_bytes * 0.7)

        # Determine target device for PyTorch model execution (Metal MPS on macOS)
        # Determine target device for PyTorch model execution (Metal MPS on macOS)
        dev_str = self.device.type if self.device else "cpu"

        # 3. Model Loading with RAM Safety Guard & GPU Acceleration Target
        dtype = torch.float16 if dev_str in ["mps", "cuda"] else torch.float32
        cap_gb = max(1.5, float(max_ram_gb or 8))
        logger.info(f"Loading model '{model_name}' with dtype {dtype} for GPU device '{dev_str}' (Model size: {round(target_size_bytes/(1024**3),1)} GB, RAM Cap: {cap_gb} GB, Avail RAM: {round(available_ram_bytes/(1024**3),1)} GB)...")
        
        self.is_cpu_offloaded = False
        try:
            offload_dir = self.get_offload_dir()
            if os.path.exists(offload_dir):
                try:
                    import shutil
                    shutil.rmtree(offload_dir, ignore_errors=True)
                except Exception:
                    pass
            os.makedirs(offload_dir, exist_ok=True)

            if target_size_bytes > ram_threshold_bytes:
                logger.warning(f"Model size ({round(target_size_bytes/(1024**3),1)} GB) exceeds RAM Cap threshold ({cap_gb} GB)! Enabling CPU offloader...")
                try:
                    if dev_str == "mps":
                        # Maximize PyTorch CPU threading & memory mapping for offloading
                        try:
                            torch.set_num_threads(max(4, os.cpu_count() or 4))
                        except Exception:
                            pass
                        cpu_mem_limit_gb = max(1.2, round(cap_gb * 0.8, 2))
                        max_mem = {"cpu": f"{cpu_mem_limit_gb}GiB"}
                        self.model = AutoModelForCausalLM.from_pretrained(
                            target_path,
                            config=self.config,
                            torch_dtype=dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto",
                            max_memory=max_mem,
                            offload_folder=offload_dir,
                            offload_state_dict=True
                        )
                        self.is_cpu_offloaded = True
                        logger.info(f"⚡ Model loaded with Accelerate RAM cap on CPU ({cpu_mem_limit_gb} GB RAM limit) to honor {cap_gb} GB RAM Cap.")


                    else:
                        gpu_cap = max(1.5, round(cap_gb * 0.7, 2))
                        cpu_cap = max(1.5, round(cap_gb * 0.9, 2))
                        max_mem = {dev_str: f"{gpu_cap}GiB", "cpu": f"{cpu_cap}GiB"}
                        self.model = AutoModelForCausalLM.from_pretrained(
                            target_path,
                            config=self.config,
                            torch_dtype=dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="sequential",
                            max_memory=max_mem,
                            offload_folder=offload_dir,
                            offload_state_dict=True
                        )
                        logger.info(f"⚡ Model loaded with Accelerate sequential map on {dev_str} (GPU Cap: {gpu_cap} GB, CPU Cap: {cpu_cap} GB, Total Cap: {cap_gb} GB)")
                except Exception as offload_err:
                    logger.warning(f"Offload loading failed ({offload_err}), attempting direct load...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        target_path,
                        config=self.config,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )

            else:
                # Safe CPU load followed by GPU transfer when within RAM Cap
                self.model = AutoModelForCausalLM.from_pretrained(
                    target_path,
                    config=self.config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

        except Exception as model_err:
            logger.warning(f"Local model load for {target_path} failed ({model_err}), trying resolved model '{resolved_name}'...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    resolved_name,
                    config=self.config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except Exception as hf_err:
                logger.error(f"Failed to load model '{model_name}' ({hf_err}). Model is not downloaded locally.")
                self.memory_tracker.active_model = ""
                self.memory_tracker.total_layers = 0
                return False

        # Transfer model to GPU acceleration device (Metal MPS / CUDA) if not CPU offloaded
        if self.device and self.device.type in ["mps", "cuda"]:
            try:
                if not getattr(self.model, "hf_device_map", None) and not self.is_cpu_offloaded:
                    logger.info(f"Transferring model execution to target GPU device: {self.device}...")
                    self.model = self.model.to(self.device)
                    logger.info(f"⚡ Model 100% active on Metal GPU: {self.device}")
                elif self.is_cpu_offloaded:
                    logger.info(f"⚡ Model pinned to CPU (0.0 GB VRAM) per RAM Cap requirement [{cap_gb} GB]")
                else:
                    logger.info(f"⚡ Model managed by Accelerate device map on: {self.device}")
            except Exception as dev_err:
                logger.warning(f"Could not transfer full model to {self.device} ({dev_err}), running in hybrid CPU mode.")


        # Immediate post-load host RAM & VRAM cache purge
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


        self.model.eval()
        self.current_model_name = model_name
        self.current_ram_cap = max_ram_gb

        # Determine actual transformer layers from loaded model architecture
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            transformer_layers = self.model.transformer.h

        if transformer_layers is not None:
            actual_num_layers = len(transformer_layers)
        else:
            actual_num_layers = getattr(self.config, "num_hidden_layers", getattr(self.config, "n_layer", num_layers))

        self.memory_tracker.total_layers = actual_num_layers

        dmap = getattr(self.model, "hf_device_map", {})
        if dmap:
            disk_c = 0
            ram_c = 0
            if transformer_layers is not None:
                for idx in range(actual_num_layers):
                    dev = dmap.get(f"model.layers.{idx}", dmap.get(f"transformer.h.{idx}", dmap.get(f"layers.{idx}", "cpu")))
                    if str(dev).lower() == "disk":
                        disk_c += 1
                    else:
                        ram_c += 1
            else:
                disk_c = sum(1 for k, d in dmap.items() if ("layer" in k.lower() or ".h." in k.lower()) and str(d).lower() == "disk")
                ram_c = max(0, actual_num_layers - disk_c)

            self.memory_tracker.offload_active = (disk_c > 0)
            self.memory_tracker.disk_layers_count = disk_c
            self.memory_tracker.ram_layers_count = ram_c
            self.memory_tracker.base_ram_count = ram_c
        else:
            self.memory_tracker.offload_active = False
            self.memory_tracker.disk_layers_count = 0
            self.memory_tracker.ram_layers_count = actual_num_layers
            self.memory_tracker.base_ram_count = actual_num_layers

        logger.info(f"Successfully loaded model {model_name} ({self.memory_tracker.total_layers} layers) active on GPU ({dev_str}) [RAM Cap: {max_ram_gb} GB, Offload: {self.memory_tracker.offload_active}]")
        return True

    def unload_model(self):
        """Unload current model completely and release 100% of RAM/VRAM back to OS."""
        self.memory_tracker.offload_active = False
        self.memory_tracker.disk_layers_count = 0
        self.memory_tracker.ram_layers_count = 0
        self.memory_tracker.base_ram_count = 0
        self.memory_tracker.total_layers = 0
        self.memory_tracker.active_model = ""
        self.memory_tracker.model_size_bytes = 0

        if hasattr(self, "mlx_model") and self.mlx_model is not None:
            del self.mlx_model
            self.mlx_model = None
        if hasattr(self, "mlx_tokenizer") and self.mlx_tokenizer is not None:
            del self.mlx_tokenizer
            self.mlx_tokenizer = None
        self.is_mlx_model = False
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass


        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except Exception:
                pass

        if self.model is not None:
            try:
                for param in self.model.parameters():
                    param.data = torch.empty(0)
            except Exception:
                pass
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if hasattr(self, "config") and self.config is not None:
            del self.config
            self.config = None

        self.current_model_name = ""
        self.current_ram_cap = None
        self.memory_tracker.current_layer = -1

        self.memory_tracker.total_layers = 0
        self.memory_tracker.active_model = ""

        # Deep multi-pass garbage collection
        gc.collect()
        gc.collect()

        # Flush PyTorch CUDA & Metal MPS caches with queue synchronization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:
                pass

        # Short pause so Apple Metal ARC garbage collector finishes deallocating MPSGraph
        time.sleep(0.15)


        # Clean up stale offload cache files from disk
        offload_dir = self.get_offload_dir()
        if os.path.exists(offload_dir):
            try:
                import shutil
                shutil.rmtree(offload_dir, ignore_errors=True)
            except Exception:
                pass

        logger.info("RAM and GPU/MPS memory completely purged.")



    def delete_model(self, model_name: str) -> bool:
        """Delete model files completely from disk."""
        import shutil
        from airollama.ollama_registry import resolve_ollama_to_hf

        resolved_name = resolve_ollama_to_hf(model_name)

        if self.current_model_name in [model_name, resolved_name]:
            logger.info(f"Unloading active model '{self.current_model_name}' before deletion...")
            self.unload_model()

        safe_folder = model_name.replace("/", "---")
        safe_resolved = resolved_name.replace("/", "---")

        target_paths = [
            os.path.join(self.models_dir, safe_folder),
            os.path.join(self.models_dir, safe_resolved)
        ]

        deleted_any = False
        for p in target_paths:
            if os.path.exists(p):
                try:
                    shutil.rmtree(p)
                    logger.info(f"Removed model directory: {p}")
                    deleted_any = True
                except Exception as e:
                    logger.error(f"Error removing model directory {p}: {e}")

        return deleted_any


    def generate_stream(
        self,
        prompt: str,
        model_name: str = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: List[str] = None,
        max_ram_gb: Optional[float] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generates text token-by-token with layer-by-layer tracking.
        Yields chunk dictionaries containing token text, layer status, and memory stats.
        """
        self.stop_requested = False
        target_model = model_name or self.current_model_name

        if target_model and (self.current_model_name != target_model or (max_ram_gb is not None and getattr(self, "current_ram_cap", None) != max_ram_gb)):

            yield {"response": f"⌛ Preparing model architecture '{target_model}' for layer streaming...\n\n", "done": False}
            success = self.load_model(target_model, max_ram_gb=max_ram_gb)

            if not success:
                yield {"error": f"Model '{target_model}' is not downloaded yet. Please pull it from the Models tab first!", "done": True}
                return


        if not self.is_mlx_model and (not self.model or not self.tokenizer):
            yield {"error": "No model loaded. Please select or pull a model first.", "done": True}
            return

        if self.is_mlx_model and self.mlx_model and self.mlx_tokenizer:
            try:
                from mlx_lm import stream_generate
                logger.info(f"⚡ Streaming generation using Apple MLX native engine for prompt len={len(prompt)}...")
                tot_l = self.memory_tracker.total_layers or 32
                token_step = 0
                for response in stream_generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=max_new_tokens
                ):
                    if self.stop_requested:
                        break
                    self.memory_tracker.current_layer = token_step % tot_l
                    token_step += 1
                    yield {
                        "response": response.text,
                        "done": False,
                        "memory": self.memory_tracker.get_system_memory()
                    }
                self.memory_tracker.current_layer = -1
                yield {"response": "", "done": True, "memory": self.memory_tracker.get_system_memory()}
                return
            except Exception as gen_err:
                self.memory_tracker.current_layer = -1
                logger.error(f"MLX generation error: {gen_err}")
                yield {"error": f"MLX generation error: {gen_err}", "done": True}
                return


        stop_sequences = stop_sequences or []
        device = next(self.model.parameters()).device

        # Determine comprehensive set of EOS/stop token IDs
        eos_ids = set()
        if isinstance(self.tokenizer.eos_token_id, int):
            eos_ids.add(self.tokenizer.eos_token_id)
        elif isinstance(self.tokenizer.eos_token_id, (list, tuple, set)):
            eos_ids.update(self.tokenizer.eos_token_id)

        for stop_str in ["<|im_end|>", "<|end_of_text|>", "<end_of_turn>", "<eos>", "<|eot_id|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(stop_str)

            if tid is not None and isinstance(tid, int) and tid > 0:
                eos_ids.add(tid)

        # Encode input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        prompt_len = input_ids.shape[1]
        start_time = time.time()
        tokens_generated = 0
        total_layers = self.memory_tracker.total_layers

        # Solution 4 & 5: Asynchronous Layer Prefetcher & OS Page Cache pre-warmer
        offload_dir = self.get_offload_dir()
        prefetcher = AsyncLayerPrefetcher(offload_dir) if getattr(self.memory_tracker, "offload_active", False) else None

        # Register forward hook to monitor layer execution and prefetch upcoming offloaded layers
        def create_layer_hook(layer_idx):
            def hook(module, input, output):
                self.memory_tracker.current_layer = layer_idx
                if prefetcher:
                    prefetcher.prefetch_layer(layer_idx + 1)
                if self.layer_callback:
                    self.layer_callback(layer_idx, total_layers)
            return hook

        hooks = []
        try:
            transformer_layers = None
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                transformer_layers = self.model.model.layers
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                transformer_layers = self.model.transformer.h

            if transformer_layers:
                for idx, layer in enumerate(transformer_layers):
                    hooks.append(layer.register_forward_hook(create_layer_hook(idx)))

            curr_input = input_ids
            past_key_values = None

            recent_buffer = ""
            self.stop_requested = False
            for i in range(max_new_tokens):
                if getattr(self, "stop_requested", False):
                    logger.info("🛑 Generation cancelled by user stop request.")
                    self.stop_requested = False
                    break

                with torch.no_grad():

                    outputs = self.model(
                        input_ids=curr_input if past_key_values is None else curr_input[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    # Temperature & Top-P Sampling
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
                        if top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                dim=1, index=sorted_indices, src=sorted_indices_to_remove
                            )
                            probs[indices_to_remove] = 0.0
                            probs = probs / probs.sum(dim=-1, keepdim=True)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                curr_input = torch.cat([curr_input, next_token], dim=-1)
                tokens_generated += 1
                token_id = next_token.item()

                if tokens_generated % 15 == 0:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    elif torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass


                # Check stop criteria before decoding or yielding
                if token_id in eos_ids:
                    yield {
                        "response": "",
                        "done": True,
                        "token_id": token_id,
                        "eval_count": tokens_generated,
                        "eval_duration_sec": round(time.time() - start_time, 2),
                        "memory": self.memory_tracker.get_system_memory()
                    }
                    break

                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                recent_buffer += token_text

                # Check multi-token stop string leakage in rolling buffer
                is_stop = False
                for stop_str in ["<|im_end|>", "<|end_of_text|>", "<end_of_turn>", "<eos>", "<|eot_id|>"]:
                    if stop_str in recent_buffer:
                        is_stop = True
                        break

                if is_stop:
                    yield {
                        "response": "",
                        "done": True,
                        "token_id": token_id,
                        "eval_count": tokens_generated,
                        "eval_duration_sec": round(time.time() - start_time, 2),
                        "memory": self.memory_tracker.get_system_memory()
                    }
                    break

                # Strip control tokens from single chunk
                for stop_str in ["<|im_end|>", "<|end_of_text|>", "<end_of_turn>", "<eos>", "<|eot_id|>"]:
                    token_text = token_text.replace(stop_str, "")





                if token_text:
                    yield {
                        "response": token_text,
                        "done": False,
                        "token_id": token_id,
                        "eval_count": tokens_generated,
                        "memory": self.memory_tracker.get_system_memory()
                    }






            if tokens_generated >= max_new_tokens:
                yield {
                    "response": "",
                    "done": True,
                    "eval_count": tokens_generated,
                    "eval_duration_sec": round(time.time() - start_time, 2),
                    "memory": self.memory_tracker.get_system_memory()
                }

        except Exception as gen_err:
            logger.error(f"❌ Exception during generation: {gen_err}", exc_info=True)
            yield {
                "response": f"\n\n⚠️ Execution Notice: {str(gen_err)}",
                "done": True,
                "eval_count": tokens_generated,
                "memory": self.memory_tracker.get_system_memory()
            }
        finally:
            for h in hooks:
                h.remove()
            self.memory_tracker.current_layer = -1




    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt using tokenizer template if available."""
        if not messages:
            return ""

        # Sanitize messages to remove any control token tags stored in message strings
        formatted_messages = []
        for m in messages:
            content = m.get("content", "")
            for tag in ["<|im_end|>", "<|im_start|>", "<|end_of_text|>", "<end_of_turn>", "<eos>", "<|eot_id|>"]:
                content = content.replace(tag, "")
            formatted_messages.append({"role": m.get("role", "user"), "content": content.strip()})

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                # Handle models whose chat templates reject system role (e.g. Gemma)
                if any(m.get("role") == "system" for m in formatted_messages):
                    sys_texts = [m["content"] for m in formatted_messages if m.get("role") == "system"]
                    non_sys = [m for m in formatted_messages if m.get("role") != "system"]
                    sys_summary = "\n".join(sys_texts)
                    if non_sys and non_sys[0].get("role") == "user":
                        non_sys[0] = dict(non_sys[0])
                        non_sys[0]["content"] = f"Instructions: {sys_summary}\n\n{non_sys[0]['content']}"
                    formatted_messages = non_sys
                    try:
                        return self.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        pass

        # Robust Fallback ChatML formatting
        formatted = ""
        for msg in formatted_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted


