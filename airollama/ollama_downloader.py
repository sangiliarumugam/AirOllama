import os
import json
import logging
import requests
import queue
import time
from typing import Generator, Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi
from requests.adapters import HTTPAdapter

logger = logging.getLogger("AirOllama.Downloader")

OLLAMA_REGISTRY_BASE = "https://registry.ollama.ai/v2"
CPU_CORES = os.cpu_count() or 8
DEFAULT_MAX_WORKERS = max(64, CPU_CORES * 8)

# Configure ultra-high throughput HTTP session pool with TCP keep-alive
def create_high_performance_session(pool_size: int = 128) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=3
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

GLOBAL_HTTP_SESSION = create_high_performance_session(pool_size=128)

def fetch_manifest(model_repo: str, tag: str = "latest") -> Tuple[bool, Dict[str, Any], str]:
    """
    Fetch manifest from registry.ollama.ai.
    Tries both 'namespace/repo' and 'library/repo'.
    """
    headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    
    path1 = f"{OLLAMA_REGISTRY_BASE}/{model_repo}/manifests/{tag}"
    try:
        r = GLOBAL_HTTP_SESSION.get(path1, headers=headers, timeout=10)
        if r.status_code == 200:
            return True, r.json(), model_repo
    except Exception:
        pass

    path2 = f"{OLLAMA_REGISTRY_BASE}/library/{model_repo}/manifests/{tag}"
    try:
        r = GLOBAL_HTTP_SESSION.get(path2, headers=headers, timeout=10)
        if r.status_code == 200:
            return True, r.json(), f"library/{model_repo}"
    except Exception:
        pass

    return False, {}, ""

def _download_chunk_range_direct(
    model_name: str,
    chunk_idx: int,
    total_chunks: int,
    blob_url: str,
    dest_file: str,
    start_byte: int,
    end_byte: int,
    filename: str,
    progress_queue: queue.Queue
):
    """
    Ultra-Fast Direct-Offset Writer:
    Streams HTTP Range chunks directly into pre-allocated byte offsets of dest_file.
    Eliminates disk I/O file concatenation overhead entirely!
    """
    from airollama.engine import ACTIVE_PULL_CANCELLATIONS

    chunk_size = (end_byte - start_byte) + 1
    headers = {
        "User-Agent": "AirOllama-Downloader/2.0",
        "Accept": "application/octet-stream",
        "Range": f"bytes={start_byte}-{end_byte}"
    }

    try:
        r = GLOBAL_HTTP_SESSION.get(blob_url, headers=headers, stream=True, timeout=120, allow_redirects=True)
        if r.status_code not in (200, 206):
            progress_queue.put({"error": f"Failed byte range {start_byte}-{end_byte}: HTTP {r.status_code}"})
            return

        downloaded = 0
        last_yield_time = time.time()
        buf_size = 8 * 1024 * 1024  # High-throughput 8MB socket buffer

        with open(dest_file, "r+b") as f:
            f.seek(start_byte)
            for chunk in r.iter_content(chunk_size=buf_size):
                if model_name in ACTIVE_PULL_CANCELLATIONS:
                    progress_queue.put({
                        "layer_idx": chunk_idx,
                        "filename": f"{filename} (Part {chunk_idx+1}/{total_chunks})",
                        "status": "Cancelled",
                        "cancelled": True
                    })
                    return

                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_yield_time > 0.2 or downloaded == chunk_size:
                        last_yield_time = now
                        pct = round((downloaded / chunk_size) * 100, 1) if chunk_size > 0 else 0
                        progress_queue.put({
                            "layer_idx": chunk_idx,
                            "filename": f"{filename} [Stream {chunk_idx+1}/{total_chunks}]",
                            "size_mb": round(chunk_size / (1024*1024), 1),
                            "downloaded_mb": round(downloaded / (1024*1024), 1),
                            "percent": pct,
                            "status": f"⚡ Thread {chunk_idx+1}/{total_chunks}: {pct}% ({round(downloaded/(1024*1024),1)} MB / {round(chunk_size/(1024*1024),1)} MB)"
                        })

        progress_queue.put({
            "layer_idx": chunk_idx,
            "filename": f"{filename} [Stream {chunk_idx+1}/{total_chunks}]",
            "percent": 100,
            "size_mb": round(chunk_size / (1024*1024), 1),
            "downloaded_mb": round(chunk_size / (1024*1024), 1),
            "status": f"✅ Thread {chunk_idx+1}/{total_chunks} complete"
        })
    except Exception as e:
        progress_queue.put({"error": f"Thread {chunk_idx+1} download failed: {e}"})

def _download_single_layer(
    model_name: str,
    idx: int,
    total_layers: int,
    layer: Dict[str, Any],
    found_path: str,
    target_dir: str,
    progress_queue: queue.Queue,
    num_parallel_threads: int = DEFAULT_MAX_WORKERS
):
    """Download layer blob using simultaneous multi-range parallel threads with direct offset writing."""
    media_type = layer.get("mediaType", "")
    digest = layer.get("digest", "")
    size = layer.get("size", 0)

    if not digest:
        return

    if "model" in media_type:
        filename = "model.safetensors"
    elif "template" in media_type:
        filename = "template.txt"
    elif "system" in media_type:
        filename = "system.txt"
    elif "params" in media_type:
        filename = "params.json"
    elif "license" in media_type:
        filename = f"license_{idx}.txt"
    else:
        filename = f"blob_{idx}.bin"

    dest_file = os.path.join(target_dir, filename)
    blob_url = f"{OLLAMA_REGISTRY_BASE}/{found_path}/blobs/{digest}"

    if size < 10 * 1024 * 1024 or num_parallel_threads <= 1:
        with open(dest_file, "wb") as f:
            pass
        _download_chunk_range_direct(
            model_name=model_name,
            chunk_idx=idx,
            total_chunks=total_layers,
            blob_url=blob_url,
            dest_file=dest_file,
            start_byte=0,
            end_byte=size - 1 if size > 0 else 0,
            filename=filename,
            progress_queue=progress_queue
        )
        return

    # Pre-allocate destination file for direct offset concurrent writing
    with open(dest_file, "wb") as f:
        f.truncate(size)

    chunk_size = size // num_parallel_threads

    progress_queue.put({
        "status": f"⚡ Launching {num_parallel_threads} SIMULTANEOUS range threads for {filename} ({round(size/(1024*1024), 1)} MB)..."
    })

    with ThreadPoolExecutor(max_workers=num_parallel_threads) as chunk_executor:
        futures = []
        for c_idx in range(num_parallel_threads):
            start = c_idx * chunk_size
            end = (start + chunk_size - 1) if c_idx < num_parallel_threads - 1 else size - 1

            futures.append(
                chunk_executor.submit(
                    _download_chunk_range_direct,
                    model_name,
                    c_idx,
                    num_parallel_threads,
                    blob_url,
                    dest_file,
                    start,
                    end,
                    filename,
                    progress_queue
                )
            )

        for f in as_completed(futures):
            f.result()

    progress_queue.put({"status": f"✅ Downloaded {filename} successfully!"})

def download_ollama_model(model_name: str, target_dir: str, max_workers: int = DEFAULT_MAX_WORKERS) -> Generator[Dict[str, Any], None, None]:
    """
    Multi-threaded SIMULTANEOUS parallel model downloader for Ollama OCI registry.
    """
    from airollama.ollama_registry import parse_ollama_name
    
    clean_repo, tag = parse_ollama_name(model_name)
    success, manifest, found_path = fetch_manifest(clean_repo, tag)

    if not success:
        yield {"error": f"Model '{model_name}' not found on Ollama registry"}
        return

    os.makedirs(target_dir, exist_ok=True)
    manifest_path = os.path.join(target_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    layers = manifest.get("layers", [])
    total_layers = len(layers)

    yield {"status": f"🚀 Launching {max_workers} SIMULTANEOUS parallel download threads across CPU cores..."}

    progress_q = queue.Queue()
    
    for idx, layer in enumerate(layers):
        from airollama.engine import ACTIVE_PULL_CANCELLATIONS
        if model_name in ACTIVE_PULL_CANCELLATIONS:
            import shutil
            try:
                shutil.rmtree(target_dir, ignore_errors=True)
            except Exception:
                pass
            yield {"status": "❌ Download cancelled and partial files deleted."}
            return

        with ThreadPoolExecutor(max_workers=1) as layer_executor:
            future = layer_executor.submit(
                _download_single_layer,
                model_name,
                idx,
                total_layers,
                layer,
                found_path,
                target_dir,
                progress_q,
                max_workers
            )

            while not future.done() or not progress_q.empty():
                if model_name in ACTIVE_PULL_CANCELLATIONS:
                    import shutil
                    try:
                        shutil.rmtree(target_dir, ignore_errors=True)
                    except Exception:
                        pass
                    yield {"status": "❌ Download cancelled and partial files deleted."}
                    return
                try:
                    msg = progress_q.get(timeout=0.1)
                    if "error" in msg:
                        import shutil
                        try:
                            shutil.rmtree(target_dir, ignore_errors=True)
                        except Exception:
                            pass
                    yield msg
                except queue.Empty:
                    pass

    # Drain any remaining progress queue items
    while not progress_q.empty():
        try:
            yield progress_q.get_nowait()
        except queue.Empty:
            break

    yield {"status": "success"}

def _download_hf_file_thread(
    model_name: str,
    idx: int,
    total_files: int,
    repo_id: str,
    fname: str,
    target_dir: str,
    progress_q: queue.Queue
):
    """Download single HF file with real-time 8MB chunked streaming progress updates."""
    from airollama.engine import ACTIVE_PULL_CANCELLATIONS
    if model_name in ACTIVE_PULL_CANCELLATIONS:
        return

    dest_file = os.path.join(target_dir, fname)
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)

    file_url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
    headers = {"User-Agent": "AirOllama-Downloader/2.0"}

    try:
        r = GLOBAL_HTTP_SESSION.get(file_url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        if r.status_code != 200:
            progress_q.put({"error": f"HTTP {r.status_code} for {fname}"})
            return

        size = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        last_yield_time = time.time()

        with open(dest_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if model_name in ACTIVE_PULL_CANCELLATIONS:
                    progress_q.put({
                        "layer_idx": idx,
                        "filename": fname,
                        "status": "Cancelled",
                        "cancelled": True
                    })
                    return

                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_yield_time > 0.2 or downloaded == size:
                        last_yield_time = now
                        pct = round((downloaded / size) * 100, 1) if size > 0 else 0
                        progress_q.put({
                            "layer_idx": idx,
                            "filename": fname,
                            "size_mb": round(size / (1024*1024), 1),
                            "downloaded_mb": round(downloaded / (1024*1024), 1),
                            "percent": pct,
                            "status": f"⚡ Thread {idx+1}/{total_files} [{fname}]: {pct}% ({round(downloaded/(1024*1024),1)} MB / {round(size/(1024*1024),1)} MB)"
                        })

        progress_q.put({
            "layer_idx": idx,
            "filename": fname,
            "percent": 100,
            "size_mb": round(size / (1024*1024), 1),
            "downloaded_mb": round(size / (1024*1024), 1),
            "status": f"✅ Downloaded {fname}"
        })
    except Exception as e:
        progress_q.put({"error": f"Failed downloading {fname}: {e}"})

def _download_hf_file_parallel_range(
    model_name: str,
    file_idx: int,
    repo_id: str,
    sfname: str,
    target_dir: str,
    workers_per_file: int,
    progress_q: queue.Queue
):
    """
    Ultra-Fast Direct-Offset Multi-Core Writer:
    Downloads a safetensor shard file using multi-threaded parallel HTTP Range chunks
    written directly into pre-allocated byte offsets of dest_sf.
    """
    from airollama.engine import ACTIVE_PULL_CANCELLATIONS

    sf_url = f"https://huggingface.co/{repo_id}/resolve/main/{sfname}"
    dest_sf = os.path.join(target_dir, sfname)

    if os.path.exists(dest_sf):
        progress_q.put({"status": f"✅ {sfname} already downloaded"})
        return

    try:
        r_head = GLOBAL_HTTP_SESSION.head(sf_url, headers={"User-Agent": "AirOllama-Downloader/2.0"}, allow_redirects=True, timeout=10)
        total_bytes = int(r_head.headers.get("Content-Length", 0))
    except Exception:
        total_bytes = 0

    if total_bytes < 10 * 1024 * 1024 or workers_per_file <= 1:
        _download_hf_file_thread(model_name, file_idx, 1, repo_id, sfname, target_dir, progress_q)
        return

    # Pre-allocate destination file for direct offset concurrent writing across CPU cores
    with open(dest_sf, "wb") as f:
        f.truncate(total_bytes)

    chunk_size = total_bytes // workers_per_file

    with ThreadPoolExecutor(max_workers=workers_per_file) as executor:
        futures = []
        for c_idx in range(workers_per_file):
            start = c_idx * chunk_size
            end = (start + chunk_size - 1) if c_idx < workers_per_file - 1 else total_bytes - 1
            global_chunk_idx = (file_idx * workers_per_file) + c_idx
            stream_label = f"[{sfname}] Stream {c_idx+1}/{workers_per_file}"

            futures.append(
                executor.submit(
                    _download_chunk_range_direct,
                    model_name,
                    global_chunk_idx,
                    workers_per_file,
                    sf_url,
                    dest_sf,
                    start,
                    end,
                    stream_label,
                    progress_q
                )
            )

        for f in as_completed(futures):
            if model_name in ACTIVE_PULL_CANCELLATIONS:
                try:
                    os.remove(dest_sf)
                except Exception:
                    pass
                return

    progress_q.put({"status": f"✅ Downloaded {sfname} successfully!"})

def download_hf_model_in_parallel(model_name: str, repo_id: str, target_dir: str, max_workers: int = DEFAULT_MAX_WORKERS) -> Generator[Dict[str, Any], None, None]:
    """
    Optimized Hugging Face Downloader with Direct-Offset Multi-Core Sharding.
    Dedicates all parallel worker streams across CPU cores and network bandwidth.
    """
    from airollama.engine import ACTIVE_PULL_CANCELLATIONS

    yield {"status": f"🔍 Querying Hugging Face repository metadata for {repo_id}..."}

    api = HfApi()
    try:
        repo_files = api.list_repo_files(repo_id=repo_id)
    except Exception as e:
        yield {"error": f"Failed to list files for HF repo {repo_id}: {e}"}
        return

    ignore_suffixes = (".msgpack", ".h5", ".ot", ".pt", ".pth")
    filtered_files = [
        f for f in repo_files 
        if not f.startswith("onnx/") and not f.endswith(ignore_suffixes)
    ]

    metadata_files = [f for f in filtered_files if not f.endswith(".safetensors") and not f.endswith(".bin")]
    safetensor_files = [f for f in filtered_files if f.endswith(".safetensors") or f.endswith(".bin")]

    os.makedirs(target_dir, exist_ok=True)
    progress_q = queue.Queue()

    # Step 1: Download non-safetensor metadata files in parallel threads
    if metadata_files:
        yield {"status": f"📦 Downloading {len(metadata_files)} metadata files in parallel..."}
        with ThreadPoolExecutor(max_workers=min(16, len(metadata_files))) as meta_executor:
            m_futures = [
                meta_executor.submit(_download_hf_file_thread, model_name, idx, len(metadata_files), repo_id, mfile, target_dir, progress_q)
                for idx, mfile in enumerate(metadata_files)
            ]
            while any(not f.done() for f in m_futures) or not progress_q.empty():
                if model_name in ACTIVE_PULL_CANCELLATIONS:
                    yield {"status": "❌ Download cancelled"}
                    return
                try:
                    msg = progress_q.get(timeout=0.1)
                    yield msg
                except queue.Empty:
                    pass

        yield {"status": f"✅ Metadata ready! Launching {max_workers} SIMULTANEOUS parallel range streams across {len(safetensor_files)} weight artifacts..."}

    # Step 2: Distribute max_workers parallel range threads across all safetensor files with direct offset writing
    if safetensor_files:
        num_files = len(safetensor_files)
        workers_per_file = max(4, max_workers // num_files)
        total_active_workers = num_files * workers_per_file

        yield {"status": f"⚡ Active Download: {total_active_workers} PARALLEL TCP RANGE STREAMS ({workers_per_file} streams per shard) across {num_files} weight shards..."}

        with ThreadPoolExecutor(max_workers=num_files) as file_executor:
            file_futures = [
                file_executor.submit(_download_hf_file_parallel_range, model_name, file_idx, repo_id, sfname, target_dir, workers_per_file, progress_q)
                for file_idx, sfname in enumerate(safetensor_files)
            ]

            while any(not f.done() for f in file_futures) or not progress_q.empty():
                if model_name in ACTIVE_PULL_CANCELLATIONS:
                    import shutil
                    try:
                        shutil.rmtree(target_dir, ignore_errors=True)
                    except Exception:
                        pass
                    yield {"status": "❌ Download cancelled and partial files deleted."}
                    return
                try:
                    msg = progress_q.get(timeout=0.1)
                    if "error" in msg:
                        import shutil
                        try:
                            shutil.rmtree(target_dir, ignore_errors=True)
                        except Exception:
                            pass
                    yield msg
                except queue.Empty:
                    pass

        # Drain any remaining messages from progress queue
        while not progress_q.empty():
            try:
                yield progress_q.get_nowait()
            except queue.Empty:
                break

        # Validate download integrity - ensure all expected files exist with non-zero size
        download_valid = True
        for sfname in safetensor_files:
            sf_path = os.path.join(target_dir, sfname)
            if not os.path.exists(sf_path) or os.path.getsize(sf_path) == 0:
                download_valid = False
                break

        if not download_valid or model_name in ACTIVE_PULL_CANCELLATIONS:
            import shutil
            try:
                shutil.rmtree(target_dir, ignore_errors=True)
            except Exception:
                pass
            yield {"error": "Download failed or incomplete. Partial files deleted from disk."}
            return

        yield {"status": "✅ All model weight artifacts downloaded and verified!"}

    yield {"status": "download complete"}
    yield {"status": "success"}
