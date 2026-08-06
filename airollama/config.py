import os
import json
import logging

logger = logging.getLogger("AirOllama.Config")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
LEGACY_CONFIG_PATH = os.path.expanduser("~/.airollama/config.json")

def get_config_path() -> str:
    """Returns the persistent config file path within the project folder."""
    return PROJECT_CONFIG_PATH

def load_config() -> dict:
    """
    Loads JSON configuration containing models_dir and hf_token from project folder.
    """
    path = get_config_path()
    config = {
        "models_dir": "",
        "hf_token": "",
        "offload_dir": ""
    }

    # Migrate from legacy home directory config if present and project config doesn't exist
    if not os.path.exists(path) and os.path.exists(LEGACY_CONFIG_PATH):
        try:
            with open(LEGACY_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    config.update(saved)
            # Persist to project directory
            save_config(config)
            logger.info(f"Migrated legacy configuration to {path}")
        except Exception as e:
            logger.warning(f"Could not migrate legacy config file: {e}")

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    config.update(saved)
        except Exception as e:
            logger.warning(f"Could not read config file at {path}: {e}")

    # Fallback to environment variable if hf_token is not set in file
    if not config.get("hf_token") and os.environ.get("HF_TOKEN"):
        config["hf_token"] = os.environ["HF_TOKEN"]

    return config

def get_offload_dir() -> str:
    """Returns the persistent offload directory path on disk."""
    cfg = load_config()
    offload_dir = cfg.get("offload_dir", "").strip()
    if not offload_dir:
        import tempfile
        offload_dir = os.path.join(tempfile.gettempdir(), "airollama_offload")
    return os.path.abspath(os.path.expanduser(offload_dir))

def save_config(updates: dict) -> dict:
    """
    Updates and saves configuration to config.json in project directory.
    """

    config = load_config()
    for k, v in updates.items():
        if v is not None:
            config[k] = v

    path = get_config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {path}")
    except Exception as e:
        logger.error(f"Failed saving config to {path}: {e}")

    return config

def apply_config_environment():
    """Applies stored configuration (e.g. HF_TOKEN env & HF login) on startup."""
    cfg = load_config()
    hf_token = cfg.get("hf_token", "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Hugging Face API Token authenticated from config file.")
        except Exception as e:
            logger.warning(f"HF login with configured token failed: {e}")
