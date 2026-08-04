import sys
import logging
import argparse
import uvicorn
import requests
import json


DEFAULT_PORT = 11211
DEFAULT_HOST = "0.0.0.0"

def main():
    parser = argparse.ArgumentParser(description="AirOllama: Layer-by-Layer On-Demand LLM Server for macOS on Port 11211")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the AirOllama API server")
    serve_parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind server (default: {DEFAULT_HOST})")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT})")
    serve_parser.add_argument("--api-only", action="store_true", help="Run in API-only / Headless mode (for OpenCode Agent / AI clients)")
    serve_parser.add_argument("--no-ui", action="store_true", help="Alias for --api-only")

    # Command: list / tags
    subparsers.add_parser("list", help="List locally cached models")

    # Command: pull
    pull_parser = subparsers.add_parser("pull", help="Pull a model from Hugging Face or Ollama registry")
    pull_parser.add_argument("model", help="Model name or tag")

    # Command: rm / delete
    rm_parser = subparsers.add_parser("rm", help="Remove a downloaded model completely")
    rm_parser.add_argument("model", help="Model name or tag to delete")

    # Command: run / chat
    run_parser = subparsers.add_parser("run", help="Run prompt on a model via AirOllama server")

    run_parser.add_argument("model", help="Model name")
    run_parser.add_argument("prompt", nargs="?", help="Prompt text")
    run_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port of server (default: {DEFAULT_PORT})")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        port = getattr(args, "port", DEFAULT_PORT)
        host = getattr(args, "host", DEFAULT_HOST)
        api_only = getattr(args, "api_only", False) or getattr(args, "no_ui", False) or ("--api-only" in sys.argv) or ("--no-ui" in sys.argv)
        
        if api_only:
            print("==========================================================")
            print(f"🚀 AirOllama Server (API-Only Mode / OpenCode Agent Ready)")
            print(f"📡 Ollama API: http://{host}:{port}/api/")
            print(f"🔌 OpenAI API: http://{host}:{port}/v1/")
            print(f"⚡ UI Dashboard Disabled (--api-only)")
            print("==========================================================")
        else:
            print(f"🚀 Starting AirOllama server on http://{host}:{port} (Web Dashboard & Ollama API)")

        # Filter out repetitive 1-second /api/ps status polling logs from console output
        class EndpointFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "/api/ps" not in record.getMessage()

        logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
        uvicorn.run("airollama.server:app", host=host, port=port, reload=False)


    elif args.command == "list":
        try:
            res = requests.get(f"http://127.0.0.1:{DEFAULT_PORT}/api/tags")
            data = res.json()
            models = data.get("models", [])
            print(f"NAME\t\t\t\t\tSIZE\t\tFORMAT")
            print("-" * 65)
            for m in models:
                size_mb = round(m.get("size", 0) / (1024 * 1024), 1)
                print(f"{m['name']:<40}\t{size_mb} MB\t{m['details']['format']}")
        except Exception as e:
            print(f"Failed to connect to AirOllama server on port {DEFAULT_PORT}: {e}")

    elif args.command == "pull":
        try:
            print(f"📥 Requesting pull for {args.model}...")
            res = requests.post(f"http://127.0.0.1:{DEFAULT_PORT}/api/pull", json={"name": args.model, "stream": True}, stream=True)
            for line in res.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    print(f"-> {data.get('status', data.get('error', ''))}")
        except Exception as e:
            print(f"Failed to pull model: {e}")

    elif args.command == "rm":
        try:
            print(f"🗑️ Deleting model '{args.model}'...")
            res = requests.delete(f"http://127.0.0.1:{DEFAULT_PORT}/api/delete", json={"name": args.model})
            if res.status_code == 200:
                print(f"✅ Successfully deleted model '{args.model}'")
            else:
                print(f"❌ Failed: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            print(f"Error deleting model: {e}")


    elif args.command == "run":
        port = args.port
        prompt = args.prompt
        if not prompt:
            prompt = input(f"[{args.model}] Prompt: ")
        
        try:
            print(f"\n--- AirOllama Streaming Response ({args.model}) ---")
            res = requests.post(
                f"http://127.0.0.1:{port}/api/generate",
                json={"model": args.model, "prompt": prompt, "stream": True},
                stream=True
            )
            for line in res.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    sys.stdout.write(token)
                    sys.stdout.flush()
            print("\n")
        except Exception as e:
            print(f"Error running prompt: {e}")

if __name__ == "__main__":
    main()
