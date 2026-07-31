from airollama.ollama_downloader import download_ollama_model

print("Testing download_ollama_model initialization...")
gen = download_ollama_model("tinyllama", "/Users/sangili/.gemini/antigravity/scratch/airollama/models/test_dir", max_workers=2)
first_item = next(gen)
print("Result ->", first_item)
print("✅ NameError scoping fixed!")
