import requests

print("1. Listing models before delete...")
r1 = requests.get("http://127.0.0.1:11211/api/tags")
print("Models before delete:", r1.json())

# Delete model if available
models = r1.json().get("models", [])
if models:
    target_to_delete = models[0]["name"]
    print(f"\n2. Deleting model '{target_to_delete}'...")
    r_del = requests.delete("http://127.0.0.1:11211/api/delete", json={"name": target_to_delete})
    print("Delete response:", r_del.json())

    print("\n3. Listing models after delete...")
    r2 = requests.get("http://127.0.0.1:11211/api/tags")
    print("Models after delete:", r2.json())
else:
    print("No models available to test deletion.")
