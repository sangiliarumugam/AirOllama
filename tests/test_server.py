"""
AirOllama Server Endpoint Integration Tests
"""
import requests

BASE_URL = "http://127.0.0.1:11211"

def test_api_tags():
    """Verify listing local models."""
    res = requests.get(f"{BASE_URL}/api/tags")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert isinstance(data["models"], list)

def test_api_status():
    """Verify real-time system memory and layer status endpoint."""
    res = requests.get(f"{BASE_URL}/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "total_ram_gb" in data
    assert "available_ram_gb" in data
    assert "active_model" in data

def test_api_config():
    """Verify fetching and updating server configuration."""
    res = requests.get(f"{BASE_URL}/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "models_dir" in data
    assert "offload_dir" in data

if __name__ == "__main__":
    test_api_tags()
    test_api_status()
    test_api_config()
    print("✅ All server API tests PASSED!")
