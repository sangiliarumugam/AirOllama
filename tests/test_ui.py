"""
AirOllama Web Dashboard UI Automation Tests
"""
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:11211"

def test_dashboard_ui_layout():
    """Verify Web Dashboard telemetry layout and navigation."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"{BASE_URL}/dashboard", wait_until="domcontentloaded")
        
        # Verify top row cards
        top_model = page.query_selector(".grid-top-row #stat-model")
        top_layer = page.query_selector(".grid-top-row #stat-layer")
        assert top_model is not None, "Top row model tile missing"
        assert top_layer is not None, "Top row layer tile missing"

        # Verify bottom row metrics
        model_ram = page.query_selector(".grid-stats #stat-model-ram")
        sys_ram = page.query_selector(".grid-stats #stat-sysram")
        vram = page.query_selector(".grid-stats #stat-vram")
        offload = page.query_selector(".grid-stats #stat-offload")
        assert all([model_ram, sys_ram, vram, offload]), "Bottom row metrics missing"

        browser.close()

if __name__ == "__main__":
    test_dashboard_ui_layout()
    print("✅ Web Dashboard UI test PASSED!")
