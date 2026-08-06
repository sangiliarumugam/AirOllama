"""
Playwright Automation Test Script for AirOllama Agentic Coding Interface
"""
import time
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:11211"

def test_agentic_coding_page_layout():
    """Verify Agentic Coding page layout, project creation modal, sidebar controls, and floating prompt pill bar."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("📡 Navigating to AirOllama Dashboard...")
        page.goto(f"{BASE_URL}/dashboard", wait_until="networkidle")

        # 1. Click Agentic Coding navbar button
        print("👆 Clicking Agentic Coding tab...")
        agentic_tab = page.wait_for_selector("button.tab-btn:has-text('Agentic Coding')", timeout=10000)
        assert agentic_tab is not None, "Agentic Coding tab button missing from navbar"
        agentic_tab.click()

        # 2. Wait for modular agentic view template to load and initialize SQLite state
        print("⌛ Waiting for Agentic template load and SQLite initialization...")
        prompt_input = page.wait_for_selector("#agentic-prompt-input", timeout=10000)
        assert prompt_input is not None, "Prompt textarea input missing"
        time.sleep(1.0)

        # 3. Verify sidebar components
        new_conv_btn = page.query_selector("button:has-text('New Conversation')")
        assert new_conv_btn is not None, "New Conversation button missing"

        conv_list = page.query_selector("#agentic-conv-list")
        proj_list = page.query_selector("#agentic-project-list")
        assert conv_list is not None, "Pinned conversations list container missing"
        assert proj_list is not None, "Projects list container missing"

        # 4. Test Project Creation `+` button and Modal
        print("📁 Testing Create Project `+` button...")
        add_proj_btn = page.query_selector("button[title='New Project']")
        assert add_proj_btn is not None, "New Project `+` button missing"
        add_proj_btn.click()
        time.sleep(0.5)

        proj_modal = page.wait_for_selector("#agentic-project-modal", timeout=5000)
        assert proj_modal is not None, "Create Project modal missing or not rendered"

        # Fill modal details
        page.fill("#modal-proj-name", "Test Project Auto")
        page.fill("#modal-proj-path", "/Users/sangili/Projects/airollama")
        
        # Click Create Project inside modal
        page.click("#agentic-project-modal button:has-text('Create Project')")
        time.sleep(1.0)

        assert page.query_selector("#agentic-project-list") is not None, "Projects list container missing"
        print("✅ Project creation modal and button flow PASSED!")

        # 5. Verify top header control bar elements
        active_title = page.query_selector("#agentic-active-title")
        active_proj_tag = page.query_selector("#agentic-active-project-tag")
        assert active_title is not None, "Active conversation title element missing"
        assert active_proj_tag is not None, "Active project tag element missing"

        # 6. Verify floating input pill controls
        model_select = page.query_selector("#agentic-model-select")
        mode_select = page.query_selector("#agentic-mode-select")
        send_btn = page.query_selector("#agentic-btn-send")

        assert model_select is not None, "Model selector dropdown missing"
        assert mode_select is not None, "Agent mode selector dropdown missing"
        assert send_btn is not None, "Send button missing"

        # 7. Test prompt input typing
        print("💬 Typing test prompt in prompt input textarea...")
        prompt_input.fill("Write a python script to check server status")
        assert prompt_input.input_value() == "Write a python script to check server status", "Prompt input value mismatch"

        # 8. Test send button click trigger
        print("🚀 Clicking Send button...")
        send_btn.click()
        time.sleep(0.5)

        # Verify messages container exists and is active
        container = page.query_selector("#agentic-messages-container")
        assert container is not None, "Messages container missing"

        print("✅ Agentic Coding Playwright UI Automation Test PASSED!")
        browser.close()

if __name__ == "__main__":
    test_agentic_coding_page_layout()
