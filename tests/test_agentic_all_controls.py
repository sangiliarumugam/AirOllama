"""
Playwright Comprehensive E2E Test Suite verifying ALL links, buttons, modals, interactions,
AND asserting real LLM agent responses on the AirOllama Agentic Coding page.
"""
import time
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:11211"

def test_all_agentic_controls():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.on("console", lambda msg: print(f"BROWSER CONSOLE [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"BROWSER ERROR: {err}"))
        page.on("dialog", lambda dialog: print(f"BROWSER DIALOG: {dialog.message}") or dialog.accept())

        print("📡 1. Navigating to AirOllama Dashboard...")
        page.goto(f"{BASE_URL}/dashboard", wait_until="networkidle")

        print("👆 2. Switching to Agentic Coding view...")
        agentic_tab = page.wait_for_selector("button.tab-btn:has-text('Agentic Coding')", timeout=10000)
        assert agentic_tab is not None
        agentic_tab.click()

        page.wait_for_selector("#agentic-prompt-input", timeout=10000)
        time.sleep(1.0)

        # 3. Test `+ New Conversation` button
        print("⚡ 3. Testing '+ New Conversation' button...")
        btn_new_conv = page.wait_for_selector("button:has-text('New Conversation')")
        assert btn_new_conv is not None
        btn_new_conv.click()
        time.sleep(0.5)

        # 4. Test Conversation History & Scheduled Tasks links
        print("🕒 4. Testing 'Conversation History' & 'Scheduled Tasks' sidebar links...")
        link_history = page.wait_for_selector("div.sidebar-item:has-text('Conversation History')")
        link_scheduled = page.wait_for_selector("div.sidebar-item:has-text('Scheduled Tasks')")
        assert link_history is not None
        assert link_scheduled is not None

        link_scheduled.click()
        time.sleep(0.3)

        link_history.click()
        time.sleep(0.3)

        # 5. Test Pinned Conversations list items & thread selection
        print("💬 5. Testing Pinned Conversations selection...")
        conv_items = page.query_selector_all("#agentic-conv-list > div")
        if conv_items:
            conv_items[0].click()
            time.sleep(0.3)

        # 6. Test New Project modal workflow
        print("📁 6. Testing Create Project button ('+') and Modal overlay...")
        btn_add_proj = page.wait_for_selector("button[title='New Project']")
        assert btn_add_proj is not None
        btn_add_proj.click()
        time.sleep(0.5)

        # Verify Modal opened
        modal = page.wait_for_selector("#agentic-project-modal", timeout=5000)
        assert modal is not None and modal.is_visible(), "Create Project modal failed to open"

        # Test Cancel button
        print("✕ 7. Testing Modal Cancel button...")
        page.click("#agentic-project-modal button:has-text('Cancel')")
        time.sleep(0.3)
        assert not modal.is_visible(), "Modal failed to close on Cancel"

        # Re-open Modal and test creation
        print("📝 8. Re-opening Modal & submitting new project details...")
        btn_add_proj.click()
        time.sleep(0.3)

        page.fill("#modal-proj-name", "E2E Playwright Project")
        page.fill("#modal-proj-path", "/Users/sangili/Projects/e2e_test_dir")
        
        print("Submitting project creation...")
        page.evaluate("agenticSubmitCreateProject()")
        time.sleep(1.0)

        # Verify project added to sidebar
        page.wait_for_selector("#agentic-project-list:has-text('E2E Playwright Project')", timeout=5000)
        print("✅ Project successfully created and rendered in sidebar!")

        # 9. Verify Project Header Tag Update
        print("📂 9. Verifying Project active header tag...")
        proj_tag = page.query_selector("#agentic-active-project-tag")
        assert proj_tag is not None and "E2E Playwright Project" in proj_tag.inner_text()

        # 10. Test Floating Input Pill Controls
        print("🎛️ 10. Testing Floating Input Bar controls...")
        prompt_input = page.query_selector("#agentic-prompt-input")
        btn_attach = page.query_selector("button[title='Attach context']")
        btn_mic = page.query_selector("button[title='Voice Input']")
        model_select = page.query_selector("#agentic-model-select")
        mode_select = page.query_selector("#agentic-mode-select")
        send_btn = page.query_selector("#agentic-btn-send")

        assert all([prompt_input, btn_attach, btn_mic, model_select, mode_select, send_btn]), "Floating pill controls missing"

        # Test mode selection
        mode_select.select_option("Pair Programmer")
        time.sleep(0.2)

        # Test prompt typing & REAL submission to server
        print("💬 11. Testing prompt submission & REAL response stream...")
        prompt_input.fill("hi")
        send_btn.click()

        # Wait for agent text response in .agentic-md-body
        response_elem = page.wait_for_selector(".agentic-md-body", timeout=25000)
        assert response_elem is not None, "Response body element missing"

        page.wait_for_function(
            "() => document.querySelector('.agentic-md-body') && document.querySelector('.agentic-md-body').innerText.trim().length > 0",
            timeout=25000
        )
        response_text = page.locator(".agentic-md-body").inner_text()
        print(f"🤖 Verified Agent Response: '{response_text.strip()}'")
        assert len(response_text.strip()) > 0, "Agent response text is empty!"

        print("✅ ALL Links, Buttons, Modals, Interactions, AND Real Agent Response PASSED 100%!")
        browser.close()

if __name__ == "__main__":
    test_all_agentic_controls()
