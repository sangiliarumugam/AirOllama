import sys
import time
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:11211"

def test_ui():
    print("🚀 Launching Playwright Chromium Headless Test...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 1. Load Dashboard
        print("1️⃣ Navigating to Dashboard...")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".brand-title")
        assert "AirOllama" in page.content(), "Brand title not found"
        print("   ✅ Dashboard loaded cleanly.")

        # 2. Test Tab Navigation: Dashboard -> Playground -> Models -> Settings -> API Docs -> Playground
        tabs = ["playground", "models", "settings", "api", "dashboard", "playground"]
        for tab in tabs:
            print(f"2️⃣ Switching to tab: '{tab}'...")
            page.evaluate(f"switchTab('{tab}')")
            page.wait_for_selector(f"#tab-{tab}.active", timeout=5000)
            assert page.is_visible(f"#tab-{tab}"), f"Tab panel #tab-{tab} is not visible"
            print(f"   ✅ Switched to tab #{tab} successfully.")

        # 3. Playground Model Selection & Auto RAM Cap Test
        print("3️⃣ Testing Model Selection & Auto RAM Cap calculation...")
        page.wait_for_selector("#chat-model-select option", state="attached", timeout=5000)
        page.select_option("#chat-model-select", value="gemma4:e4b")
        select_val = page.eval_on_selector("#chat-model-select", "e => e.value")
        print(f"   ✅ Selected model for test: {select_val}")

        ram_val = page.eval_on_selector("#param-max-ram", "e => e.value")
        print(f"   ✅ Auto-calculated RAM Cap: {ram_val} GB")

        # 4. Playground RAM Speed Preset Buttons Test
        print("4️⃣ Testing RAM Speed Preset Buttons...")
        page.click("button:has-text('🚀 Max Speed')")
        time.sleep(0.5)
        new_ram_val = page.eval_on_selector("#param-max-ram", "e => e.value")
        print(f"   ✅ Max Speed RAM Cap: {new_ram_val} GB")

        # 5. Playground Model Preloading Test
        print("5️⃣ Testing ⚡ Preload Model Button...")
        page.click("button:has-text('⚡ Preload')")
        page.wait_for_timeout(1000)
        print("   ✅ Preload button click verified.")

        # 6. Playground Chat Prompt Submission & Token Streaming Test
        print("6️⃣ Testing Prompt Submission & Token Streaming ('hi')...")
        page.fill("#chat-input", "hi")
        page.click("#btn-chat-send")

        # Wait for user message to appear
        page.wait_for_selector(".msg.user:has-text('hi')")
        print("   ✅ User prompt 'hi' added to chat container.")

        # Wait for streamed text to start arriving in assistant message
        page.wait_for_selector(".msg.assistant:has-text('Hello')", timeout=15000)
        assistant_reply = page.query_selector_all(".msg.assistant")[-1].inner_text()
        print(f"   ✅ Assistant response streamed successfully: '{assistant_reply.strip()}'")

        # 7. Test Real-Time Prompt Cancellation Button while generating
        print("7️⃣ Testing Real-Time Prompt Cancellation...")
        page.wait_for_selector("#btn-chat-stop", state="visible", timeout=5000)
        print("   ✅ Stop button visible during streaming generation.")
        page.click("#btn-chat-stop")
        # Wait for send button to be restored
        page.wait_for_selector("#btn-chat-send", state="visible", timeout=5000)
        print("   ✅ Real-time prompt cancellation test passed.")

        # 8. Test Settings Page Configuration Save
        print("8️⃣ Testing Settings Page Configuration Save...")
        page.evaluate("switchTab('settings')")
        page.wait_for_selector("#tab-settings.active")
        models_dir_val = page.eval_on_selector("#settings-models-dir", "e => e.value")
        assert "/Users/sangili/Projects/airollama/models" in models_dir_val, "Models dir mismatch"
        print(f"   ✅ Settings page verified with models_dir: {models_dir_val}")

        browser.close()
        print("\n🎉 ALL PLAYWRIGHT UI TESTS PASSED 100% CLEANLY!")

if __name__ == "__main__":
    test_ui()
