import sys
import time
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:11211"

def capture_console_and_chat():
    print("🌐 Launching Playwright with Console & Network Listener...")
    
    console_logs = []
    network_events = []

    with sync_playwright() as p:
        # Launch browser (headless=True for speed, or set to False if desired)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Capture browser console messages
        def handle_console(msg):
            log_line = f"🖥️ [BROWSER CONSOLE {msg.type.upper()}] {msg.text}"
            console_logs.append(log_line)
            print(log_line)

        # Capture network requests/responses
        def handle_response(response):
            if "/api/" in response.url:
                net_line = f"📡 [NETWORK HTTP {response.status}] {response.request.method} {response.url}"
                network_events.append(net_line)
                print(net_line)

        page.on("console", handle_console)
        page.on("response", handle_response)

        # 1. Load Dashboard
        print("1️⃣ Navigating to http://127.0.0.1:11211...")
        page.goto(BASE_URL)
        page.wait_for_selector(".brand-title")

        # 2. Switch to Playground
        print("2️⃣ Switching to Playground tab...")
        page.evaluate("switchTab('playground')")
        page.wait_for_selector("#tab-playground.active")

        # 3. Check selected model & RAM Cap
        page.wait_for_selector("#chat-model-select option", state="attached", timeout=5000)
        model = page.eval_on_selector("#chat-model-select", "e => e.value")
        ram_cap = page.eval_on_selector("#param-max-ram", "e => e.value")
        print(f"3️⃣ Active Model: {model} | RAM Cap: {ram_cap} GB")

        # 4. Fill Prompt 'hi' and click Send
        print("4️⃣ Submitting prompt 'hi'...")
        page.fill("#chat-input", "hi")
        page.click("#btn-chat-send")

        # 5. Wait up to 15 seconds while listening to stream
        print("5️⃣ Listening to real-time chat stream...")
        start = time.time()
        reply_received = False

        while time.time() - start < 15:
            page.wait_for_timeout(500)
            msgs = page.query_selector_all(".msg.assistant")
            if msgs:
                txt = msgs[-1].inner_text()
                if txt and not txt.startswith("Thinking"):
                    print(f"\n🎉 REAL-TIME RESPONSE CAPTURED IN DOM: '{txt.strip()}'\n")
                    reply_received = True
                    break

        if not reply_received:
            print("\n⚠️ Timed out waiting for non-Thinking text. Final Assistant Box Text:")
            msgs = page.query_selector_all(".msg.assistant")
            for idx, m in enumerate(msgs):
                print(f"   Box #{idx+1}: {m.inner_text()}")

        browser.close()

        print("\n--- SUMMARY OF CAPTURED CONSOLE LOGS ---")
        for log in console_logs:
            print(log)

if __name__ == "__main__":
    capture_console_and_chat()
