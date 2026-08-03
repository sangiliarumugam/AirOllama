import AppKit
import WebKit
import Foundation

// MARK: - AirOllama Native macOS Application

class AirOllamaApp: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    
    var window: NSWindow!
    var webView: WKWebView!
    var statusItem: NSStatusItem!
    var serverProcess: Process?
    let serverURL = URL(string: "http://127.0.0.1:11211")!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        checkAndStartServer()
        setupWindow()
    }
    
    func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.title = "⚡ AirOllama"
            button.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        }
        
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "🖥️ Show AirOllama Dashboard", action: #selector(showDashboard), keyEquivalent: "d"))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "⚡ Server: http://127.0.0.1:11211", action: nil, keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "📂 Open Models Directory", action: #selector(openModelsFolder), keyEquivalent: "m"))
        menu.addItem(NSMenuItem(title: "📄 View Logs", action: #selector(openLogsFolder), keyEquivalent: "l"))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "🚪 Quit AirOllama", action: #selector(quitApp), keyEquivalent: "q"))
        
        statusItem.menu = menu
    }
    
    func setupWindow() {
        let windowMask: NSWindow.StyleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 840), styleMask: windowMask, backing: .buffered, defer: false)
        window.center()
        window.title = "AirOllama"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.backgroundColor = NSColor(red: 0.05, green: 0.07, blue: 0.12, alpha: 1.0)
        window.isMovableByWindowBackground = true
        window.delegate = self
        
        let config = WKWebViewConfiguration()
        let customUserAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) AirOllamaMac/1.0"
        config.applicationNameForUserAgent = customUserAgent
        
        webView = WKWebView(frame: window.contentView!.bounds, configuration: config)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.setValue(false, forKey: "drawsBackground")
        
        window.contentView!.addSubview(webView)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        
        loadDashboard()
    }
    
    func loadDashboard() {
        let request = URLRequest(url: serverURL, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 10.0)
        webView.load(request)
    }
    
    func checkAndStartServer() {
        let task = URLSession.shared.dataTask(with: URL(string: "http://127.0.0.1:11211/api/version")!) { [weak self] (data, response, error) in
            if error != nil {
                print("⚡ Server not detected on 11211, starting background server process...")
                DispatchQueue.main.async {
                    self?.launchServerScript()
                }
            } else {
                print("✅ AirOllama server is active on http://127.0.0.1:11211")
            }
        }
        task.resume()
    }
    
    func launchServerScript() {
        let currentDir = FileManager.default.currentDirectoryPath
        let scriptPath = (currentDir as NSString).appendingPathComponent("run_server.sh")
        
        if FileManager.default.fileExists(atPath: scriptPath) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/bash")
            proc.arguments = [scriptPath, "11211"]
            proc.currentDirectoryURL = URL(fileURLWithPath: currentDir)
            do {
                try proc.run()
                self.serverProcess = proc
                print("🚀 Launched AirOllama server process (PID: \(proc.processIdentifier))")
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                    self?.loadDashboard()
                }
            } catch {
                print("Error launching server process: \(error)")
            }
        }
    }
    
    @objc func showDashboard() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        loadDashboard()
    }
    
    @objc func openModelsFolder() {
        let modelsDir = (FileManager.default.currentDirectoryPath as NSString).appendingPathComponent("models")
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: modelsDir)
    }
    
    @objc func openLogsFolder() {
        let logPath = (FileManager.default.currentDirectoryPath as NSString).appendingPathComponent("airollama.log")
        if FileManager.default.fileExists(atPath: logPath) {
            NSWorkspace.shared.open(URL(fileURLWithPath: logPath))
        }
    }
    
    @objc func quitApp() {
        if let proc = serverProcess, proc.isRunning {
            proc.terminate()
        }
        NSApp.terminate(nil)
    }
    
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        window.orderOut(nil)
        return false
    }
    
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        let offlineHTML = """
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            body { background: #0b0f19; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; text-align: center; }
            .badge { background: linear-gradient(135deg, #00f2fe, #4facfe); color: #000; font-weight: bold; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; margin-bottom: 1rem; }
            h2 { margin: 0 0 0.5rem 0; font-size: 1.5rem; }
            p { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; }
            button { background: rgba(0, 242, 254, 0.15); border: 1px solid rgba(0, 242, 254, 0.4); color: #00f2fe; padding: 10px 20px; border-radius: 10px; font-weight: bold; cursor: pointer; font-size: 0.9rem; }
            button:hover { background: rgba(0, 242, 254, 0.25); }
          </style>
        </head>
        <body>
          <div class="badge">AirOllama Native App</div>
          <h2>⌛ Connecting to AirOllama Server...</h2>
          <p>Listening on http://127.0.0.1:11211</p>
          <button onclick="location.reload()">Retry Connection</button>
          <script>
            setTimeout(() => { location.reload(); }, 2000);
          </script>
        </body>
        </html>
        """
        webView.loadHTMLString(offlineHTML, baseURL: nil)
    }
}

let app = NSApplication.shared
let delegate = AirOllamaApp()
app.delegate = delegate
app.run()
