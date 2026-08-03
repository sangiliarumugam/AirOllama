#!/bin/bash
set -e

echo "=========================================================="
echo "🔨 Building Native AirOllama macOS Application Bundle"
echo "=========================================================="

APP_DIR="dist/AirOllama.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RESOURCES_DIR="$APP_DIR/Contents/Resources"

mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

echo "⚡ Compiling Swift AppKit + WebKit Native Binary..."
swiftc -O mac_app/main.swift -framework AppKit -framework WebKit -o "$MACOS_DIR/AirOllama"

echo "📄 Creating Info.plist..."
cat << 'EOF' > "$RESOURCES_DIR/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>AirOllama</string>
    <key>CFBundleIdentifier</key>
    <string>com.sangiliarumugam.AirOllama</string>
    <key>CFBundleName</key>
    <string>AirOllama</string>
    <key>CFBundleDisplayName</key>
    <string>AirOllama</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
</dict>
</plist>
EOF

chmod +x "$MACOS_DIR/AirOllama"
ln -sfh "$(pwd)/$APP_DIR" ./AirOllama.app

echo "=========================================================="
echo "✅ Native macOS App Built Successfully!"
echo "📦 App Location: ./dist/AirOllama.app (Shortcut: ./AirOllama.app)"
echo "🚀 To launch: open dist/AirOllama.app"
echo "=========================================================="
