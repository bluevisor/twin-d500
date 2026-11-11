#!/bin/bash
# Builds TwinD500.app. Usage: ./build.sh [--run | --bench]
set -euo pipefail
cd "$(dirname "$0")"

APP=build/TwinD500.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

swiftc -O -swift-version 5 -o "$APP/Contents/MacOS/TwinD500" Sources/main.swift \
  -framework Cocoa -framework Metal -framework QuartzCore
cp Sources/Shaders.metal "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Twin D500</string>
  <key>CFBundleDisplayName</key><string>Twin D500</string>
  <key>CFBundleIdentifier</key><string>org.bluevisor.twin-d500</string>
  <key>CFBundleExecutable</key><string>TwinD500</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP" >/dev/null
echo "built $APP"

case "${1:-}" in
  --run)   open "$APP" ;;
  --bench) "$APP/Contents/MacOS/TwinD500" --bench out ;;
esac
