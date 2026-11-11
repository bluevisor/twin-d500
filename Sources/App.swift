// The window, HUD, input handling and render loop.

import Cocoa
import Metal
import QuartzCore

final class MetalView: NSView {
    let metalLayer = CAMetalLayer()
    var onKey: ((NSEvent) -> Void)?
    var onDrag: ((CGFloat, CGFloat) -> Void)?
    var onScroll: ((CGFloat) -> Void)?

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
    }
    required init?(coder: NSCoder) { fatalError() }

    override func makeBackingLayer() -> CALayer { metalLayer }
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) { onKey?(event) }
    override func mouseDragged(with event: NSEvent) { onDrag?(event.deltaX, event.deltaY) }
    override func scrollWheel(with event: NSEvent) { onScroll?(event.scrollingDeltaY) }
    override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); updateDrawableSize() }
    override func viewDidChangeBackingProperties() { super.viewDidChangeBackingProperties(); updateDrawableSize() }

    func updateDrawableSize() {
        let scale = window?.backingScaleFactor ?? 1
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: max(bounds.width * scale, 16), height: max(bounds.height * scale, 16))
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    let gpus: GPUs
    let pixels: PixelRenderer
    let galaxy: NBody
    let selftestDir: URL?
    var window: NSWindow!
    var view: MetalView!
    var hud: NSTextField!
    var running = true
    var paused = false
    var time: Float = 0
    var scene: Scene = .mandelbulb
    var mode: GPUMode
    private let camLock = NSLock()
    private var yaw: Float = 0.6, pitch: Float = 0.18, dist: Float = 3.4
    private var lastInput: CFTimeInterval = 0

    init(gpus: GPUs, selftestDir: URL?) throws {
        self.gpus = gpus
        pixels = try PixelRenderer(gpus: gpus)
        galaxy = try NBody(gpus: gpus)
        self.selftestDir = selftestDir
        mode = gpus.hasTwo ? .both : .displayGPU
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        let frame = NSRect(x: 0, y: 0, width: 1600, height: 900)
        window = NSWindow(contentRect: frame, styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Twin D500"
        window.collectionBehavior = [.fullScreenPrimary]
        window.center()

        view = MetalView(frame: frame)
        let l = view.metalLayer
        l.device = gpus.devA
        l.pixelFormat = .bgra8Unorm
        l.framebufferOnly = false
        l.maximumDrawableCount = 3
        l.displaySyncEnabled = true
        l.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        window.contentView = view

        hud = NSTextField(labelWithString: "")
        hud.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .medium)
        hud.textColor = .white
        hud.drawsBackground = true
        hud.backgroundColor = NSColor(white: 0, alpha: 0.55)
        hud.frame = NSRect(x: 14, y: frame.height - 14 - 168, width: 700, height: 168)
        hud.autoresizingMask = [.minYMargin]
        hud.maximumNumberOfLines = 9
        view.addSubview(hud)

        view.onKey = { [weak self] e in self?.handleKey(e) }
        view.onDrag = { [weak self] dx, dy in
            guard let self else { return }
            self.camLock.lock()
            self.yaw -= Float(dx) * 0.006
            self.pitch = min(max(self.pitch + Float(dy) * 0.004, -0.15), 1.3)
            self.lastInput = CACurrentMediaTime()
            self.camLock.unlock()
        }
        view.onScroll = { [weak self] d in
            guard let self else { return }
            self.camLock.lock()
            self.dist = min(max(self.dist - Float(d) * 0.02, 1.6), 7)
            self.lastInput = CACurrentMediaTime()
            self.camLock.unlock()
        }

        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        view.updateDrawableSize()
        NSApp.activate(ignoringOtherApps: true)

        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.updateHUD() }
        let t = Thread { [weak self] in self?.renderLoop() }
        t.qualityOfService = .userInteractive
        t.start()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { running = false }

    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Toggle Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Twin D500", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        NSApp.mainMenu = main
    }

    private func setMode(_ m: GPUMode) {
        mode = gpus.hasTwo ? m : .displayGPU
        pixels.mode = mode
        galaxy.mode = mode
    }

    private func handleKey(_ e: NSEvent) {
        guard let c = e.charactersIgnoringModifiers?.lowercased() else { return }
        if let digit = Int(c), digit >= 1, digit <= Scene.allCases.count {
            scene = Scene.allCases[digit - 1]
            pixels.scene = scene
        } else {
            switch c {
            case "g": setMode(mode.next)
            case "t": pixels.tint.toggle(); galaxy.tint = pixels.tint
            case "q": pixels.quality = (pixels.quality + 1) % qualities.count
            case "m": pixels.morph.toggle()
            case "p": pixels.pipelined.toggle()
            case "r": galaxy.reset()
            case "v": view.metalLayer.displaySyncEnabled.toggle()
            case " ": paused.toggle()
            case "f": window.toggleFullScreen(nil)
            case "h": hud.isHidden.toggle()
            default:
                if e.keyCode == 53, window.styleMask.contains(.fullScreen) { window.toggleFullScreen(nil) }
                return
            }
        }
        pixels.timing.reset()
        galaxy.timing.reset()
    }

    private func updateHUD() {
        let isGalaxy = scene == .galaxy
        let s = isGalaxy ? galaxy.timing.snapshot() : pixels.timing.snapshot()
        let unit = isGalaxy ? "of stars" : "of rows "
        func line(_ name: String, _ share: Double, _ ms: Double) -> String {
            let msText = ms > 0 ? String(format: "%5.1f ms", ms) : "   idle"
            return String(format: " %@ %3d%% %@  %@", name, Int((share * 100).rounded()), unit, msText)
        }
        var lines: [String] = []
        var head = String(format: " TWIN D500   %@   %dx%d   %.1f fps", scene.title, s.width, s.height, s.fps)
        if !view.metalLayer.displaySyncEnabled { head += "  (vsync off)" }
        lines.append(head)
        var sub = " \(mode.label)"
        if isGalaxy {
            sub += String(format: "   %d stars, %.1f billion interactions/s", galaxy.n,
                          Double(galaxy.n) * Double(galaxy.n) * s.fps / 1e9)
        } else {
            sub += "   \(qualities[pixels.quality].name)"
            if pixels.morph && (scene == .mandelbulb || scene == .gallery) { sub += ", morphing" }
            sub += pixels.pipelined ? ", pipelined" : ", CPU-synced"
        }
        lines.append(sub)
        var a = line("GPU 0 display  ", s.shareA, s.msA)
        if isGalaxy && s.extraMs > 0 { a += String(format: " + %.1f ms drawing", s.extraMs) }
        lines.append(a)
        if gpus.hasTwo {
            var b = line("GPU 1 headless ", 1 - s.shareA, s.msB)
            if s.shareA < 1 { b += String(format: "   %.0f MB/s over PCIe", s.mbps) }
            lines.append(b)
        }
        lines.append("")
        lines.append(" [1-6] scene: bulb, canyon, tunnel, nebula, galaxy, gallery")
        lines.append(" [G] GPUs  [T] show split  [Q] quality  [M] morph  [P] pipelining  [R] restart galaxy")
        lines.append(" [V] vsync  [Space] pause  [F] full screen  [H] hide   Drag to orbit, scroll to zoom")
        hud.stringValue = lines.joined(separator: "\n")
    }

    private func camera(dt: Float, forGalaxy: Bool) -> CameraState {
        camLock.lock(); defer { camLock.unlock() }
        if !paused && CACurrentMediaTime() - lastInput > 3 { yaw += dt * 0.08 }
        if forGalaxy {
            let c = orbitCamera(yaw: yaw, pitch: max(pitch, 0.25), dist: dist * 3.6)
            return CameraState(eye: c.eye, target: .zero)
        }
        return orbitCamera(yaw: yaw, pitch: pitch, dist: dist)
    }

    private func drawFrame(_ drawable: CAMetalDrawable, dt: Float, present: Bool) -> MTLCommandBuffer {
        if scene == .galaxy {
            if paused { galaxy.waitIdle() }
            return galaxy.frame(target: drawable.texture, cam: camera(dt: dt, forGalaxy: true), present: present ? drawable : nil)
        }
        return pixels.renderFrame(target: drawable.texture, time: time, cam: camera(dt: dt, forGalaxy: false),
                                  present: present ? drawable : nil)
    }

    private func renderLoop() {
        var last = CACurrentMediaTime()
        var phaseStart = last
        var phase = 0
        var report: [String] = []
        if let dir = selftestDir {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            scene = Scene.allCases[0]
            pixels.scene = scene
        }

        while running {
            autoreleasepool {
                let now = CACurrentMediaTime()
                let dt = Float(min(now - last, 0.1))
                last = now
                if !paused { time += dt }
                guard let drawable = view.metalLayer.nextDrawable() else { return }

                if let dir = selftestDir, now - phaseStart > 4 {
                    // Capture the real drawable (proves the on-screen path), log stats, move to the next scene.
                    let s = scene == .galaxy ? galaxy.timing.snapshot() : pixels.timing.snapshot()
                    report.append(String(format: "%@: %.1f fps at %dx%d, GPU0 %.0f%% %.1f ms, GPU1 %.0f%% %.1f ms",
                                         scene.title, s.fps, s.width, s.height, s.shareA * 100, s.msA,
                                         (1 - s.shareA) * 100, s.msB))
                    drawFrame(drawable, dt: dt, present: false).waitUntilCompleted()
                    let data = gpus.readback(drawable.texture, present: drawable)
                    savePNG(data, width: drawable.texture.width, height: drawable.texture.height,
                            to: dir.appendingPathComponent("window-\(scene.rawValue + 1).png"))
                    phase += 1
                    phaseStart = now
                    if phase >= Scene.allCases.count {
                        try? (report.joined(separator: "\n") + "\n").write(
                            to: dir.appendingPathComponent("selftest.txt"), atomically: true, encoding: .utf8)
                        running = false
                        DispatchQueue.main.async { NSApp.terminate(nil) }
                        return
                    }
                    scene = Scene.allCases[phase]
                    pixels.scene = scene
                    pixels.timing.reset()
                    galaxy.timing.reset()
                    return
                }
                _ = drawFrame(drawable, dt: dt, present: true)
            }
        }
        pixels.waitIdle()
        galaxy.waitIdle()
    }
}
