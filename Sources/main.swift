// Twin D500: split-frame rendering across both FirePro D500s in a Mac Pro 6,1.
//
// The GPU that drives the display renders the top band of each frame straight
// into the window. The headless GPU renders the bottom band into a page-aligned
// block of system memory that both GPUs map (no CPU copy). Once the headless
// GPU finishes, the display GPU copies that band into the frame and presents.
// The split line moves every frame so both GPUs finish at the same time.

import Cocoa
import Metal
import QuartzCore
import ImageIO
import UniformTypeIdentifiers

struct Uniforms {
    var resTime: SIMD4<Float>
    var camPos: SIMD4<Float>
    var camTarget: SIMD4<Float>
    var rows: SIMD4<Int32>
    var flags: SIMD4<Int32>
}

enum Mode: Int {
    case displayGPU = 1, both = 2, headlessGPU = 3
    var label: String {
        switch self {
        case .displayGPU: return "Display GPU only"
        case .both: return "Both GPUs, split frame"
        case .headlessGPU: return "Headless GPU only"
        }
    }
}

struct Quality { let name: String; let steps: Int32; let samples: Int32; let extras: Int32 }
let qualities = [
    Quality(name: "Fast", steps: 110, samples: 1, extras: 0),
    Quality(name: "High", steps: 180, samples: 1, extras: 1),
    Quality(name: "Ultra 4x AA", steps: 220, samples: 2, extras: 1),
]

struct CameraState { var eye: SIMD3<Float>; var target: SIMD3<Float> }

struct Stats {
    var fps = 0.0, msA = 0.0, msB = 0.0, mbps = 0.0
    var rowsA = 0, rowsB = 0, width = 0, height = 0
}

struct RenderError: Error, CustomStringConvertible { let description: String }

final class SharedSlot {
    let a: MTLBuffer          // display GPU's view of the memory
    let b: MTLBuffer?         // headless GPU's view of the same memory
    var lastUse: MTLCommandBuffer?
    init(a: MTLBuffer, b: MTLBuffer?) { self.a = a; self.b = b }
}

final class Renderer {
    let devA: MTLDevice
    let devB: MTLDevice?
    let queueA: MTLCommandQueue
    let queueB: MTLCommandQueue?
    let texPipe: MTLComputePipelineState
    let bufPipe: MTLComputePipelineState?
    let maxW = 4096, maxH = 2304
    private var slots: [SharedSlot] = []
    private var slotIndex = 0

    var mode: Mode
    var quality = 1
    var tint = false
    var splitFraction = 0.5   // share of rows the display GPU renders in .both

    private let lock = NSLock()
    private var _stats = Stats()
    private var rateA = 0.0, rateB = 0.0   // rows per ms, latest measurement
    private var lastCall: CFTimeInterval = 0

    var stats: Stats { lock.lock(); defer { lock.unlock() }; return _stats }

    init() throws {
        let all = MTLCopyAllDevices()
        guard !all.isEmpty else { throw RenderError(description: "no Metal devices") }
        let display = CGDirectDisplayCopyCurrentMetalDevice(CGMainDisplayID())
        let a = all.first { $0.registryID == display?.registryID } ?? all[0]
        devA = a
        devB = all.first { $0.registryID != a.registryID }

        let src = try Renderer.shaderSource()
        let libA = try a.makeLibrary(source: src, options: nil)
        texPipe = try a.makeComputePipelineState(function: libA.makeFunction(name: "renderTex")!)
        queueA = a.makeCommandQueue()!
        if let b = devB {
            let libB = try b.makeLibrary(source: src, options: nil)
            bufPipe = try b.makeComputePipelineState(function: libB.makeFunction(name: "renderBuf")!)
            queueB = b.makeCommandQueue()
        } else {
            bufPipe = nil
            queueB = nil
        }
        mode = devB == nil ? .displayGPU : .both

        let page = Int(getpagesize())
        let len = (maxW * maxH * 4 + page - 1) / page * page
        for _ in 0..<2 {
            var p: UnsafeMutableRawPointer?
            guard posix_memalign(&p, page, len) == 0, let mem = p else {
                throw RenderError(description: "posix_memalign failed")
            }
            memset(mem, 0, len)
            guard let ba = a.makeBuffer(bytesNoCopy: mem, length: len, options: .storageModeShared, deallocator: nil) else {
                throw RenderError(description: "display GPU could not map shared memory")
            }
            let bb = devB?.makeBuffer(bytesNoCopy: mem, length: len, options: .storageModeShared, deallocator: nil)
            if devB != nil && bb == nil { throw RenderError(description: "headless GPU could not map shared memory") }
            slots.append(SharedSlot(a: ba, b: bb))
        }
    }

    static func shaderSource() throws -> String {
        var candidates: [URL] = []
        if let u = Bundle.main.url(forResource: "Shaders", withExtension: "metal") { candidates.append(u) }
        let exeDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        candidates.append(exeDir.appendingPathComponent("../Resources/Shaders.metal"))
        candidates.append(URL(fileURLWithPath: "Sources/Shaders.metal"))
        for u in candidates {
            if let s = try? String(contentsOf: u, encoding: .utf8) { return s }
        }
        throw RenderError(description: "Shaders.metal not found")
    }

    var hasTwoGPUs: Bool { devB != nil }

    private func uniforms(w: Int, h: Int, time: Float, cam: CameraState, first: Int, count: Int, gpu: Int32) -> Uniforms {
        let q = qualities[quality]
        let power: Float = 8.0 + 1.6 * sin(time * 0.21)
        return Uniforms(
            resTime: SIMD4(Float(w), Float(h), time, power),
            camPos: SIMD4(cam.eye, 0),
            camTarget: SIMD4(cam.target, 0),
            rows: SIMD4(Int32(first), Int32(count), q.samples, q.steps),
            flags: SIMD4(tint ? 1 : 0, gpu, q.extras, Int32(w)))
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, w: Int, rows: Int) {
        enc.dispatchThreadgroups(MTLSize(width: (w + 7) / 8, height: (rows + 7) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }

    private func record(gpu: Int, rows: Int, cb: MTLCommandBuffer) {
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        guard ms > 0 else { return }
        lock.lock()
        if gpu == 0 { _stats.msA = _stats.msA * 0.8 + ms * 0.2; rateA = Double(rows) / ms }
        else { _stats.msB = _stats.msB * 0.8 + ms * 0.2; rateB = Double(rows) / ms }
        lock.unlock()
    }

    /// Renders one frame into `target` (a drawable or an offscreen texture).
    /// Returns the final command buffer on the display GPU (already committed).
    @discardableResult
    func renderFrame(target: MTLTexture, time: Float, cam: CameraState, present drawable: MTLDrawable?) -> MTLCommandBuffer {
        let w = min(target.width, maxW), h = min(target.height, maxH)
        let m: Mode = hasTwoGPUs ? mode : .displayGPU

        if m == .both {
            lock.lock(); let ra = rateA, rb = rateB; lock.unlock()
            if ra > 0 && rb > 0 {
                let ideal = ra / (ra + rb)
                splitFraction += (ideal - splitFraction) * 0.15
                splitFraction = min(max(splitFraction, 0.05), 0.95)
            }
        }
        let rowsA: Int
        switch m {
        case .displayGPU: rowsA = h
        case .headlessGPU: rowsA = 0
        case .both: rowsA = min(max(Int((Double(h) * splitFraction).rounded()), 8), h - 8)
        }
        let rowsB = h - rowsA

        let slot = slots[slotIndex]
        slotIndex = (slotIndex + 1) % slots.count
        slot.lastUse?.waitUntilCompleted()   // the copy that last read this memory must be done

        var cbB: MTLCommandBuffer?
        if rowsB > 0, let qB = queueB, let pipe = bufPipe, let bb = slot.b {
            let cb = qB.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipe)
            var u = uniforms(w: w, h: h, time: time, cam: cam, first: rowsA, count: rowsB, gpu: 1)
            enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            enc.setBuffer(bb, offset: 0, index: 1)
            dispatch(enc, w: w, rows: rowsB)
            enc.endEncoding()
            cb.addCompletedHandler { [weak self] cb in self?.record(gpu: 1, rows: rowsB, cb: cb) }
            cb.commit()
            cbB = cb
        }
        if rowsA > 0 {
            let cb = queueA.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(texPipe)
            var u = uniforms(w: w, h: h, time: time, cam: cam, first: 0, count: rowsA, gpu: 0)
            enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            enc.setTexture(target, index: 0)
            dispatch(enc, w: w, rows: rowsA)
            enc.endEncoding()
            cb.addCompletedHandler { [weak self] cb in self?.record(gpu: 0, rows: rowsA, cb: cb) }
            cb.commit()
        }

        cbB?.waitUntilCompleted()
        let cbC = queueA.makeCommandBuffer()!
        if rowsB > 0 {
            let blit = cbC.makeBlitCommandEncoder()!
            blit.copy(from: slot.a, sourceOffset: 0, sourceBytesPerRow: w * 4, sourceBytesPerImage: w * 4 * rowsB,
                      sourceSize: MTLSize(width: w, height: rowsB, depth: 1),
                      to: target, destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: rowsA, z: 0))
            blit.endEncoding()
        }
        if let d = drawable { cbC.present(d) }
        cbC.commit()
        slot.lastUse = cbC

        let now = CACurrentMediaTime()
        lock.lock()
        if lastCall > 0 {
            let inst = 1.0 / max(now - lastCall, 1e-6)
            _stats.fps = _stats.fps == 0 ? inst : _stats.fps * 0.9 + inst * 0.1
        }
        _stats.rowsA = rowsA; _stats.rowsB = rowsB; _stats.width = w; _stats.height = h
        if m == .displayGPU { _stats.msB = 0 }
        if m == .headlessGPU { _stats.msA = 0 }
        _stats.mbps = Double(rowsB * w * 4) * _stats.fps / 1e6
        lock.unlock()
        lastCall = now
        return cbC
    }

    func resetTiming() {
        lock.lock(); _stats = Stats(); rateA = 0; rateB = 0; lock.unlock()
        lastCall = 0
    }

    /// Copies a texture into CPU memory (BGRA8). Optionally presents a drawable afterward.
    func readback(_ tex: MTLTexture, present: MTLDrawable? = nil) -> Data {
        let w = tex.width, h = tex.height
        let buf = devA.makeBuffer(length: w * h * 4, options: .storageModeShared)!
        let cb = queueA.makeCommandBuffer()!
        let bl = cb.makeBlitCommandEncoder()!
        bl.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: w, height: h, depth: 1),
                to: buf, destinationOffset: 0, destinationBytesPerRow: w * 4, destinationBytesPerImage: w * h * 4)
        bl.endEncoding()
        if let p = present { cb.present(p) }
        cb.commit()
        cb.waitUntilCompleted()
        return Data(bytes: buf.contents(), count: w * h * 4)
    }
}

func savePNG(_ data: Data, width: Int, height: Int, to url: URL) {
    guard let provider = CGDataProvider(data: data as CFData),
          let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

func orbitCamera(yaw: Float, pitch: Float, dist: Float) -> CameraState {
    let target = SIMD3<Float>(0, -0.15, 0)
    let eye = target + dist * SIMD3<Float>(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
    return CameraState(eye: eye, target: target)
}

func gpuLine(_ name: String, _ share: Int, _ total: Int, _ ms: Double) -> String {
    let pct = total > 0 ? Int((Double(share) / Double(total) * 100).rounded()) : 0
    let msText = ms > 0 ? String(format: "%5.1f ms", ms) : "   idle"
    return String(format: "%@ %3d%% of rows  %@", name, pct, msText)
}

// MARK: - Offscreen benchmark (no window): raw throughput per mode + a correctness check.

func runBench(outDir: URL) throws {
    let r = try Renderer()
    let w = 2560, h = 1440
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.usage = [.shaderWrite, .shaderRead]
    desc.storageMode = .private
    let tex = r.devA.makeTexture(descriptor: desc)!
    let cam = orbitCamera(yaw: 0.6, pitch: 0.18, dist: 3.4)

    print("Display GPU:  \(r.devA.name) (registry \(r.devA.registryID))")
    print("Headless GPU: \(r.devB.map { "\($0.name) (registry \($0.registryID))" } ?? "none")")
    print("Frame: \(w)x\(h)\n")

    var report: [String] = []
    for qi in [0, 1, 2] {
        r.quality = qi
        var fpsByMode: [Mode: Double] = [:]
        let modes: [Mode] = r.hasTwoGPUs ? [.displayGPU, .headlessGPU, .both] : [.displayGPU]
        for mode in modes {
            r.mode = mode
            r.splitFraction = 0.5
            r.resetTiming()
            var t: Float = 2.0
            let frames = qi == 2 ? 12 : 40
            for _ in 0..<(frames / 3) { r.renderFrame(target: tex, time: t, cam: cam, present: nil).waitUntilCompleted(); t += 1 / 60 }
            let start = CACurrentMediaTime()
            var last: MTLCommandBuffer?
            for _ in 0..<frames { last = r.renderFrame(target: tex, time: t, cam: cam, present: nil); t += 1 / 60 }
            last?.waitUntilCompleted()
            let fps = Double(frames) / (CACurrentMediaTime() - start)
            fpsByMode[mode] = fps
            let s = r.stats
            var line = qualities[qi].name.padding(toLength: 12, withPad: " ", startingAt: 0) + mode.label.padding(toLength: 25, withPad: " ", startingAt: 0) + String(format: "%6.1f fps", fps)
            if mode == .both { line += String(format: "   split %d/%d rows, %.0f MB/s over PCIe", s.rowsA, s.rowsB, s.mbps) }
            print(line); report.append(line)
        }
        if let one = fpsByMode[.displayGPU], let two = fpsByMode[.both] {
            let line = qualities[qi].name.padding(toLength: 12, withPad: " ", startingAt: 0) + String(format: "speedup with both GPUs: %.2fx\n", two / one)
            print(line); report.append(line)
        }
    }

    // Correctness: the split frame must match a single-GPU frame pixel for pixel (within rounding).
    r.quality = 1
    r.tint = false
    r.mode = .displayGPU
    r.renderFrame(target: tex, time: 5, cam: cam, present: nil).waitUntilCompleted()
    let single = r.readback(tex)
    var dual = single
    if r.hasTwoGPUs {
        r.mode = .both
        r.splitFraction = 0.5
        r.resetTiming()
        r.renderFrame(target: tex, time: 5, cam: cam, present: nil).waitUntilCompleted()
        dual = r.readback(tex)
        var maxDiff = 0, differing = 0
        single.withUnsafeBytes { (a: UnsafeRawBufferPointer) in
            dual.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
                for i in 0..<a.count {
                    let d = abs(Int(a[i]) - Int(b[i]))
                    if d > 0 { differing += 1; maxDiff = max(maxDiff, d) }
                }
            }
        }
        let line = "Correctness: split frame vs single GPU, \(differing) of \(single.count) bytes differ, max difference \(maxDiff)/255"
        print(line); report.append(line)
    }
    savePNG(dual, width: w, height: h, to: outDir.appendingPathComponent("frame.png"))
    r.tint = true
    r.renderFrame(target: tex, time: 5, cam: cam, present: nil).waitUntilCompleted()
    savePNG(r.readback(tex), width: w, height: h, to: outDir.appendingPathComponent("frame-split-tinted.png"))
    try (report.joined(separator: "\n") + "\n").write(to: outDir.appendingPathComponent("bench.txt"), atomically: true, encoding: .utf8)
    print("\nWrote \(outDir.path)/frame.png, frame-split-tinted.png, bench.txt")
}

// MARK: - Window app

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
    let renderer: Renderer
    let selftestDir: URL?
    var window: NSWindow!
    var view: MetalView!
    var hud: NSTextField!
    var running = true
    var paused = false
    var time: Float = 0
    private let camLock = NSLock()
    private var yaw: Float = 0.6, pitch: Float = 0.18, dist: Float = 3.4
    private var lastInput: CFTimeInterval = 0

    init(renderer: Renderer, selftestDir: URL?) {
        self.renderer = renderer
        self.selftestDir = selftestDir
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        let frame = NSRect(x: 0, y: 0, width: 1600, height: 900)
        window = NSWindow(contentRect: frame, styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "Twin D500"
        window.collectionBehavior = [.fullScreenPrimary]
        window.center()

        view = MetalView(frame: frame)
        let l = view.metalLayer
        l.device = renderer.devA
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
        hud.frame = NSRect(x: 14, y: frame.height - 14 - 150, width: 640, height: 150)
        hud.autoresizingMask = [.minYMargin]
        hud.maximumNumberOfLines = 8
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

    private func handleKey(_ e: NSEvent) {
        guard let c = e.charactersIgnoringModifiers?.lowercased() else { return }
        switch c {
        case "1": renderer.mode = .displayGPU
        case "2": if renderer.hasTwoGPUs { renderer.mode = .both }
        case "3": if renderer.hasTwoGPUs { renderer.mode = .headlessGPU }
        case "t": renderer.tint.toggle()
        case "q": renderer.quality = (renderer.quality + 1) % qualities.count
        case "v": view.metalLayer.displaySyncEnabled.toggle()
        case " ": paused.toggle()
        case "f": window.toggleFullScreen(nil)
        case "h": hud.isHidden.toggle()
        default:
            if e.keyCode == 53, window.styleMask.contains(.fullScreen) { window.toggleFullScreen(nil) }
            return
        }
        renderer.resetTiming()
    }

    private func updateHUD() {
        let s = renderer.stats
        let q = qualities[renderer.quality]
        var lines: [String] = []
        lines.append(String(format: " TWIN D500   %dx%d   %@   %.1f fps%@", s.width, s.height, q.name, s.fps,
                            view.metalLayer.displaySyncEnabled ? "" : "  (vsync off)"))
        lines.append(" Mode: \(renderer.mode.label)")
        lines.append(" " + gpuLine("GPU 0 display  (top)   ", s.rowsA, s.height, s.msA))
        if renderer.hasTwoGPUs {
            var l = " " + gpuLine("GPU 1 headless (bottom)", s.rowsB, s.height, s.msB)
            if s.rowsB > 0 { l += String(format: "  %.0f MB/s PCIe", s.mbps) }
            lines.append(l)
        }
        lines.append("")
        lines.append(" [1] display GPU  [2] both  [3] headless GPU  [T] tint split")
        lines.append(" [Q] quality  [V] vsync  [Space] pause  [F] full screen  [H] hide")
        lines.append(" Drag to orbit, scroll to zoom")
        hud.stringValue = lines.joined(separator: "\n")
    }

    private func camera(dt: Float) -> CameraState {
        camLock.lock(); defer { camLock.unlock() }
        if !paused && CACurrentMediaTime() - lastInput > 3 { yaw += dt * 0.08 }
        return orbitCamera(yaw: yaw, pitch: pitch, dist: dist)
    }

    private func renderLoop() {
        var last = CACurrentMediaTime()
        var frames = 0
        var phaseStart = last
        var phase = 0
        let phases: [Mode] = renderer.hasTwoGPUs ? [.displayGPU, .headlessGPU, .both] : [.displayGPU]
        var report: [String] = []
        if let dir = selftestDir {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            renderer.mode = phases[0]
        }

        while running {
            autoreleasepool {
                let now = CACurrentMediaTime()
                let dt = Float(min(now - last, 0.1))
                last = now
                if !paused { time += dt }
                let cam = camera(dt: dt)
                guard let drawable = view.metalLayer.nextDrawable() else { return }

                if let dir = selftestDir, now - phaseStart > 4 {
                    // Capture the real drawable (proves the on-screen path), log stats, advance.
                    let s = renderer.stats
                    report.append(String(format: "%@: %.1f fps, %dx%d, GPU0 %.1f ms (%d rows), GPU1 %.1f ms (%d rows)",
                                         renderer.mode.label, s.fps, s.width, s.height, s.msA, s.rowsA, s.msB, s.rowsB))
                    renderer.renderFrame(target: drawable.texture, time: time, cam: cam, present: nil).waitUntilCompleted()
                    let data = renderer.readback(drawable.texture, present: drawable)
                    savePNG(data, width: drawable.texture.width, height: drawable.texture.height,
                            to: dir.appendingPathComponent("window-\(renderer.mode.rawValue).png"))
                    phase += 1
                    phaseStart = now
                    if phase >= phases.count {
                        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                        try? (report.joined(separator: "\n") + "\n").write(to: dir.appendingPathComponent("selftest.txt"), atomically: true, encoding: .utf8)
                        running = false
                        DispatchQueue.main.async { NSApp.terminate(nil) }
                        return
                    }
                    renderer.mode = phases[phase]
                    renderer.resetTiming()
                    return
                }
                renderer.renderFrame(target: drawable.texture, time: time, cam: cam, present: drawable)
                frames += 1
            }
        }
    }
}

// MARK: - Entry point

let args = CommandLine.arguments
func argValue(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag) else { return nil }
    return i + 1 < args.count ? args[i + 1] : "out"
}

if let dir = argValue("--bench") {
    do { try runBench(outDir: URL(fileURLWithPath: dir)) } catch { print("error: \(error)"); exit(1) }
    exit(0)
}

let renderer: Renderer
do { renderer = try Renderer() } catch { print("error: \(error)"); exit(1) }
let app = NSApplication.shared
let delegate = AppDelegate(renderer: renderer, selftestDir: argValue("--selftest").map { URL(fileURLWithPath: $0) })
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
