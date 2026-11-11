// Offscreen benchmark: throughput per scene and GPU mode, plus correctness checks.

import Metal
import QuartzCore
import simd

private func pad(_ s: String, _ n: Int) -> String { s.padding(toLength: n, withPad: " ", startingAt: 0) }

func runBench(outDir: URL, only: String?) throws {
    let gpus = try GPUs()
    let pixels = try PixelRenderer(gpus: gpus)
    let galaxy = try NBody(gpus: gpus, bodies: Int(argValue("--bodies") ?? "") ?? 49152)
    let w = 2560, h = 1440
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.usage = [.shaderWrite, .shaderRead]
    desc.storageMode = .private
    let tex = gpus.devA.makeTexture(descriptor: desc)!
    let cam = orbitCamera(yaw: 0.6, pitch: 0.18, dist: 3.4)
    let galaxyCam = CameraState(eye: orbitCamera(yaw: 0.6, pitch: 0.35, dist: 12).eye, target: .zero)
    let modes: [GPUMode] = gpus.hasTwo ? [.displayGPU, .headlessGPU, .both] : [.displayGPU]

    var report: [String] = []
    func say(_ s: String) { print(s); report.append(s) }
    say("Display GPU:  \(gpus.devA.name)")
    say("Headless GPU: \(gpus.devB?.name ?? "none")")
    say("Frame: \(w)x\(h), quality High\n")

    /// Frames per second for the current settings (after a warm-up that lets the split settle).
    func measurePixels(frames: Int, time t0: Float = 2) -> Double {
        pixels.timing.reset()
        pixels.splitFraction = 0.5
        var t = t0
        for _ in 0..<max(frames / 3, 4) { pixels.renderFrame(target: tex, time: t, cam: cam, present: nil); t += 1 / 60 }
        pixels.waitIdle()
        let start = CACurrentMediaTime()
        var last: MTLCommandBuffer?
        for _ in 0..<frames { last = pixels.renderFrame(target: tex, time: t, cam: cam, present: nil); t += 1 / 60 }
        last?.waitUntilCompleted()
        return Double(frames) / (CACurrentMediaTime() - start)
    }

    let pixelScenes: [Scene] = [.mandelbulb, .canyon, .sponge, .nebula, .gallery]
        .filter { only == nil || $0.title.lowercased().contains(only!) }
    if !pixelScenes.isEmpty {
        say(pad("Scene", 22) + pad("display GPU", 13) + pad("headless GPU", 14) + pad("both GPUs", 11) + "speedup")
    }
    for scene in pixelScenes {
        pixels.scene = scene
        pixels.quality = 1
        var fps: [GPUMode: Double] = [:]
        for m in modes { pixels.mode = m; fps[m] = measurePixels(frames: 30) }
        var row = pad(scene.title, 22)
        for m in modes { row += pad(String(format: "%5.1f fps", fps[m]!), m == .both ? 11 : (m == .displayGPU ? 13 : 14)) }
        if let one = fps[.displayGPU], let two = fps[.both] { row += String(format: "%.2fx", two / one) }
        say(row)
        if gpus.hasTwo {
            pixels.mode = .both
            pixels.tint = false
            pixels.renderFrame(target: tex, time: 6, cam: cam, present: nil).waitUntilCompleted()
            savePNG(gpus.readback(tex), width: w, height: h,
                    to: outDir.appendingPathComponent("scene-\(scene.rawValue + 1).png"))
        }
    }

    if gpus.hasTwo && (only == nil || only == "mandelbulb") {
        say("\nMandelbulb, both GPUs:")
        pixels.scene = .mandelbulb
        pixels.mode = .both
        pixels.morph = true; pixels.pipelined = false
        say(String(format: "  trig formula, CPU waits for GPU 1       %5.1f fps", measurePixels(frames: 30)))
        pixels.pipelined = true
        say(String(format: "  + GPUs wait on each other (pipelined)   %5.1f fps", measurePixels(frames: 30)))
        pixels.morph = false
        say(String(format: "  + trig-free power-8 formula             %5.1f fps", measurePixels(frames: 30)))
    }

    // Correctness: a split frame must match a single-GPU frame.
    if gpus.hasTwo && only == nil {
        say("")
        for scene in [Scene.mandelbulb, .gallery] {
            pixels.scene = scene
            pixels.tint = false
            pixels.mode = .displayGPU
            pixels.renderFrame(target: tex, time: 5, cam: cam, present: nil).waitUntilCompleted()
            let single = gpus.readback(tex)
            pixels.clearSlots()
            pixels.mode = .both
            pixels.renderFrame(target: tex, time: 5, cam: cam, present: nil).waitUntilCompleted()
            let dual = gpus.readback(tex)
            var maxDiff = 0, differing = 0
            single.withUnsafeBytes { (a: UnsafeRawBufferPointer) in
                dual.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
                    for i in 0..<a.count {
                        let d = abs(Int(a[i]) - Int(b[i]))
                        if d > 0 { differing += 1; maxDiff = max(maxDiff, d) }
                    }
                }
            }
            say("Check \(scene.title): split frame vs one GPU, \(differing) of \(single.count) bytes differ, max \(maxDiff)/255")
        }
        pixels.tint = true
        pixels.scene = .gallery
        pixels.renderFrame(target: tex, time: 6, cam: cam, present: nil).waitUntilCompleted()
        savePNG(gpus.readback(tex), width: w, height: h, to: outDir.appendingPathComponent("gallery-split-tinted.png"))
        pixels.tint = false
    }

    // Galaxy collision.
    if only == nil || only == "galaxy" {
        say("\nGalaxy collision, \(galaxy.n) stars (all-pairs gravity):")
        var sps: [GPUMode: Double] = [:]
        for m in modes {
            galaxy.mode = m
            galaxy.split = 0.5
            galaxy.timing.reset()
            for _ in 0..<15 { galaxy.frame(target: tex, cam: galaxyCam, present: nil) }
            galaxy.waitIdle()
            let frames = 40
            let start = CACurrentMediaTime()
            var last: MTLCommandBuffer?
            for _ in 0..<frames { last = galaxy.frame(target: tex, cam: galaxyCam, present: nil) }
            last?.waitUntilCompleted()
            let v = Double(frames) / (CACurrentMediaTime() - start)
            sps[m] = v
            let inter = Double(galaxy.n) * Double(galaxy.n) * v / 1e9
            var row = "  " + pad(m.label, 20) + String(format: "%5.1f steps/s  %5.1f billion interactions/s", v, inter)
            if m == .both { row += String(format: "  (GPU 0 takes %.0f%% of stars)", galaxy.timing.snapshot().shareA * 100) }
            say(row)
        }
        if let one = sps[.displayGPU], let two = sps[.both] { say(String(format: "  speedup with both GPUs: %.2fx", two / one)) }

        if gpus.hasTwo {
            // Same start, same number of steps: one GPU vs both must give the same stars.
            galaxy.mode = .displayGPU
            galaxy.reset()
            for _ in 0..<60 { galaxy.frame(target: tex, cam: galaxyCam, present: nil) }
            let a = galaxy.readPositions()
            galaxy.mode = .both
            galaxy.split = 0.5
            galaxy.reset()
            for _ in 0..<60 { galaxy.frame(target: tex, cam: galaxyCam, present: nil) }
            let b = galaxy.readPositions()
            var maxErr: Float = 0, differing = 0
            for i in 0..<a.count {
                let e = simd_length(SIMD3(a[i].x, a[i].y, a[i].z) - SIMD3(b[i].x, b[i].y, b[i].z))
                if e > 0 { differing += 1; maxErr = max(maxErr, e) }
            }
            say("  Check: after 60 steps, \(differing) of \(a.count) stars differ between one GPU and both (max \(maxErr))")
        }
        galaxy.mode = gpus.hasTwo ? .both : .displayGPU
        galaxy.reset()
        for _ in 0..<260 { galaxy.frame(target: tex, cam: galaxyCam, present: nil) }
        galaxy.waitIdle()
        savePNG(gpus.readback(tex), width: w, height: h, to: outDir.appendingPathComponent("scene-5.png"))
    }

    try (report.joined(separator: "\n") + "\n").write(to: outDir.appendingPathComponent("bench.txt"), atomically: true, encoding: .utf8)
    print("\nWrote \(outDir.path)/bench.txt and scene PNGs")
}
