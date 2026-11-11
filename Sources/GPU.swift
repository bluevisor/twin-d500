// Shared types, the two-GPU setup, timing and small utilities.

import Cocoa
import Metal
import ImageIO
import UniformTypeIdentifiers
import simd

struct Uniforms {
    var resTime: SIMD4<Float>
    var camPos: SIMD4<Float>
    var camTarget: SIMD4<Float>
    var rows: SIMD4<Int32>
    var flags: SIMD4<Int32>
    var scene: SIMD4<Int32>
}

enum GPUMode: Int {
    case displayGPU = 1, both = 2, headlessGPU = 3
    var label: String {
        switch self {
        case .displayGPU: return "Display GPU only"
        case .both: return "Both GPUs"
        case .headlessGPU: return "Headless GPU only"
        }
    }
    var next: GPUMode {
        switch self {
        case .displayGPU: return .both
        case .both: return .headlessGPU
        case .headlessGPU: return .displayGPU
        }
    }
}

enum Scene: Int, CaseIterable {
    case mandelbulb, canyon, sponge, nebula, galaxy, gallery
    var title: String {
        switch self {
        case .mandelbulb: return "Mandelbulb"
        case .canyon: return "Canyon flyover"
        case .sponge: return "Menger tunnel"
        case .nebula: return "Nebula"
        case .galaxy: return "Galaxy collision"
        case .gallery: return "Gallery"
        }
    }
    var shaderID: Int32 {
        switch self {
        case .mandelbulb: return 0
        case .canyon: return 1
        case .sponge: return 2
        case .nebula: return 3
        case .gallery: return 4
        case .galaxy: return -1
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
    var fps = 0.0, msA = 0.0, msB = 0.0, extraMs = 0.0, mbps = 0.0
    var shareA = 0.0
    var width = 0, height = 0
}

struct RenderError: Error, CustomStringConvertible { let description: String }

/// Per-GPU timings, work rates for load balancing, and frame rate.
final class Timing {
    private let lock = NSLock()
    private var stats = Stats()
    private var rateA = 0.0, rateB = 0.0
    private var lastFrame: CFTimeInterval = 0

    private static func ema(_ old: Double, _ new: Double) -> Double { old == 0 ? new : old * 0.8 + new * 0.2 }

    /// Records one GPU's command buffer: `work` = rows or bodies it processed.
    func record(gpu: Int, work: Int, cb: MTLCommandBuffer) {
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        guard ms > 0, work > 0 else { return }
        lock.lock()
        if gpu == 0 { stats.msA = Timing.ema(stats.msA, ms); rateA = Double(work) / ms }
        else { stats.msB = Timing.ema(stats.msB, ms); rateB = Double(work) / ms }
        lock.unlock()
    }

    /// Records display-GPU work that isn't split (the galaxy's drawing pass).
    func recordExtra(cb: MTLCommandBuffer) {
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        guard ms > 0 else { return }
        lock.lock(); stats.extraMs = Timing.ema(stats.extraMs, ms); lock.unlock()
    }

    func rates() -> (a: Double, b: Double, extraMs: Double) {
        lock.lock(); defer { lock.unlock() }
        return (rateA, rateB, stats.extraMs)
    }

    func frame(shareA: Double, width: Int, height: Int, bytesToDisplayGPU: Double) {
        let now = CACurrentMediaTime()
        lock.lock()
        if lastFrame > 0 {
            let inst = 1.0 / max(now - lastFrame, 1e-6)
            stats.fps = stats.fps == 0 ? inst : stats.fps * 0.92 + inst * 0.08
        }
        lastFrame = now
        stats.shareA = shareA
        stats.width = width
        stats.height = height
        stats.mbps = bytesToDisplayGPU * stats.fps / 1e6
        if shareA >= 1 { stats.msB = 0 }
        if shareA <= 0 { stats.msA = 0 }
        lock.unlock()
    }

    func snapshot() -> Stats { lock.lock(); defer { lock.unlock() }; return stats }

    func reset() {
        lock.lock(); stats = Stats(); rateA = 0; rateB = 0; lastFrame = 0; lock.unlock()
    }
}

/// Page-aligned system memory that both GPUs map as their own MTLBuffer (no copies between them).
struct SharedBlock {
    let a: MTLBuffer
    let b: MTLBuffer?
    let mem: UnsafeMutableRawPointer
    let length: Int
}

/// Both D500s: the one driving the display (A) and the headless one (B).
final class GPUs {
    let devA: MTLDevice
    let devB: MTLDevice?
    let queueA: MTLCommandQueue    // display GPU: rendering and simulation
    let queueA2: MTLCommandQueue   // display GPU: assembling and presenting frames
    let queueB: MTLCommandQueue?    // headless GPU: rendering and simulation
    let queueB2: MTLCommandQueue?   // headless GPU: copying results out to shared memory
    let libA: MTLLibrary
    let libB: MTLLibrary?

    init() throws {
        let all = MTLCopyAllDevices()
        guard !all.isEmpty else { throw RenderError(description: "no Metal devices") }
        let display = CGDirectDisplayCopyCurrentMetalDevice(CGMainDisplayID())
        let a = all.first { $0.registryID == display?.registryID } ?? all[0]
        devA = a
        devB = all.first { $0.registryID != a.registryID }
        let src = try GPUs.shaderSource()
        libA = try a.makeLibrary(source: src, options: nil)
        queueA = a.makeCommandQueue()!
        queueA2 = a.makeCommandQueue()!
        if let b = devB {
            libB = try b.makeLibrary(source: src, options: nil)
            queueB = b.makeCommandQueue()
            queueB2 = b.makeCommandQueue()
        } else {
            libB = nil
            queueB = nil
            queueB2 = nil
        }
    }

    var hasTwo: Bool { devB != nil }

    func mapShared(length: Int) throws -> SharedBlock {
        let page = Int(getpagesize())
        let len = (length + page - 1) / page * page
        var p: UnsafeMutableRawPointer?
        guard posix_memalign(&p, page, len) == 0, let mem = p else { throw RenderError(description: "posix_memalign failed") }
        memset(mem, 0, len)
        guard let ba = devA.makeBuffer(bytesNoCopy: mem, length: len, options: .storageModeShared, deallocator: nil) else {
            throw RenderError(description: "display GPU could not map shared memory")
        }
        var bb: MTLBuffer?
        if let b = devB {
            bb = b.makeBuffer(bytesNoCopy: mem, length: len, options: .storageModeShared, deallocator: nil)
            if bb == nil { throw RenderError(description: "headless GPU could not map shared memory") }
        }
        return SharedBlock(a: ba, b: bb, mem: mem, length: len)
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

    /// Copies a texture into CPU memory (BGRA8). Optionally presents a drawable afterward.
    func readback(_ tex: MTLTexture, present: MTLDrawable? = nil) -> Data {
        let w = tex.width, h = tex.height
        let buf = devA.makeBuffer(length: w * h * 4, options: .storageModeShared)!
        let cb = queueA2.makeCommandBuffer()!
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

func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float> = [0, 1, 0]) -> simd_float4x4 {
    let f = simd_normalize(target - eye)
    let s = simd_normalize(simd_cross(f, up))
    let u = simd_cross(s, f)
    return simd_float4x4(columns: (
        SIMD4(s.x, u.x, -f.x, 0),
        SIMD4(s.y, u.y, -f.y, 0),
        SIMD4(s.z, u.z, -f.z, 0),
        SIMD4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)))
}

func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let ys = 1 / tan(fovY * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return simd_float4x4(columns: (
        SIMD4(xs, 0, 0, 0),
        SIMD4(0, ys, 0, 0),
        SIMD4(0, 0, zs, -1),
        SIMD4(0, 0, zs * near, 0)))
}
