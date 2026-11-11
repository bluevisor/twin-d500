// Two colliding galaxies: an all-pairs N-body simulation split across both GPUs.
//
// Each GPU keeps its own full copy of positions and velocities in VRAM. Every
// step, GPU A integrates bodies [0, countA) and GPU B integrates [countA, n),
// each reading all n positions from its own copy (4 threads per body). Each writes its results to
// shared system memory too; then each copies the other's half back into its own
// copy. Shared events order the steps between the GPUs; the CPU never waits.
// GPU A also draws the stars, so it gets fewer bodies to keep both GPUs busy.

import Metal
import QuartzCore
import simd

final class NBody {
    let gpus: GPUs
    let n: Int
    let timing = Timing()
    var mode: GPUMode
    var split = 0.5   // share of bodies GPU A integrates
    var tint = false
    let dt: Float = 0.01
    let eps2: Float = 0.0036

    private let stepA: MTLComputePipelineState
    private let stepB: MTLComputePipelineState?
    private let pointPipe: MTLRenderPipelineState
    private let tonemapPipe: MTLComputePipelineState
    private var posA: [MTLBuffer] = [], velA: [MTLBuffer] = []
    private var posB: [MTLBuffer] = [], velB: [MTLBuffer] = []
    private let shared: SharedBlock
    private let evSA: MTLSharedEvent, evSB: MTLSharedEvent, evXA: MTLSharedEvent, evXB: MTLSharedEvent
    private var stepNo: UInt64 = 0
    private let inflight = DispatchSemaphore(value: 2)
    private var lastA: MTLCommandBuffer?, lastB: MTLCommandBuffer?
    private var hdr: MTLTexture?
    private var resetPending = true
    private var lastCountA = 0

    init(gpus: GPUs, bodies: Int = 49152) throws {
        self.gpus = gpus
        n = max(bodies / 256, 2) * 256
        mode = gpus.hasTwo ? .both : .displayGPU
        stepA = try gpus.devA.makeComputePipelineState(function: gpus.libA.makeFunction(name: "nbodyStep")!)
        if let b = gpus.devB, let lib = gpus.libB {
            stepB = try b.makeComputePipelineState(function: lib.makeFunction(name: "nbodyStep")!)
        } else {
            stepB = nil
        }
        let rp = MTLRenderPipelineDescriptor()
        rp.vertexFunction = gpus.libA.makeFunction(name: "pointVertex")
        rp.fragmentFunction = gpus.libA.makeFunction(name: "pointFragment")
        let ca = rp.colorAttachments[0]!
        ca.pixelFormat = .rgba16Float
        ca.isBlendingEnabled = true
        ca.rgbBlendOperation = .add
        ca.alphaBlendOperation = .add
        ca.sourceRGBBlendFactor = .one
        ca.destinationRGBBlendFactor = .one
        ca.sourceAlphaBlendFactor = .one
        ca.destinationAlphaBlendFactor = .one
        pointPipe = try gpus.devA.makeRenderPipelineState(descriptor: rp)
        tonemapPipe = try gpus.devA.makeComputePipelineState(function: gpus.libA.makeFunction(name: "galaxyTonemap")!)

        let len = n * 16
        for _ in 0..<2 {
            posA.append(gpus.devA.makeBuffer(length: len, options: .storageModePrivate)!)
            velA.append(gpus.devA.makeBuffer(length: len, options: .storageModePrivate)!)
            if let b = gpus.devB {
                posB.append(b.makeBuffer(length: len, options: .storageModePrivate)!)
                velB.append(b.makeBuffer(length: len, options: .storageModePrivate)!)
            }
        }
        shared = try gpus.mapShared(length: len * 2)
        evSA = gpus.devA.makeSharedEvent()!
        evSB = gpus.devA.makeSharedEvent()!
        evXA = gpus.devA.makeSharedEvent()!
        evXB = gpus.devA.makeSharedEvent()!
    }

    func reset() { resetPending = true }

    func waitIdle() {
        lastA?.waitUntilCompleted()
        lastB?.waitUntilCompleted()
    }

    /// Two disk galaxies with central black holes on a glancing collision course.
    private func initialState() -> (pos: [SIMD4<Float>], vel: [SIMD4<Float>]) {
        var rng = SplitMix64(seed: 2013)
        var pos = [SIMD4<Float>](repeating: .zero, count: n)
        var vel = [SIMD4<Float>](repeating: .zero, count: n)
        let half = n / 2
        let galaxies: [(c: SIMD3<Float>, v: SIMD3<Float>, axis: SIMD3<Float>)] = [
            ([-3.5, 0.3, -0.8], [0.32, 0, 0.12], simd_normalize([0.2, 1, 0.1])),
            ([3.5, -0.3, 0.8], [-0.32, 0, -0.12], simd_normalize([-0.5, 0.8, 0.4])),
        ]
        let hole: Float = 1.0, starMass: Float = 0.6 / Float(half - 1), rMax: Float = 2.5
        for (g, gal) in galaxies.enumerated() {
            let e1 = simd_normalize(simd_cross(gal.axis, abs(gal.axis.x) < 0.9 ? [1, 0, 0] : [0, 0, 1]))
            let e2 = simd_cross(gal.axis, e1)
            let base = g * half
            pos[base] = SIMD4(gal.c, hole)
            vel[base] = SIMD4(gal.v, 0)
            for i in 1..<half {
                let r = min(0.25 + 2.0 * -log(1 - rng.unit() * 0.95) / 3, rMax)
                let a = rng.unit() * 2 * .pi
                let radial = cos(a) * e1 + sin(a) * e2
                let p = gal.c + r * radial + rng.gauss() * 0.04 * gal.axis
                let enclosed = hole + 0.6 * pow(min(r / rMax, 1), 1.5)
                let v = gal.v + sqrt(enclosed / r) * simd_normalize(simd_cross(gal.axis, radial))
                pos[base + i] = SIMD4(p, starMass)
                vel[base + i] = SIMD4(v, 0)
            }
        }
        return (pos, vel)
    }

    private func upload(into idx: Int) {
        let (pos, vel) = initialState()
        let len = n * 16
        func put(_ dev: MTLDevice, _ queue: MTLCommandQueue, _ p: MTLBuffer, _ v: MTLBuffer) {
            let sp = dev.makeBuffer(bytes: pos, length: len, options: .storageModeShared)!
            let sv = dev.makeBuffer(bytes: vel, length: len, options: .storageModeShared)!
            let cb = queue.makeCommandBuffer()!
            let bl = cb.makeBlitCommandEncoder()!
            bl.copy(from: sp, sourceOffset: 0, to: p, destinationOffset: 0, size: len)
            bl.copy(from: sv, sourceOffset: 0, to: v, destinationOffset: 0, size: len)
            bl.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        put(gpus.devA, gpus.queueA, posA[idx], velA[idx])
        if let b = gpus.devB, let q = gpus.queueB { put(b, q, posB[idx], velB[idx]) }
    }

    private func encodeStep(_ cb: MTLCommandBuffer, pipe: MTLComputePipelineState, pos: [MTLBuffer], vel: [MTLBuffer],
                            sharedBuf: MTLBuffer, inI: Int, outI: Int, start: Int, count: Int) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(pos[inI], offset: 0, index: 0)
        enc.setBuffer(vel[inI], offset: 0, index: 1)
        enc.setBuffer(pos[outI], offset: 0, index: 2)
        enc.setBuffer(vel[outI], offset: 0, index: 3)
        enc.setBuffer(sharedBuf, offset: 0, index: 4)
        var range = SIMD4<UInt32>(UInt32(start), UInt32(count), UInt32(n), 0)
        var sim = SIMD4<Float>(dt, eps2, 0, 0)
        enc.setBytes(&range, length: 16, index: 5)
        enc.setBytes(&sim, length: 16, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: count / 64, height: 1, depth: 1),   // 64 bodies x 4 threads each
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Copies bodies [start, start + count) from shared memory into this GPU's own copy.
    private func encodeExchange(_ cb: MTLCommandBuffer, sharedBuf: MTLBuffer, pos: MTLBuffer, vel: MTLBuffer, start: Int, count: Int) {
        guard count > 0 else { return }
        let bl = cb.makeBlitCommandEncoder()!
        bl.copy(from: sharedBuf, sourceOffset: start * 16, to: pos, destinationOffset: start * 16, size: count * 16)
        bl.copy(from: sharedBuf, sourceOffset: (n + start) * 16, to: vel, destinationOffset: start * 16, size: count * 16)
        bl.endEncoding()
    }

    /// Bodies GPU A integrates this step: balances simA + drawing against simB.
    private func countForA(_ m: GPUMode) -> Int {
        switch m {
        case .displayGPU: return n
        case .headlessGPU: return 0
        case .both:
            let r = timing.rates()
            if r.a > 0 && r.b > 0 {
                let ideal = (Double(n) / r.b - r.extraMs) / (1 / r.a + 1 / r.b) / Double(n)
                split += (ideal - split) * 0.15
                split = min(max(split, 0.05), 0.95)
            }
            let blocks = n / 256
            return min(max(Int((Double(blocks) * split).rounded()), 1), blocks - 1) * 256
        }
    }

    /// Advances one step and draws it into `target`. Returns the display GPU's last command buffer.
    @discardableResult
    func frame(target: MTLTexture, cam: CameraState, present drawable: MTLDrawable?) -> MTLCommandBuffer {
        inflight.wait()
        if resetPending {
            waitIdle()
            upload(into: Int(stepNo % 2))
            resetPending = false
        }
        stepNo += 1
        let s = stepNo
        let inI = Int((s - 1) % 2), outI = Int(s % 2)
        let m: GPUMode = gpus.hasTwo ? mode : .displayGPU
        let countA = countForA(m)
        let countB = n - countA
        lastCountA = countA

        // Headless GPU: integrate its bodies, then pull in GPU A's results.
        if let qB = gpus.queueB, let pipe = stepB, let sb = shared.b {
            if s > 1 {
                let w = qB.makeCommandBuffer()!   // GPU A must be done reading B's last results
                w.encodeWaitForEvent(evXA, value: s - 1)
                w.commit()
            }
            let cb = qB.makeCommandBuffer()!
            if countB > 0 {
                encodeStep(cb, pipe: pipe, pos: posB, vel: velB, sharedBuf: sb, inI: inI, outI: outI, start: countA, count: countB)
            }
            cb.encodeSignalEvent(evSB, value: s)
            cb.addCompletedHandler { [timing] cb in timing.record(gpu: 1, work: countB, cb: cb) }
            cb.commit()
            let x = qB.makeCommandBuffer()!
            x.encodeWaitForEvent(evSA, value: s)
            encodeExchange(x, sharedBuf: sb, pos: posB[outI], vel: velB[outI], start: 0, count: countA)
            x.encodeSignalEvent(evXB, value: s)
            x.commit()
            lastB = x
        }

        // Display GPU: integrate its bodies, pull in GPU B's results, draw.
        let qA = gpus.queueA
        if gpus.hasTwo && s > 1 {
            let w = qA.makeCommandBuffer()!
            w.encodeWaitForEvent(evXB, value: s - 1)
            w.commit()
        }
        let cbA = qA.makeCommandBuffer()!
        if countA > 0 {
            encodeStep(cbA, pipe: stepA, pos: posA, vel: velA, sharedBuf: shared.a, inI: inI, outI: outI, start: 0, count: countA)
        }
        cbA.encodeSignalEvent(evSA, value: s)
        cbA.addCompletedHandler { [timing] cb in timing.record(gpu: 0, work: countA, cb: cb) }
        cbA.commit()
        if gpus.hasTwo {
            let x = qA.makeCommandBuffer()!
            x.encodeWaitForEvent(evSB, value: s)
            encodeExchange(x, sharedBuf: shared.a, pos: posA[outI], vel: velA[outI], start: countA, count: countB)
            x.encodeSignalEvent(evXA, value: s)
            x.commit()
        }

        let r = qA.makeCommandBuffer()!
        draw(r, target: target, cam: cam, pos: posA[outI], vel: velA[outI], countA: countA)
        if let d = drawable { r.present(d) }
        r.addCompletedHandler { [timing, inflight] cb in
            timing.recordExtra(cb: cb)
            inflight.signal()
        }
        r.commit()
        lastA = r
        timing.frame(shareA: Double(countA) / Double(n), width: target.width, height: target.height,
                     bytesToDisplayGPU: Double(countB * 32))
        return r
    }

    private func draw(_ cb: MTLCommandBuffer, target: MTLTexture, cam: CameraState, pos: MTLBuffer, vel: MTLBuffer, countA: Int) {
        let w = target.width, h = target.height
        if hdr == nil || hdr!.width != w || hdr!.height != h {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            hdr = gpus.devA.makeTexture(descriptor: d)
        }
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = hdr
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        rp.colorAttachments[0].storeAction = .store
        let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
        enc.setRenderPipelineState(pointPipe)
        enc.setVertexBuffer(pos, offset: 0, index: 0)
        enc.setVertexBuffer(vel, offset: 0, index: 1)
        var vp = perspective(fovY: 0.85, aspect: Float(w) / Float(h), near: 0.05, far: 200)
            * lookAt(eye: cam.eye, target: cam.target)
        enc.setVertexBytes(&vp, length: MemoryLayout<simd_float4x4>.stride, index: 2)
        var extra = SIMD4<Float>(tint ? 1 : 0, Float(countA), 0, 0)
        enc.setVertexBytes(&extra, length: 16, index: 4)
        let scale = Float(h) / 1080
        for (size, intensity) in [(Float(7), Float(0.03)), (Float(1.6), Float(0.3))] {   // soft glow, then sharp stars
            var style = SIMD4<Float>(size, intensity, Float(n / 2), scale)
            enc.setVertexBytes(&style, length: 16, index: 3)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: n)
        }
        enc.endEncoding()

        let ce = cb.makeComputeCommandEncoder()!
        ce.setComputePipelineState(tonemapPipe)
        ce.setTexture(hdr, index: 0)
        ce.setTexture(target, index: 1)
        var params = SIMD4<Float>(1.4, 0, 0, 0)
        ce.setBytes(&params, length: 16, index: 0)
        ce.dispatchThreadgroups(MTLSize(width: (w + 7) / 8, height: (h + 7) / 8, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        ce.endEncoding()
    }

    /// Current positions as GPU A sees them (for the correctness check).
    func readPositions() -> [SIMD4<Float>] {
        waitIdle()
        let len = n * 16
        let out = gpus.devA.makeBuffer(length: len, options: .storageModeShared)!
        let cb = gpus.queueA.makeCommandBuffer()!
        let bl = cb.makeBlitCommandEncoder()!
        bl.copy(from: posA[Int(stepNo % 2)], sourceOffset: 0, to: out, destinationOffset: 0, size: len)
        bl.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: n))
    }
}

struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    mutating func gauss() -> Float {
        let u1 = max(unit(), 1e-7), u2 = unit()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}
