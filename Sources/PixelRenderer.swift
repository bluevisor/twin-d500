// Split-frame rendering of the ray-marched scenes across both GPUs.
//
// Frame n, pipelined (default):
//   GPU B  renders rows [split, h) into its own texture (slot n % 3), signals evBr = n
//   GPU B  (second queue) waits for evBr, copies that band to shared memory, signals evB = n
//   GPU A  renders rows [0, split) straight into the frame, signals evA = n
//   GPU A  (second queue) waits for evA and evB, copies the band into the frame,
//          presents, signals evC = n
// The GPUs wait on each other through shared events (~0.1 ms), so the CPU never
// blocks and each GPU starts frame n + 1 while frame n is still being copied.
// The copy-out runs as a bulk blit: 5.5 GB/s, versus 1.6 GB/s when the shader
// writes pixels to system memory directly.
// `pipelined = false` keeps the original scheme, where the CPU waits for GPU B.

import Metal
import QuartzCore

final class PixelRenderer {
    let gpus: GPUs
    let timing = Timing()
    let maxW = 4096, maxH = 2304

    var mode: GPUMode
    var scene: Scene = .mandelbulb
    var quality = 1
    var tint = false
    var morph = false
    var pipelined = true
    var splitFraction = 0.5   // share of rows the display GPU renders

    private var pipesA: [String: MTLComputePipelineState] = [:]
    private var pipesB: [String: MTLComputePipelineState] = [:]
    private var slots: [SharedBlock] = []
    private var texB: [MTLTexture] = []
    private let evA: MTLSharedEvent, evB: MTLSharedEvent, evBr: MTLSharedEvent, evC: MTLSharedEvent
    private var frameNo: UInt64 = 0
    private let inflight = DispatchSemaphore(value: 3)
    private var lastFrame: MTLCommandBuffer?

    private static func key(_ scene: Int32, _ morph: Bool) -> String {
        "\(scene)-\(morph && (scene == 0 || scene == 4))"
    }

    init(gpus: GPUs) throws {
        self.gpus = gpus
        evA = gpus.devA.makeSharedEvent()!
        evB = gpus.devA.makeSharedEvent()!
        evBr = gpus.devA.makeSharedEvent()!
        evC = gpus.devA.makeSharedEvent()!
        mode = gpus.hasTwo ? .both : .displayGPU
        for _ in 0..<3 { slots.append(try gpus.mapShared(length: maxW * maxH * 4)) }
        if let b = gpus.devB {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: maxW, height: maxH, mipmapped: false)
            d.usage = [.shaderWrite, .shaderRead]
            d.storageMode = .private
            for _ in 0..<3 { texB.append(b.makeTexture(descriptor: d)!) }
        }
        try compilePipelines()
    }

    /// One specialized pipeline per scene (and per formula for the Mandelbulb), on both GPUs.
    private func compilePipelines() throws {
        let variants: [(Int32, Bool)] = [(0, false), (0, true), (1, false), (2, false), (3, false), (4, false), (4, true)]
        var jobs: [(gpu: Int, scene: Int32, morph: Bool)] = []
        for v in variants {
            jobs.append((0, v.0, v.1))
            if gpus.hasTwo { jobs.append((1, v.0, v.1)) }
        }
        let lock = NSLock()
        var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
            let job = jobs[i]
            do {
                let fc = MTLFunctionConstantValues()
                var s = job.scene, m = job.morph
                fc.setConstantValue(&s, type: .int, index: 0)
                fc.setConstantValue(&m, type: .bool, index: 1)
                let lib = job.gpu == 0 ? gpus.libA : gpus.libB!
                let dev = job.gpu == 0 ? gpus.devA : gpus.devB!
                let pipe = try dev.makeComputePipelineState(function: try lib.makeFunction(name: "renderTex", constantValues: fc))
                lock.lock()
                if job.gpu == 0 { pipesA[Self.key(job.scene, job.morph)] = pipe } else { pipesB[Self.key(job.scene, job.morph)] = pipe }
                lock.unlock()
            } catch {
                lock.lock(); firstError = firstError ?? error; lock.unlock()
            }
        }
        if let e = firstError { throw e }
    }

    func waitIdle() { lastFrame?.waitUntilCompleted() }

    /// Zeroes the shared slots, so a frame that skipped waiting for GPU B would show up as black.
    func clearSlots() {
        waitIdle()
        for s in slots { memset(s.mem, 0, s.length) }
    }

    private func uniforms(w: Int, h: Int, time: Float, cam: CameraState, first: Int, count: Int, gpu: Int32) -> Uniforms {
        let q = qualities[quality]
        let power: Float = morph ? 8.0 + 1.6 * sin(time * 0.21) : 8.0
        return Uniforms(
            resTime: SIMD4(Float(w), Float(h), time, power),
            camPos: SIMD4(cam.eye, 0),
            camTarget: SIMD4(cam.target, 0),
            rows: SIMD4(Int32(first), Int32(count), q.samples, q.steps),
            flags: SIMD4(tint ? 1 : 0, gpu, q.extras, Int32(w)),
            scene: SIMD4(scene.shaderID, morph ? 1 : 0, Int32(quality), 0))
    }

    private func encodeBand(_ cb: MTLCommandBuffer, pipe: MTLComputePipelineState, target: MTLTexture, u: Uniforms, w: Int, rows: Int) {
        var u = u
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setTexture(target, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: (w + 7) / 8, height: (rows + 7) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }

    /// Renders one frame into `target` (a drawable's texture or an offscreen texture).
    /// Returns the last command buffer of the frame (already committed).
    @discardableResult
    func renderFrame(target: MTLTexture, time: Float, cam: CameraState, present drawable: MTLDrawable?) -> MTLCommandBuffer {
        inflight.wait()
        frameNo += 1
        let n = frameNo
        let w = min(target.width, maxW), h = min(target.height, maxH)
        let m: GPUMode = gpus.hasTwo ? mode : .displayGPU
        let key = Self.key(scene.shaderID, morph)

        if m == .both {
            let r = timing.rates()
            if r.a > 0 && r.b > 0 {
                splitFraction += (r.a / (r.a + r.b) - splitFraction) * 0.15
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
        let si = Int(n % UInt64(slots.count))
        let slot = slots[si]

        var bDone: MTLCommandBuffer?
        if rowsB > 0, let qB = gpus.queueB, let qB2 = gpus.queueB2, let pipe = pipesB[key], let sb = slot.b {
            let tb = texB[si]
            let cb = qB.makeCommandBuffer()!
            if pipelined && n > UInt64(slots.count) {
                cb.encodeWaitForEvent(evC, value: n - UInt64(slots.count))   // last reader of this slot is done
            }
            let render = pipelined ? qB.makeCommandBuffer()! : cb
            if pipelined { cb.commit() }
            encodeBand(render, pipe: pipe, target: tb,
                       u: uniforms(w: w, h: h, time: time, cam: cam, first: rowsA, count: rowsB, gpu: 1), w: w, rows: rowsB)
            render.addCompletedHandler { [timing] cb in timing.record(gpu: 1, work: rowsB, cb: cb) }
            let copy: MTLCommandBuffer
            if pipelined {
                render.encodeSignalEvent(evBr, value: n)
                render.commit()
                copy = qB2.makeCommandBuffer()!
                copy.encodeWaitForEvent(evBr, value: n)
            } else {
                copy = render
            }
            let bl = copy.makeBlitCommandEncoder()!
            bl.copy(from: tb, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: rowsA, z: 0),
                    sourceSize: MTLSize(width: w, height: rowsB, depth: 1),
                    to: sb, destinationOffset: 0, destinationBytesPerRow: w * 4, destinationBytesPerImage: w * 4 * rowsB)
            bl.endEncoding()
            copy.encodeSignalEvent(evB, value: n)
            copy.commit()
            bDone = copy
        }
        if rowsA > 0, let pipe = pipesA[key] {
            let cb = gpus.queueA.makeCommandBuffer()!
            encodeBand(cb, pipe: pipe, target: target,
                       u: uniforms(w: w, h: h, time: time, cam: cam, first: 0, count: rowsA, gpu: 0), w: w, rows: rowsA)
            cb.encodeSignalEvent(evA, value: n)
            cb.addCompletedHandler { [timing] cb in timing.record(gpu: 0, work: rowsA, cb: cb) }
            cb.commit()
        }

        let cbC: MTLCommandBuffer
        if pipelined {
            cbC = gpus.queueA2.makeCommandBuffer()!
            if rowsA > 0 { cbC.encodeWaitForEvent(evA, value: n) }
            if rowsB > 0 { cbC.encodeWaitForEvent(evB, value: n) }
        } else {
            bDone?.waitUntilCompleted()
            cbC = gpus.queueA.makeCommandBuffer()!
        }
        if rowsB > 0 {
            let blit = cbC.makeBlitCommandEncoder()!
            blit.copy(from: slot.a, sourceOffset: 0, sourceBytesPerRow: w * 4, sourceBytesPerImage: w * 4 * rowsB,
                      sourceSize: MTLSize(width: w, height: rowsB, depth: 1),
                      to: target, destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: rowsA, z: 0))
            blit.endEncoding()
        }
        if let d = drawable { cbC.present(d) }
        cbC.encodeSignalEvent(evC, value: n)
        cbC.addCompletedHandler { [inflight] _ in inflight.signal() }
        cbC.commit()
        lastFrame = cbC

        timing.frame(shareA: Double(rowsA) / Double(h), width: w, height: h, bytesToDisplayGPU: Double(rowsB * w * 4))
        return cbC
    }
}
