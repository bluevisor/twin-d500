# Twin D500

Real-time graphics that run on **both** FirePro D500s of a Mac Pro 6,1 (2013) at once: four ray-marched scenes, a gallery that shows all four together, and a 49,152-star galaxy collision.

macOS exposes the two D500s as separate Metal devices: one drives the display, the other is headless, and there is no GPU-to-GPU link. Twin D500 splits the work two ways:

- **Pixel scenes (split frame).** GPU 0 renders the top band of each frame straight into the window. GPU 1 renders the bottom band into its own memory, then copies it to page-aligned system memory that both GPUs map. GPU 0 copies that band into the frame and presents it. The split line moves every frame, using each GPU's measured rows per millisecond, so both finish together.
- **Galaxy collision (split simulation).** Each GPU keeps a full copy of all stars and integrates half of them with all-pairs gravity. After each step, each GPU sends its half's results through shared memory, and both copy in the other half. GPU 0 also draws the stars, so it takes fewer of them.

The GPUs wait on each other through Metal shared events (~0.1 ms per handoff), so the CPU never blocks and each GPU can start the next frame while the previous one is still being assembled.

## Results

2560x1440, High quality, offscreen (`./build.sh --bench`):

| Scene | One GPU | Both GPUs | Speedup |
|---|---|---|---|
| Mandelbulb | 30.7 fps | 55.7 fps | 1.81x |
| Canyon flyover | 11.1 fps | 20.8 fps | 1.87x |
| Menger tunnel | 45.2 fps | 74.7 fps | 1.65x |
| Nebula | 7.1 fps | 13.9 fps | 1.95x |
| Gallery (all four) | 12.1 fps | 23.6 fps | 1.95x |
| Galaxy, 49,152 stars | 17.7 steps/s | 33.4 steps/s | 1.89x |

A split frame is bit-identical to a single-GPU frame, and the galaxy gives bit-identical star positions on one GPU or two (the benchmark checks both).

### What made it faster

Measured on the Mandelbulb with both GPUs (first version: 30 fps):

- **One pipeline per scene** (Metal function constants). A single kernel holding every scene needs the register budget of the heaviest one, so each compute unit keeps fewer pixels in flight: about 25% slower across the board.
- **Trig-free power-8 Mandelbulb.** The polynomial form drops `acos`, `atan2`, `pow`, `sin` and `cos`, which GCN runs at quarter rate: 30 → 56 fps. Press M to switch back to the trig form, which can morph the shape.
- **GPU 1 renders to its own memory, then bulk-copies.** Writing pixels from the shader straight into system memory runs at 1.6 GB/s; a blit runs at 5.5 GB/s, on a second queue so it overlaps the next frame.
- **Four threads per star in the N-body kernel.** With ~24k stars per GPU, one thread per star leaves the D500 too few waves to hide memory latency (1.47x speedup); splitting each star's sum across 4 threads gives 1.89x, and also makes a single GPU 40% faster.
- **GPUs synchronize with each other instead of through the CPU.** This matters little in throughput (~1%) because the handoff is cheap, but it keeps the CPU free.

## Run

```
./build.sh --run     # build and open the app
./build.sh --bench   # offscreen benchmark and correctness checks, writes out/
```

Needs only the Command Line Tools: shaders compile at runtime, so the `metal` compiler isn't required. Works on a single-GPU Mac too (it just won't split).

## Controls

| Key | Action |
|---|---|
| 1-6 | Mandelbulb, canyon, Menger tunnel, nebula, galaxy collision, gallery |
| G | GPUs: display only, both, headless only |
| T | Tint each GPU's share (rows, or stars in the galaxy) |
| Q | Quality: Fast, High, Ultra (4x AA) |
| M | Morphing Mandelbulb (trig formula, slower) |
| P | Pipelining on/off |
| R | Restart the galaxy collision |
| V | Vsync on/off |
| Space | Pause |
| F | Full screen |
| H | Hide the HUD |
| Drag / scroll | Orbit / zoom |

## License

MIT
