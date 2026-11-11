# Twin D500

A real-time ray-marched Mandelbulb that renders every frame on **both** FirePro D500s in a Mac Pro 6,1 at once.

macOS exposes the two D500s as separate Metal devices: one drives the display, the other is headless, and there is no GPU-to-GPU link. Twin D500 splits each frame by rows:

- **GPU 0 (display)** renders the top band straight into the window.
- **GPU 1 (headless)** renders the bottom band into a page-aligned block of system memory that both GPUs map, so no CPU copy is needed.
- GPU 0 then copies that band over PCIe into the frame and presents it.
- The split line moves every frame, using each GPU's measured rows-per-millisecond, so both GPUs finish together.

## Results (2560x1440, offscreen, `--bench`)

| Quality | One GPU | Both GPUs | Speedup |
|---|---|---|---|
| Fast | 41 fps | 67 fps | 1.65x |
| High | 16 fps | 30 fps | 1.87x |
| Ultra (4x AA) | 4 fps | 7.6 fps | 1.88x |

The split frame matches a single-GPU frame to within 1/255 on a handful of pixels (float rounding).
In the default 1600x900 window, both GPUs together hold 60 fps (vsync) versus 39 fps for either GPU alone.

## Run

```
./build.sh --run     # build and open the app
./build.sh --bench   # offscreen benchmark, writes out/frame.png and out/bench.txt
```

Needs only the Command Line Tools: shaders compile at runtime, so the `metal` compiler isn't required.

## Controls

| Key | Action |
|---|---|
| 1 / 2 / 3 | Display GPU only / both GPUs / headless GPU only |
| T | Tint each GPU's band and draw the split line |
| Q | Cycle quality (Fast, High, Ultra 4x AA) |
| V | Toggle vsync (off shows uncapped fps) |
| Space | Pause |
| F | Full screen |
| H | Hide the HUD |
| Drag / scroll | Orbit / zoom |
