# ImageEditor

A small Windows-native painterly image editor. Loads a photo, repaints it as
a multi-scale brush-stroke composition (Hertzmann-style), and saves the
result as PNG or JPEG. Written in modern C++ (C++20) with Dear ImGui on
DirectX 11.

```
+--------------------------------------------------------------+
| Load | Process | Save | [x] A/B | [Cancel] [================]|
|--------------------------------------------------------------|
|                     |                  |                     |
|  Brush Stroke       |    Input image   |    Output image     |
|  - Gradient         |                  |                     |
|  - Multi-scale      |   (or split A/B  |                     |
|  - Strokes          |       view)      |                     |
|  - Compositing      |                  |                     |
|  - Random           |                  |                     |
|                     |                  |                     |
+--------------------------------------------------------------+
```

---

## Features

- **Multi-scale painterly stylize** — coarse-to-fine brush passes with
  gradient-aligned stroke walking, configurable curvature smoothing,
  color-tolerance stroke cutoff, and per-pass refill for coverage.
- **Runs off the UI thread** — Stylize executes on a worker; the toolbar
  shows a live progress bar and a Cancel button.
- **Parallelized inner loops** — gradient, blur, Sobel, gradient
  magnitude, error-pixel scans, and the final straggler dab are all
  `#pragma omp parallel for`.
- **Summed-area table** for color sampling — O(1) box average at any
  brush radius, so coarse brushes get a representative color.
- **Grid-jittered seed sampling** at the coarsest scale — better
  coverage with far fewer strokes than uniform random.
- **Zoom & pan in both viewers** — mouse wheel zooms (anchored at the
  cursor), drag pans, double-click resets to fit.
- **A/B compare** — toggle the A/B checkbox to slide a divider between
  the input and output sharing one image rect.
- **Drag-and-drop** — drop a `.png` / `.jpg` / `.bmp` / `.tga` onto the
  window to load it.
- **Per-monitor DPI awareness V2** — non-client area and ImGui scale
  with the monitor's DPI at startup.
- **Keyboard shortcuts** — `Ctrl+O` open, `Ctrl+S` save, `Space` process.

---

## Algorithm

The stylize pass is a streamlined Hertzmann "painterly rendering" with a
few practical tweaks. For each brush radius from large to small:

1. **Pick seed points.**
   - At the coarsest radius: one jittered candidate per `R x R` grid
     cell, then keep a `samplingRatio` fraction.
   - At finer radii: only pixels whose canvas color differs from the
     source by more than `errorThreshold` (in RGB distance) — this
     focuses detail passes on regions that still look wrong.
2. **For each seed, walk a stroke forward and backward.**
   - The stroke tangent is the perpendicular of the local image
     gradient (so strokes ride along edges).
   - In low-gradient regions, the tangent falls back to a global
     `textureAngleDeg` canvas direction.
   - `curvatureFilter` blends the new tangent with the previous one;
     the stroke stops when the source color at the next step differs
     from the brush color by more than `colorTolerance`.
3. **At the coarsest scale only, run refill passes** that paint
   stragglers until every pixel is covered.

Color at the seed is sampled as the box average over a `(2R+1)` window
around the seed via a precomputed summed-area table — O(1) per query.

---

## Build

### Requirements
- Windows 10 1703 or later (for `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`).
- Visual Studio 2022 / 2026 with the "Desktop development with C++" workload.
- A vcpkg manifest providing `imgui[dx11-binding,win32-binding]`, `stb`.
  The project already integrates with vcpkg if it's installed
  (`vcpkg integrate install`).

### From Visual Studio
1. Open `ImageEditor.slnx`.
2. Select the `x64 / Release` configuration.
3. Build (`Ctrl+Shift+B`) and run (`F5`).

### From the command line
```powershell
& "$env:ProgramFiles\Microsoft Visual Studio\<edition>\<sku>\MSBuild\Current\Bin\MSBuild.exe" `
  ImageEditor\ImageEditor.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```
The executable lands at `ImageEditor\x64\Release\ImageEditor.exe`.

OpenMP is enabled in the vcxproj (`<OpenMPSupport>true</OpenMPSupport>`)
across all four configurations. If you build with `/openmp` disabled the
code still compiles — the pragmas become no-ops and `ParallelCollect`
falls back to a serial path.

---

## Controls

| Action          | Mouse / key                                  |
|-----------------|----------------------------------------------|
| Load image      | `Ctrl+O`, drag-drop, or **Load** button      |
| Process         | `Space` or **Process** button                |
| Save output     | `Ctrl+S` or **Save** button                  |
| Cancel run      | **Cancel** button (visible while processing) |
| Zoom            | Mouse wheel inside a viewer                  |
| Pan             | Left-drag inside a viewer                    |
| Reset view      | Double-click inside a viewer                 |
| A/B compare     | **A/B** checkbox, then **right-drag** divider|

---

## Parameter cheat-sheet

| Group         | Parameter             | What it does                                                          |
|---------------|-----------------------|-----------------------------------------------------------------------|
| Gradient      | Kernel (3/5/7/9)      | Box size used to blur luma before Sobel                               |
|               | Edge thresh           | `\|grad\|` below this is treated as "no direction" -> texture angle    |
| Multi-scale   | Max brush R           | Radius of the coarsest brush                                          |
|               | Num scales            | How many radii are tried (each half the previous)                     |
|               | Error thresh          | RGB distance above which a pixel becomes a seed at finer scales       |
| Strokes       | Sampling              | Fraction of grid cells / error pixels kept as seeds                   |
|               | Length (max)          | Max stroke length at the coarsest scale (scales down with R)          |
|               | Curve smooth          | 0 = always keep previous direction, 1 = snap to local tangent         |
|               | Color tol             | Stop a stroke when source diverges from the brush color by more       |
|               | Texture deg           | Canvas direction used in low-gradient regions                         |
| Compositing   | Blend / Alpha         | Alpha-blend brush color over previously painted pixels                |
|               | Refill passes         | Extra coarse-scale passes to fill canvas gaps                         |
| Random        | Seed / Randomize      | RNG seed for grid jitter and error-pixel shuffles                     |

---

## Project layout

```
ImageEditor.slnx
ImageEditor/
  ImageEditor.vcxproj          # MSBuild project (C++20, OpenMP, /SDL)
  ImageEditor.vcxproj.filters  # Solution Explorer filters
  main.cpp                     # Win32 + DX11 host, ImGui UI, file I/O
  image.h                      # RGBA8 Image POD
  stylize.h                    # BrushStrokeParams, StylizeContext, Stylize()
  stylize.cpp                  # SAT, parallel scans, multi-scale stroke loop
```

`main.cpp` is the only translation unit that includes the stb single-headers
(`stb_image.h`, `stb_image_write.h`), so they only compile once.

---

## Implementation notes

- **Threading model.** A single `std::thread` worker runs Stylize on a
  snapshot of the input image and the current params. Progress is published
  via `std::atomic<float>`; the user can request cancellation via
  `std::atomic<bool>` which the worker polls at scale and refill boundaries
  and once per 4096 strokes. The main thread checks `done` at the top of
  every frame, joins, and uploads the result.
- **GPU texture lifetime.** `GpuTexture::Upload` takes a fast path:
  if the existing texture's dimensions already match the new image, the
  pixels are streamed in with `UpdateSubresource` and no D3D objects are
  recreated.
- **No race in seed generation.** Grid-jittered seeding is serial (cheap).
  High-error and refill-empties scans collect into per-thread vectors that
  are concatenated after the parallel region, so the critical section
  vanishes.
- **DPI.** The window is created under `PER_MONITOR_AWARE_V2`. The initial
  DPI is sampled once via `GetDpiForWindow` and used to scale the ImGui
  style and font global scale. `WM_DPICHANGED` is acknowledged with
  `SetWindowPos` to the suggested rect.

---

## Roadmap

- Live low-resolution preview while sliders move.
- Save / load parameter presets.
- Spatial partitioning to safely parallelize stroke painting itself.
- Optional GPU compute path for gradient and SAT.

---

## License

The original code in this repository is provided under the MIT License
(see `LICENSE` if present). Third-party dependencies retain their own
licenses: Dear ImGui (MIT), stb_image / stb_image_write (public domain
or MIT, at your option).
