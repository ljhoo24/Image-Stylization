#pragma once
#include <atomic>
#include <cstdint>
#include "image.h"

struct BrushStrokeParams
{
    // Gradient
    int      gradientKernel    = 5;     // 3, 5, 7, 9 (odd)
    float    gradientThreshold = 25.f;  // |grad| above which we follow the tangent

    // Sampling / strokes
    float    samplingRatio     = 0.50f; // 0.10..1.0 of candidate pixel count
    int      strokeLength      = 24;    // max stroke length (largest scale); scaled
                                        // proportionally for smaller brushes
    // Multi-scale brushes (Hertzmann-style)
    int      maxBrushRadius    = 8;     // largest brush radius
    int      numScales         = 3;     // number of brush sizes (radii halved each step)
    float    errorThreshold    = 40.f;  // per-pixel RGB distance (0..441) gating which
                                        // pixels become seeds at scales > 0

    // Stroke shaping
    float    curvatureFilter   = 0.5f;  // 0..1; fc - 0 = keep prev dir, 1 = snap to local
    float    colorTolerance    = 60.f;  // 0..441; stop a stroke if |src - brush| exceeds
    float    textureAngleDeg   = 0.f;   // canvas direction used in low-gradient regions

    // Compositing
    bool     blend             = false;
    float    blendAlpha        = 0.60f;

    // Coverage
    int      maxRefillPasses   = 6;     // applied at the largest scale only
    uint32_t seed              = 12345u;
};

// Optional progress + cancellation channel. Pass nullptr for fire-and-forget.
// progress is monotonically advanced in [0,1]; cancel is polled at safe points
// and, if set, makes Stylize() return an empty Image.
struct StylizeContext
{
    std::atomic<float> progress{ 0.f };
    std::atomic<bool>  cancel{ false };
};

Image Stylize(const Image& src, const BrushStrokeParams& p, StylizeContext* ctx = nullptr);
