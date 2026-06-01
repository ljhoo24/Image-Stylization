#include "stylize.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

namespace {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline uint8_t clamp8(float v)
{
    int i = (int)std::lround(v);
    return (uint8_t)(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// ---- gray + gaussian blur + Sobel ----------------------------------------
static std::vector<float> ToGrayLuma(const Image& img)
{
    std::vector<float> g(size_t(img.w) * img.h);
    const uint8_t* p = img.rgba.data();
    for (size_t i = 0, n = g.size(); i < n; ++i, p += 4)
        g[i] = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    return g;
}

static std::vector<float> Gaussian1DKernel(float sigma)
{
    int radius = std::max(1, (int)std::ceil(3.f * sigma));
    int n = 2 * radius + 1;
    std::vector<float> k(n);
    float s = 0.f;
    const float inv2s2 = 1.f / (2.f * sigma * sigma);
    for (int i = -radius; i <= radius; ++i)
    {
        float v = std::exp(-(i * i) * inv2s2);
        k[i + radius] = v;
        s += v;
    }
    for (auto& v : k) v /= s;
    return k;
}

static void BlurSeparable(std::vector<float>& buf, int w, int h, float sigma)
{
    if (sigma <= 0.f) return;
    auto kernel = Gaussian1DKernel(sigma);
    int radius = (int)kernel.size() / 2;
    std::vector<float> tmp(buf.size());
    for (int y = 0; y < h; ++y)
    {
        const float* row = &buf[size_t(y) * w];
        float* out = &tmp[size_t(y) * w];
        for (int x = 0; x < w; ++x)
        {
            float acc = 0.f;
            for (int t = -radius; t <= radius; ++t)
                acc += row[clampi(x + t, 0, w - 1)] * kernel[t + radius];
            out[x] = acc;
        }
    }
    for (int y = 0; y < h; ++y)
    {
        float* out = &buf[size_t(y) * w];
        for (int x = 0; x < w; ++x)
        {
            float acc = 0.f;
            for (int t = -radius; t <= radius; ++t)
                acc += tmp[size_t(clampi(y + t, 0, h - 1)) * w + x] * kernel[t + radius];
            out[x] = acc;
        }
    }
}

static void Sobel3x3(const std::vector<float>& gray, int w, int h,
                     std::vector<float>& gx, std::vector<float>& gy)
{
    gx.assign(gray.size(), 0.f);
    gy.assign(gray.size(), 0.f);
    for (int y = 1; y < h - 1; ++y)
    {
        for (int x = 1; x < w - 1; ++x)
        {
            const size_t i = size_t(y) * w + x;
            const float a = gray[i - w - 1], b = gray[i - w], c = gray[i - w + 1];
            const float d = gray[i     - 1],                  f = gray[i     + 1];
            const float g = gray[i + w - 1], hh= gray[i + w], k = gray[i + w + 1];
            gx[i] = (-a + c - 2.f * d + 2.f * f - g + k) * 0.125f;
            gy[i] = (-a - 2.f * b - c + g + 2.f * hh + k) * 0.125f;
        }
    }
}

// ---- canvas / drawing primitives ----------------------------------------
struct Canvas
{
    uint8_t* dst;
    uint8_t* painted;
    int w, h;
};

inline void PaintDisk(Canvas& c, int cx, int cy, int r,
                      uint8_t br, uint8_t bg, uint8_t bb,
                      bool blend, float alpha)
{
    const int x0 = std::max(0, cx - r), x1 = std::min(c.w - 1, cx + r);
    const int y0 = std::max(0, cy - r), y1 = std::min(c.h - 1, cy + r);
    const int rr = r * r;
    for (int y = y0; y <= y1; ++y)
    {
        const int dy = y - cy;
        for (int x = x0; x <= x1; ++x)
        {
            const int dx = x - cx;
            if (dx * dx + dy * dy > rr) continue;
            const size_t idx = size_t(y) * c.w + x;
            uint8_t* p = c.dst + idx * 4;
            if (blend && c.painted[idx])
            {
                p[0] = clamp8((1.f - alpha) * p[0] + alpha * br);
                p[1] = clamp8((1.f - alpha) * p[1] + alpha * bg);
                p[2] = clamp8((1.f - alpha) * p[2] + alpha * bb);
            }
            else
            {
                p[0] = br; p[1] = bg; p[2] = bb;
            }
            p[3] = 255;
            c.painted[idx] = 1;
        }
    }
}

inline void AverageColor3x3(const Image& src, int cx, int cy,
                            uint8_t& outR, uint8_t& outG, uint8_t& outB)
{
    int rs = 0, gs = 0, bs = 0, n = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        const int y = clampi(cy + dy, 0, src.h - 1);
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int x = clampi(cx + dx, 0, src.w - 1);
            const uint8_t* p = src.rgba.data() + (size_t(y) * src.w + x) * 4;
            rs += p[0]; gs += p[1]; bs += p[2];
            ++n;
        }
    }
    outR = uint8_t(rs / n);
    outG = uint8_t(gs / n);
    outB = uint8_t(bs / n);
}

// Walks one half of a stroke from (sx,sy) along (ptx,pty).
// Does NOT paint at the seed itself - the caller is responsible for that
// so forward+backward halves don't double-paint the start.
struct WalkInputs
{
    const Image*               src;
    const std::vector<float>*  gx;
    const std::vector<float>*  gy;
    const std::vector<float>*  mag;
    Canvas*                    canvas;
    int                        w;
    int                        h;
    float                      gradientThreshold;
    float                      curvatureFilter; // 0..1
    float                      colorTolSq;      // squared
    int                        radius;
    int                        steps;
    bool                       blend;
    float                      alpha;
};

static void WalkHalf(const WalkInputs& W,
                     float sx, float sy, float ptx, float pty,
                     uint8_t br, uint8_t bg, uint8_t bb)
{
    float x = sx, y = sy;
    for (int s = 0; s < W.steps; ++s)
    {
        x += ptx;
        y += pty;
        const int ix = (int)std::lround(x);
        const int iy = (int)std::lround(y);
        if (ix < 0 || ix >= W.w || iy < 0 || iy >= W.h) break;

        const size_t idx = size_t(iy) * W.w + ix;
        const uint8_t* sp = W.src->rgba.data() + idx * 4;
        const float dr = float(sp[0]) - br;
        const float dg = float(sp[1]) - bg;
        const float db = float(sp[2]) - bb;
        if (dr * dr + dg * dg + db * db > W.colorTolSq) break;

        PaintDisk(*W.canvas, ix, iy, W.radius, br, bg, bb, W.blend, W.alpha);

        // Tangent update - only when local gradient is significant.
        if ((*W.mag)[idx] >= W.gradientThreshold)
        {
            float lx = -(*W.gy)[idx];
            float ly =  (*W.gx)[idx];
            const float L = std::sqrt(lx * lx + ly * ly);
            if (L > 1e-6f)
            {
                lx /= L; ly /= L;
                if (lx * ptx + ly * pty < 0.f) { lx = -lx; ly = -ly; }
                const float fc = W.curvatureFilter;
                float nx = fc * lx + (1.f - fc) * ptx;
                float ny = fc * ly + (1.f - fc) * pty;
                const float NL = std::sqrt(nx * nx + ny * ny);
                if (NL > 1e-6f) { ptx = nx / NL; pty = ny / NL; }
            }
        }
        // else: keep (ptx,pty) - coast through low-gradient regions.
    }
}

static void RunStroke(const WalkInputs& W,
                      int sx, int sy,
                      float ctx, float cty)
{
    if (sx < 0 || sx >= W.w || sy < 0 || sy >= W.h) return;

    uint8_t br, bg, bb;
    AverageColor3x3(*W.src, sx, sy, br, bg, bb);

    // Seed dab (covers the case where both halves break immediately).
    PaintDisk(*W.canvas, sx, sy, W.radius, br, bg, bb, W.blend, W.alpha);

    const size_t idx = size_t(sy) * W.w + sx;
    float tx, ty;
    if ((*W.mag)[idx] >= W.gradientThreshold)
    {
        tx = -(*W.gy)[idx];
        ty =  (*W.gx)[idx];
        const float L = std::sqrt(tx * tx + ty * ty);
        if (L < 1e-6f) { tx = ctx; ty = cty; }
        else { tx /= L; ty /= L; }
    }
    else
    {
        tx = ctx; ty = cty;
    }

    WalkHalf(W, (float)sx, (float)sy,  tx,  ty, br, bg, bb);
    WalkHalf(W, (float)sx, (float)sy, -tx, -ty, br, bg, bb);
}

} // namespace

// --------------------------------------------------------------------------
Image Stylize(const Image& src, const BrushStrokeParams& p)
{
    Image dst;
    if (!src.valid()) return dst;
    const int w = src.w, h = src.h;
    const size_t N = size_t(w) * h;
    dst.w = w; dst.h = h;
    dst.rgba.assign(N * 4, 0);

    // 1) gradient (luma -> blur -> Sobel 3x3)
    std::vector<float> gray = ToGrayLuma(src);
    const int kernel = std::max(3, p.gradientKernel | 1);
    if (kernel > 3)
        BlurSeparable(gray, w, h, (kernel - 1) * 0.25f);
    std::vector<float> gx, gy;
    Sobel3x3(gray, w, h, gx, gy);
    std::vector<float> mag(N);
    for (size_t i = 0; i < N; ++i)
        mag[i] = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);

    // Canvas + painted mask
    std::vector<uint8_t> painted(N, 0);
    Canvas canvas{ dst.rgba.data(), painted.data(), w, h };

    // Texture (canvas) direction for low-gradient regions
    const float ang = p.textureAngleDeg * float(std::numbers::pi) / 180.f;
    const float ctx = std::cos(ang);
    const float cty = std::sin(ang);

    // Build the brush radii ladder (geometric halving, descending)
    std::vector<int> radii;
    {
        const int numScales = std::max(1, p.numScales);
        const int maxR = std::max(1, p.maxBrushRadius);
        float r = (float)maxR;
        for (int i = 0; i < numScales; ++i)
        {
            radii.push_back(std::max(1, (int)std::lround(r)));
            r *= 0.5f;
        }
    }
    const int maxR = radii.front();

    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<int> ux(0, w - 1), uy(0, h - 1);

    // Reusable WalkInputs (radius/steps mutated per scale)
    WalkInputs W{};
    W.src = &src; W.gx = &gx; W.gy = &gy; W.mag = &mag;
    W.canvas = &canvas;
    W.w = w; W.h = h;
    W.gradientThreshold = p.gradientThreshold;
    W.curvatureFilter   = std::clamp(p.curvatureFilter, 0.f, 1.f);
    W.colorTolSq        = p.colorTolerance * p.colorTolerance;
    W.blend             = p.blend;
    W.alpha             = p.blendAlpha;

    // ---- Multi-scale loop (largest to smallest) -------------------------
    for (size_t scaleIdx = 0; scaleIdx < radii.size(); ++scaleIdx)
    {
        const int R = radii[scaleIdx];
        const int strokeLen = std::max(2, (int)std::lround((double)p.strokeLength * (double)R / (double)maxR));
        W.radius = R;
        W.steps  = std::max(1, strokeLen / 2);

        // Build the seed list for this scale.
        std::vector<int> seeds;
        if (scaleIdx == 0)
        {
            const int seedCount = std::max(1, (int)std::lround(p.samplingRatio * (double)N));
            seeds.reserve(seedCount);
            for (int i = 0; i < seedCount; ++i)
                seeds.push_back(uy(rng) * w + ux(rng));
        }
        else
        {
            // Detail pass: only seed from pixels where the canvas diverges
            // from the source by more than errorThreshold.
            const float errSq = p.errorThreshold * p.errorThreshold;
            std::vector<int> highError;
            highError.reserve(N / 4);
            for (size_t i = 0; i < N; ++i)
            {
                const uint8_t* a = dst.rgba.data() + i * 4;
                const uint8_t* b = src.rgba.data() + i * 4;
                const float dr = float(a[0]) - float(b[0]);
                const float dg = float(a[1]) - float(b[1]);
                const float db = float(a[2]) - float(b[2]);
                if (dr * dr + dg * dg + db * db > errSq)
                    highError.push_back((int)i);
            }
            if (highError.empty()) continue;
            std::shuffle(highError.begin(), highError.end(), rng);
            const int take = std::max(1, (int)std::lround(p.samplingRatio * (double)highError.size()));
            seeds.assign(highError.begin(), highError.begin() + std::min<size_t>(take, highError.size()));
        }

        for (int idx : seeds)
        {
            const int sy = idx / w;
            const int sx = idx % w;
            RunStroke(W, sx, sy, ctx, cty);
        }

        // Refill only on the first (coarsest) scale to guarantee coverage.
        if (scaleIdx == 0)
        {
            for (int pass = 0; pass < p.maxRefillPasses; ++pass)
            {
                std::vector<int> empties;
                empties.reserve(N / 8);
                for (size_t i = 0; i < N; ++i)
                    if (!painted[i]) empties.push_back((int)i);
                if (empties.empty()) break;
                std::shuffle(empties.begin(), empties.end(), rng);
                for (int idx : empties)
                {
                    if (painted[idx]) continue;
                    RunStroke(W, idx % w, idx / w, ctx, cty);
                }
            }
            // Final 1-px dab for stragglers (rare; keeps the canvas covered).
            for (size_t i = 0; i < N; ++i)
            {
                if (painted[i]) continue;
                const int y = (int)(i / w), x = (int)(i % w);
                uint8_t br, bg, bb;
                AverageColor3x3(src, x, y, br, bg, bb);
                uint8_t* q = dst.rgba.data() + i * 4;
                q[0] = br; q[1] = bg; q[2] = bb; q[3] = 255;
                painted[i] = 1;
            }
        }
    }

    return dst;
}
