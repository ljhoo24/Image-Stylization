#include "stylize.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline uint8_t clamp8(float v)
{
    int i = (int)std::lround(v);
    return (uint8_t)(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// ---- summed-area table (integral image) over RGB ------------------------
// Stored as (w+1) x (h+1) with the zeroth row/col padded with zeros, so the
// box query x in [x0..x1], y in [y0..y1] is just four lookups.
struct RgbSAT
{
    int w = 0, h = 0;
    int sw = 0;                       // stride = w + 1
    std::vector<uint64_t> r, g, b;    // (w+1) * (h+1)
};

static RgbSAT BuildRgbSAT(const Image& img)
{
    RgbSAT s;
    s.w = img.w; s.h = img.h; s.sw = img.w + 1;
    const size_t n = size_t(s.sw) * size_t(img.h + 1);
    s.r.assign(n, 0);
    s.g.assign(n, 0);
    s.b.assign(n, 0);

    // Row-wise running sums fed into column-wise accumulation -- O(w*h).
    for (int y = 0; y < img.h; ++y)
    {
        uint64_t rowR = 0, rowG = 0, rowB = 0;
        const uint8_t* p = img.rgba.data() + size_t(y) * img.w * 4;
        const size_t rowBase = size_t(y + 1) * s.sw;
        const size_t upBase  = size_t(y    ) * s.sw;
        for (int x = 0; x < img.w; ++x)
        {
            rowR += p[0]; rowG += p[1]; rowB += p[2];
            p += 4;
            const size_t idx = rowBase + (x + 1);
            const size_t up  = upBase  + (x + 1);
            s.r[idx] = s.r[up] + rowR;
            s.g[idx] = s.g[up] + rowG;
            s.b[idx] = s.b[up] + rowB;
        }
    }
    return s;
}

// Average RGB inside the axis-aligned box [cx-half..cx+half] x [cy-half..cy+half],
// clipped to the image. O(1).
inline void BoxAverage(const RgbSAT& s, int cx, int cy, int half,
                       uint8_t& outR, uint8_t& outG, uint8_t& outB)
{
    const int x0 = std::max(0, cx - half);
    const int y0 = std::max(0, cy - half);
    const int x1 = std::min(s.w - 1, cx + half);
    const int y1 = std::min(s.h - 1, cy + half);
    if (x1 < x0 || y1 < y0) { outR = outG = outB = 0; return; }
    const int n = (x1 - x0 + 1) * (y1 - y0 + 1);
    const size_t A = size_t(y0)     * s.sw;
    const size_t B = size_t(y1 + 1) * s.sw;
    const int    L = x0;
    const int    R = x1 + 1;
    const uint64_t sumR = s.r[B + R] - s.r[A + R] - s.r[B + L] + s.r[A + L];
    const uint64_t sumG = s.g[B + R] - s.g[A + R] - s.g[B + L] + s.g[A + L];
    const uint64_t sumB = s.b[B + R] - s.b[A + R] - s.b[B + L] + s.b[A + L];
    outR = uint8_t(sumR / uint64_t(n));
    outG = uint8_t(sumG / uint64_t(n));
    outB = uint8_t(sumB / uint64_t(n));
}

// ---- gray + gaussian blur + Sobel ----------------------------------------
static std::vector<float> ToGrayLuma(const Image& img)
{
    std::vector<float> g(size_t(img.w) * img.h);
    const int total = (int)g.size();
    const uint8_t* base = img.rgba.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total; ++i)
    {
        const uint8_t* p = base + size_t(i) * 4;
        g[i] = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    }
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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

static void RunStroke(const WalkInputs& W, const RgbSAT& sat,
                      int sx, int sy, int colorHalf,
                      float texDirX, float texDirY)
{
    if (sx < 0 || sx >= W.w || sy < 0 || sy >= W.h) return;

    uint8_t br, bg, bb;
    BoxAverage(sat, sx, sy, colorHalf, br, bg, bb);

    // Seed dab (covers the case where both halves break immediately).
    PaintDisk(*W.canvas, sx, sy, W.radius, br, bg, bb, W.blend, W.alpha);

    const size_t idx = size_t(sy) * W.w + sx;
    float tx, ty;
    if ((*W.mag)[idx] >= W.gradientThreshold)
    {
        tx = -(*W.gy)[idx];
        ty =  (*W.gx)[idx];
        const float L = std::sqrt(tx * tx + ty * ty);
        if (L < 1e-6f) { tx = texDirX; ty = texDirY; }
        else { tx /= L; ty /= L; }
    }
    else
    {
        tx = texDirX; ty = texDirY;
    }

    WalkHalf(W, (float)sx, (float)sy,  tx,  ty, br, bg, bb);
    WalkHalf(W, (float)sx, (float)sy, -tx, -ty, br, bg, bb);
}

// Build the seed list at the coarsest scale: one jittered seed per RxR cell.
// Then a uniform-random keep at p.samplingRatio. This gives much better
// coverage than purely uniform pixel sampling at the same stroke count.
static std::vector<int> BuildGridJitteredSeeds(int w, int h, int spacing,
                                               float keepRatio, std::mt19937& rng)
{
    spacing = std::max(1, spacing);
    const int cellsX = (w + spacing - 1) / spacing;
    const int cellsY = (h + spacing - 1) / spacing;
    std::vector<int> seeds;
    seeds.reserve(size_t(cellsX) * size_t(cellsY));

    std::uniform_real_distribution<float> uj(-0.5f, 0.5f);
    for (int cy = 0; cy < cellsY; ++cy)
    {
        for (int cx = 0; cx < cellsX; ++cx)
        {
            const float gx = (cx + 0.5f) * spacing + uj(rng) * spacing;
            const float gy = (cy + 0.5f) * spacing + uj(rng) * spacing;
            const int sx = clampi((int)std::lround(gx), 0, w - 1);
            const int sy = clampi((int)std::lround(gy), 0, h - 1);
            seeds.push_back(sy * w + sx);
        }
    }
    if (keepRatio < 0.999f)
    {
        std::shuffle(seeds.begin(), seeds.end(), rng);
        const size_t keep = std::max<size_t>(1, (size_t)std::lround(keepRatio * (double)seeds.size()));
        if (keep < seeds.size()) seeds.resize(keep);
    }
    return seeds;
}

// Parallel collect of pixel indices where pred(i) holds. pred is called from
// many threads concurrently so it must be thread-safe (read-only here).
template <class Pred>
static std::vector<int> ParallelCollect(int total, Pred pred)
{
#ifdef _OPENMP
    const int T = std::max(1, omp_get_max_threads());
    std::vector<std::vector<int>> partials(T);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local = partials[tid];
        local.reserve(size_t(total) / size_t(T * 8));
#pragma omp for nowait schedule(static)
        for (int i = 0; i < total; ++i)
            if (pred(i)) local.push_back(i);
    }
    size_t n = 0;
    for (auto& v : partials) n += v.size();
    std::vector<int> out;
    out.reserve(n);
    for (auto& v : partials) out.insert(out.end(), v.begin(), v.end());
    return out;
#else
    std::vector<int> out;
    out.reserve(size_t(total) / 8);
    for (int i = 0; i < total; ++i)
        if (pred(i)) out.push_back(i);
    return out;
#endif
}

} // namespace

// --------------------------------------------------------------------------
Image Stylize(const Image& src, const BrushStrokeParams& p, StylizeContext* ctx)
{
    Image dst;
    if (!src.valid()) return dst;
    const int w = src.w, h = src.h;
    const size_t N = size_t(w) * h;
    dst.w = w; dst.h = h;
    dst.rgba.assign(N * 4, 0);

    auto report    = [&](float v) { if (ctx) ctx->progress.store(v, std::memory_order_relaxed); };
    auto cancelled = [&]() { return ctx && ctx->cancel.load(std::memory_order_relaxed); };
    report(0.f);

    // 1) gradient (luma -> blur -> Sobel 3x3) + |grad|
    std::vector<float> gray = ToGrayLuma(src);
    const int kernel = std::max(3, p.gradientKernel | 1);
    if (kernel > 3)
        BlurSeparable(gray, w, h, (kernel - 1) * 0.25f);
    std::vector<float> gx, gy;
    Sobel3x3(gray, w, h, gx, gy);

    std::vector<float> mag(N);
    {
        const int total = (int)N;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < total; ++i)
            mag[i] = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    }

    // 2) SAT for fast box-average color sampling at any brush radius
    const RgbSAT sat = BuildRgbSAT(src);

    // Canvas + painted mask
    std::vector<uint8_t> painted(N, 0);
    Canvas canvas{ dst.rgba.data(), painted.data(), w, h };

    // Texture (canvas) base direction for low-gradient regions.
    const float baseAngle = p.textureAngleDeg * float(std::numbers::pi) / 180.f;
    const float jitterRad = std::clamp(p.textureJitterDeg, 0.f, 90.f) * float(std::numbers::pi) / 180.f;

    // Smooth jitter field: uniform random per pixel, heavily blurred, then renormalized
    // to roughly [-1, 1]. Each pixel's fallback stroke direction is rotated by
    // jitterField[i] * jitterRad off baseAngle, so flat regions get organic flow
    // instead of one global angle. Built only when jitter > 0.
    std::vector<float> jitterField;
    if (jitterRad > 0.f)
    {
        jitterField.assign(N, 0.f);
        std::mt19937 jrng(p.seed ^ 0xC0FFEEu);
        std::uniform_real_distribution<float> uds(-1.f, 1.f);
        for (size_t i = 0; i < N; ++i) jitterField[i] = uds(jrng);
        // Blur sigma scales with image size: ~5% of the shorter side, floor 8 px.
        const float sigma = std::max(8.f, float(std::min(w, h)) * 0.05f);
        BlurSeparable(jitterField, w, h, sigma);
        float maxAbs = 0.f;
        for (float v : jitterField) maxAbs = std::max(maxAbs, std::abs(v));
        if (maxAbs > 1e-6f)
            for (float& v : jitterField) v /= maxAbs;
    }

    // Returns the per-pixel fallback direction (cos/sin) at the given pixel index.
    auto fallbackDir = [&](size_t idx, float& outX, float& outY)
    {
        float ang = baseAngle;
        if (!jitterField.empty()) ang += jitterField[idx] * jitterRad;
        outX = std::cos(ang);
        outY = std::sin(ang);
    };

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

    // Progress budget: 5% prep + 85% scales (equal split) + 10% refill (scale 0).
    report(0.05f);
    if (cancelled()) return Image{};

    const float kScalesShare = 0.85f;
    const float kRefillShare = 0.10f;
    const float perScaleShare = kScalesShare / float(radii.size());

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
    float progressBase = 0.05f;
    for (size_t scaleIdx = 0; scaleIdx < radii.size(); ++scaleIdx)
    {
        if (cancelled()) return Image{};

        const int R = radii[scaleIdx];
        const int strokeLen = std::max(2, (int)std::lround((double)p.strokeLength * (double)R / (double)maxR));
        W.radius = R;
        W.steps  = std::max(1, strokeLen / 2);
        const int colorHalf = std::max(1, R);

        // Build the seed list for this scale.
        std::vector<int> seeds;
        if (scaleIdx == 0)
        {
            // Grid-jittered: one candidate per RxR cell, then keep samplingRatio of them.
            seeds = BuildGridJitteredSeeds(w, h, std::max(1, R), p.samplingRatio, rng);
        }
        else
        {
            // Detail pass: only seed from pixels where the canvas diverges
            // from the source by more than errorThreshold.
            const float errSq = p.errorThreshold * p.errorThreshold;
            const uint8_t* dstBase = dst.rgba.data();
            const uint8_t* srcBase = src.rgba.data();
            std::vector<int> highError = ParallelCollect((int)N, [&](int i)
            {
                const uint8_t* a = dstBase + size_t(i) * 4;
                const uint8_t* b = srcBase + size_t(i) * 4;
                const float dr = float(a[0]) - float(b[0]);
                const float dg = float(a[1]) - float(b[1]);
                const float db = float(a[2]) - float(b[2]);
                return (dr * dr + dg * dg + db * db) > errSq;
            });
            if (highError.empty()) { progressBase += perScaleShare; report(progressBase); continue; }
            std::shuffle(highError.begin(), highError.end(), rng);
            const size_t take = std::max<size_t>(1, (size_t)std::lround(p.samplingRatio * (double)highError.size()));
            seeds.assign(highError.begin(), highError.begin() + std::min<size_t>(take, highError.size()));
        }

        for (size_t i = 0; i < seeds.size(); ++i)
        {
            if ((i & 4095u) == 4095u)
            {
                if (cancelled()) return Image{};
                report(progressBase + perScaleShare * float(i + 1) / float(seeds.size()));
            }
            const int idx = seeds[i];
            const int sy = idx / w;
            const int sx = idx % w;
            float fx, fy;
            fallbackDir(size_t(idx), fx, fy);
            RunStroke(W, sat, sx, sy, colorHalf, fx, fy);
        }
        progressBase += perScaleShare;
        report(progressBase);

        // Refill only on the first (coarsest) scale to guarantee coverage.
        if (scaleIdx == 0)
        {
            const int passes = std::max(0, p.maxRefillPasses);
            const uint8_t* paintedBase = painted.data();
            for (int pass = 0; pass < passes; ++pass)
            {
                if (cancelled()) return Image{};
                std::vector<int> empties = ParallelCollect((int)N, [&](int i) { return !paintedBase[i]; });
                if (empties.empty()) break;
                std::shuffle(empties.begin(), empties.end(), rng);
                for (int idx : empties)
                {
                    if (painted[idx]) continue;
                    float fx, fy;
                    fallbackDir(size_t(idx), fx, fy);
                    RunStroke(W, sat, idx % w, idx / w, colorHalf, fx, fy);
                }
                report(progressBase + kRefillShare * float(pass + 1) / float(passes));
            }
            // Final 1-px dab for stragglers (rare; keeps the canvas covered).
            const int totalPx = (int)N;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < totalPx; ++i)
            {
                if (painted[i]) continue;
                const int y = i / w, x = i % w;
                uint8_t br, bg, bb;
                BoxAverage(sat, x, y, 1, br, bg, bb);
                uint8_t* q = dst.rgba.data() + size_t(i) * 4;
                q[0] = br; q[1] = bg; q[2] = bb; q[3] = 255;
                painted[i] = 1;
            }
            progressBase += kRefillShare;
            report(progressBase);
        }
    }

    report(1.f);
    return dst;
}
