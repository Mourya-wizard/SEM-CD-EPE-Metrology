// Sub-pixel edge refinement: shared per-point core (also #included by edge_refine.cu with
// ER_CORE_ONLY defined) + C++/OpenMP CPU driver. Same algorithm as classical.refine_edges
// (docs/interfaces.md "Edge refinement kernel"), single precision.
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define ER_HD __host__ __device__ __forceinline__
#else
#define ER_HD inline
#endif

struct RefineParams {
    float w_in, w_out, step, frac, min_prom;
    int K, kin;  // K = round((w_in+w_out)/step)+1 ; kin = round(2*w_in/step)+1
};

// Bilinear sample with coordinates clamped to [0, n-1] == scipy map_coordinates(order=1, mode="nearest").
ER_HD float bilinear_clamped(const float* img, int H, int W, float x, float y) {
    x = fminf(fmaxf(x, 0.f), float(W - 1));
    y = fminf(fmaxf(y, 0.f), float(H - 1));
    int x0 = int(x), y0 = int(y);  // x, y >= 0 so truncation == floor
    int x1 = x0 + 1 < W ? x0 + 1 : x0, y1 = y0 + 1 < H ? y0 + 1 : y0;
    float fx = x - float(x0), fy = y - float(y0);
    const float* r0 = img + (size_t)y0 * W;
    const float* r1 = img + (size_t)y1 * W;
    float a = r0[x0] + fx * (r0[x1] - r0[x0]);
    float b = r1[x0] + fx * (r1[x1] - r1[x0]);
    return a + fy * (b - a);
}

// Refine one edge point. `sample(x, y)` returns the (bilinear) image value at pixel coords (x, y).
// One pass computes peak (first argmax over the first kin samples) and base = min(v[kp:]) online
// (the running min restarts whenever the argmax moves); a second pass finds the first crossing.
template <class Sampler>
ER_HD void refine_one(const Sampler& sample, float px, float py, float nx, float ny, float t0,
                      const RefineParams& p, float* t_out, uint8_t* valid_out) {
    int kp = 0;
    float vp = -INFINITY, base = INFINITY;
    for (int k = 0; k < p.K; ++k) {
        float t = t0 + (-p.w_in + k * p.step);
        float v = sample(px + nx * t, py + ny * t);
        if (k < p.kin && v > vp) { vp = v; kp = k; base = v; }
        else if (k >= kp) base = fminf(base, v);
    }
    float level = base + p.frac * (vp - base);
    float va = vp, tc = 0.f;
    bool has = false;
    for (int k = kp + 1; k < p.K; ++k) {
        float t = t0 + (-p.w_in + k * p.step);
        float vb = sample(px + nx * t, py + ny * t);
        if (vb < level) {
            float d = va - vb;
            tc = (-p.w_in + (k - 1) * p.step) + p.step * (va - level) / (d != 0.f ? d : 1.f);
            has = true;
            break;
        }
        va = vb;
    }
    bool ok = has && (vp - base >= p.min_prom) && kp > 0;
    *t_out = ok ? t0 + tc : NAN;
    *valid_out = ok ? 1 : 0;
}

#ifndef ER_CORE_ONLY
#include <omp.h>

struct CpuSampler {
    const float* img; int H, W;
    float operator()(float x, float y) const { return bilinear_clamped(img, H, W, x, y); }
};

// pts, nrm: (N,2) [x,y]; returns via t (N,), valid (N,). threads <= 0 -> OpenMP default.
void refine_edges_cpu(const float* img, int H, int W, const float* pts, const float* nrm, const float* t0,
                      int64_t N, RefineParams p, float* t, uint8_t* valid, int threads) {
    CpuSampler s{img, H, W};
    int nt = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t i = 0; i < N; ++i)
        refine_one(s, pts[2 * i], pts[2 * i + 1], nrm[2 * i], nrm[2 * i + 1], t0[i], p, t + i, valid + i);
}
#endif
