// CUDA sub-pixel edge refinement: one thread per edge point.
//   manual : bilinear interpolation in software from a linear float buffer (read-only __ldg path)
//   texture: hardware bilinear filtering (cudaFilterModeLinear, cudaAddressModeClamp, unnormalised
//            coords, +0.5 texel-centre offset; weights have 8 fractional bits in hardware)
#define ER_CORE_ONLY
#include "edge_refine_cpu.cpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

struct ManualSampler {
    const float* __restrict__ img; int H, W;
    __device__ float operator()(float x, float y) const {
        x = fminf(fmaxf(x, 0.f), float(W - 1));
        y = fminf(fmaxf(y, 0.f), float(H - 1));
        int x0 = int(x), y0 = int(y);
        int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
        float fx = x - float(x0), fy = y - float(y0);
        const float* r0 = img + (size_t)y0 * W;
        const float* r1 = img + (size_t)y1 * W;
        float a00 = __ldg(r0 + x0), a01 = __ldg(r0 + x1), a10 = __ldg(r1 + x0), a11 = __ldg(r1 + x1);
        float a = a00 + fx * (a01 - a00), b = a10 + fx * (a11 - a10);
        return a + fy * (b - a);
    }
};

struct TexSampler {
    cudaTextureObject_t tex;
    // Clamp addressing at texel centres reproduces mode="nearest" outside [0, n-1].
    __device__ float operator()(float x, float y) const { return tex2D<float>(tex, x + 0.5f, y + 0.5f); }
};

template <class S>
__global__ void refine_kernel(S s, const float2* __restrict__ pts, const float2* __restrict__ nrm,
                              const float* __restrict__ t0, int N, RefineParams p, float* t, uint8_t* valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float2 q = pts[i], n = nrm[i];
    refine_one(s, q.x, q.y, n.x, n.y, t0[i], p, t + i, valid + i);
}

// ---------------------------------------------------------------------------- device-resident problem
struct GpuProblem {
    int H = 0, W = 0, N = 0;
    float* d_img = nullptr;           // linear image (manual path)
    cudaArray_t arr = nullptr;        // cudaArray + texture object (texture path)
    cudaTextureObject_t tex = 0;
    float2 *d_pts = nullptr, *d_nrm = nullptr;
    float *d_t0 = nullptr, *d_t = nullptr;
    uint8_t* d_valid = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    GpuProblem() {
        CK(cudaStreamCreate(&stream));
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    }
    ~GpuProblem() {
        free_image(); free_points();
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(stream);
    }
    float elapsed() { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms; }
    void free_image() {
        if (tex) cudaDestroyTextureObject(tex);
        if (arr) cudaFreeArray(arr);
        if (d_img) cudaFree(d_img);
        tex = 0; arr = nullptr; d_img = nullptr;
    }
    void free_points() {
        cudaFree(d_pts); cudaFree(d_nrm); cudaFree(d_t0); cudaFree(d_t); cudaFree(d_valid);
        d_pts = d_nrm = nullptr; d_t0 = d_t = nullptr; d_valid = nullptr;
    }
    // Upload image (linear buffer always; cudaArray + texture object if `texture`). Returns ms (events).
    // Buffers / texture object are reused when the size and mode are unchanged (only the copy is repeated).
    float set_image(const float* img, int H_, int W_, bool texture) {
        bool reuse = H == H_ && W == W_ && (texture ? arr != nullptr : d_img != nullptr);
        if (!reuse) { free_image(); H = H_; W = W_; }
        CK(cudaEventRecord(e0, stream));
        if (texture && reuse) {
            CK(cudaMemcpy2DToArrayAsync(arr, 0, 0, img, W * sizeof(float), W * sizeof(float), H,
                                        cudaMemcpyHostToDevice, stream));
        } else if (texture) {
            cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
            CK(cudaMallocArray(&arr, &cd, W, H));
            CK(cudaMemcpy2DToArrayAsync(arr, 0, 0, img, W * sizeof(float), W * sizeof(float), H,
                                        cudaMemcpyHostToDevice, stream));
            cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
            cudaTextureDesc td = {};
            td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
            td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeElementType;
            td.normalizedCoords = 0;
            CK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
        } else {
            if (!reuse) CK(cudaMalloc(&d_img, (size_t)H * W * sizeof(float)));
            CK(cudaMemcpyAsync(d_img, img, (size_t)H * W * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        CK(cudaEventRecord(e1, stream)); CK(cudaStreamSynchronize(stream));
        return elapsed();
    }
    float set_points(const float* pts, const float* nrm, const float* t0, int N_) {
        bool reuse = d_pts != nullptr && N_ == N;
        if (!reuse) { free_points(); N = N_; }
        CK(cudaEventRecord(e0, stream));
        if (!reuse) {
            CK(cudaMalloc(&d_pts, N * sizeof(float2))); CK(cudaMalloc(&d_nrm, N * sizeof(float2)));
            CK(cudaMalloc(&d_t0, N * sizeof(float))); CK(cudaMalloc(&d_t, N * sizeof(float)));
            CK(cudaMalloc(&d_valid, N));
        }
        CK(cudaMemcpyAsync(d_pts, pts, N * sizeof(float2), cudaMemcpyHostToDevice, stream));
        CK(cudaMemcpyAsync(d_nrm, nrm, N * sizeof(float2), cudaMemcpyHostToDevice, stream));
        CK(cudaMemcpyAsync(d_t0, t0, N * sizeof(float), cudaMemcpyHostToDevice, stream));
        CK(cudaEventRecord(e1, stream)); CK(cudaStreamSynchronize(stream));
        return elapsed();
    }
    // Kernel only (inputs already on device). Returns ms measured with CUDA events.
    float run(bool texture, const RefineParams& p, int block) {
        if (texture ? !tex : !d_img) throw std::runtime_error("image not uploaded for this mode");
        int grid = (N + block - 1) / block;
        CK(cudaEventRecord(e0, stream));
        if (texture) refine_kernel<<<grid, block, 0, stream>>>(TexSampler{tex}, d_pts, d_nrm, d_t0, N, p, d_t, d_valid);
        else refine_kernel<<<grid, block, 0, stream>>>(ManualSampler{d_img, H, W}, d_pts, d_nrm, d_t0, N, p, d_t, d_valid);
        CK(cudaGetLastError());
        CK(cudaEventRecord(e1, stream)); CK(cudaStreamSynchronize(stream));
        return elapsed();
    }
    float download(float* t, uint8_t* valid) {
        CK(cudaEventRecord(e0, stream));
        CK(cudaMemcpyAsync(t, d_t, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CK(cudaMemcpyAsync(valid, d_valid, N, cudaMemcpyDeviceToHost, stream));
        CK(cudaEventRecord(e1, stream)); CK(cudaStreamSynchronize(stream));
        return elapsed();
    }
};

// Opaque C-style API used by python_binding.cpp (compiled by the host compiler).
void* gp_create() { return new GpuProblem(); }
void gp_destroy(void* g) { delete static_cast<GpuProblem*>(g); }
float gp_set_image(void* g, const float* img, int H, int W, bool tex) { return static_cast<GpuProblem*>(g)->set_image(img, H, W, tex); }
float gp_set_points(void* g, const float* p, const float* n, const float* t0, int N) { return static_cast<GpuProblem*>(g)->set_points(p, n, t0, N); }
float gp_run(void* g, bool tex, RefineParams p, int block) { return static_cast<GpuProblem*>(g)->run(tex, p, block); }
float gp_download(void* g, float* t, uint8_t* v) { return static_cast<GpuProblem*>(g)->download(t, v); }
int gp_n(void* g) { return static_cast<GpuProblem*>(g)->N; }
