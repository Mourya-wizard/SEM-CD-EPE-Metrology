// pybind11 module `edge_refine_ext`: numpy in, numpy out. No torch headers (nvcc+MSVC issue).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <omp.h>
#define ER_CORE_ONLY
#include "edge_refine_cpu.cpp"  // RefineParams only; the CPU driver is its own TU

namespace py = pybind11;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

void refine_edges_cpu(const float*, int, int, const float*, const float*, const float*, int64_t, RefineParams,
                      float*, uint8_t*, int);
void* gp_create();
void gp_destroy(void*);
float gp_set_image(void*, const float*, int, int, bool);
float gp_set_points(void*, const float*, const float*, const float*, int);
float gp_run(void*, bool, RefineParams, int);
float gp_download(void*, float*, uint8_t*);
int gp_n(void*);

static RefineParams make_params(double w_in, double w_out, double step, double frac, double min_prom) {
    RefineParams p{float(w_in), float(w_out), float(step), float(frac), float(min_prom), 0, 0};
    p.K = int(std::lround((w_in + w_out) / step)) + 1;   // same rounding as the Python reference
    p.kin = int(std::lround(2 * w_in / step)) + 1;
    return p;
}

static void check(const F32& img, const F32& pts, const F32& nrm, const F32& t0) {
    if (img.ndim() != 2) throw std::invalid_argument("img must be (H,W)");
    int64_t N = t0.size();
    if (pts.ndim() != 2 || pts.shape(1) != 2 || pts.shape(0) != N || nrm.ndim() != 2 || nrm.shape(1) != 2 || nrm.shape(0) != N)
        throw std::invalid_argument("pts and normals must be (N,2), t0 (N,)");
}

static py::tuple cpu(F32 img, F32 pts, F32 nrm, F32 t0, double w_in, double w_out, double step, double frac,
                     double min_prom, int threads) {
    check(img, pts, nrm, t0);
    int64_t N = t0.size();
    py::array_t<float> t(N);
    py::array_t<uint8_t> v(N);
    RefineParams p = make_params(w_in, w_out, step, frac, min_prom);
    {
        py::gil_scoped_release r;
        refine_edges_cpu(img.data(), int(img.shape(0)), int(img.shape(1)), pts.data(), nrm.data(), t0.data(), N, p,
                         t.mutable_data(), v.mutable_data(), threads);
    }
    return py::make_tuple(t, v);
}

// Device-resident problem: upload once, time the kernel alone, download.
struct Gpu {
    void* g;
    Gpu() : g(gp_create()) {}
    ~Gpu() { gp_destroy(g); }
    float set_image(F32 img, bool texture) {
        if (img.ndim() != 2) throw std::invalid_argument("img must be (H,W)");
        return gp_set_image(g, img.data(), int(img.shape(0)), int(img.shape(1)), texture);
    }
    float set_points(F32 pts, F32 nrm, F32 t0) {
        check(F32(py::array_t<float>({1, 1})), pts, nrm, t0);
        return gp_set_points(g, pts.data(), nrm.data(), t0.data(), int(t0.size()));
    }
    float run(bool texture, double w_in, double w_out, double step, double frac, double min_prom, int block) {
        return gp_run(g, texture, make_params(w_in, w_out, step, frac, min_prom), block);
    }
    py::tuple download() {
        int N = gp_n(g);
        py::array_t<float> t(N);
        py::array_t<uint8_t> v(N);
        float ms = gp_download(g, t.mutable_data(), v.mutable_data());
        return py::make_tuple(t, v, ms);
    }
};

// End-to-end GPU call from host numpy arrays: alloc + H2D (+ texture setup) + kernel + D2H.
// Returns (t, valid, timings) with per-phase CUDA-event ms and host wall-clock total ms.
static py::tuple cuda(F32 img, F32 pts, F32 nrm, F32 t0, double w_in, double w_out, double step, double frac,
                      double min_prom, const std::string& mode, int block) {
    check(img, pts, nrm, t0);
    bool tex = mode == "texture";
    if (!tex && mode != "manual") throw std::invalid_argument("mode must be 'manual' or 'texture'");
    auto t_start = std::chrono::steady_clock::now();
    Gpu g;
    float ms_img = g.set_image(img, tex);
    float ms_pts = g.set_points(pts, nrm, t0);
    float ms_k = g.run(tex, w_in, w_out, step, frac, min_prom, block);
    py::tuple r = g.download();
    double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    py::dict tm;
    tm["image_upload_ms"] = ms_img; tm["points_upload_ms"] = ms_pts; tm["kernel_ms"] = ms_k;
    tm["download_ms"] = r[2]; tm["total_wall_ms"] = total;
    return py::make_tuple(r[0], r[1], tm);
}

PYBIND11_MODULE(edge_refine_ext, m) {
    m.doc() = "Sub-pixel edge refinement (classical.refine_edges) on CPU/OpenMP and CUDA (manual / texture bilinear)";
    m.def("cpu", &cpu, py::arg("img"), py::arg("pts"), py::arg("normals"), py::arg("t0"), py::arg("w_in"),
          py::arg("w_out"), py::arg("step"), py::arg("frac"), py::arg("min_prom"), py::arg("threads") = 0);
    m.def("cuda", &cuda, py::arg("img"), py::arg("pts"), py::arg("normals"), py::arg("t0"), py::arg("w_in"),
          py::arg("w_out"), py::arg("step"), py::arg("frac"), py::arg("min_prom"), py::arg("mode") = "manual",
          py::arg("block") = 128);
    m.def("omp_max_threads", [] { return omp_get_max_threads(); });
    py::class_<Gpu>(m, "GpuProblem")
        .def(py::init<>())
        .def("set_image", &Gpu::set_image, py::arg("img"), py::arg("texture"))
        .def("set_points", &Gpu::set_points, py::arg("pts"), py::arg("normals"), py::arg("t0"))
        .def("run", &Gpu::run, py::arg("texture"), py::arg("w_in"), py::arg("w_out"), py::arg("step"),
             py::arg("frac"), py::arg("min_prom"), py::arg("block") = 128)
        .def("download", &Gpu::download);
}
