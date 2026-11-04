"""M8 kernel parity tests vs classical.refine_edges (float64). Requires the extension built into build/
(see cuda/build.bat); skipped otherwise. GPU tests are short (< a few seconds)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import bench_kernel as bk  # noqa: E402

ext = pytest.importorskip("edge_refine_ext", reason="build the extension first (cuda/CMakeLists.txt -> build/)")
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")


@pytest.fixture(scope="module")
def problem():
    img, pts, nrm, t0 = bk.synthetic_problem(384, 8000, seed=7, border_frac=0.1)
    t_ref, v_ref = bk.reference(img, pts, nrm, t0, bk.RP)
    assert 0.5 < v_ref.mean() < 1.0  # both valid and invalid points are exercised
    return img, pts, nrm, t0, t_ref, v_ref


def _check(problem, t, v, tol_max, tol_p999, min_agree=1.0):
    img, pts, nrm, t0, t_ref, v_ref = problem
    assert t.dtype == np.float32 and v.dtype == np.uint8 and t.shape == v.shape == t0.shape
    assert np.all(np.isnan(t[v == 0])) and np.all(np.isfinite(t[v == 1]))
    r = bk.compare(t_ref, v_ref, t, v)
    assert r["valid_agree"] >= min_agree, r
    assert r["max_abs_px"] <= tol_max and r["p999_abs_px"] <= tol_p999, r


def test_cpu_matches_reference(problem):
    img, pts, nrm, t0 = problem[:4]
    t, v = ext.cpu(img, pts, nrm, t0, *bk.P(bk.RP), threads=4)
    _check(problem, t, v, 1e-3, 1e-4)


def test_cpu_thread_count_invariant(problem):
    img, pts, nrm, t0 = problem[:4]
    a = ext.cpu(img, pts, nrm, t0, *bk.P(bk.RP), threads=1)
    b = ext.cpu(img, pts, nrm, t0, *bk.P(bk.RP), threads=4)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[0], b[0])


def test_clamping_outside_image():
    """Points far outside the image: profile is constant -> no prominence -> invalid, like the reference."""
    img = np.random.default_rng(0).random((64, 80)).astype(np.float32)
    pts = np.array([[-50, 10], [200, 30], [40, -90], [40, 500]], np.float32)
    nrm = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32)
    t0 = np.zeros(4, np.float32)
    t_ref, v_ref = bk.reference(img, pts, nrm, t0, bk.RP)
    t, v = ext.cpu(img, pts, nrm, t0, *bk.P(bk.RP), threads=1)
    np.testing.assert_array_equal(v.astype(bool), v_ref)


@cuda_only
def test_cuda_manual_matches_reference(problem):
    img, pts, nrm, t0 = problem[:4]
    t, v, tm = ext.cuda(img, pts, nrm, t0, *bk.P(bk.RP), mode="manual")
    _check(problem, t, v, 1e-3, 1e-4)
    assert tm["kernel_ms"] > 0


@cuda_only
def test_cuda_texture_close_to_reference(problem):
    # hardware filtering uses 8-bit fractional weights: ~1e-3 px typical error, rare larger argmax flips
    img, pts, nrm, t0 = problem[:4]
    t, v, _ = ext.cuda(img, pts, nrm, t0, *bk.P(bk.RP), mode="texture")
    _check(problem, t, v, 10.0, 0.01, min_agree=0.999)


@cuda_only
def test_device_resident_problem(problem):
    img, pts, nrm, t0 = problem[:4]
    g = ext.GpuProblem()
    g.set_image(img, False)
    g.set_points(pts, nrm, t0)
    assert g.run(False, *bk.P(bk.RP)) > 0
    t, v, _ = g.download()
    t2, v2, _ = ext.cuda(img, pts, nrm, t0, *bk.P(bk.RP), mode="manual")
    np.testing.assert_array_equal(v, v2)
    np.testing.assert_array_equal(t, t2)
    # buffer reuse (same sizes): new data must be uploaded into the existing buffers / cudaArray
    for tex in (False, True):
        g.set_image(img, tex); g.set_points(pts, nrm, t0); g.run(tex, *bk.P(bk.RP))
        g.set_image(img[::-1].copy(), tex); g.set_points(pts, nrm, t0 + 0.25); g.run(tex, *bk.P(bk.RP))
        t3, v3, _ = g.download()
        t4, v4, _ = ext.cuda(img[::-1].copy(), pts, nrm, t0 + 0.25, *bk.P(bk.RP), mode="texture" if tex else "manual")
        np.testing.assert_array_equal(v3, v4)
        np.testing.assert_array_equal(t3, t4)


@cuda_only
def test_torch_reference_implementation(problem):
    img, pts, nrm, t0 = problem[:4]
    t, v = bk.run_torch_numpy(img, pts, nrm, t0, bk.RP)
    _check(problem, t, v.astype(np.uint8), 1e-3, 2e-4)
