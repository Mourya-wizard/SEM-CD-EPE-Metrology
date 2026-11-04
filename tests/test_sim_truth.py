"""M2 checkpoint tests for the SEM simulator and truth (needs data/ and a CUDA GPU)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import sem_sim as ss  # noqa: E402
from data_lithobench import load_config  # noqa: E402
from gauges import design_edge_points  # noqa: E402

CFG = load_config()
SC, S, M = CFG["sim"], CFG["lithobench"]["crop_px"], CFG["sim"]["margin_px"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = json.load(open(os.path.join(CFG["paths"]["results"], "splits.json")))
TILES = [("MetalSet", SPLITS["val"]["MetalSet"][0]), ("ViaSet", SPLITS["val"]["ViaSet"][0])]


@pytest.fixture(scope="module")
def crop():
    s, t = TILES[0]
    y0, x0 = ss.choose_crop(CFG, s, t, S)
    return ss.load_crop(CFG, s, t, y0, x0, S)


def _recovery(mode, **over):
    return np.concatenate([ss.check_profile_recovery(CFG, s, t, DEV, mode, **over) for s, t in TILES])


def test_seed_reproduces_image_and_truth(crop):
    a, _ = crop
    p = ss.sample_params(np.random.default_rng(3), SC, 1)
    o1 = ss.simulate(a[None], p, SC, S, seed=11, device=DEV, frames=[4])
    o2 = ss.simulate(a[None], p, SC, S, seed=11, device=DEV, frames=[4])
    o3 = ss.simulate(a[None], p, SC, S, seed=12, device=DEV, frames=[4])
    assert np.array_equal(o1["F"], o2["F"]) and np.array_equal(o1["clean"], o2["clean"])
    assert np.array_equal(o1["noisy"][4], o2["noisy"][4])
    assert not np.array_equal(o1["noisy"][4], o3["noisy"][4])
    # noise is reproducible from (seed, frames) on top of a cached clean image
    g = torch.Generator(device=DEV).manual_seed(ss.noise_seed(11, 4, 0))
    n = ss.add_noise(torch.as_tensor(o1["clean"], device=DEV)[:, None], 4, SC, g)[:, 0].cpu().numpy()
    assert np.array_equal(n, o1["noisy"][4])


def test_truth_recovery_noise_free_no_offset():
    for mode, over in (("peak", dict(contrast=0.0)), ("level", dict(bloom_amp=0.0))):
        e = _recovery(mode, **over)
        assert e.size > 500
        assert np.median(np.abs(e)) <= 0.1, (mode, np.median(np.abs(e)))


def test_known_bloom_offset_gives_expected_bias():
    for off in (1.0, -1.0):
        e = _recovery("peak", contrast=0.0, bloom_offset_px=off)
        assert abs(np.median(e) - (-off)) <= 0.1, (off, np.median(e))


def test_exposure_sweep_is_monotone_and_pattern_dependent():
    s, t = TILES[0]
    th = [0.205, 0.215, 0.225, 0.235, 0.245]
    sh = ss.edge_shift_vs_threshold(CFG, s, [t], th)
    med = [np.median(sh[x]) for x in th]
    assert all(med[i] > med[i + 1] for i in range(len(med) - 1)), med     # higher threshold -> smaller features
    lo = sh[0.205]
    assert np.subtract(*np.percentile(lo, [95, 5])) > 1.0                  # not a uniform shift


def test_local_bump_moves_truth_edge(crop):
    a, _ = crop
    sc = dict(SC, bumps_per_crop=[1, 1])
    p = ss.to_torch(ss.nominal_params(SC, 1), DEV)
    at = torch.as_tensor(a, device=DEV)[None, None]
    F1, bl = ss.build_truth(at, p, sc, torch.Generator(device=DEV).manual_seed(5), bumps=True)
    F0, _ = ss.build_truth(at, p, sc, torch.Generator(device=DEV).manual_seed(5), bumps=False)
    cy, cx, dsh, _ = bl[0][0]
    f0 = F0[0, 0].cpu().numpy()
    gy, gx = np.gradient(f0)
    n = -np.array([gx[int(cy), int(cx)], gy[int(cy), int(cx)]])
    n /= np.linalg.norm(n)                                                  # outward normal
    pt = np.array([[cx, cy]])
    t0 = ss.crossings_along(f0, pt, n[None], 12.0)[0][0]
    t1 = ss.crossings_along(F1[0, 0].cpu().numpy(), pt, n[None], 12.0)[0][0]
    assert abs((t1 - t0) - dsh) <= 0.2 * abs(dsh) + 0.15, (t1 - t0, dsh)


def test_roughness_sigma(crop):
    a, tm = crop
    at = torch.as_tensor(a, device=DEV)[None, None]
    pts, nrm, _ = design_edge_points(tm, 10, stride=2, border_px=M + 12)
    p0 = ss.to_torch(ss.nominal_params(SC, 1), DEV)
    p1 = ss.to_torch(ss.nominal_params(SC, 1, rough_sigma_nm=2.0, rough_corr_nm=10.0), DEV)
    F0, _ = ss.build_truth(at, p0, SC, torch.Generator(device=DEV).manual_seed(1), bumps=False)
    F1, _ = ss.build_truth(at, p1, SC, torch.Generator(device=DEV).manual_seed(1), bumps=False)
    t0, n0 = ss.crossings_along(F0[0, 0].cpu().numpy(), pts, nrm, 10.0)
    t1, n1 = ss.crossings_along(F1[0, 0].cpu().numpy(), pts, nrm, 10.0)
    ok = (n0 == 1) & (n1 == 1)
    sd = np.std(t1[ok] - t0[ok])
    assert 1.4 <= sd <= 2.6, sd


def test_noise_statistics():
    clean = torch.full((1, 1, 256, 256), 0.3, device=DEV)
    fr = 8
    x = ss.add_noise(clean, fr, SC, torch.Generator(device=DEV).manual_seed(0)).cpu().numpy()
    var_exp = 0.3 / (fr * SC["electrons_per_frame"]) + SC["detector_sigma"] ** 2
    assert abs(x.mean() - 0.3) < 0.005
    assert abs(x.var() / var_exp - 1) < 0.05


def test_stress_ranges_outside_training():
    rng = np.random.default_rng(0)
    pc = ss.sample_params(rng, SC, 200, "stress_contrast")
    pb = ss.sample_params(rng, SC, 200, "stress_blur")
    assert pc["contrast"].max() < SC["train"]["contrast"][0]
    assert pb["blur_px"].min() > SC["train"]["blur_px"][1]
