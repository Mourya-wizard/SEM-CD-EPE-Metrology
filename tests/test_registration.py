"""M3 registration and gauge tests (needs data/ and a CUDA GPU)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import gauges as gg  # noqa: E402
import registration as rg  # noqa: E402
import sem_sim as ss  # noqa: E402
from data_lithobench import load_config  # noqa: E402

CFG = load_config()
SC, RC, GC = CFG["sim"], CFG["registration"], CFG["gauges"]
S, M = CFG["lithobench"]["crop_px"], CFG["sim"]["margin_px"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPL = json.load(open(os.path.join(CFG["paths"]["results"], "splits.json")))
TILES = [("MetalSet", t) for t in SPL["val"]["MetalSet"][40:44]] + [("ViaSet", t) for t in SPL["val"]["ViaSet"][40:44]]


@pytest.fixture(scope="module")
def batch():
    crops = [ss.load_crop(CFG, s, t, *ss.choose_crop(CFG, s, t, S), S) for s, t in TILES]
    a = np.stack([c[0] for c in crops]); T = np.stack([c[1] for c in crops])
    p = ss.sample_params(np.random.default_rng(11), SC, len(TILES))
    return T, ss.simulate(a, p, SC, S, seed=21, device=DEV, frames=[2, 64])


def test_coarse_recovers_integer_shift():
    s, t = TILES[0]
    _, T = ss.load_crop(CFG, s, t, *ss.choose_crop(CFG, s, t, S), S)
    d = torch.as_tensor(T.astype(np.float32), device=DEV)[None, None]
    for dx, dy in ((32, 32), (37, 27), (26, 39)):
        est = rg._coarse(rg.smooth_norm(d[..., dy:dy + S, dx:dx + S], 2.0), rg.smooth_norm(d, 2.0), M).cpu().numpy()[0]
        assert np.abs(est - [dx, dy]).max() < 0.1, (dx, dy, est)


def test_known_registration_recovered_with_printed_template(batch):
    """Against a template of the actually printed pattern the algorithm recovers stage offset + distortion."""
    T, o = batch
    tm, wt = rg.design_template(o["F"] > 0, dict(RC, corner_weight_px=0.0))
    for fr in (2, 64):
        A, b = rg.register(torch.as_tensor(o["noisy"][fr], device=DEV)[:, None], torch.as_tensor(tm, device=DEV)[:, None], RC, M)
        ec, em = rg.registration_error(A.cpu().numpy(), b.cpu().numpy(), o["A"], o["b"], S)
        assert np.median(ec) < 0.6, (fr, ec)
        assert np.median(em) < 2.0, (fr, em)


def test_cad_registration_deterministic_and_bounded(batch):
    """CAD-template error includes printed-vs-design placement (systematic EPE); only sanity-bounded here."""
    T, o = batch
    tm, wt = rg.design_template(T, RC)
    x = torch.as_tensor(o["noisy"][64], device=DEV)[:, None]
    args = (torch.as_tensor(tm, device=DEV)[:, None], RC, M)
    A1, b1 = rg.register(x, *args, weight=torch.as_tensor(wt, device=DEV)[:, None])
    A2, b2 = rg.register(x, *args, weight=torch.as_tensor(wt, device=DEV)[:, None])
    assert torch.equal(A1, A2) and torch.equal(b1, b2)
    ec, _ = rg.registration_error(A1.cpu().numpy(), b1.cpu().numpy(), o["A"], o["b"], S)
    assert np.median(ec) < 6.0, ec


def test_uniform_pattern_shift_is_absorbed():
    """A uniform pattern shift changes true EPE but registration removes it: not a valid sensitivity test."""
    import evaluate as ev
    cp = ev.classical_params(CFG)
    df = ev.uniform_shift_demo(CFG, cp, shift_px=3.0, n_tiles=2, device=DEV)
    assert (np.abs(df.true_dEPE_signed - 3.0) < 0.3).all(), df
    assert (np.abs(df.measured_dEPE_signed) < 1.0).all(), df


def test_gauges_zero_offset_reproduce_design_and_gds_roundtrip(tmp_path, batch):
    T, o = batch
    for i, is_c in ((0, False), (len(TILES) - 1, True)):
        pr = gg.build_gauges(T[i], GC, is_c, GC["border_px"])
        g0 = gg.gauge_values(pr, np.zeros(len(pr)), np.ones(len(pr), bool))
        d = pr.groupby("gauge_id").design_cd.first()
        sel = g0.kind.isin(["cd", "contact"])
        assert np.allclose(g0[sel].value.values, d.loc[g0[sel].gauge_id].values, atol=1e-6)
        layers = {1: gg.contours_from_field(o["F"][i])}
        path = str(tmp_path / f"t{i}.gds")
        gg.export_gds(path, layers, (1000, 2000))
        ok, md, n = gg.gds_roundtrip_check(path, layers, (1000, 2000))
        assert ok and n > 0, (md, n)
