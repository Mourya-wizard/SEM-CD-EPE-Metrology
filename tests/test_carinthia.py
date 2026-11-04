"""M7 Carinthia-S tests: split, mask binarisation, border handling, metrics, and a tiny end-to-end smoke run."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import carinthia as ca  # noqa: E402
from data_lithobench import load_config  # noqa: E402


def _df():
    n = {1: 55, 2: 8, 3: 400, 4: 29, 5: 4, 6: 23}
    lab = [c for c, k in n.items() for _ in range(k)]
    return pd.DataFrame({"filename": [f"f{i:04d}" for i in range(len(lab))], "label": lab})


def test_split_stratified_disjoint_deterministic():
    df = _df()
    s = ca.make_split(df, 7)
    sets = [set(s[k]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(df.filename)
    lab = df.set_index("filename").label
    for c, g in df.groupby("label"):
        nv, nt = (lab[s["val"]] == c).sum(), (lab[s["test"]] == c).sum()
        assert nv >= 1 and nt >= 1                              # rare classes (2: n=8, 5: n=4) reach every split
        assert abs(nv - 0.15 * len(g)) <= 1 and abs(nt - 0.15 * len(g)) <= 1
    assert s == ca.make_split(df, 7) and s != ca.make_split(df, 8)


def test_real_split_file_if_present():
    if not os.path.exists(ca.SPLIT_JSON):
        pytest.skip("split not written")
    s = json.load(open(ca.SPLIT_JSON))
    a, b, c = (set(s[k]) for k in ("train", "val", "test"))
    assert len(a | b | c) == 4591 and not (a & b or a & c or b & c)
    assert all(s["counts"][k].get("5", 0) >= 1 and s["counts"][k].get("2", 0) >= 1 for k in s["counts"])


def test_binarise_nonbinary_mask():
    m = np.array([[0, 127, 128, 255], [3, 200, 126, 130]], np.uint8)
    assert ca.binarise(m).tolist() == [[False, False, True, True], [False, True, False, True]]


def test_border_fill_and_valid():
    img = np.full((6, 8), 100, np.uint8)
    img[:, -3:] = 0
    x = ca.fill_border(img, 3)
    assert np.all(x == 100) and x.dtype == np.float32
    v = ca.valid_region((6, 8), 3)
    assert v[:, :5].all() and not v[:, 5:].any()
    pred, gt = np.zeros((6, 8), bool), np.zeros((6, 8), bool)
    pred[:, -1] = True                                         # prediction only in the border: ignored
    assert ca.seg_stats(pred, gt, v)["iou"] == 1.0


def test_seg_stats_toy():
    gt = np.zeros((4, 4), bool); gt[:2, :2] = True              # 4 px
    pred = np.zeros((4, 4), bool); pred[:2, :3] = True          # 6 px, 4 overlap
    r = ca.seg_stats(pred, gt)
    assert r["inter"] == 4 and r["union"] == 6
    assert r["iou"] == pytest.approx(4 / 6) and r["dice"] == pytest.approx(8 / 10)
    z = np.zeros((4, 4), bool)
    assert ca.seg_stats(z, z)["dice"] == 1.0 and ca.seg_stats(z, gt)["dice"] == 0.0


def test_threshold_capture_nuisance():
    neg = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    t = ca.pick_threshold(neg, 0.10)                            # 1 of 10 negatives may be flagged
    assert ca.flag_rate(neg, t) == pytest.approx(0.1) and 0.9 < t <= 0.95
    assert ca.flag_rate(neg, ca.pick_threshold(neg, 0.0)) == 0.0
    assert ca.pick_threshold(neg, 1.0) == -np.inf
    pos = np.array([0.99, 0.96, 0.5, 0.92])
    assert ca.flag_rate(pos, t) == pytest.approx(0.75)
    tt, cap, nui = ca.curve(pos, neg)
    assert cap[0] == 1 and nui[0] == 1 and cap[-1] == 0 and nui[-1] == 0 and np.all(np.diff(cap) <= 0)


def test_bootstrap_ci():
    x = np.r_[np.zeros(50), np.ones(50)]
    lo, hi = ca.bootstrap_ci(x)
    assert lo < 0.5 < hi and hi - lo < 0.3
    assert ca.bootstrap_ci(np.ones(5)) == [1.0, 1.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_train_aggregate_smoke(tmp_path, monkeypatch):
    """Tiny (64x64, 3 iters) end-to-end run with fake arrays: train -> per-image csv/probs -> aggregate."""
    files = [f"f{i}" for i in range(30)]
    lab = {f: (6 if i % 3 == 0 else 1 + i % 5) for i, f in enumerate(files)}

    def fake(cfg, fs, threads=3):
        X, Y = [], []
        for f in fs:
            r = np.random.default_rng(int(f[1:]))
            y = np.zeros((64, 64), bool)
            if lab[f] != 6:
                y[20:30, 20:34] = True
            X.append((r.normal(90, 10, (64, 64)) + 60 * y).clip(0, 255).astype(np.uint8)); Y.append(y)
        return np.stack(X), np.stack(Y), np.array([lab[f] for f in fs])

    split = {"train": files[:18], "val": files[18:24], "test": files[24:], "counts": {}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    base = yaml.safe_load(open(ca.BASE_YAML)); base["train"].update(iters=3, batch=2, val_every=2, log_every=1)
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "run.yaml").write_text("name: carinthia_conv_s1\nop: conv\nseed: 1\n")
    monkeypatch.setattr(ca, "load_arrays", fake)
    monkeypatch.setattr(ca, "SPLIT_JSON", str(tmp_path / "split.json"))
    monkeypatch.setattr(ca, "BASE_YAML", str(tmp_path / "base.yaml"))
    cfg = load_config()
    cfg["paths"]["data_root_abs"] = str(tmp_path / "data"); cfg["paths"]["results"] = str(tmp_path / "results")
    ca.train(cfg, str(tmp_path / "run.yaml"))
    d = tmp_path / "checkpoints" / "carinthia_conv_s1"
    assert json.loads((d / "status.json").read_text())["state"] == "done"
    te = pd.read_csv(d / "test_per_image.csv")
    assert len(te) == 6 and te.max_prob.between(0, 1).all()
    ca.aggregate(cfg)
    s = json.loads((tmp_path / "results" / "carinthia" / "carinthia_summary.json").read_text())
    assert s["runs"][0]["test_dice"]["n"] == 4 and (tmp_path / "results" / "carinthia" / "carinthia_grid.png").exists()
