"""E17 Max Planck tests: SVG label rasterisation (tracks with holes, vias) and per-component track-width measurement."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import maxplanck as mp  # noqa: E402
import maxplanck_prepare as prep  # noqa: E402


def test_rasterise_track_hole_and_via(tmp_path):
    svg = tmp_path / "label.svg"
    svg.write_text('<svg><path d="M100,100 L300,100 L300,300 L100,300 Z M150,150 L150,250 L250,250 L250,150 Z" fill="#fff"/>'
                   '<circle cx="1000" cy="1000" r="20"/></svg>')
    m = prep.rasterise(str(svg))
    assert m[120, 120] & 1 and not m[200, 200] & 1          # filled track, opposite-orientation subpath is a hole
    assert m[1000, 1000] & 2 and not m[1000, 1000] & 1      # via bit only
    assert not m[50, 50]


def test_width_table_measures_and_flags_miss():
    gt = np.zeros((256, 256), bool)
    gt[100:112, 20:236] = True                               # 12 px wide horizontal track
    pr = np.zeros_like(gt)
    pr[101:111, 20:236] = True                               # predicted 2 px narrower
    (area, w_gt, w_pr), = mp.width_table(gt, pr)
    assert area == 12 * 216
    assert abs(w_gt - 12) <= 1.5 and abs((w_gt - w_pr) - 2) <= 1.0
    (_, _, w_miss), = mp.width_table(gt, np.zeros_like(gt))
    assert np.isnan(w_miss)
