"""Smoke check used by `run_all.sh --smoke`: classical measurement on the 2+2-tile smoke split and a sanity
assertion on the CD regression slope. A script file (not stdin) so Windows multiprocessing can spawn workers."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

if __name__ == "__main__":
    import evaluate as ev
    from data_lithobench import load_config
    cfg = load_config()
    tj = [p for p in (os.path.join(cfg["paths"]["results"], d, "classical_tuning.json") for d in ("classical_baseline", "m3"))
          if os.path.exists(p)]
    cp = ev.classical_params(cfg, json.load(open(tj[0])) if tj else None)
    meas = ev.run_classical(cfg, "smoke", cp, "est", workers=4, label="smoke")
    s = ev.summarize(cfg, meas, ev.load_truth(cfg, "smoke"), "classical", n_boot=0)
    r = s[(s.kind == "cd") & (s.dose == 8)].iloc[0]
    print(f"[run_all] smoke classical CD @8 frames: slope {r.inside_slope:.3f} TMU-style {r.inside_tmu3s:.2f} nm fail {r.inside_fail_rate:.3f}")
    assert 0.8 < r.inside_slope < 1.2, "smoke regression slope out of range"
