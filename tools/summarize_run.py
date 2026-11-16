"""Summarise one TEST measurement file (per kind x dose, tile-bootstrap CIs) into results/registration_test/test_summary_<out>.csv.
  python tools/summarize_run.py <meas_name> <out_name>      e.g. cadtest_dcnv3_semonly_s1_regbias__learned baseline_bias
Kept apart from results/model_eval/ so that retrained models never overwrite the published summaries."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import evaluate as ev  # noqa: E402
from data_lithobench import load_config  # noqa: E402

if __name__ == "__main__":
    meas, out = sys.argv[1], sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    cfg = load_config()
    s = ev.summarize(cfg, ev.load_meas(cfg, split, meas), ev.load_truth(cfg, split), out, n_boot=1000)
    d = os.path.join(cfg["paths"]["results"], "registration_test")
    os.makedirs(d, exist_ok=True)
    s.to_csv(os.path.join(d, f"{split}_summary_{out}.csv"), index=False)
    print("saved", out)
