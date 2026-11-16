"""Classical baseline (validation-tuned, frozen) on the TEST set under both registrations -> results/registration_test/test_summary_classical_{cad,bias}.csv
  python tools/classical_test.py [workers]"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import evaluate as ev  # noqa: E402
from data_lithobench import load_config  # noqa: E402

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_config()
    cp = ev.classical_params(cfg, json.load(open(os.path.join(cfg["paths"]["results"], "classical_baseline", "classical_tuning.json"))))
    truth = ev.load_truth(cfg, "test")
    d = os.path.join(cfg["paths"]["results"], "registration_test")
    os.makedirs(d, exist_ok=True)
    for mode in ("cad", "bias"):
        cfg["registration"]["mode"] = mode
        meas = ev.run_classical(cfg, "test", cp, "est", workers=workers, label=f"test-{mode}")
        ev.save_meas(meas, cfg, "test", f"classical_est_reg{mode}")
        ev.summarize(cfg, meas, truth, f"classical_{mode}", n_boot=1000).to_csv(os.path.join(d, f"test_summary_classical_{mode}.csv"), index=False)
        print("done", mode, flush=True)
