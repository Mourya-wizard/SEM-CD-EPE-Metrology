"""LER/LWR of a trained model on the documented line/space set (same seeds, doses, noise and oracle registration
as the classical roughness analysis), for one roughness level.

  python tools/roughness_learned.py <run_name> <rough_sigma_nm>   -> results/<figures>/roughness_<run>_sigma<s>.csv
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

if __name__ == "__main__":
    import torch

    import evaluate as ev
    import measure as ms
    import train as tr
    from data_lithobench import load_config
    from models import build_model
    run, sigma = sys.argv[1], float(sys.argv[2])
    cfg = load_config()
    cfg["roughness"]["rough_sigma_nm"] = [sigma]
    ck = torch.load(os.path.join(ROOT, "checkpoints", run, "best.pt"), map_location="cuda", weights_only=False)
    cad = ck["run"]["cad"]
    model = build_model(cfg["model"], ck["run"]["op"], 2 if cad else 1).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    _, T = ev.linespace_set(cfg)
    S, trunc, hw = cfg["lithobench"]["crop_px"], cfg["model"]["tsdf_trunc_px"], cfg["gauges"]["probe_half_window_px"]

    def fn(img, pr, A, b, fr):
        c = [tr.cad_in_sem(T, A, b, S, "cuda")] if cad else None
        tsdf = tr.predict_tsdf(model, [img], c, "cuda")[0] * trunc
        return ms.measure_learned(tsdf, img, pr, A, b, hw)["learned"]

    df = ev.roughness_eval(cfg, fn)
    df["model"] = run
    out = next(os.path.join(ROOT, "results", d) for d in ("figures", "final") if os.path.isdir(os.path.join(ROOT, "results", d)))
    df.to_csv(os.path.join(out, f"roughness_{run}_sigma{sigma:g}.csv"), index=False)
    print("saved", run, sigma, len(df))
