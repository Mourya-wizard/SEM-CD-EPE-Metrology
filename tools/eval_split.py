"""Run one learned-model evaluation as parallel parts (one per tile set) and merge them.

The evaluation code path (evaluate.run_learned) is unchanged; each part evaluates a disjoint subset of the
split's tiles with the same deterministic noise, so the merged result equals a single-process run.

  python tools/eval_split.py part  <run> <split> <set_name>   # e.g. part cadtest_dcnv3_semonly_s2 test StdMetal
  python tools/eval_split.py merge <run> <split>
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

if __name__ == "__main__":
    import pandas as pd
    import torch

    import evaluate as ev
    from data_lithobench import load_config
    cfg = load_config()
    mode, run, split = sys.argv[1], sys.argv[2], sys.argv[3]
    d = os.path.join(cfg["paths"]["results"], "raw", "meas", split)
    if mode == "part":
        set_name = sys.argv[4]
        torch.cuda.set_per_process_memory_fraction(float(cfg["queue"]["mem_fraction"]))
        orig = ev.save_meas
        ev.save_meas = lambda df, c, sp, name: orig(df, c, sp, name.replace(f"{run}__", f"{run}__part_{set_name}__"))
        tiles = [x for x in ev.eval_tiles(cfg, split) if x[0] == set_name]
        _, ms = ev.run_learned(cfg, run, split, tiles=tiles)
        json.dump({"infer_ms_per_img_bs8": ms}, open(os.path.join(d, f"{run}__part_{set_name}__timing.json"), "w"))
    else:
        for key in ("learned", "refine"):
            parts = sorted(glob.glob(os.path.join(d, f"{run}__part_*__{key}.pkl.gz")))
            assert len(parts) == len({s for s, _ in ev.eval_tiles(cfg, split)}), parts
            df = pd.concat([pd.read_pickle(p, compression="gzip") for p in parts], ignore_index=True)
            ev.save_meas(df, cfg, split, f"{run}__{key}")
        t = [json.load(open(p))["infer_ms_per_img_bs8"] for p in glob.glob(os.path.join(d, f"{run}__part_*__timing.json"))]
        json.dump({"ckpt": run, "split": split, "reg": "est", "infer_ms_per_img_bs8": sum(t) / len(t), "parts": len(t)},
                  open(os.path.join(d, f"{run}__timing.json"), "w"))
        print("merged", run, split)
