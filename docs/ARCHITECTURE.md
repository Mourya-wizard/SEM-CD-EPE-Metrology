# Architecture and interfaces

These conventions are frozen. Changing any of them invalidates cached data and results; if a change is unavoidable,
document the reason and rerun the affected evaluations.

## Coordinates and units
- 1 px = 1 nm everywhere (LithoBench tiles, design crops, SEM images).
- Arrays are `[row = y (down), col = x]`; pixel centres at integer coordinates; points are `[x, y]`.
- Tile → design crop: the SEM field of view (FOV) is tile rows/cols `[y0, y0+512) x [x0, x0+512)`;
  the design crop is the FOV plus a 32 px margin: tile `[y0-32, y0+544)`. `origin = (x0-32, y0-32)` in tile px.
- SEM → design-crop mapping: `p_design = A @ [u, v] + b` (u = SEM col, v = SEM row). True `(A, b)` come from
  the simulator; estimated `(A, b)` come from `registration.register` (coarse translation; A = I).
- Truth field `F` (design-crop px): `F > 0` inside printed features; truth contour = zero level set of the
  bilinear interpolant of F (`sem_sim.crossings_along(..., order=1)`).
- Normals point OUT of the feature. Edge offsets `t` are signed along the outward normal (+ = feature larger).

## Gauges (`gauges.build_gauges`)
DataFrame `probes`: `gauge_id:int, kind:str in {epe1d, epe2d, cd, contact}, x, y, nx, ny, design_cd, cx, cy`.
- `epe1d`: straight design segments, >= 10 px from corners; `epe2d`: remaining boundary points; stride 4 px.
- `cd`: two opposite 1D probes on the same straight feature section; stride 8 px; value = design_cd + t_a + t_b.
- `contact`: 32 rays from the design centroid, anchored on the area-equivalent design circle;
  value = area-equivalent CD of the polygon of valid rays (>= 75% rays valid).
- Truth per probe: crossing of F within +-16 px along the normal; valid iff exactly one crossing.
- `gauges.gauge_values(probes, t, valid)` -> `gauge_id, kind, value (nm), valid`.

## Evaluation conditions (`evaluate.conditions`)
Index order: thresholds inside `[0.205, 0.215, 0.225, 0.235, 0.245]`, outside `[0.190, 0.200, 0.250, 0.260]`,
then `stress_contrast`, `stress_blur` (both at 0.225). Nominal = `thr0.225` (index 2). Each tile has one set of
imaging parameters (seeded by `tile_seed(set, tile)`) shared across conditions; local bumps are on.
Doses: frames `[2, 4, 8, 16, 32, 64]`; rep 0 for every condition; reps 0..19 on the nominal condition for the
first 30 tiles per set (precision tiles).

## Files (all under `results/raw/`, git-ignored, regenerable)
- `cache/<split>/<set>/<tile>.npz`: `clean (C,512,512) f16`, `F (C,576,576) f16`, `A (C,2,2)`, `b (C,2)`,
  `T (576,576) bool`, `origin (2)`, `seed`, `params (json str)`, `probes (records)`, `truth (records:
  gauge_id, kind, value, valid, cond)`.
- `reg/<split>/<set>/<tile>.npz`: `jobs (J,3) [cond, frames, rep]`, `A (J,2,2)`, `b (J,2)` (estimated).
- Noisy image for a job: `evaluate.eval_noisy(clean[cond], frames, seed*100+cond, rep, cfg['sim'])`
  (CPU torch generator, deterministic in any process).
- `meas/<split>/<method>.pkl.gz`: DataFrame `set, tile, cond, dose, rep, gauge_id, kind, value, valid`.

## Method API (for M5 learned models)
A method must produce, per job, a per-probe edge offset `t` (design px, + outward) and a validity flag, using the
estimated registration from `reg/` (oracle registration only for the documented EPE decomposition and roughness).
- Learned models output a truncated signed distance map (TSDF, + inside, px, truncation in config) in the SEM
  frame. CAD input = design raster warped into the SEM frame with the ESTIMATED registration
  (`registration.warp_design_to_sem`).
- `measure.measure_learned(tsdf, sem, probes, A, b, half_window, refine_params, sigma)` returns
  `{"learned": (t, v), "learned+refine": (t, v)}`.
- Selection metric (frozen before training): mean validation TMU-style CD at 8 frames (`eval.selection_dose`),
  with validation failure rate reported beside it.

## Edge refinement kernel (for `cuda/`)
`classical.refine_edges(img, pts, normals, t0, rp)` is the reference (float64). Inputs: float32 image (H,W)
(already Gaussian-filtered), points (N,2) `[x, y]` in image px, unit normals (N,2), start offsets t0 (N,),
params `w_in, w_out, step, frac, min_prom`. Output: offset t (N,) and valid (N,). Algorithm:
1. sample `K = round((w_in + w_out)/step) + 1` points at `t0 - w_in + k*step` (bilinear, coordinates clamped
   to the image like `map_coordinates(mode="nearest")`);
2. peak = argmax over the first `round(2*w_in/step) + 1` samples;
3. base = min over samples from the peak to the end; level = base + frac*(peak - base);
4. first sample after the peak with value < level; linear interpolation between it and the previous sample;
5. valid = crossing found and (peak - base) >= min_prom and peak index > 0.

## Carinthia-S (`src/carinthia.py`)
CSV `;`-separated `image_path, mask_path, filename, label`; images 480x480 uint8; masks binarised with `> 127`;
class 6 = no visible defect (negative class for capture vs nuisance); crop/mask the 3 px black right border.
