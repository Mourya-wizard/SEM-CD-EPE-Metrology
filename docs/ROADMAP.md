# Roadmap v2: incremental improvements driven by measured results

This roadmap lists what has been measured so far (E1–E10), what each result implies, and the experiments that follow
from it. The experiments form a **chain**: each one changes a single component, has one gate that was written down
before the run, and its decision becomes the starting point of the next. There is no hyper-parameter grid; every fixed
value is justified in advance. Decisions use the **validation** set; the test set is evaluated once per frozen
configuration at the end.

## 1. What has been run and what it implies

| # | Experiment | Measured result | Implication |
|---|---|---|---|
| E1 | Truth and simulator checks | Truth recovered to 0.018 px; peak location with material contrast is biased by −0.60 px; a known bloom offset of ±1 px gives −1.017 / +1.038 px bias | The bright peak is not the edge. The simulator draws the peak offset per image (±1.5 px), so a per-image ambiguity can set an error floor (→ E13) |
| E2 | Registration | Printed-pattern template: 0.28–0.32 px median error; CAD template: ~2.6 px; vias print (−1.0, +2.0) px away from the design; a uniform +3 nm shift is absorbed | Registration against the raw design is biased by printing effects (→ E12) |
| E3 | Classical baseline | CD slope 0.995, TMU-style 6.73 nm, precision 0.88 nm at 64 frames | Reference for every comparison |
| E4 | Operator screening (validation) | DCNv3 best (5.49 nm); U-Net statistically tied (+0.03, CI −0.04…+0.11); classical profile refinement worsens every model | Keep DCNv3 as the working operator; drop refinement |
| E5 | Test-set metrology | Learned TMU-style 6.29–6.30 vs 6.73 nm; precision 0.45 vs 0.88 nm (≈4x dose); **TMU at 64 frames (6.21 nm) is nearly the same as at 8 frames (6.27 nm) although precision improves 0.43 → 1.17 nm**; EPE TMU 8.25 nm vs 2.66 nm with ideal registration; outside-range CD failures 46% vs 1.2% | The accuracy floor is systematic, not noise. Registration (E12) and the per-image edge ambiguity (E13) are the two measured candidates |
| E6 | CAD vs SEM-only | Accuracy equal (6.27–6.31 vs 6.30 nm); contact slope 0.948–0.955 with CAD vs 0.968–0.969 without, consistent across seeds | **Decision taken: SEM-only is the default model** for everything below |
| E7 | Roughness and LCDU | Learned models report LER/LWR 4.5–6.8% below truth at 64 frames; classical stays within ±0.5%; LCDU excess is systematic | Learned models smooth real roughness (→ E14) |
| E8 | Confidence | Seed spread + profile quality do not rank errors (Spearman −0.07…+0.11) | Not pursued now |
| E9 | Carinthia-S (real SEM) | DCNv3 beats U-Net on every seed (Dice 0.956 vs 0.951); capture 100% at 2.9–5.9% nuisance | DCN helps on irregular real defects but not on smooth simulated edges: E11 looks at why |
| E10 | C++/CUDA kernel | 62x faster than 48 CPU threads; the refinement path is not used by the best model | No further kernel work |

## 2. The chain

```
E11 Grad-CAM + DCN offsets (understand)  ──►  where do U-Net / DCN models look, do DCN offsets deviate?
        │                                       (applied again at every later stage)
E12 bias-aware registration  ──decision──►  evaluation protocol used by E13, E14 and the final test
        │
E13 edge-offset supervision  ──decision──►  training recipe used by E14
        │
E14 roughness spectrum loss (built on the E13 winner)
        │
Final: lock the recipe, seeds 2–3, one test evaluation, README update
```
Deferred, not run: E15 Monte Carlo SEM check (Nebula) and any backbone scaling or multi-task heads.

## 3. E11: Grad-CAM and DCN offset maps (explainability at each stage)
- **Question:** where does each model look when it measures a CD, and do the deformable operators sample differently
  from a regular convolution?
- **Method:** Grad-CAM with a metrology target. For simulated SEM the scalar is the sum of predicted CD gauge values
  (differentiable zero crossing of the TSDF along the gauge normals); for Carinthia-S it is the summed defect logit.
  CAMs are taken at the last deformable block and at the final decoder layer. For DCNv2/v3/v4 the learned sampling
  offsets are visualised as magnitude maps. For models with a CAD channel, gradient×input gives the share of
  attribution on the CAD channel.
- **Quantities reported:** share of CAM mass within 4 px and 12 px of the true contour; mean offset magnitude near
  edges vs in flat regions; CAD-channel attribution share.
- **Stages covered:** U-Net, DCNv2, DCNv3, DCNv4 (SEM + CAD); SEM-only DCNv3; the E13 and E14 models; Carinthia-S models.

## 4. E12: bias-aware registration
- **Problem (E2, E5):** registering against the raw design mis-locks when printing changes feature sizes.
- **Change:** the template is rendered from the design with a global edge bias *b* (grown or shrunk by *b* px),
  and translation and *b* are estimated jointly: the best template over *b* ∈ {−14, −12, …, +14} px by correlation
  peak. A peak-to-sidelobe ratio (PSR) flags unreliable locks. Registration is not trained against the simulator's
  true offsets, because that would learn LithoBench's placement habits and hide real placement error.
- **Gates (validation, SEM-only DCNv3 seed 1, dose 8):**
  1. outside-range CD failure rate ≤ 50% of the CAD-registration value;
  2. inside-range EPE TMU-style ≤ CAD-registration value + 0.10 nm;
  3. PSR flag: ROC AUC ≥ 0.85 for detecting a registration centre error above 3 px.
- **Decision:** adopt the new registration if gates 1 and 2 hold; adopt the flag if gate 3 holds. Otherwise keep CAD
  registration and report the failure. EPE is also reported split into global placement and local shape.

## 5. E13: edge-offset supervision
- **Hypothesis (E1, E5):** the per-image bloom offset (±1.5 px) cannot be separated from the edge position at low
  contrast and sets part of the CD floor (a ±1.5 px uniform offset alone implies a CD 3σ near 5.2 nm).
- **Step 0, no training (baseline model, ideal registration, inside range, dose 8):** split the CD error into a
  per-(tile, condition) component and a per-gauge residual. Proceed only if the per-image component carries ≥ 30% of
  the error variance **and** its Pearson correlation with the bloom offset has |r| ≥ 0.5. Otherwise the hypothesis is
  rejected and the experiment stops with that finding.
- **Change:** an auxiliary head on the bottleneck features predicts the per-image offset (L1, normalised, weight
  0.05, fixed in advance). Same data, schedule and seed as the baseline.
- **Gates (validation, seed 1, dose 8):** CD TMU-style with ideal registration ≤ baseline − 0.30 nm (≈ 7x the seed
  range seen in E5); estimated-registration TMU ≤ baseline + 0.05 nm; low-contrast stress RMSE ≤ baseline + 0.10 nm;
  offset-head R² ≥ 0.5 (the ambiguity can be resolved from the image).
- **Decision:** if all hold the auxiliary head becomes part of the recipe; otherwise the baseline recipe stays and the
  ambiguity is reported as an irreducible edge-model term.

## 6. E14: roughness spectrum loss
- **Problem (E7):** learned contours are smoother than the true edges.
- **Change (on the E13 winner):** add a log-spectrum matching term on the TSDF band along rows and columns for
  wavelengths of 4–64 px (weight 0.05, fixed in advance). The training signal is a proxy; the evaluation uses the real
  LER/LWR analysis on the line/space set with ideal registration, so a proxy that does not transfer is detected.
- **Gates (validation):** mean |LER bias| at 64 frames over σ = 1 and 2 nm ≤ 2.5% (baseline ≈ 5%); CD TMU-style
  (estimated registration) ≤ baseline + 0.10 nm; CD precision at 8 frames ≤ 1.10 × baseline.
- **Decision:** adopt if all hold; otherwise keep the previous recipe and publish a calibrated LER correction factor
  with its uncertainty.

## 7. Final stage
Lock the recipe chosen by E12–E14. If it differs from the baseline, train seeds 2 and 3 and run one test evaluation
of each seed with the chosen registration. Report a stage table (baseline → +E12 → +E13 → +E14) with seed ranges, and
regenerate the Grad-CAM figures for the final model.

## 8. Results

Validation set, SEM-only DCNv3 baseline (seed 1), dose 8 frames, unless stated otherwise. Files are in `results/registration_e12/` and `results/roadmap_gates/`.

### E12: bias-aware registration (gates evaluated 2026-11-15)
| Validation, SEM-only DCNv3 | CAD template | Bias-aware | Ideal registration |
|---|:--:|:--:|:--:|
| CD failure outside the training range | 43.5% | **2.6%** | 1.9% |
| EPE TMU-style, inside range (nm) | 7.75 | **5.29** | 2.91 |
| CD TMU-style, inside range (nm) | 6.71 | **6.02** | 5.56 |
| CD failure inside range | 4.9% | **1.2%** | 0.8% |

| Registration accuracy (validation, all doses at 8 frames) | CAD template | Bias-aware |
|---|:--:|:--:|
| Registrations off by more than 3 px | 50.5% | **16.7%** |
| Median centre error (px) | 3.08 | **1.94** |

- **Gate 1** (outside-range failures ≤ 50% of the CAD value): 43.5% → 2.6%. **Passed.**
- **Gate 2** (inside EPE TMU-style not worse by more than 0.10 nm): 7.75 → 5.29 nm. **Passed.**
- **Gate 3** (PSR flag ROC AUC ≥ 0.85 for detecting a registration error above 3 px): AUC 0.55 (0.59 outside the range). **Failed.**
- **Decision:** the bias-aware registration replaces the CAD-template registration in all later evaluations (E13, E14 and the final test). The PSR flag is not adopted.
- **By-product:** the estimated global edge bias follows the true exposure shift. After removing the constant 2.4 px offset between
  printed and design edges, estimated and true shifts agree within 1.4 px over the whole sweep (for example +11.7 vs +10.3 px at threshold 0.190 and
  −10.6 vs −10.7 px at 0.260), so the registration also acts as an exposure-bias monitor.
- **What remains:** the gap to ideal registration (EPE 5.29 vs 2.91 nm) is placement that cannot be separated from printing effects using the SEM image
  and the design alone.

### E13 step 0: is the CD error a per-image offset? (baseline, ideal registration, inside range)
- The per-image component carries **91.2%** of the CD error variance (300 images, 15,478 gauges).
- The per-image mean error correlates with the image's bloom offset at **r = −0.979**, with a slope of −1.87 nm per px of offset
  (a peak-based edge finder would give −2).
- Both pre-registered conditions (share ≥ 30%, |r| ≥ 0.5) hold, so the training experiment proceeds. The hypothesis that the CD floor is
  an edge-offset ambiguity is confirmed for the baseline model.

### E13: edge-offset supervision (gates evaluated 2026-11-15)
| Validation, SEM-only DCNv3, dose 8 | Baseline | Auxiliary offset head |
|---|:--:|:--:|
| CD TMU-style, ideal registration (nm) | 5.562 | 5.539 |
| CD TMU-style, bias-aware registration (nm) | 6.015 | 5.994 |
| Low-contrast stress RMSE (nm) | 2.129 | 2.131 |
| Offset-head R² on validation images | n/a | **−0.008** |

- **Gate 1** (ideal-registration TMU drop ≥ 0.30 nm): −0.02 nm. **Failed.** **Gate 2** (estimated-registration TMU not worse by more than 0.05 nm): passed.
  **Gate 3** (stress RMSE not worse by more than 0.10 nm): passed. **Gate 4** (offset-head R² ≥ 0.5): **failed**.
- **Decision:** the auxiliary head is not adopted; the baseline recipe stays and E14 is built on it.
- **Interpretation:** step 0 showed that the CD error is almost entirely a per-image offset that tracks the bloom offset (r = −0.98). The head
  shows that this offset cannot be recovered from an 8-frame image by this network (R² ≈ 0), so for an uncalibrated tool the ambiguity is an
  irreducible edge-model term. Removing the per-image component from the validation errors would leave a residual variance of 8.8% of the
  total, which corresponds to a CD 3σ of about 1.7 nm instead of 5.6 nm (derived from the decomposition, not measured with a model). In a real
  recipe the edge-model offset is calibrated against reference metrology (for example cross-section TEM or AFM); this simulation randomises it per image
  on purpose, so the TMU-style floor here is the cost of an uncalibrated edge model.

### E14: roughness spectrum loss (gates evaluated 2026-11-15)
| Validation, SEM-only DCNv3, line/space set, 64 frames | Baseline | + spectrum loss (0.05) |
|---|:--:|:--:|
| Mean abs. LER bias, σ = 1 and 2 nm (gate metric) | 70.6% (invalid, see below) | **4.6%** |
| Signed LER bias, σ = 1 nm (mean over repeats) | n/a | −4.7% |
| CD TMU-style, bias-aware registration (nm) | 6.015 | 5.950 |
| CD precision, 8 frames (nm, 3σ) | 1.264 | 1.265 |

- **Gate 1** (|LER bias| ≤ 2.5%): 4.6%, **failed**. **Gate 2** (TMU not worse by more than 0.10 nm) and **Gate 3** (precision ≤ 1.10×): passed.
- **Decision:** not adopted under the pre-registered rule; the recipe stays the baseline.
- **Caveat on the baseline number.** The retrained SEM-only baseline produced a failed measurement on 38–69% of the line/space gauges (E14 model: 0–0.2%), so
  its 70.6% LER bias is not a roughness measurement and the gate comparison against it is not informative. The meaningful reference is the earlier
  CAD-input models on the same set (−5.7% for DCNv3 and −4.9% for U-Net at 64 frames, σ = 1 nm). Against that reference the spectrum loss moves the
  bias only from about −5% to about −4.7% at 64 frames, i.e. it does **not** close the gap, which agrees with the gate decision. At 2 frames the E14 model over-reports (+31%, noise).
  Why the SEM-only baseline fails on this set was not investigated (the other SEM-only checkpoint, E14, does not).
- **Interpretation:** a log-spectrum proxy on the TSDF band does not make the learned contour recover the true roughness amplitude. The calibrated
  LER correction factor from E7 remains the way to report roughness with these models.

## 9. Final stage (decided 2026-11-15)
No new recipe was adopted beyond E12 (an evaluation-time change): E13 and E14 failed their gates, so there are no extra training seeds to run.

| Stage (SEM-only DCNv3, seed 1, test set, dose 8) | CD TMU-style (nm) | CD failure in / out of range | EPE TMU-style (nm) | EPE slope |
|---|:--:|:--:|:--:|:--:|
| Baseline, CAD-template registration | 6.32 | 4.6% / 46.9% | 8.33 | 0.862 |
| + E12 bias-aware registration | 5.56 | 1.0% / 2.1% | 5.15 | 0.953 |
| + E13 offset head | not adopted (R² −0.008) | | | |
| + E14 spectrum loss | not adopted (LER bias 4.6% vs gate 2.5%) | | | |
| Classical tracer, CAD → bias-aware registration | 6.73 → 6.12 | 6.8% / 47.6% → 2.5% / 4.0% | 8.17 → 5.25 | 0.850 → 0.951 |

Single seed (1); no seed range is available for E12, which was evaluated on one retrained model and on the classical tracer. With
bias-aware registration the learned and classical EPE agree (5.15 vs 5.25 nm), so the earlier EPE gap was registration, not the edge model.
The learned model's advantage that remains is precision (0.44 vs 0.89 nm 3σ at 64 frames).

## 10. E11 results: Grad-CAM at each stage (2026-11-15)
Files: `results/gradcam/`. Scalar = sum of predicted CD-gauge edge offsets; "inside features" = CAM mass inside printed shapes (area share 10.9%).
- **Operators (SEM + CAD, simulated):** at the last deformable block and final decoder layer all four operators put ~50% of CAM mass inside
  features (4.6× chance) and none in the far background. The deepest block differs: U-Net 5% and DCNv2 21% inside features, DCNv3 3% and DCNv4 1%
  with 41% / 26% in the far background, i.e. the coarse deepest block of the DCN models carries a large background (context) component.
- **DCN learned offsets (last deformable block, input px):** DCNv3 3.7 near / 3.0 far from edges, DCNv2 6.0 / 7.2, DCNv4 5.1 / 7.6. Offsets are not
  concentrated at edges, so there is no sign in these maps that deformable sampling is edge-adaptive here, consistent with the null result of E4/E5.
- **Real SEM (Carinthia-S):** CAM lies in the defect for 91–97% of its mass for all operators; the DCNv3 offset is 1.4× larger near the defect boundary
  than elsewhere, DCNv4 1.0×, DCNv2 0.9×.
- **CAD vs SEM-only:** with CAD input the deepest block's CAM is 2.9% inside features vs 29.1% for SEM-only (far background 41% vs 16%); the CAD channel carries 3.9% of input attribution.
- **Stages (baseline, +E13, +E14):** final-layer CAMs are nearly identical (CAM within 4 px of edges 0.6% for all three); neither change altered where the model looks.

## 11. E16: equal-width operator comparison (pre-registered 2026-11-15, before any E16 run)
**Question.** Does deformable sampling beat a regular convolution at the same channel width? Earlier comparisons matched the parameter count
(about 5.6 M), which gave DCNv3 and DCNv4 a base width of 36 and conv and DCNv2 a base width of 32. Here every operator runs at the same width.

**Design (SEM-only input, E12 bias-aware registration for all evaluations).**
- Width 36 for all four operators. At width 36 the parameter counts are conv 7.09 M, DCNv2 7.37 M, DCNv3 5.56 M and DCNv4 5.55 M, so conv carries 27% more
  parameters than DCNv3, and a DCN advantage cannot be explained by capacity. A conv control at width 32 (5.60 M, the earlier setting) isolates the effect of width alone.
- Simulated SEM: 2 seeds per model (conv-32, conv-36, DCNv2-36, DCNv3-36, DCNv4-36), same recipe (6000 iterations, batch 12). DCNv3-36 seed 1 is the existing baseline run.
- Real SEM (Carinthia-S): 3 seeds for conv-36 and DCNv2-36 (new), DCNv4 seeds 2–3 (default width is already 36); DCNv3 (width 36) and conv-32 use their existing runs.
- Per-run `width:` override in the run yaml (`models.build_for_run`, `train.py`, `carinthia.py`); nothing else in the recipe changes.

**Why these metrics.** E13 showed that the CD TMU-style floor is a per-image edge offset that no network recovers, so TMU-style is only a "not worse" check.
Precision at low dose, failure rate and Carinthia Dice are where an operator can still differ. Line/space failure rate and LER bias (E14) are recorded because
the current SEM-only baseline failed on many line/space gauges.

**Gates (validation only; the test set is not used in E16).**
- **G0 (informational):** conv-32 vs conv-36, CD precision 3σ at 8 frames. Tells whether width alone changes anything.
- **G1 (simulated advantage of an operator over conv-36):** all of (a) mean CD precision at 8 frames ≤ 0.95 × the conv-36 mean and below the smaller conv-36 seed on both seeds;
  (b) CD TMU-style (8 frames) not worse by more than 0.10 nm; (c) CD failure rate (8 frames) not worse by more than 0.5 points.
- **G2 (real-SEM advantage):** mean validation Dice ≥ conv-36 mean + 0.003 and higher than conv-36 on every seed (paired by seed).
- **Also recorded, not gated:** precision at 2 frames, stress-contrast RMSE, line/space failure rate and signed LER bias at 64 frames, inference latency, learned-offset statistics.

**Decision rule.** An operator that passes G1 or G2 is carried into the next experiment at width 36. If none passes, the earlier parity was a width effect or a genuine tie,
and the placement experiment (DCN at the two highest-resolution levels) is run with the best-performing operator on real SEM, DCNv2 breaking ties because its compiled kernel gives the only realistic latency.
Evaluation: `tools/e16_analysis.py` (results in `results/roadmap_gates/e16-*`).

### E16 results (validation, evaluated 2026-11-15)
Simulated SEM: SEM-only models, bias-aware registration, 2 seeds; values are the mean with the seed range in brackets. Real SEM: Carinthia-S, 3 seeds, best validation Dice on defect images.
Files: `results/roadmap_gates/e16-sim-summary.csv`, `e16-real-summary.csv`, `e16-gates.json`.

| Model (width) | Params | CD prec. 3σ, 8 fr (nm) | CD prec. 3σ, 2 fr (nm) | CD TMU, 8 fr (nm) | CD failure, 8 fr | Stress RMSE, 8 fr (nm) | Carinthia val Dice |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Conv (32) | 5.60 M | 1.335 [1.325, 1.345] | 2.449 [2.391, 2.506] | 5.99 | 1.7% | 2.33 | 0.9456 |
| Conv (36) | 7.09 M | 1.314 [1.311, 1.317] | 2.378 [2.372, 2.385] | 5.98 | 1.8% | 2.25 | 0.9446 |
| DCNv2 (36) | 7.37 M | 1.283 [1.264, 1.303] | 2.344 [2.325, 2.362] | 5.98 | 1.8% | 2.38 | 0.9455 |
| DCNv3 (36) | 5.56 M | 1.272 [1.264, 1.281] | 2.347 [2.325, 2.370] | 5.98 | 1.7% | 2.23 | **0.9485** |
| DCNv4 (36) | 5.55 M | **1.270** [1.257, 1.284] | 2.341 [2.292, 2.390] | 5.96 | 1.8% | 2.40 | 0.9477 |

Carinthia values per seed: conv-32 0.9448 / 0.9462 / 0.9457; conv-36 0.9436 / 0.9459 / 0.9442; DCNv2-36 0.9474 / 0.9455 / 0.9437; DCNv3 0.9488 / 0.9485 / 0.9483; DCNv4 0.9470 / 0.9475 / 0.9485.
DCNv3 and conv-32 use the published runs for seeds 2–3 (their checkpoints no longer exist); seed 1 of each was retrained.

**Gates as pre-registered.**
- **G0 (width alone, conv-32 → conv-36):** precision at 8 frames 1.335 → 1.314 nm (−1.6%); Carinthia Dice 0.9456 → 0.9446 (no gain). Width helps the simulated precision slightly and the real-SEM Dice not at all.
- **G1 (simulated advantage over conv-36): failed for all three operators.** Precision at 8 frames improves by 2.3% (DCNv2), 3.2% (DCNv3) and 3.3% (DCNv4), below the 5% requirement.
  Each DCN seed is below the smaller conv-36 seed, so the direction is consistent, but the effect is small. TMU-style and failure rate are unchanged (G1b and G1c hold).
- **G2 (real-SEM advantage): passed by DCNv3 (+0.0039) and DCNv4 (+0.0031), failed by DCNv2 (+0.0009).** DCNv3 and DCNv4 are higher than conv-36 on every seed.

**Observations (not gated).**
- At equal width the parameter-matching explanation does not hold: DCNv3 and DCNv4 beat conv-36 on real SEM with 22% fewer parameters (5.5 M vs 7.1 M), and conv-36 does no better than conv-32.
- The simulated gain is about 3% of precision, the same size as the effect of width alone for conv (1.6%), and CD TMU-style stays at 5.96–5.99 nm for every model, as E13 predicted for an edge-offset-limited metric.
- Stress-contrast RMSE does not rank the operators consistently (2.23–2.40 nm, seed ranges overlap).
- **Line/space roughness:** the failure of the retrained SEM-only DCNv3 baseline on line/space gauges (E14 caveat) is specific to seed 1 (38.9% failed, +56% LER bias); seed 2 has 0% failures. All other 8 models have 0–0.7% failures.
  The earlier E14 comparison therefore used an unstable baseline checkpoint, not a general DCNv3 property.
  Signed LER bias at 64 frames (σ = 1 and 2 nm): conv-32 −4.2 / −4.0%, conv-36 −0.6 / −2.2%, DCNv2 −5.2 / −3.6%, DCNv3 (seed 2) −5.6%, DCNv4 −5.0 / −2.9%.
  Conv at width 36 under-reports roughness least; the DCN models are near the earlier −5%.
- Inference latency was not re-measured at equal width. The earlier batch-1 timings (SEM + CAD models, PyTorch paths for DCNv3 and DCNv4) are conv 4.76 ms, DCNv2 6.31 ms, DCNv3 11.88 ms, DCNv4 11.42 ms.

**Decision.** No operator is adopted on simulated precision (G1 failed). DCNv3 and DCNv4 show a consistent real-SEM Dice gain (G2 passed) and a small, consistent precision gain,
so both are carried into the placement experiment at width 36. DCNv2 is not (G2 failed, and its precision gain is the smallest).
