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
