# Limitations

This is a resume project that demonstrates metrology reasoning on **simulated** data. Read every number with the
limits below in mind.

## What the data can and cannot show
- **CD, EPE, TMU-style, precision, roughness and LCDU results all use simulated SEM images.** The simulator is a
  controlled forward model (material step + edge bloom over a level-set truth, Gaussian beam blur, low-order
  distortion, stage offset, Poisson shot noise + Gaussian detector noise). It is not an electron-matter simulator.
  Nothing here has been validated on a real CD-SEM.
- **LithoBench is 32/45 nm-class data** (1 nm/px tiles, aerial images thresholded at 0.225). No claim is made about
  advanced nodes, production use or fab readiness.
- **Carinthia-S is real SEM data, but only for defect segmentation/detection.** It has no CD or EPE truth. Its noise
  is not comparable to the simulator: relative noise 0.018 vs 0.079-1.056 simulated, and spatially correlated
  (lag-1 correlation 0.68, JPEG/unknown processing), so the comparison is indicative only.
- **TMU-style** = 3 x residual sd about the OLS line of measured vs simulated truth, with reference uncertainty set to
  zero because simulated truth is exact. A real TMU needs a reference measurement system with its own uncertainty
  (e.g. AFM/TEM cross-sections) and Mandel regression.

## Physics not modelled
- **Charging**, beam-induced **resist shrinkage**, **focus** variation and topography/shadowing are absent.
- **Real distortion** (scan non-linearity, drift during frame integration) is replaced by a static low-order affine.
- **Tool matching and drift** between tools/days are not modelled; all "repeatability" here is pure shot noise on a fixed geometry.
- **Edge bloom offset** is a free parameter (uniform +-1.5 px in training). Its true value depends on material and beam
  conditions; the measured bias of peak-based edge finding (about -0.6 px with material contrast) is a property of this model.

## Method and evaluation limits
- **Registration**: the CAD-template registration is limited by printed-vs-design placement (LithoBench via sets have
  a systematic printed offset of about (-1, +2) px) and fails at outer exposure values; at thresholds 0.19/0.26 the
  CD failure rate rises to >50% for every method with estimated registration, versus 1-12% for classical with oracle
  registration (validation). EPE numbers therefore include registration error by design; oracle-registration numbers
  are reported separately.
- **Uniform pattern shift is not a valid EPE sensitivity test** here (registration absorbs it: a true +3 nm shift is
  measured as -0.7..+0.0 nm).
- **Roughness correction**: the PSD high-frequency-plateau method assumes white measurement noise; filtering and blur
  correlate the noise along the edge, so the plateau under-estimates it. A repeat-noise correction is reported next
  to it, and results with SNR < 2 are flagged unreliable.
- **Seeds and statistics**: confidence intervals are tile-level bootstraps (resampling test tiles), not training-seed
  variation. Rows with one seed are single-seed results. See the README Limitations section for which runs have fewer seeds
  than planned because of the compute-time budget.
- **DCNv4** is a PyTorch reimplementation of the published module semantics, not the official CUDA operator (its build
  failed on this Windows/PyTorch 2.11 setup). Its speed says nothing about official DCNv4. DCNv3 uses the official
  InternImage PyTorch fallback, not the compiled operator.
- **Model capacity/budget**: about 5.6 M parameters, 6000 iterations; Carinthia models 3000 iterations (best
  checkpoint was the last one in every run, so they may be under-trained).

## What real-tool validation would require
Real CD-SEM images with traceable reference metrology on the same features (AFM or cross-section TEM), a calibrated
edge model per material stack, static and dynamic repeatability across days, tool-to-tool matching, charging and
shrinkage characterisation at the doses used, and the registration recipe validated against measured overlay/placement.
