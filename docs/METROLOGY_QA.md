# Interview notes (answers use this repo's measured results; all CD/EPE data are simulated)

**Why are precision and accuracy different in CD-SEM?**
Precision is how much repeated measurements of the *same* feature scatter (here: 20 noise seeds on fixed geometry);
accuracy is how far the average measurement sits from the true value. They come apart: the peak-based edge finder
was perfectly repeatable on noise-free images yet biased by -0.6 px because of the material step (M2), and a known
+1 px bloom offset shifted results by -1.02 px without changing precision. On test, classical CD precision at 64
frames is 0.88 nm 3 sigma, but its TMU-style figure is 6.73 nm: most of the error is systematic, not noise.

**What does TMU add beyond 3 sigma?**
TMU regresses measured vs reference values across many real features and process conditions and reports the
residual scatter about the fitted line, so it includes linearity errors and pattern-dependent bias that a
repeatability test on one feature never sees. Here the learned models' precision at 64 frames is 0.45-0.47 nm, but
their TMU-style CD is 6.29-6.30 nm (classical 6.73 nm). My "TMU-style" assumes zero reference uncertainty because
simulated truth is exact; a real TMU needs a reference tool (AFM/TEM) and Mandel regression.

**Why can CAD input hide real EPE?**
If the model sees the design, it can learn to draw the design instead of the wafer, pulling measurements toward
nominal. Test: train the selected DCNv3 with and without CAD and compare sensitivity. Accuracy was the same (TMU-style
6.27 vs 6.30 nm), but the contact-CD regression slope dropped from 0.968 [0.962, 0.974] without CAD to 0.948
[0.941, 0.954] with CAD: the CAD model under-responds to real contact-size changes by about 2%. Single seed for the
SEM-only model, so this is weak evidence, but it is the right test to run.

**Why does registration matter, and why was uniform shift invalid as a test?**
EPE is measured in design coordinates, so any registration error goes straight into EPE. With the CAD-template
registration, EPE1d TMU-style is 8.2 nm; with oracle registration it is 2.7-3.1 nm. At outer exposure values the
printed pattern differs so much from the design that registration mis-locks and CD failures jump to 46-48% for every
method (1.2-3.8% with oracle registration). A uniform pattern shift cannot be told apart from a stage offset, so
registration removes it: a true +3 nm EPE change was measured as -0.7..+0.0 nm. Sensitivity therefore has to be
tested with exposure-like (pattern-dependent) changes and local programmed edge changes.

**Why is the bright edge peak not necessarily the true edge?**
The secondary-electron peak sits where the beam interacts with the sidewall, offset from the physical edge by an
amount that depends on material, topography and beam conditions. In the simulator the bloom offset is a free
parameter (+-1.5 px in training); a +-1 px offset moved peak-based edges by -1.02/+1.04 px, and even with zero offset
the material-contrast step biased the peak by -0.6 px. The classical baseline therefore needed a calibrated edge
offset (about 2.0 px from validation), exactly like a real recipe calibrated against reference metrology.

**Why does raw LER or LCDU overstate true variation?**
Measurement noise adds variance: raw = true + noise. At 2 frames, raw LER was 2.98 nm vs a true 2.70 nm (3 sigma).
Subtracting a noise floor helps only if the floor is estimated correctly: the PSD high-frequency plateau assumes
white noise, but blur and filtering correlate the noise along the edge, so it under-estimates the noise; a
repeat-noise estimate (two noise realisations) gave 2.61 nm with SNR 1.9, flagged unreliable. For LCDU the excess
(raw 10.1-11.1 vs true 8.3 nm) was mostly systematic per-image error, not noise, so noise correction could not fix it.

**What did DCNv2, DCNv3 and DCNv4 change here, and which result would you trust?**
Same U-Net shell, same data and schedule, parameters matched within 10%, only the operator swapped. On simulated
metrology nothing changed measurably: DCNv3 (3 seeds) TMU-style 6.27-6.31 nm vs U-Net (3 seeds) 6.28-6.32 nm; DCNv2
and DCNv4 (single seed) 6.29-6.30 nm. On real Carinthia-S defect segmentation DCNv3 was slightly but consistently
better (Dice 0.9558-0.9569 vs 0.9500-0.9525 over 3 seeds each). I trust the 3-seed comparisons, treat DCNv2/DCNv4 as
single-seed indications, and note that DCNv4 is a PyTorch reimplementation, not the official kernel.

**What changes when moving from simulated to real tool data?**
No exact truth: accuracy needs reference metrology (AFM, cross-section TEM) with its own uncertainty, so TMU needs
Mandel regression. Real noise differs: Carinthia-S relative noise is about 0.018 and spatially correlated, far from
the simulator's 0.08-1.06 white shot noise. Charging, shrinkage, scan distortion, drift and tool matching appear;
the bloom offset must be calibrated per stack; and registration must be validated against real placement.

**Why C++/CUDA here, and when would the optimisation not be worth it?**
Edge refinement runs per gauge point, so full-wafer inspection means millions of points per image. At 1e6 points
on a 4096 px image the texture kernel takes 1.75 ms (manual 6.60 ms) vs 108.2 ms for 48-thread C++/OpenMP and
38.8 ms for PyTorch grid_sample. It is not worth it for small jobs: at 1e4 points on a 4096 px image, copying the
image to the GPU makes it 5x slower than the CPU, and even at 1e6 points transfer is most of the GPU time (7.2 ms
end-to-end). Hardware texture filtering is fastest, but its 8-bit weights produced a 5.4 px outlier on synthetic
data, so the manual path is the safe choice when parity with the reference matters.
