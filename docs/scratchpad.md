# Scratchpad: research framing and inputs

> Running log of the author's framing and questions, kept so the docs can be checked against intent.
> Newest first. Answers live in [`RESEARCH.md`](RESEARCH.md) and [`PROOFS.md`](PROOFS.md).

## 2026-09-24: research direction

**Intent.** The repo was meant as a *principled* way to fit and run spatially varying deconvolution. It never reached publication.

**Where it got to.**
- Eigen-PSFs (PCA) of bead PSFs, used to represent roughly 5σ of the PSF, reconciled "well enough" on beads.

**Open questions raised.**
1. Is there anything better than PCA eigen-PSFs? PCA "isn't principled for microscopy".
   → [RESEARCH.md §2–4](RESEARCH.md), [PROOFS.md T3](PROOFS.md#t3-pca-of-the-whole-field-is-the-hilbertschmidt-optimal-product-convolution-model)
2. Zernikes were considered but "seem weak".
   → strong as a *parametrisation of field dependence* (Hopkins), weak as a *linear PSF basis* ([PROOFS.md T6](PROOFS.md#t6-identifiability-and-the-parametric-physics-fit))
3. Statistics-based PSF variation seems interesting.
   → astronomy PSF-field literature (PSFEx, RCA/MCCD, PIFF, WaveDiff); GP/kriging ([RESEARCH.md §3](RESEARCH.md))
4. Conjecture: coefficients on orthogonal vectors will be smooth, but this was never proved.
   → **proved** ([PROOFS.md T2](PROOFS.md#t2-any-fixed-orthonormal-basis-inherits-the-fields-smoothness-your-conjecture))
5. How to use smooth, orthogonal, summed convolving PSFs for **blind** deconvolution, i.e. to extract sensible PSFs?
   → [RESEARCH.md §5](RESEARCH.md)
6. What about PSFs that are hard to measure: **STED, SIM**?
   → [RESEARCH.md §6](RESEARCH.md)

## 2026-09-23: session start
- "go", with no further spec. Default chosen: make the pipeline runnable, add tests and docs (`jrl_impute.py`, `tests/`).
