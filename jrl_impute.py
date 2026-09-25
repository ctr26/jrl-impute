"""jrl_impute: Richardson-Lucy deconvolution with a spatially varying PSF
whose unmeasured entries are imputed from a sparse set of "bead" PSFs.

Self-contained (numpy / scipy / scikit-learn / scikit-image); see
docs/PROOFS.md for the maths and README.md for usage.

Pipeline
--------
1. ``radial_sigma`` + ``psf_stack``  -> one PSF per object pixel, ``(N, k, k)``
2. ``forward_matrix``                -> sparse ``H`` (N_p x N_v), column v = PSF of pixel v
3. ``bead_mask``                     -> which PSFs are "measured"
4. ``impute_psfs``                   -> fill in the rest from position
5. ``richardson_lucy``               -> recover x from b = H x (+ noise)
   (``ProductConvolution`` + ``richardson_lucy_pc``: rank-r FFT operator for large images)

Run ``python jrl_impute.py --help`` for the end-to-end demo.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

IMPUTE_METHODS = ("knn", "poly", "iterative", "mean")


# --------------------------------------------------------------------------
# PSF model
# --------------------------------------------------------------------------

def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Normalised, centred 2D Gaussian on an odd ``size x size`` pixel grid."""
    if size % 2 == 0:
        raise ValueError("PSF size must be odd so it has a centre pixel")
    r = np.arange(size) - size // 2
    g = np.exp(-(r**2) / (2.0 * sigma**2))
    psf = np.outer(g, g)
    return psf / psf.sum()


def radial_sigma(shape: tuple[int, int], sigma_min: float = 0.8,
                 sigma_max: float = 2.5) -> np.ndarray:
    """PSF width per pixel, growing linearly with distance from the centre
    (a stand-in for field-dependent aberration)."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]), indexing="ij")
    r = np.sqrt(xx**2 + yy**2) / np.sqrt(2.0)
    return sigma_min + (sigma_max - sigma_min) * r


def psf_stack(sigmas: np.ndarray, size: int) -> np.ndarray:
    """One Gaussian PSF per pixel: ``(N, size, size)`` for ``N = sigmas.size``."""
    return np.stack([gaussian_psf(size, s) for s in sigmas.ravel()])


# --------------------------------------------------------------------------
# Forward model
# --------------------------------------------------------------------------

def forward_matrix(psfs: np.ndarray, shape: tuple[int, int]) -> sparse.csr_array:
    """Sparse measurement matrix ``H`` with ``b = H @ x`` (both flattened).

    Column ``v`` is the image of a point source at pixel ``v``: ``psfs[v]``
    centred on ``v`` and cropped at the border (zero boundary). For any single
    (odd-sized) PSF this equals ``ndimage.convolve(x, psf, mode="constant")``.
    """
    h, w = shape
    n, k, _ = psfs.shape
    if n != h * w:
        raise ValueError(f"need {h * w} PSFs for shape {shape}, got {n}")
    r = k // 2
    vy, vx = np.divmod(np.arange(n), w)                       # source pixel
    dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    py = vy[:, None, None] + dy[None]                         # image pixel
    px = vx[:, None, None] + dx[None]
    inside = (py >= 0) & (py < h) & (px >= 0) & (px < w)
    rows = (py * w + px)[inside]
    cols = np.broadcast_to(np.arange(n)[:, None, None], inside.shape)[inside]
    return sparse.csr_array((psfs[inside], (rows, cols)), shape=(n, n))


def blur(x: np.ndarray, H: sparse.sparray) -> np.ndarray:
    """Apply ``H`` to a 2D image."""
    return (H @ x.ravel()).reshape(x.shape)


# --------------------------------------------------------------------------
# Sparse measurement + imputation
# --------------------------------------------------------------------------

def bead_mask(n: int, n_beads: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the ``n_beads`` pixels whose PSF is measured."""
    known = np.zeros(n, dtype=bool)
    known[rng.choice(n, size=n_beads, replace=False)] = True
    return known


def pixel_coords(shape: tuple[int, int]) -> np.ndarray:
    """``(N, 2)`` array of (y, x) pixel positions scaled to [-1, 1]."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]), indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def impute_psfs(psfs: np.ndarray, shape: tuple[int, int], known: np.ndarray,
                method: str = "knn", n_neighbors: int = 5, degree: int = 4,
                max_iter: int = 10, random_state: int = 0) -> np.ndarray:
    """Fill in the PSFs of pixels where ``known`` is False, as a function of position.

    * ``knn``       - inverse-distance mean of the ``n_neighbors`` nearest beads
    * ``poly``      - ridge regression of every PSF pixel on a ``degree``-order
                      polynomial in (y, x)
    * ``iterative`` - sklearn ``IterativeImputer`` (BayesianRidge) on the table
                      ``[y, x, psf...]``; the approach of the original scripts.
                      Slow, and each PSF pixel is regressed mostly on other,
                      mean-filled PSF pixels, so it stays close to ``mean``.
    * ``mean``      - the average bead PSF everywhere (spatially invariant baseline)

    Measured PSFs are kept exactly; imputed ones are clipped at 0 and
    renormalised to unit sum.
    """
    if method not in IMPUTE_METHODS:
        raise ValueError(f"method must be one of {IMPUTE_METHODS}")
    n, k, _ = psfs.shape
    flat = psfs.reshape(n, k * k).astype(float)
    xy = pixel_coords(shape)

    if method == "knn":
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        filled = model.fit(xy[known], flat[known]).predict(xy)
    elif method == "poly":
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
        filled = model.fit(xy[known], flat[known]).predict(xy)
    elif method == "iterative":
        table = np.hstack([xy, np.where(known[:, None], flat, np.nan)])
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,
                               random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            filled = imp.fit_transform(table)[:, 2:]
    else:
        filled = np.broadcast_to(flat[known].mean(axis=0), flat.shape).copy()

    filled[known] = flat[known]                               # keep measured PSFs exact
    filled = np.clip(filled, 0.0, None)
    filled /= np.maximum(filled.sum(axis=1, keepdims=True), 1e-12)
    return filled.reshape(n, k, k)


# --------------------------------------------------------------------------
# Deconvolution
# --------------------------------------------------------------------------

def richardson_lucy(H: sparse.sparray, b: np.ndarray, n_iter: int = 50,
                    x0: np.ndarray | None = None, background: float = 0.0,
                    eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Richardson-Lucy for ``b ~ Poisson(H x + background)``.

    ``x <- x / (H^T 1) * H^T (b / (H x + background))``

    Returns ``(x, rel_residual)`` where ``rel_residual[i] = ||b - H x_i|| / ||b||``.
    ``b`` may be 1D or 2D; ``x`` is returned with the same shape.
    """
    shape = b.shape
    b = np.clip(b.ravel().astype(float), 0.0, None)
    Ht = H.T.tocsr()
    norm = np.maximum(Ht @ np.ones(H.shape[0]), eps)
    x = np.full(H.shape[1], max(b.mean(), eps)) if x0 is None else x0.ravel().astype(float)
    x = np.clip(x, eps, None)
    b_norm = np.linalg.norm(b)
    history = np.empty(n_iter)
    for i in range(n_iter):
        Hx = H @ x
        x = x * (Ht @ (b / (Hx + background + eps))) / norm
        history[i] = np.linalg.norm(b - H @ x) / b_norm
    return x.reshape(shape), history


# --------------------------------------------------------------------------
# Product-convolution operator (low-rank, FFT-based; docs/PROOFS.md T3/T4)
# --------------------------------------------------------------------------

class ProductConvolution:
    """Rank-r column-varying operator ``H x = sum_i e_i (*) (c_i . x)``.

    Built from a PSF field by uncentred truncated SVD over the *whole* field,
    the Hilbert-Schmidt-optimal rank-r model (docs/PROOFS.md T3). The
    discarded energy ``tail`` = sum_{i>r} sigma_i^2 is an upper bound on the
    squared HS error (exact for interior columns). Uses zero-padded FFTs, so
    it matches ``forward_matrix`` (zero boundary) at full rank. A truncated
    operator can have negative entries; ``richardson_lucy_pc`` clamps H x.
    """

    def __init__(self, psfs: np.ndarray, shape: tuple[int, int], rank: int):
        n, k, _ = psfs.shape
        if n != shape[0] * shape[1]:
            raise ValueError(f"need {shape[0] * shape[1]} PSFs for shape {shape}, got {n}")
        M = psfs.reshape(n, k * k)
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        self.rank, self.shape, self.k = rank, shape, k
        self.sigma = s
        self.tail = float((s[rank:] ** 2).sum())
        self.E = Vt[:rank].reshape(rank, k, k)                         # basis PSFs
        self.C = (M @ Vt[:rank].T).T.reshape(rank, *shape)             # coefficient maps
        self._pad = (shape[0] + k - 1, shape[1] + k - 1)
        self._Ef = np.fft.rfft2(self.E, s=self._pad)

    def _crop(self, full: np.ndarray) -> np.ndarray:
        r = self.k // 2
        return full[..., r:r + self.shape[0], r:r + self.shape[1]]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """``H x``: convolve each weighted image with its basis PSF and sum."""
        xf = np.fft.rfft2(self.C * x.reshape(self.shape), s=self._pad)
        return self._crop(np.fft.irfft2((self._Ef * xf).sum(0), s=self._pad))

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """``H^T y``: correlate with each basis PSF, weight, and sum."""
        yp = np.zeros(self._pad)
        r = self.k // 2
        yp[r:r + self.shape[0], r:r + self.shape[1]] = y.reshape(self.shape)
        corr = np.fft.irfft2(np.conj(self._Ef) * np.fft.rfft2(yp)[None], s=self._pad)
        corr = np.roll(corr, (r, r), axis=(-2, -1))                    # undo kernel offset
        return (self.C * self._crop(corr)).sum(0)


def richardson_lucy_pc(op: ProductConvolution, b: np.ndarray, n_iter: int = 50,
                       background: float = 0.0, eps: float = 1e-12) -> np.ndarray:
    """Richardson-Lucy with a :class:`ProductConvolution` operator (``r`` FFT pairs/iter)."""
    b = np.clip(b.astype(float), 0.0, None)
    norm = np.maximum(op.rmatvec(np.ones(op.shape)), eps)
    x = np.full(op.shape, max(b.mean(), eps))
    for _ in range(n_iter):
        x = x * op.rmatvec(b / np.maximum(op.matvec(x) + background, eps)) / norm
        x = np.clip(x, eps, None)
    return x


# --------------------------------------------------------------------------
# End-to-end demo
# --------------------------------------------------------------------------

def test_image(size: int) -> np.ndarray:
    """Astronaut test card, greyscale, resized to ``size x size``, in [0, 1]."""
    from skimage import color, data
    from skimage.transform import resize
    img = resize(color.rgb2gray(data.astronaut()), (size, size), anti_aliasing=True)
    return (img - img.min()) / (img.max() - img.min())


def run_demo(size: int = 64, psf_size: int = 9, n_beads: int = 100,
             photons: float = 200.0, n_iter: int = 50, seed: int = 0,
             methods: tuple[str, ...] = IMPUTE_METHODS) -> dict:
    """Simulate, drop PSFs, impute, deconvolve. Returns images and metrics."""
    from skimage.metrics import normalized_root_mse, structural_similarity

    rng = np.random.default_rng(seed)
    shape = (size, size)
    truth = test_image(size)
    psfs_true = psf_stack(radial_sigma(shape), psf_size)
    H_true = forward_matrix(psfs_true, shape)
    b = rng.poisson(blur(truth, H_true) * photons) / photons
    known = bead_mask(truth.size, n_beads, rng)

    def score(img):
        return {"nrmse": float(normalized_root_mse(truth, img)),
                "ssim": float(structural_similarity(truth, img, data_range=1.0))}

    results = {"truth": truth, "blurred": b, "known": known.reshape(shape),
               "images": {}, "psf_error": {}, "metrics": {"blurred": score(b)}}

    runs = {"true H": psfs_true}
    for m in methods:
        runs[f"{m}-imputed H"] = impute_psfs(psfs_true, shape, known, method=m)
    for name, psfs in runs.items():
        x, _ = richardson_lucy(forward_matrix(psfs, shape), b, n_iter=n_iter)
        results["images"][name] = x
        results["metrics"][name] = score(x)
        miss = ~known
        results["psf_error"][name] = float(
            np.linalg.norm(psfs[miss] - psfs_true[miss]) / np.linalg.norm(psfs_true[miss]))
    return results


def _plot(results: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("ground truth", results["truth"]), ("blurred + noise", results["blurred"])]
    panels += list(results["images"].items())
    ncol = (len(panels) + 1) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 6.8))
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        m = results["metrics"].get(name.replace("blurred + noise", "blurred"))
        ax.set_title(name + (f"\nSSIM {m['ssim']:.3f}" if m else ""), fontsize=9)
        ax.axis("off")
    by, bx = np.nonzero(results["known"])
    axes[1].scatter(bx, by, s=4, c="tab:red", label="beads")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--size", type=int, default=64, help="image side length (pixels)")
    p.add_argument("--psf-size", type=int, default=9, help="PSF window (odd)")
    p.add_argument("--beads", type=int, default=100, help="number of measured PSFs")
    p.add_argument("--photons", type=float, default=200.0, help="peak photon count")
    p.add_argument("--iters", type=int, default=50, help="Richardson-Lucy iterations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=list(IMPUTE_METHODS),
                   choices=IMPUTE_METHODS)
    p.add_argument("--plot", default=None, help="save a comparison figure here")
    a = p.parse_args(argv)

    res = run_demo(a.size, a.psf_size, a.beads, a.photons, a.iters, a.seed,
                   tuple(a.methods))
    print(f"{'run':<20}{'SSIM':>8}{'NRMSE':>8}{'PSF err':>9}")
    for name, m in res["metrics"].items():
        err = res["psf_error"].get(name)
        print(f"{name:<20}{m['ssim']:>8.3f}{m['nrmse']:>8.3f}"
              + (f"{err:>9.3f}" if err is not None else f"{'':>9}"))
    if a.plot:
        _plot(res, a.plot)
        print(f"figure -> {a.plot}")
    return res


if __name__ == "__main__":
    main()
