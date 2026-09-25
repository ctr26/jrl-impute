#!/usr/bin/env python3
"""E3: blind (bead-free) spatially varying deconvolution with defocus diversity.

A thin sample (scikit-image ``cells3d`` slice) is imaged at focus offsets
``planes`` (known defocus, rad RMS on Z4) through a spatially varying PSF field.
Unknowns: the object x and the field parameters theta of a physics model
(Seidel-5 or quadratic-Zernike-31). The operator is an anchor-grid
product-convolution with bilinear (convex) weights:

    b_p = sum_a  h_a(theta, dz_p) (*) (w_a . x)

Alternating scheme: multi-plane Richardson-Lucy for x, then Gauss-Newton on
theta (Anscombe residuals, analytic Jacobian through the pupil model).

Compared against: oracle field, bead-calibrated field (E1 hybrid, 32 beads),
invariant diffraction-limited PSF, and blind WITHOUT diversity (in-focus plane
only; docs/PROOFS.md T6 predicts the even-aberration sign flip).

    uv run experiments/e3_blind.py [--size 96] [--photons 2000]

Outputs: results/e3.csv, results/e3_summary.md, results/e3.png
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jrl_impute as J  # noqa: E402
import psf_fields as pf  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
warnings.filterwarnings("ignore")


class AnchorOperator:
    """b = sum_a h_a (*) (w_a . x) with bilinear weights on a g x g anchor grid."""

    def __init__(self, optics: pf.Optics, shape, g: int = 8):
        self.o, self.shape, self.g = optics, shape, g
        t = np.linspace(-1, 1, g)
        self.anchors = np.stack(np.meshgrid(t, t), -1).reshape(-1, 2)      # (x, y)
        # bilinear hat weights over the image (partition of unity, convex)
        u = np.linspace(0, g - 1, shape[1])
        v = np.linspace(0, g - 1, shape[0])
        hx = np.maximum(0, 1 - np.abs(u[None] - np.arange(g)[:, None]))    # (g, W)
        hy = np.maximum(0, 1 - np.abs(v[None] - np.arange(g)[:, None]))    # (g, H)
        self.W = np.einsum("iy,jx->ijyx", hy, hx).reshape(g * g, *shape)   # anchor (row i=y, col j=x)
        k = optics.k
        self.pad = (shape[0] + k - 1, shape[1] + k - 1)
        self.r = k // 2

    def _conv(self, K, X):
        """sum over leading axis of K (*) X, zero boundary; K (..., k, k), X (..., H, W)."""
        f = np.fft.irfft2(np.fft.rfft2(K, s=self.pad) * np.fft.rfft2(X, s=self.pad), s=self.pad)
        return f[..., self.r:self.r + self.shape[0], self.r:self.r + self.shape[1]]

    def _corr(self, K, Y):
        Yp = np.zeros(Y.shape[:-2] + self.pad)
        Yp[..., self.r:self.r + self.shape[0], self.r:self.r + self.shape[1]] = Y
        f = np.fft.irfft2(np.conj(np.fft.rfft2(K, s=self.pad)) * np.fft.rfft2(Yp, s=self.pad), s=self.pad)
        return np.roll(f, (self.r, self.r), axis=(-2, -1))[..., self.r:self.r + self.shape[0],
                                                              self.r:self.r + self.shape[1]]

    def anchor_psfs(self, A):
        return self.o.psf(A)                                                # (na, k, k)

    def forward(self, h, x):
        return self._conv(h, self.W * x).sum(0)

    def adjoint(self, h, y):
        return (self.W * self._corr(h, y[None])).sum(0)


def rl_multi(op, hs, bs, x0, iters, eps=1e-9):
    """Multi-image RL: x <- x * sum_p H_p^T(b_p / H_p x) / sum_p H_p^T 1."""
    norm = sum(op.adjoint(h, np.ones(op.shape)) for h in hs)
    x = x0.copy()
    for _ in range(iters):
        x = x * sum(op.adjoint(h, b / np.maximum(op.forward(h, x), eps)) for h, b in zip(hs, bs))
        x = np.clip(x / np.maximum(norm, eps), eps, None)
    return x


def planes_A(op, basis, theta, planes):
    A = basis.coeffs(op.anchors, theta)
    i4 = op.o.index(4)
    out = []
    for dz in planes:
        Ap = A.copy()
        Ap[:, i4] += dz
        out.append(Ap)
    return out


def fit_theta(op, basis, x, bs, planes, theta0, bg, nfev=30):
    D = basis.design(op.anchors)                                   # (na, J, p)
    sb = [np.sqrt(b + 3 / 8) for b in bs]
    Wx = op.W * x                                                  # (na, H, W)

    def model(theta, want_jac):
        rs, js = [], []
        for Ap, s in zip(planes_A(op, basis, theta, planes), sb):
            if want_jac:
                h, dh = op.o.psf_and_jac(Ap)                       # (na,k,k), (na,J,k,k)
            else:
                h = op.o.psf(Ap)
            m = op.forward(h, x) + bg
            sq = np.sqrt(np.maximum(m, 0) + 3 / 8)
            rs.append((sq - s).ravel())
            if want_jac:
                G = op._conv(dh, Wx[:, None])                      # (na, J, H, W)
                Jm = np.einsum("ajhw,ajq->qhw", G, D) / (2 * sq)
                js.append(Jm.reshape(basis.p, -1).T)
        return np.concatenate(rs), (np.concatenate(js) if want_jac else None)

    res = least_squares(lambda t: model(t, False)[0], theta0, jac=lambda t: model(t, True)[1],
                        method="trf", x_scale="jac", max_nfev=nfev)
    return res.x


def blind(op, basis, bs, planes, bg, outer=8, rl_iters=30, seed=0):
    theta = np.zeros(basis.p)
    x = np.full(op.shape, max(np.mean(bs[0]) - bg, 1e-3))
    hist = []
    for it in range(outer):
        hs = [op.anchor_psfs(A) for A in planes_A(op, basis, theta, planes)]
        x = rl_multi(op, hs, [b - bg for b in bs], x, rl_iters)
        theta = fit_theta(op, basis, x, bs, planes, theta, bg)
        hist.append(theta.copy())
    hs = [op.anchor_psfs(A) for A in planes_A(op, basis, theta, planes)]
    x = rl_multi(op, hs, [b - bg for b in bs], x, 2 * rl_iters)
    return x, theta, hist


def field_error(o, basis_or_afun, theta, afun_true):
    t = np.linspace(-1, 1, 16)
    Y = np.stack(np.meshgrid(t, t), -1).reshape(-1, 2)
    T = o.psf(afun_true(Y))
    P = o.psf(basis_or_afun(Y, theta)) if theta is not None else basis_or_afun(Y)
    return float(np.linalg.norm(P - T) / np.linalg.norm(T))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--photons", type=float, default=2000.0)
    ap.add_argument("--bg", type=float, default=2.0)
    ap.add_argument("--field", default="decentred:1")
    ap.add_argument("--outer", type=int, default=8)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    o = pf.Optics()
    kind, level = a.field.split(":")
    afun = pf.truth_field(o, kind, float(level), seed=100)
    cells = skdata.cells3d()
    obj = cells[30, 0, 64:64 + a.size, 64:64 + a.size].astype(float)
    obj = np.clip((obj - np.percentile(obj, 1)) / np.ptp(obj), 0, None)
    obj /= obj.max()
    shape = obj.shape
    op = AnchorOperator(o, shape)
    rng = np.random.default_rng(0)
    div_planes = (-1.0, 0.0, 1.0)

    # data: exact per-pixel field (sparse H, one PSF per pixel), not the anchor
    # operator used for reconstruction (avoids an "inverse crime")
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    Apix = afun(np.column_stack([xx.ravel(), yy.ravel()]))

    def render(planes):
        out = []
        for dz in planes:
            Ap = Apix.copy()
            Ap[:, o.index(4)] += dz
            H = J.forward_matrix(o.psf(Ap), shape)
            out.append(rng.poisson(J.blur(obj * a.photons, H) + a.bg).astype(float))
        return out

    b3 = render(div_planes)
    b1 = [b3[1]]
    rows = []

    def record(name, x, field_err, secs, extra=None):
        xe = x / a.photons
        rows.append(dict(method=name, psnr=float(peak_signal_noise_ratio(obj, np.clip(xe, 0, 1), data_range=1)),
                         ssim=float(structural_similarity(obj, xe, data_range=1)),
                         psf_rel_l2=field_err, seconds=secs, **(extra or {})))
        return xe

    imgs = {"truth": obj, "in-focus data": (b3[1] - a.bg) / a.photons}
    # oracle
    t0 = time.time()
    hs = [o.psf(A) for A in planes_A(op, pf.FieldBasis("seidel5", o), np.zeros(5), div_planes)]
    A = afun(op.anchors)
    hs_true = []
    for dz in div_planes:
        Ap = A.copy(); Ap[:, o.index(4)] += dz
        hs_true.append(o.psf(Ap))
    x = rl_multi(op, hs_true, [b - a.bg for b in b3], np.full(shape, b3[1].mean()), 300)
    imgs["oracle"] = record("oracle field (3 planes)", x, 0.0, time.time() - t0)
    # diffraction-limited invariant PSF
    t0 = time.time()
    x = rl_multi(op, hs, [b - a.bg for b in b3], np.full(shape, b3[1].mean()), 300)
    record("diffraction-limited PSF (3 planes)", x,
           field_error(o, lambda Y: o.psf(np.zeros((len(Y), o.J))), None, afun), time.time() - t0)
    # bead-calibrated hybrid (32 beads), same data
    t0 = time.time()
    beads = pf.simulate_beads(o, afun, 32, np.random.default_rng(5))
    basis_q = pf.FieldBasis("zpoly2", o, kind="zpoly", deg=2)
    theta_b = pf.fit_parametric(beads, o, basis_q)
    hs_b = [o.psf(Ap) for Ap in planes_A(op, basis_q, theta_b, div_planes)]
    x = rl_multi(op, hs_b, [b - a.bg for b in b3], np.full(shape, b3[1].mean()), 300)
    record("bead-calibrated Zquad-31 (32 beads, 3 planes)", x,
           field_error(o, basis_q.coeffs, theta_b, afun), time.time() - t0)
    # blind
    for bname, basis in (("Seidel-5", pf.FieldBasis("seidel5", o)), ("Zquad-31", basis_q)):
        for label, bs, planes in (("3 planes (diversity)", b3, div_planes), ("in-focus only", b1, (0.0,))):
            t0 = time.time()
            x, theta, _ = blind(op, basis, bs, planes, a.bg, outer=a.outer)
            name = f"BLIND {bname}, {label}"
            xe = record(name, x, field_error(o, basis.coeffs, theta, afun), time.time() - t0,
                        {"theta": np.round(theta, 3).tolist() if basis.p <= 5 else ""})
            if bname == "Zquad-31":
                imgs[name] = xe
            print(f"  {name}: {rows[-1]['psnr']:.2f} dB, field err {rows[-1]['psf_rel_l2']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3.csv", index=False)
    txt = [f"# E3 summary: blind spatially varying deconvolution ({a.field} field, {a.size}² cells3d crop, "
           f"{a.photons:g} peak photons, bg {a.bg:g})\n",
           df[["method", "psnr", "ssim", "psf_rel_l2", "seconds"]].round(4).to_markdown(index=False)]
    if kind == "centred":
        true5 = (float(level) * np.array([0.10, 0.25, 0.20, 0.15, 0.08])).round(3).tolist()
        txt.append(f"\nTrue Seidel-5 parameters: {true5}. Seidel-5 fitted parameters are in results/e3.csv.")
    (OUT / "e3_summary.md").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(imgs), figsize=(2.6 * len(imgs), 2.9))
    for ax, (n, im) in zip(axes, imgs.items()):
        ax.imshow(im, cmap="magma", vmin=0, vmax=1)
        ax.set_title(n, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "e3.png", dpi=130)


if __name__ == "__main__":
    main()
