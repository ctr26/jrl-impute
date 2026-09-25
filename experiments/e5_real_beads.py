#!/usr/bin/env python3
"""E5: real field-dependent PSFs: held-out bead prediction on public widefield bead data.

Data: localize-psf bead_data.zarr.zip (Brown et al., QI2lab; Zenodo 10.5281/zenodo.10022862,
CC-BY-4.0): widefield, NA 1.3, 65 nm pixels, 18 z-planes at 0.25 um, 2048^2 FOV
(133 um), blue channel (lambda ~ 0.515 um assumed, n = 1.51).

Protocol
* Detect isolated, unsaturated beads; cut 21 x 21 windows at planes z0-2, z0, z0+2
  (z0 = the modal best-focus plane; +-0.5 um diversity).
* 200 beads held out; methods are trained on n random others (3 repeats).
* Target: the in-focus-plane (z0) PSF of each held-out bead. The prediction is
  aligned to the data with an optimal amplitude, offset and sub-pixel shift
  (identical for every method); metric = relative L2 residual.
* Non-parametric methods use the registered, flux-normalised z0 windows.
  Physics models fit all three planes with per-bead flux, background and tilt
  (block-coordinate Gauss-Newton), a scalar pupil with NA/lambda sampling, and
  stage defocus projected onto Zernikes.

Also reports the measured focus field (best-focus plane vs position).

    uv run experiments/e5_real_beads.py

Download (84 MB) first into data/localize_psf/:
    https://zenodo.org/records/10022862/files/bead_data.zarr.zip
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import fourier_shift
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

sys.path.insert(0, str(Path(__file__).parent))
import psf_fields as pf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "localize_psf" / "bead_data.zarr.zip"
OUT = ROOT / "results"
warnings.filterwarnings("ignore")

NA, LAM, N_IMM, DX, DZ = 1.3, 0.515, 1.51, 0.065, 0.2496
K = 21


# ------------------------------------------------------------------ data ---

def load_stack():
    store = zarr.storage.ZipStore(str(DATA), mode="r")
    g = zarr.open_group(store, mode="r", path="bead_data.zarr")
    return np.asarray(g["cam1/widefield_blue"][0, 0, :, 0, 0]).astype(float)


def find_beads(st):
    bg = np.median(st)
    mp = st.max(0)
    pk = peak_local_max(mp, min_distance=12, threshold_abs=bg + 1500, exclude_border=K)
    d, _ = cKDTree(pk).query(pk, k=2)
    ok = (d[:, 1] > 24) & (mp[pk[:, 0], pk[:, 1]] < 38000)
    return pk[ok], bg


def focus_field(st, P):
    prof = np.stack([st[:, y - 1:y + 2, x - 1:x + 2].sum((1, 2)) for y, x in P])
    zf = []
    for p in prof:
        i = int(np.clip(p.argmax(), 1, len(p) - 2))
        y0, y1, y2 = p[i - 1:i + 2]
        zf.append(i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    zf = np.array(zf)
    Y = P[:, ::-1] / (st.shape[-1] / 2) - 1                     # (x, y) in [-1, 1]
    A = np.c_[np.ones(len(Y)), Y[:, 0], Y[:, 1], (Y**2).sum(1)]
    c, *_ = np.linalg.lstsq(A, zf, rcond=None)
    return zf, c, float(np.std(zf - A @ c))


def windows(st, P, planes):
    r = K // 2
    return np.stack([st[planes, y - r:y + r + 1, x - r:x + r + 1] for y, x in P])  # (nb, P, K, K)


def border_bg(w):
    m = np.ones((K, K), bool)
    m[2:-2, 2:-2] = False
    return np.median(w[..., m], axis=-1)


def register(img, ref=None):
    """Sub-pixel centroid shift to the window centre (Fourier shift)."""
    ref = img if ref is None else ref
    t = np.clip(ref - 0.2 * ref.max(), 0, None)
    yy, xx = np.mgrid[:K, :K]
    cy, cx = (t * yy).sum() / t.sum(), (t * xx).sum() / t.sum()
    s = (K // 2 - cy, K // 2 - cx)
    return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(img), s))), s


# ---------------------------------------------------------------- metric ---

def aligned_error(pred, data):
    """min over (amplitude, offset, sub-pixel shift) of ||a*shift(pred)+c - data|| / ||data - c*||."""
    Pf = np.fft.fft2(pred)

    def fit(s):
        sh = np.real(np.fft.ifft2(fourier_shift(Pf, s)))
        A = np.c_[sh.ravel(), np.ones(K * K)]
        coef, *_ = np.linalg.lstsq(A, data.ravel(), rcond=None)
        return np.sum((A @ coef - data.ravel()) ** 2), coef

    res = minimize(lambda s: fit(s)[0], x0=[0.0, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-9))
    err, coef = fit(res.x)
    return float(np.sqrt(err) / np.linalg.norm(data - coef[1]))


# ---------------------------------------------------------- physics model ---

def optics_for_data():
    o = pf.Optics(n=64, radius=NA / LAM * 64 * DX, jmax=22, k=K, first=2)   # keep tip/tilt for per-bead shifts
    # stage defocus (per um): W = dz (n - sqrt(n^2 - NA^2 rho^2)), projected on the modes
    yy, xx = (np.mgrid[:64, :64] - 32) / o.radius
    rho2 = np.clip(xx**2 + yy**2, 0, 1)
    W = (N_IMM - np.sqrt(N_IMM**2 - NA**2 * rho2)) * 2 * np.pi / LAM
    inside = o.P > 0
    Wc = W - W[inside].mean()
    v = np.array([(Wc * Z)[inside].mean() for Z in o.Z])       # rad per um, per mode
    return o, v


class RealFit:
    """Global field theta (FieldBasis) + per-bead flux, background, tilt; block-coordinate GN."""

    def __init__(self, o, v, basis, Y, D, dz):
        self.o, self.v, self.basis, self.Y, self.D, self.dz = o, v, basis, Y, D, np.asarray(dz)
        self.Dd = basis.design(Y)
        nb = len(Y)
        self.theta = np.zeros(basis.p)
        self.flux = D[:, len(dz) // 2].sum((1, 2)) - border_bg(D[:, len(dz) // 2]) * K * K
        self.bg = border_bg(D).mean(1)
        self.tilt = np.zeros((nb, 2))
        self.sqrtD = np.sqrt(np.maximum(D, 0) + 3 / 8)

    def coeffs(self, theta=None, tilt=None, Y=None):
        theta = self.theta if theta is None else theta
        A = (self.Dd if Y is None else self.basis.design(Y)) @ theta
        if tilt is not None:
            A[:, 0] += tilt[:, 0]                                 # modes start at Noll 2 (tilts)
            A[:, 1] += tilt[:, 1]
        return A

    def planes(self, A):
        return [A + dz * self.v for dz in self.dz]

    def model(self, theta, tilt, jac_theta=False, jac_tilt=False):
        hs, dhs = [], []
        for Ap in self.planes(self.coeffs(theta, tilt)):
            if jac_theta or jac_tilt:
                h, dh = self.o.psf_and_jac(Ap)
                dhs.append(dh)
            else:
                h = self.o.psf(Ap)
            hs.append(h)
        h = np.stack(hs, 1)                                       # (nb, P, K, K)
        return h, (np.stack(dhs, 1) if dhs else None)

    def resid(self, h):
        m = self.flux[:, None, None, None] * h + self.bg[:, None, None, None]
        return np.sqrt(np.maximum(m, 0) + 3 / 8) - self.sqrtD, m

    def step_theta(self, nfev=8):
        def r(t):
            h, _ = self.model(t, self.tilt)
            return self.resid(h)[0].ravel()

        def j(t):
            h, dh = self.model(t, self.tilt, jac_theta=True)
            _, m = self.resid(h)
            s = np.sqrt(np.maximum(m, 0) + 3 / 8)
            dr = self.flux[:, None, None, None, None] * dh / (2 * s[:, :, None])
            return np.einsum("npjxy,njq->npxyq", dr, self.Dd).reshape(-1, self.basis.p)

        self.theta = least_squares(r, self.theta, jac=j, method="trf", x_scale="jac",
                                   max_nfev=nfev).x

    def step_nuisance(self, iters=3):
        """Batched Gauss-Newton on (flux, bg, tilt_x, tilt_y) per bead."""
        for _ in range(iters):
            h, dh = self.model(self.theta, self.tilt, jac_tilt=True)
            res, m = self.resid(h)
            s = np.sqrt(np.maximum(m, 0) + 3 / 8)
            nb = len(self.Y)
            Jf = h / (2 * s)
            Jb = 1 / (2 * s)
            Jt = self.flux[:, None, None, None, None] * dh[:, :, :2] / (2 * s[:, :, None])
            Jm = np.concatenate([Jf[:, None], Jb[:, None], np.moveaxis(Jt, 2, 1)], 1)  # (nb, 4, P, K, K)
            Jm = Jm.reshape(nb, 4, -1)
            g = np.einsum("nqr,nr->nq", Jm, res.reshape(nb, -1))
            Hm = np.einsum("nqr,npr->nqp", Jm, Jm) + 1e-9 * np.eye(4)
            d = np.linalg.solve(Hm, g[..., None])[..., 0]
            self.flux = np.maximum(self.flux - d[:, 0], 1.0)
            self.bg = self.bg - d[:, 1]
            self.tilt = self.tilt - d[:, 2:]

    def fit(self, outer=6):
        self.step_nuisance()
        for _ in range(outer):
            self.step_theta()
            self.step_nuisance()
        return self

    def predict(self, Yq):
        A = self.basis.design(Yq) @ self.theta
        return self.o.psf(A)                                      # z0 plane, no tilt


def fit_with_blur(o, v, basis, Yb, Wb, sigmas=(0.9, 1.1, 1.3)):
    """Profile the global blur sigma (px) on the TRAINING beads' cost, then refine.
    The defocus direction is not identifiable from symmetric planes (PROOFS.md T6);
    it does not affect the in-focus prediction, so +v is used."""
    best = None
    for s in sigmas:
        o.blur_sigma = s
        f = RealFit(o, v, basis, Yb, Wb, dz=[-2 * DZ, 0.0, 2 * DZ]).fit(outer=3)
        h, _ = f.model(f.theta, f.tilt)
        cost = float((f.resid(h)[0] ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, s, f)
    o.blur_sigma = best[1]
    f = best[2]
    for _ in range(3):
        f.step_theta()
        f.step_nuisance()
    f.sigma = best[1]
    return f


# ------------------------------------------------------------------ main ---

def main():
    t0 = time.time()
    st = load_stack()
    P, bg0 = find_beads(st)
    zf, fc, fres = focus_field(st, P)
    z0 = int(np.round(np.median(zf)))
    planes = [z0 - 2, z0, z0 + 2]
    W = windows(st, P, planes)
    Y = P[:, ::-1] / (st.shape[-1] / 2) - 1.0                    # (x, y) in [-1, 1]
    print(f"{len(P)} beads; focus field planes = {fc.round(3)} (const, x, y, r^2), "
          f"resid {fres:.3f} planes ({fres * DZ * 1000:.0f} nm); z0={z0}; {time.time() - t0:.1f}s")

    # registered, flux-normalised in-focus windows for non-parametric methods
    Wf = W[:, 1] - border_bg(W[:, 1])[:, None, None]
    reg = np.stack([register(w)[0] for w in Wf])
    reg /= reg.sum((1, 2), keepdims=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(P))
    test, pool = perm[:200], perm[200:]
    o, v = optics_for_data()
    rows = []
    np_methods = {k: pf.ESTIMATORS[k] for k in
                  ("mean (invariant)", "nearest bead", "IDW kNN", "poly pixels (PSFEx)",
                   "PCA+poly (Jee07)", "PCA+RBF (Gentile13)", "PCA+GP (PIFF-like)")}

    class Beads:
        def __init__(self, idx):
            self.Y, self._p = Y[idx], reg[idx]

        def infocus_psf(self):
            return self._p

    class Grid:
        k = K

    for n in (8, 16, 32, 64, 128, 327):
        for rep in range(3 if n < 327 else 1):
            tr = np.random.default_rng(rep).choice(pool, n, replace=False)
            b = Beads(tr)
            preds = {}
            for name, est in np_methods.items():
                t1 = time.time()
                preds[name] = (est(b, Grid)(Y[test]), time.time() - t1)
            if n <= 128:
                for bname, basis in (
                        ("Zernike field (quadratic) + const higher modes",
                         pf.FieldBasis("zq", o, kind="zpoly", zmodes=(4, 5, 6, 7, 8), deg=2,
                                       const_modes=tuple(range(9, 23)))),):
                    t1 = time.time()
                    f = fit_with_blur(o, v, basis, Y[tr], W[tr])
                    Pm = f.predict(Y[test])
                    preds[f"physics: {bname}"] = (Pm, time.time() - t1)
                    print(f"    physics fit n={n}: blur sigma={f.sigma} px", flush=True)
                    # hybrid: registered model + GP on registered residual
                    Mtr = np.stack([register(p)[0] for p in f.predict(Y[tr])])
                    Mtr /= Mtr.sum((1, 2), keepdims=True)
                    fres_ = pf.gp_residual(Y[tr], reg[tr] - Mtr)
                    Mte = np.stack([register(p)[0] for p in Pm])
                    Mte /= Mte.sum((1, 2), keepdims=True)
                    preds["hybrid: physics + GP residual"] = (Mte + fres_(Y[test]), time.time() - t1)
            for name, (Pr, secs) in preds.items():
                errs = [aligned_error(Pr[i], Wf[j]) for i, j in enumerate(test)]
                rows.append(dict(train_beads=n, rep=rep, method=name,
                                 rel_err=float(np.mean(errs)), rel_err_median=float(np.median(errs)),
                                 seconds=secs))
            print(f"n={n} rep={rep}: " + ", ".join(f"{r['method'][:18]}={r['rel_err']:.4f}"
                                                   for r in rows[-len(preds):]), flush=True)

    # noise floor: two independent in-focus-adjacent planes are not replicates; use a
    # Poisson estimate instead (counts in ADU ~ photons x gain; gain unknown -> upper bound)
    floor = float(np.mean([np.sqrt(np.clip(Wf[j], 0, None).sum() + bg0 * K * K) / np.linalg.norm(Wf[j])
                           for j in test]))
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "e5.csv", index=False)
    g = df.groupby(["method", "train_beads"]).rel_err
    tab = (g.mean().map("{:.4f}".format) + " ± " + g.std().fillna(0).map("{:.4f}".format)).unstack()
    order = list(dict.fromkeys(df.method))
    txt = ["# E5 summary: held-out real bead PSFs (localize-psf widefield, NA 1.3)\n",
           f"{len(P)} isolated beads; 200 held out. Metric: mean relative L2 residual of the z0 "
           "PSF after optimal amplitude/offset/sub-pixel-shift alignment (lower is better).\n",
           f"Measured focus field (best-focus plane index): {fc[0]:.2f} + {fc[1]:.3f} x + {fc[2]:.3f} y "
           f"+ {fc[3]:.3f} r²; residual {fres:.3f} planes = {fres * DZ * 1000:.0f} nm "
           "(x, y in [-1, 1] over the 133 µm FOV).\n",
           f"Shot-noise floor (Poisson, gain 1 ADU/photon; indicative): ≈ {floor:.3f}\n",
           tab.loc[order].to_markdown()]
    (OUT / "e5_summary.md").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
