#!/usr/bin/env python3
"""Numerical checks of the PSF-field results in docs/PROOFS.md.

Self-contained (numpy only; matplotlib optional for --plot). Run:

    python proofs/psf_field_proofs.py            # prints a report, asserts every bound
    python proofs/psf_field_proofs.py --plot f.png

Model
-----
Scalar pupil model. Field position y in [-1, 1]^2, Zernike coefficients a(y)
with Hopkins/Seidel-style field dependence (polynomial in y), phase
phi_y = sum_j a_j(y) Z_j, amplitude u_y = DFT_unitary(P exp(i phi_y)),
PSF h_y = |u_y|^2 / ||P||^2 (unit mass by Parseval).

Claims checked (numbering as in docs/PROOFS.md)
------------------------------------------------
T1  ||h_y - h_y'||_1 <= 2 RMS_P(phi_y - phi_y')  (= 2 ||a(y) - a(y')||_2 for orthonormal Z)
T2  for ANY fixed orthonormal basis {e_i}:  |c_i(y) - c_i(y')| <= ||h_y - h_y'||_2
    and sum_i |c_i(y) - c_i(y')|^2 = ||h_y - h_y'||_2^2              (Parseval)
T3  column-varying operator with periodic boundary:
      ||H - H~||_F^2 = sum_v ||h_v - h~_v||_2^2,  ||H - H~||_{1->1} = max_v ||h_v - h~_v||_1
    and rank-r truncated SVD of the PSF field is the Frobenius-optimal
    product-convolution model (Eckart-Young), error = sum_{i>r} sigma_i^2
T4  product-convolution form: H x = sum_i e_i (*) (c_i . x),  H^T y = sum_i c_i . (e_i (star) y)
T5  nearest-bead interpolation: ||h_y - h_nn(y)||_1 <= 2 L_a |y - nn(y)|,
    L_a = sup_y ||Da(y)||_op ; any convex-combination interpolator obeys the same
    bound with |y - nn| replaced by the largest neighbour distance used.
T6  identifiability: in focus, the "twin" phase -phi(-rho) (even Zernikes negated)
    gives the identical PSF, so no in-focus bead fit or blind method can tell them
    apart; a known defocus diversity breaks the tie, and a 5-parameter
    Hopkins-field fit from a handful of noisy beads then beats non-parametric
    interpolation that uses many more beads.
"""

from __future__ import annotations

import argparse
from math import factorial

import numpy as np

RNG = np.random.default_rng(0)
TOL = 1e-10


# ------------------------------------------------------------------ Zernike ---

def noll_to_nm(j: int) -> tuple[int, int]:
    """Noll index (1-based) -> radial order n, signed azimuthal m."""
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike (unit RMS over the unit disc)."""
    n, m = noll_to_nm(j)
    am = abs(m)
    R = sum((-1) ** s * factorial(n - s)
            / (factorial(s) * factorial((n + am) // 2 - s) * factorial((n - am) // 2 - s))
            * rho ** (n - 2 * s) for s in range((n - am) // 2 + 1))
    if m == 0:
        return np.sqrt(n + 1) * R
    ang = np.cos(am * theta) if m > 0 else np.sin(am * theta)
    return np.sqrt(2 * (n + 1)) * R * ang


# ------------------------------------------------------------ Pupil model ---

class PupilModel:
    """Scalar pupil -> PSF, with Hopkins-style field-dependent Zernike coefficients."""

    # Noll indices used: defocus 4, astig 5/6, coma 7/8, spherical 11
    J = (4, 5, 6, 7, 8, 11)

    def __init__(self, n: int = 64, radius: float = 16.0,
                 defocus=0.3, curvature=1.0, astig=1.2, coma=0.8, spherical=0.4,
                 curvature4=0.0):
        yy, xx = (np.mgrid[:n, :n] - n // 2) / radius
        rho, theta = np.hypot(xx, yy), np.arctan2(yy, xx)
        self.P = (rho <= 1.0).astype(float)
        self.Z = np.stack([zernike(j, rho, theta) * self.P for j in self.J])
        # re-orthonormalise on the discrete pupil so RMS(sum a_j Z_j) == ||a||_2 exactly
        Zf = self.Z.reshape(len(self.J), -1)[:, self.P.ravel() > 0]
        L = np.linalg.cholesky(Zf @ Zf.T / Zf.shape[1])
        self.Z = np.einsum("ij,jxy->ixy", np.linalg.inv(L), self.Z)
        self.p = dict(defocus=defocus, curvature=curvature, astig=astig,
                      coma=coma, spherical=spherical, curvature4=curvature4)
        self.n = n

    def coeffs(self, y: np.ndarray) -> np.ndarray:
        """a(y) for y of shape (..., 2). Third-order (Seidel) field dependence:
        defocus ~ const + |y|^2, astig ~ |y|^2 at 2*psi, coma ~ |y| at psi, spherical const."""
        y = np.asarray(y, float)
        yx, yy = y[..., 0], y[..., 1]
        r2 = yx**2 + yy**2
        p = self.p
        return np.stack([
            p["defocus"] + p["curvature"] * r2 + p.get("curvature4", 0.0) * r2**2,  # Z4
            p["astig"] * 2 * yx * yy,                    # Z5  |y|^2 sin 2psi
            p["astig"] * (yx**2 - yy**2),                # Z6  |y|^2 cos 2psi
            p["coma"] * yy,                              # Z7  |y| sin psi
            p["coma"] * yx,                              # Z8  |y| cos psi
            np.full_like(yx, p["spherical"]),            # Z11
        ], axis=-1)

    def jacobian_norm_sup(self, grid: int = 201) -> float:
        """sup_y ||Da(y)||_op over [-1,1]^2 (analytic Jacobian on a fine grid)."""
        t = np.linspace(-1, 1, grid)
        yx, yy = np.meshgrid(t, t)
        p = self.p
        z = np.zeros_like(yx)
        J = np.stack([
            np.stack([(2 * p["curvature"] + 4 * p["curvature4"] * (yx**2 + yy**2)) * yx,
                      (2 * p["curvature"] + 4 * p["curvature4"] * (yx**2 + yy**2)) * yy], -1),
            np.stack([2 * p["astig"] * yy, 2 * p["astig"] * yx], -1),
            np.stack([2 * p["astig"] * yx, -2 * p["astig"] * yy], -1),
            np.stack([z, z + p["coma"]], -1),
            np.stack([z + p["coma"], z], -1),
            np.stack([z, z], -1),
        ], axis=-2)                                     # (g, g, 6, 2)
        return float(np.linalg.norm(J, ord=2, axis=(-2, -1)).max())

    def phase(self, a: np.ndarray) -> np.ndarray:
        return np.tensordot(a, self.Z, axes=(-1, 0))

    def psf(self, a: np.ndarray) -> np.ndarray:
        """Full-grid, unit-mass PSF(s) for coefficient vector(s) a (..., len(J))."""
        field = self.P * np.exp(1j * self.phase(a))
        u = np.fft.fftshift(np.fft.fft2(field, norm="ortho"), axes=(-2, -1))
        return np.abs(u) ** 2 / (self.P**2).sum()

    def rms(self, phi: np.ndarray) -> np.ndarray:
        w = self.P**2 / (self.P**2).sum()
        return np.sqrt((w * phi**2).sum(axis=(-2, -1)))


def crop(h: np.ndarray, k: int) -> np.ndarray:
    c, r = h.shape[-1] // 2, k // 2
    return h[..., c - r:c + r + 1, c - r:c + r + 1]


# --------------------------------------------------------- Operator helpers ---

def periodic_column_operator(psfs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Dense H (N x N): column v = psfs[v] centred on v with periodic wrap."""
    h, w = shape
    N, k, _ = psfs.shape
    r = k // 2
    H = np.zeros((N, N))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    for v in range(N):
        vy, vx = divmod(v, w)
        rows = ((vy + dy) % h) * w + (vx + dx) % w
        np.add.at(H[:, v], rows.ravel(), psfs[v].ravel())
    return H


def kernel_to_image(e: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place a centred k x k kernel at the origin of a periodic image (for FFT conv)."""
    k = e.shape[-1]
    out = np.zeros(e.shape[:-2] + shape)
    out[..., :k, :k] = e
    return np.roll(out, (-(k // 2), -(k // 2)), axis=(-2, -1))


def pc_apply(E: np.ndarray, C: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Product-convolution H x = sum_i e_i (*) (c_i . x) (periodic, via FFT)."""
    Ef = np.fft.fft2(kernel_to_image(E, x.shape))
    return np.real(np.fft.ifft2((Ef * np.fft.fft2(C * x)).sum(0)))


def pc_adjoint(E: np.ndarray, C: np.ndarray, y: np.ndarray) -> np.ndarray:
    """H^T y = sum_i c_i . (e_i (star) y)  (correlation = conjugate in Fourier)."""
    Ef = np.fft.fft2(kernel_to_image(E, y.shape))
    return (C * np.real(np.fft.ifft2(np.conj(Ef) * np.fft.fft2(y)))).sum(0)


# ------------------------------------------------------------------ Checks ---

def check_T1(model: PupilModel, n_pairs: int = 400) -> dict:
    y1 = RNG.uniform(-1, 1, (n_pairs, 2))
    y2 = np.clip(y1 + RNG.normal(0, 0.3, (n_pairs, 2)), -1, 1)
    a1, a2 = model.coeffs(y1), model.coeffs(y2)
    dh = np.abs(model.psf(a1) - model.psf(a2)).sum(axis=(-2, -1))
    rms = model.rms(model.phase(a1) - model.phase(a2))
    da = np.linalg.norm(a1 - a2, axis=-1)
    assert np.all(dh <= 2 * rms + TOL), "T1 violated"
    assert np.allclose(rms, da, rtol=1e-8), "discrete Zernikes not orthonormal"
    return {"max ||dh||_1 / (2 RMS dphi)": float((dh / (2 * rms)).max()),
            "median ratio": float(np.median(dh / (2 * rms)))}


def field_samples(model: PupilModel, g: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(-1, 1, g)
    Y = np.stack(np.meshgrid(t, t, indexing="xy"), -1).reshape(-1, 2)
    return Y, crop(model.psf(model.coeffs(Y)), k)


def check_T2(model: PupilModel, k: int = 21) -> dict:
    Y, Hs = field_samples(model, 24, k)
    M = Hs.reshape(len(Y), -1)
    U, s, Vt = np.linalg.svd(M, full_matrices=True)       # full orthonormal basis Vt
    bases = {"PCA/SVD": Vt, "random orthonormal": np.linalg.qr(RNG.normal(size=(k * k,) * 2))[0].T,
             "pixel (identity)": np.eye(k * k)}
    i, j = RNG.integers(0, len(Y), (2, 500))
    dh = M[i] - M[j]
    out = {}
    for name, B in bases.items():
        dc = dh @ B.T
        assert np.all(np.abs(dc) <= np.linalg.norm(dh, axis=1)[:, None] + TOL), f"T2 bound {name}"
        assert np.allclose((dc**2).sum(1), (dh**2).sum(1)), f"T2 Parseval {name}"
        out[name] = "ok"
    return out


def check_T3_T4(model: PupilModel, g: int = 20, k: int = 15, ranks=(1, 2, 3, 5, 8)) -> dict:
    shape = (g, g)
    Y, Hs = field_samples(model, g, k)                    # one PSF per pixel
    N = g * g
    M = Hs.reshape(N, -1)
    H = periodic_column_operator(Hs, shape)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    out = {}
    x = RNG.random(shape)
    ytest = RNG.random(shape)
    # a competing basis: Jacobian of h wrt Zernike coeffs at the field centre ("physics" basis)
    a0 = model.coeffs(np.zeros(2))
    eps = 1e-4
    Jb = [crop(model.psf(a0), k).ravel()]
    for jj in range(len(model.J)):
        da = np.zeros_like(a0); da[jj] = eps
        Jb.append(((crop(model.psf(a0 + da), k) - crop(model.psf(a0 - da), k)) / (2 * eps)).ravel())
    Qj = np.linalg.qr(np.array(Jb).T)[0].T                 # orthonormalised physics basis
    for r in ranks:
        E = Vt[:r]
        C = M @ E.T                                         # (N, r) optimal coefficients
        Mr = C @ E
        Hr = periodic_column_operator(Mr.reshape(N, k, k), shape)
        fro2 = np.linalg.norm(H - Hr) ** 2
        col2 = ((M - Mr) ** 2).sum()
        assert np.isclose(fro2, col2), "T3 HS identity"
        assert np.isclose(col2, (s[r:] ** 2).sum()), "T3 Eckart-Young tail"
        one = np.abs(H - Hr).sum(0).max()
        assert np.isclose(one, np.abs(M - Mr).sum(1).max()), "T3 1->1 norm"
        # T4: product-convolution FFT form == dense operator
        Cimg = C.T.reshape(r, *shape)
        assert np.allclose(pc_apply(E.reshape(r, k, k), Cimg, x), (Hr @ x.ravel()).reshape(shape))
        assert np.allclose(pc_adjoint(E.reshape(r, k, k), Cimg, ytest),
                           (Hr.T @ ytest.ravel()).reshape(shape))
        rr = min(r, len(Qj))
        Mq = (M @ Qj[:rr].T) @ Qj[:rr]
        out[r] = {"rel HS err PCA": float(np.sqrt(col2) / np.linalg.norm(M)),
                  "rel HS err physics-Jacobian": float(np.linalg.norm(M - Mq) / np.linalg.norm(M)),
                  "max col L1 err PCA": float(one)}
        assert out[r]["rel HS err PCA"] <= out[r]["rel HS err physics-Jacobian"] + 1e-12
    return out


def check_T5(model: PupilModel, k: int = 21, bead_counts=(10, 25, 50, 100, 200)) -> dict:
    Y, Hs = field_samples(model, 32, k)
    full = model.psf(model.coeffs(Y))                       # un-cropped for exact L1 bound
    La = model.jacobian_norm_sup()
    out = {}
    for nb in bead_counts:
        B = RNG.uniform(-1, 1, (nb, 2))
        hb_full = model.psf(model.coeffs(B))
        d = np.linalg.norm(Y[:, None] - B[None], axis=-1)
        nn = d.argmin(1)
        err_nn = np.abs(full - hb_full[nn]).sum(axis=(-2, -1))
        bound = 2 * La * d[np.arange(len(Y)), nn]
        assert np.all(err_nn <= bound + TOL), "T5 NN bound"
        # inverse-distance kNN (convex combination) and polynomial-in-y on PCA coeffs
        kk = min(5, nb)
        idx = np.argsort(d, 1)[:, :kk]
        wts = 1 / np.maximum(np.take_along_axis(d, idx, 1), 1e-12)
        wts /= wts.sum(1, keepdims=True)
        hb = crop(hb_full, k).reshape(nb, -1)
        knn = np.einsum("pk,pkq->pq", wts, hb[idx])
        err_knn_l1 = np.abs(Hs.reshape(len(Y), -1) - knn).sum(1)
        dmax = np.take_along_axis(d, idx, 1).max(1)
        cropped_true = Hs.reshape(len(Y), -1)
        assert np.all(err_knn_l1 <= 2 * La * dmax + TOL), "T5 convex-combination bound"

        def poly(Yp, deg=3):
            return np.stack([Yp[:, 0] ** i * Yp[:, 1] ** j
                             for i in range(deg + 1) for j in range(deg + 1 - i)], 1)
        r = 8
        Vt = np.linalg.svd(hb, full_matrices=False)[2][:r]
        c = hb @ Vt.T
        if nb >= poly(B).shape[1]:
            coef, *_ = np.linalg.lstsq(poly(B), c, rcond=None)
            pc = (poly(Y) @ coef) @ Vt
            e_poly = np.linalg.norm(pc - cropped_true) / np.linalg.norm(cropped_true)
        else:
            e_poly = float("nan")
        rel = lambda A: float(np.linalg.norm(A - cropped_true) / np.linalg.norm(cropped_true))
        out[nb] = {"fill dist": float(d.min(1).max()),
                   "NN rel L2": rel(crop(hb_full[nn], k).reshape(len(Y), -1)),
                   "kNN rel L2": rel(knn),
                   "PCA8+cubic rel L2": float(e_poly),
                   "NN L1 bound tightness": float((err_nn / bound).max())}
    return out


def check_T6(model: PupilModel, n_beads: int = 8, photons: float = 2e4,
             diversity: float = 1.0, k: int = 21, true_model: PupilModel | None = None) -> dict:
    """Fit the 5 third-order Hopkins parameters. If ``true_model`` is given, beads
    and the reference field come from it instead (model misspecification)."""
    from scipy.optimize import least_squares

    keys = ("defocus", "curvature", "astig", "coma", "spherical")
    even = np.array([1, 1, 1, 0, 1], bool)                  # coma is odd in rho
    truth = np.array([model.p[q] for q in keys])

    def with_params(theta):
        m = PupilModel.__new__(PupilModel)
        m.__dict__.update(model.__dict__)
        m.p = dict(zip(keys, theta), curvature4=0.0)
        return m

    gen = true_model or model

    def psfs(theta, Yb, dz):
        a = with_params(theta).coeffs(Yb)
        a = a + np.array([dz, 0, 0, 0, 0, 0])               # known defocus (Noll 4) offset
        return crop(model.psf(a), k)

    # (a) twin ambiguity: exact in focus, broken by diversity
    Yb = RNG.uniform(-1, 1, (n_beads, 2))
    twin = np.where(even, -truth, truth)
    same = np.abs(psfs(truth, Yb, 0) - psfs(twin, Yb, 0)).max()
    diff = np.abs(psfs(truth, Yb, diversity) - psfs(twin, Yb, diversity)).sum(axis=(-2, -1)).mean()
    assert same < 1e-12, "T6 twin ambiguity should be exact in focus"

    # (b) fit the 5 Hopkins parameters from noisy bead z-pairs {-dz, +dz}
    planes = (-diversity, diversity)
    def gen_psfs(Yb, dz):
        a = gen.coeffs(Yb) + np.array([dz, 0, 0, 0, 0, 0])
        return crop(model.psf(a), k)

    data = [RNG.poisson(photons * gen_psfs(Yb, dz)) / photons for dz in planes]

    def resid(theta):
        return np.concatenate([(np.sqrt(np.maximum(psfs(theta, Yb, dz), 0)) - np.sqrt(d)).ravel()
                               for dz, d in zip(planes, data)])

    best = min((least_squares(resid, RNG.normal(0, 0.7, 5)) for _ in range(6)),
               key=lambda r: r.cost)
    Y, Hs = field_samples(gen, 32, k)
    fit_field = crop(model.psf(with_params(best.x).coeffs(Y)), k)
    rel = float(np.linalg.norm(fit_field - Hs) / np.linalg.norm(Hs))
    return {"in-focus twin max|dh|": float(same),
            "diversity twin mean ||dh||_1": float(diff),
            "true params": truth.round(3).tolist(),
            "fitted params": best.x.round(3).tolist(),
            f"field rel L2 from {n_beads} beads": rel}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plot", default=None)
    a = ap.parse_args(argv)
    model = PupilModel()
    print("Zernike (n, m) for Noll j:", {j: noll_to_nm(j) for j in model.J})
    print("\nT1  phase -> PSF L1 Lipschitz:", check_T1(model))
    print("T2  coefficient bound + Parseval:", check_T2(model))
    print("\nT3/T4  rank-r product-convolution (HS identity, Eckart-Young, FFT form):")
    for r, v in check_T3_T4(model).items():
        print(f"  r={r:<2}", {kk: round(vv, 4) for kk, vv in v.items()})
    print(f"\nT5  interpolation (L_a = sup||Da|| = {model.jacobian_norm_sup():.3f}):")
    for nb, v in check_T5(model).items():
        print(f"  beads={nb:<4}", {kk: round(vv, 4) for kk, vv in v.items()})
    print("\nT6  identifiability + parametric (Hopkins) fit with defocus diversity:")
    for kk, vv in check_T6(model).items():
        print(f"  {kk}: {vv}")
    mis = PupilModel(curvature4=0.6)
    print("  -- misspecified: truth adds 0.6 |y|^4 field curvature the 5-param model lacks --")
    for kk, vv in check_T6(model, true_model=mis).items():
        if "twin" not in kk:
            print(f"  {kk}: {vv}")
    print("     (non-parametric kNN on the same misspecified field:",
          {nb: round(v["kNN rel L2"], 3) for nb, v in check_T5(mis, bead_counts=(8, 50, 200)).items()}, ")")
    print("\nall assertions passed")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.linspace(-1, 1, 5)
        fig, ax = plt.subplots(5, 5, figsize=(7, 7))
        for i, yy in enumerate(t[::-1]):
            for j, yx in enumerate(t):
                ax[i, j].imshow(np.sqrt(crop(model.psf(model.coeffs(np.array([yx, yy]))), 21)),
                                cmap="magma")
                ax[i, j].axis("off")
        fig.suptitle("PSF field (sqrt intensity), Hopkins-style aberrations")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("figure ->", a.plot)


if __name__ == "__main__":
    main()
