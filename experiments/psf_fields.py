"""PSF-field models and estimators shared by the experiments.

Self-contained (numpy / scipy / scikit-learn). Contents:

* ``Optics``            scalar pupil model with an analytic Jacobian dh/da
* ``FieldBasis``        maps parameters theta -> Zernike coefficients a(y) (Seidel-5, Zernike-polynomial)
* ``truth_field``       centred / decentred ground-truth aberration fields
* ``simulate_beads``    noisy through-focus bead stacks (Poisson + background)
* ``ESTIMATORS``        PSF-field estimators: literature baselines + physics models + hybrid

Conventions: field positions y in [-1, 1]^2; PSFs are k x k crops of a unit-mass
full-grid PSF (so a crop keeps < 1 of the mass); bead data are photon counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# ----------------------------------------------------------------- Zernikes ---

def noll_to_nm(j: int) -> tuple[int, int]:
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n, m = noll_to_nm(j)
    am = abs(m)
    R = sum((-1) ** s * factorial(n - s)
            / (factorial(s) * factorial((n + am) // 2 - s) * factorial((n - am) // 2 - s))
            * rho ** (n - 2 * s) for s in range((n - am) // 2 + 1))
    if m == 0:
        return np.sqrt(n + 1) * R
    return np.sqrt(2 * (n + 1)) * R * (np.cos(am * theta) if m > 0 else np.sin(am * theta))


# ------------------------------------------------------------------ Optics ---

class Optics:
    """Scalar pupil -> k x k PSF crop, modes Noll j = 4..jmax (piston/tilt removed).

    Zernikes are Gram-Schmidt orthonormalised on the discrete pupil (including
    piston and tilts first), so RMS phase == ||a||_2 exactly.
    """

    def __init__(self, n: int = 64, radius: float = 10.0, jmax: int = 15, k: int = 21,
                 first: int = 4):
        yy, xx = (np.mgrid[:n, :n] - n // 2) / radius
        rho, th = np.hypot(xx, yy), np.arctan2(yy, xx)
        self.P = (rho <= 1).astype(float)
        inside = self.P.ravel() > 0
        Z = np.stack([zernike(j, rho, th).ravel()[inside] for j in range(1, jmax + 1)])
        Q, _ = np.linalg.qr(Z.T)                                 # orthonormal, same span order
        Q *= np.sign(np.sum(Q * Z.T, axis=0))                     # keep sign convention
        Q *= np.sqrt(inside.sum())                                # unit RMS
        full = np.zeros((jmax, n * n))
        full[:, inside] = Q.T
        self.Z = full[first - 1:].reshape(jmax - first + 1, n, n)   # default: drop piston, tip, tilt
        self.modes = list(range(first, jmax + 1))
        self.n, self.k, self.radius = n, k, radius
        self.blur_sigma = 0.0     # extra Gaussian blur (px): bead size, pixel integration, vectorial smoothing
        self._norm = (self.P**2).sum()
        c, r = n // 2, k // 2
        self._sl = (slice(c - r, c + r + 1), slice(c - r, c + r + 1))

    @property
    def J(self) -> int:
        return len(self.modes)

    def index(self, j: int) -> int:
        return self.modes.index(j)

    def _field(self, a):
        return self.P * np.exp(1j * np.tensordot(a, self.Z, axes=(-1, 0)))

    def _fft(self, f):
        return np.fft.fftshift(np.fft.fft2(f, norm="ortho"), axes=(-2, -1))

    def _blur(self, h):
        if not self.blur_sigma:
            return h
        f = np.fft.fftfreq(self.n)
        g = np.exp(-2 * (np.pi * self.blur_sigma) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
        hs = np.fft.ifftshift(h, axes=(-2, -1))
        return np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(hs) * g)), axes=(-2, -1))

    def psf(self, a: np.ndarray, full: bool = False) -> np.ndarray:
        u = self._fft(self._field(a))
        h = self._blur(np.abs(u) ** 2 / self._norm)
        return h if full else h[(..., *self._sl)]

    def psf_and_jac(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """h (..., k, k) and dh/da (..., J, k, k): dh/da_j = 2 Re(conj(u) F[i Z_j P e^{i phi}])."""
        f = self._field(a)
        if self.blur_sigma:
            u = self._fft(f)
            du = self._fft(1j * self.Z * f[..., None, :, :])
            h = self._blur(np.abs(u) ** 2 / self._norm)[(..., *self._sl)]
            dh = self._blur(2 * np.real(np.conj(u)[..., None, :, :] * du) / self._norm)[(..., *self._sl)]
            return h, dh
        u = self._fft(f)[(..., *self._sl)]
        du = self._fft(1j * self.Z * f[..., None, :, :])[(..., *self._sl)]
        h = np.abs(u) ** 2 / self._norm
        dh = 2 * np.real(np.conj(u)[..., None, :, :] * du) / self._norm
        return h, dh


# -------------------------------------------------------------- Field bases ---

def _monomials(Y: np.ndarray, deg: int) -> np.ndarray:
    return np.stack([Y[:, 0] ** i * Y[:, 1] ** j
                     for d in range(deg + 1) for i in range(d, -1, -1) for j in [d - i]], 1)


@dataclass
class FieldBasis:
    """Linear map theta -> a(y): ``a(Y) = design(Y) @ theta``, design (M, J, p)."""
    name: str
    optics: Optics
    kind: str = "seidel5"          # "seidel5" | "zpoly"
    zmodes: tuple = (4, 5, 6, 7, 8)
    deg: int = 2
    const_modes: tuple = (11,)

    @property
    def p(self) -> int:
        if self.kind == "seidel5":
            return 5
        return len(self.zmodes) * _monomials(np.zeros((1, 2)), self.deg).shape[1] + len(self.const_modes)

    def design(self, Y: np.ndarray) -> np.ndarray:
        o = self.optics
        M = len(Y)
        D = np.zeros((M, o.J, self.p))
        if self.kind == "seidel5":
            x, y = Y[:, 0], Y[:, 1]
            i4, i5, i6, i7, i8, i11 = (o.index(j) for j in (4, 5, 6, 7, 8, 11))
            D[:, i4, 0] = 1.0                       # constant defocus
            D[:, i4, 1] = x**2 + y**2               # medial field curvature  ~ H^2
            D[:, i5, 2] = 2 * x * y                 # astigmatism             ~ H^2 sin 2psi
            D[:, i6, 2] = x**2 - y**2               #                         ~ H^2 cos 2psi
            D[:, i7, 3] = y                         # coma                    ~ H sin psi
            D[:, i8, 3] = x                         #                         ~ H cos psi
            D[:, i11, 4] = 1.0                      # spherical               const
            return D
        mon = _monomials(Y, self.deg)
        q = mon.shape[1]
        for t, j in enumerate(self.zmodes):
            D[:, o.index(j), t * q:(t + 1) * q] = mon
        for t, j in enumerate(self.const_modes):
            D[:, o.index(j), len(self.zmodes) * q + t] = 1.0
        return D

    def coeffs(self, Y, theta):
        return self.design(Y) @ theta


# ------------------------------------------------------------ Ground truth ---

def truth_field(optics: Optics, kind: str = "centred", level: float = 1.0,
                seed: int = 0):
    """Return a(y) callable. ``level`` scales every term.

    centred   : pure Seidel field (in the Seidel-5 model class)
    decentred : Seidel + nodal-aberration-theory terms (constant coma, astigmatism
                linear in y), trefoil ~|y|^3, quartic defocus, and a smooth random
                residual on all modes -> outside both parametric model classes
    """
    rng = np.random.default_rng(seed)
    base = FieldBasis("seidel5", optics)
    th = level * np.array([0.10, 0.25, 0.20, 0.15, 0.08])   # rad RMS: defocus0, curv, astig, coma, sph
    idx = optics.index
    freqs = rng.normal(0, 2.0, (6, 2))
    phases = rng.uniform(0, 2 * np.pi, 6)
    amps = rng.normal(0, 1, (6, optics.J)) * 0.02 * level

    def a(Y):
        Y = np.atleast_2d(Y)
        A = base.coeffs(Y, th)
        if kind == "decentred":
            x, y = Y[:, 0], Y[:, 1]
            r2 = x**2 + y**2
            A[:, idx(7)] += level * 0.06                         # constant coma (decentre)
            A[:, idx(8)] += level * -0.04
            A[:, idx(5)] += level * 0.08 * (x + 0.5 * y)          # astigmatism linear in H
            A[:, idx(6)] += level * 0.08 * (y - 0.3 * x)
            A[:, idx(4)] += level * 0.10 * r2**2                  # quartic defocus
            if 9 in optics.modes:
                A[:, idx(9)] += level * 0.06 * r2 * x             # trefoil ~ H^3
                A[:, idx(10)] += level * 0.06 * r2 * y
            A += np.cos(Y @ freqs.T + phases) @ amps              # smooth random residual
        return A

    return a


def aberration_stats(optics, afun):
    t = np.linspace(-1, 1, 41)
    Y = np.stack(np.meshgrid(t, t), -1).reshape(-1, 2)
    rms = np.linalg.norm(afun(Y), axis=1)
    corner = np.linalg.norm(afun(np.array([[1.0, 1.0]]))[0])
    strehl = optics.psf(afun(Y), full=True).max(axis=(-2, -1)) / optics.psf(np.zeros(optics.J), full=True).max()
    return {"rms_centre": float(np.linalg.norm(afun(np.zeros((1, 2)))[0])),
            "rms_median": float(np.median(rms)), "rms_corner": float(corner),
            "strehl_min": float(strehl.min())}


# ------------------------------------------------------------- Bead data ---

@dataclass
class Beads:
    Y: np.ndarray            # (nb, 2)
    D: np.ndarray            # (nb, nplanes, k, k) photon counts
    planes: np.ndarray       # known defocus per plane (rad RMS on Z4)
    photons: float
    bg: float

    @property
    def focus(self) -> int:
        return int(np.argmin(np.abs(self.planes)))

    def infocus_psf(self) -> np.ndarray:
        """Background-subtracted, photon-normalised in-focus bead images."""
        return (self.D[:, self.focus] - self.bg) / self.photons


def simulate_beads(optics, afun, nb, rng, photons=2e4, bg=2.0, planes=(-1.0, 0.0, 1.0)):
    Y = rng.uniform(-1, 1, (nb, 2))
    A = afun(Y)
    planes = np.asarray(planes, float)
    i4 = optics.index(4)
    D = np.empty((nb, len(planes), optics.k, optics.k))
    for p, dz in enumerate(planes):
        Ap = A.copy()
        Ap[:, i4] += dz
        D[:, p] = rng.poisson(photons * optics.psf(Ap) + bg)
    return Beads(Y, D, planes, photons, bg)


# -------------------------------------------------------------- Estimators ---
# Each estimator: fit(beads, optics) -> predict(Y) -> PSFs (M, k, k)

def _flat(h):
    return h.reshape(len(h), -1)


def est_mean(b, o):
    m = b.infocus_psf().mean(0)
    return lambda Y: np.clip(np.broadcast_to(m, (len(Y), *m.shape)), 0, None)


def _idw(b_Y, vals, k=5):
    def predict(Y):
        d = np.linalg.norm(Y[:, None] - b_Y[None], axis=-1)
        kk = min(k, len(b_Y))
        idx = np.argsort(d, 1)[:, :kk]
        w = 1 / np.maximum(np.take_along_axis(d, idx, 1), 1e-9)
        w /= w.sum(1, keepdims=True)
        return np.einsum("mk,mk...->m...", w, vals[idx])
    return predict


def est_nn(b, o):
    f = _idw(b.Y, b.infocus_psf(), k=1)
    return lambda Y: np.clip(f(Y), 0, None)


def est_idw(b, o):
    f = _idw(b.Y, b.infocus_psf(), k=5)
    return lambda Y: np.clip(f(Y), 0, None)


def _max_deg(nb, cap=3):
    deg = 0
    while deg < cap and _monomials(np.zeros((1, 2)), deg + 1).shape[1] <= nb // 2:
        deg += 1
    return deg


def est_poly_pixels(b, o):
    """PSFEx-style: every PSF pixel is a polynomial in position (least squares)."""
    deg = _max_deg(len(b.Y))
    H = _flat(b.infocus_psf())
    coef, *_ = np.linalg.lstsq(_monomials(b.Y, deg), H, rcond=None)
    return lambda Y: np.clip((_monomials(Y, deg) @ coef).reshape(len(Y), o.k, o.k), 0, None)


def _pca(b, r=8):
    H = _flat(b.infocus_psf())
    r = min(r, len(H) - 1)
    Vt = np.linalg.svd(H, full_matrices=False)[2][:r]
    return H @ Vt.T, Vt


def est_pca_poly(b, o, r=8):
    """Jee+ 2007-style: PCA coefficients as polynomials in position."""
    C, Vt = _pca(b, r)
    deg = _max_deg(len(b.Y))
    coef, *_ = np.linalg.lstsq(_monomials(b.Y, deg), C, rcond=None)
    return lambda Y: np.clip((_monomials(Y, deg) @ coef @ Vt).reshape(len(Y), o.k, o.k), 0, None)


def est_pca_rbf(b, o, r=8):
    """Gentile+ 2013 best: thin-plate RBF on PCA coefficients; smoothing by CV
    (leave-one-out up to 60 beads, 5-fold above)."""
    C, Vt = _pca(b, r)
    n = len(b.Y)
    if n < 4:
        return est_idw(b, o)
    folds = np.arange(n) if n <= 60 else np.random.default_rng(0).permutation(n) % 5
    best = None
    for s in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        err = 0.0
        for f in np.unique(folds):
            m = folds != f
            g = RBFInterpolator(b.Y[m], C[m], kernel="thin_plate_spline", smoothing=s)
            err += np.sum((g(b.Y[~m]) - C[~m]) ** 2)
        if best is None or err < best[0]:
            best = (err, s)
    f = RBFInterpolator(b.Y, C, kernel="thin_plate_spline", smoothing=best[1])
    return lambda Y: np.clip((f(Y) @ Vt).reshape(len(Y), o.k, o.k), 0, None)


def est_pca_gp(b, o, r=8):
    """PIFF-style: Gaussian-process regression of PCA coefficients (ML hyperparameters)."""
    C, Vt = _pca(b, r)
    kern = ConstantKernel(1.0) * RBF(0.5, (0.05, 5.0)) + WhiteKernel(1e-4, (1e-8, 1.0))
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2 if len(C) <= 500 else 0,
                                  random_state=0).fit(b.Y, C)
    return lambda Y: np.clip((gp.predict(Y) @ Vt).reshape(len(Y), o.k, o.k), 0, None)


def fit_parametric(b: Beads, o: Optics, basis: FieldBasis, n_starts: int = 3,
                   rng=None) -> np.ndarray:
    """Fit theta by Anscombe-style residuals sqrt(model) - sqrt(data) on all planes."""
    rng = rng or np.random.default_rng(0)
    Dd = basis.design(b.Y)                                            # (nb, J, p)
    i4 = o.index(4)
    sqrt_data = np.sqrt(b.D + 3 / 8)

    def model(theta):
        A = Dd @ theta
        hs, dhs = [], []
        for dz in b.planes:
            Ap = A.copy()
            Ap[:, i4] += dz
            h, dh = o.psf_and_jac(Ap)
            hs.append(h)
            dhs.append(dh)
        return np.stack(hs, 1), np.stack(dhs, 1)                     # (nb,P,k,k), (nb,P,J,k,k)

    def resid(theta):
        h, _ = model(theta)
        return (np.sqrt(b.photons * h + b.bg + 3 / 8) - sqrt_data).ravel()

    def jac(theta):
        h, dh = model(theta)
        s = np.sqrt(b.photons * h + b.bg + 3 / 8)
        dr_da = b.photons * dh / (2 * s[:, :, None])                 # (nb,P,J,k,k)
        Jm = np.einsum("npjxy,njq->npxyq", dr_da, Dd)
        return Jm.reshape(-1, basis.p)

    starts = [np.zeros(basis.p)] + [rng.normal(0, 0.1, basis.p) for _ in range(n_starts - 1)]
    fits = [least_squares(resid, t0, jac=jac, method="trf", x_scale="jac", max_nfev=200)
            for t0 in starts]
    return min(fits, key=lambda r: r.cost).x


def _est_param(basis_fn):
    def est(b, o):
        basis = basis_fn(o)
        theta = fit_parametric(b, o, basis)
        return lambda Y: o.psf(basis.coeffs(Y, theta))
    return est


def gp_residual(Yb, res, r=8):
    """PCA + GP (with white-noise kernel) on residual PSFs: shrinks to 0 when the
    residual is noise, follows it when it is spatially coherent."""
    R = _flat(res)
    r = min(r, len(R) - 1)
    Vt = np.linalg.svd(R, full_matrices=False)[2][:r]
    C = R @ Vt.T
    kern = ConstantKernel(1.0) * RBF(0.5, (0.05, 5.0)) + WhiteKernel(1e-2, (1e-8, 1e2))
    gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=2,
                                  random_state=0).fit(Yb, C)
    k = res.shape[-1]
    return lambda Y: (gp.predict(Y) @ Vt).reshape(len(Y), k, k)


def est_hybrid(b, o):
    """Quadratic Zernike field + GP-regressed in-focus residual."""
    basis = FieldBasis("zpoly2", o, kind="zpoly", deg=2)
    theta = fit_parametric(b, o, basis)
    res = b.infocus_psf() - o.psf(basis.coeffs(b.Y, theta))
    fres = gp_residual(b.Y, res)
    return lambda Y: np.clip(o.psf(basis.coeffs(Y, theta)) + fres(Y), 0, None)


ESTIMATORS = {
    "mean (invariant)": est_mean,
    "nearest bead": est_nn,
    "IDW kNN": est_idw,
    "poly pixels (PSFEx)": est_poly_pixels,
    "PCA+poly (Jee07)": est_pca_poly,
    "PCA+RBF (Gentile13)": est_pca_rbf,
    "PCA+GP (PIFF-like)": est_pca_gp,
    "Seidel-5 (physics)": _est_param(lambda o: FieldBasis("seidel5", o)),
    "Zernike-quad-31 (physics)": _est_param(lambda o: FieldBasis("zpoly2", o, kind="zpoly", deg=2)),
    "hybrid: Zquad + GP residual": est_hybrid,
}

PARAMETRIC = {"Seidel-5 (physics)", "Zernike-quad-31 (physics)", "hybrid: Zquad + GP residual"}
