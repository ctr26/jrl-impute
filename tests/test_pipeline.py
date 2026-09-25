"""Smoke + correctness tests for jrl_impute. Run: ``pytest -q``."""

import numpy as np
import pytest
from scipy import ndimage

import jrl_impute as J

SHAPE = (24, 24)
K = 7


@pytest.fixture(scope="module")
def varying():
    psfs = J.psf_stack(J.radial_sigma(SHAPE), K)
    return psfs, J.forward_matrix(psfs, SHAPE)


def test_gaussian_psf_is_normalised_and_centred():
    psf = J.gaussian_psf(K, 1.3)
    assert psf.sum() == pytest.approx(1.0)
    assert np.unravel_index(psf.argmax(), psf.shape) == (K // 2, K // 2)
    with pytest.raises(ValueError):
        J.gaussian_psf(8, 1.0)


def test_static_forward_matrix_matches_convolution():
    psf = J.gaussian_psf(K, 1.5)
    H = J.forward_matrix(np.repeat(psf[None], np.prod(SHAPE), axis=0), SHAPE)
    x = np.random.default_rng(0).random(SHAPE)
    np.testing.assert_allclose(J.blur(x, H), ndimage.convolve(x, psf, mode="constant"),
                               atol=1e-12)


def test_varying_forward_matrix_columns_are_point_responses(varying):
    """Column v of H is the image of a point source at v (not row v: that is H^T)."""
    psfs, H = varying
    r = K // 2
    v = (SHAPE[0] // 2) * SHAPE[1] + SHAPE[1] // 2 + 3           # interior, off-centre
    delta = np.zeros(SHAPE)
    delta.flat[v] = 1.0
    vy, vx = divmod(v, SHAPE[1])
    img = J.blur(delta, H)
    np.testing.assert_allclose(img[vy - r:vy + r + 1, vx - r:vx + r + 1], psfs[v])
    assert img.sum() == pytest.approx(1.0)
    # the PSF really does vary, so H is not symmetric
    assert abs(H - H.T).max() > 1e-3


def test_richardson_lucy_improves_on_blurred(varying):
    _, H = varying
    truth = J.test_image(SHAPE[0])
    b = J.blur(truth, H)
    x, hist = J.richardson_lucy(H, b, n_iter=40)
    assert x.shape == truth.shape and np.all(x >= 0)
    assert hist[-1] < hist[0]
    assert np.linalg.norm(x - truth) < np.linalg.norm(b - truth)


@pytest.mark.parametrize("method", J.IMPUTE_METHODS)
def test_impute_keeps_beads_and_returns_valid_psfs(varying, method):
    psfs, _ = varying
    known = J.bead_mask(psfs.shape[0], 40, np.random.default_rng(1))
    out = J.impute_psfs(psfs, SHAPE, known, method=method)
    assert out.shape == psfs.shape and np.all(np.isfinite(out)) and np.all(out >= 0)
    np.testing.assert_allclose(out.sum(axis=(1, 2)), 1.0)
    np.testing.assert_allclose(out[known], psfs[known])


def test_spatial_imputation_beats_mean(varying):
    psfs, _ = varying
    known = J.bead_mask(psfs.shape[0], 40, np.random.default_rng(1))
    miss = ~known

    def err(m):
        out = J.impute_psfs(psfs, SHAPE, known, method=m)
        return np.linalg.norm(out[miss] - psfs[miss])

    assert err("knn") < 0.5 * err("mean")
    assert err("poly") < err("mean")


def test_demo_end_to_end_ranks_methods():
    res = J.run_demo(size=32, psf_size=K, n_beads=60, n_iter=30, seed=0,
                     methods=("knn", "mean"))
    m = res["metrics"]
    assert m["true H"]["ssim"] > m["blurred"]["ssim"]
    assert m["knn-imputed H"]["ssim"] > m["mean-imputed H"]["ssim"]
