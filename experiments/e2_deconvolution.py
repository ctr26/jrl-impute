#!/usr/bin/env python3
"""E2: spatially varying Richardson-Lucy with PSF fields recovered from beads.

Object: public scikit-image ``cells3d`` (fluorescence, membrane + nuclei;
Allen Institute for Cell Science), one z-slice, 256 x 256. The image is blurred
with the TRUE spatially varying PSF field (exact sparse H, one PSF per pixel)
and Poisson noise is added. It is then deconvolved with:

* the oracle field (true H),
* each E1 estimator's field recovered from the same 32 noisy beads,
* rank-r product-convolution approximations of the true field (docs/PROOFS.md T3/T4).

    uv run experiments/e2_deconvolution.py [--beads 32] [--photons 300] [--iters 60]

Outputs: results/e2.csv, results/e2.png, results/e2_summary.md
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jrl_impute as J  # noqa: E402
import psf_fields as pf  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
warnings.filterwarnings("ignore")


def objects(size: int = 256) -> dict[str, np.ndarray]:
    cells = skdata.cells3d()                           # (z, channel, y, x): 0 membrane, 1 nuclei
    out = {}
    for name, ch in (("membrane", 0), ("nuclei", 1)):
        img = cells[30, ch, :size, :size].astype(float)
        img -= np.percentile(img, 1)
        out[name] = np.clip(img / np.percentile(img, 99.9), 0, None)
    return out


def pixel_positions(shape):
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])  # (x, y) to match field convention


def psfs_in_chunks(fn, Y, chunk=4096):
    return np.concatenate([fn(Y[i:i + chunk]) for i in range(0, len(Y), chunk)])


CHECKPOINTS = (10, 20, 40, 80, 160)


def rl_checkpoints(apply, adjoint, b, checkpoints=CHECKPOINTS, eps=1e-12):
    """Richardson-Lucy, returning iterates at the checkpoint iterations."""
    b = np.clip(b, 0, None)
    norm = np.maximum(adjoint(np.ones_like(b)), eps)
    x = np.full_like(b, max(b.mean(), eps))
    out = {}
    for it in range(1, max(checkpoints) + 1):
        x = np.clip(x * adjoint(b / np.maximum(apply(x), eps)) / norm, eps, None)
        if it in checkpoints:
            out[it] = x.copy()
    return out


def sparse_ops(H, shape):
    Ht = H.T.tocsr()
    return (lambda x: (H @ x.ravel()).reshape(shape),
            lambda y: (Ht @ y.ravel()).reshape(shape))


def score(truth, est, photons):
    e = est / photons
    return dict(psnr=float(peak_signal_noise_ratio(truth, np.clip(e, 0, None), data_range=1.0)),
                ssim=float(structural_similarity(truth, e, data_range=1.0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--beads", type=int, default=32)
    ap.add_argument("--photons", nargs="+", type=float, default=[100.0, 1000.0, 10000.0],
                    help="peak photons of the object")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fields", nargs="+", default=["centred:1", "decentred:2"],
                    help="truth fields as kind:level")
    ap.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 8, 16])
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    o = pf.Optics()
    objs = objects(a.size)
    shape = (a.size, a.size)
    Ypix = pixel_positions(shape)
    rows, panels = [], {}
    for spec in a.fields:
        kind, level = spec.split(":")
        field = f"{kind} L{level}"
        afun = pf.truth_field(o, kind, float(level), seed=100)
        t0 = time.time()
        true_psfs = psfs_in_chunks(lambda Y: o.psf(afun(Y)), Ypix)
        H_true = J.forward_matrix(true_psfs, shape)
        print(f"[{field}] true field + H: {time.time() - t0:.1f} s, nnz={H_true.nnz:,}")
        beads = pf.simulate_beads(o, afun, a.beads, np.random.default_rng(0))

        fields = {"oracle (true H)": true_psfs}
        fit_times = {}
        for name, est in pf.ESTIMATORS.items():
            t0 = time.time()
            pred = est(beads, o)
            fields[name] = psfs_in_chunks(pred, Ypix)
            fit_times[name] = time.time() - t0
            print(f"  {name:<30} field in {fit_times[name]:.1f} s")

        errs = {name: float(np.linalg.norm(psfs - true_psfs) / np.linalg.norm(true_psfs))
                for name, psfs in fields.items()}
        data = {}
        for photons in a.photons:
            for oname, obj in objs.items():
                rng = np.random.default_rng(1)
                b = rng.poisson(J.blur(obj * photons, H_true)).astype(float)
                data[(photons, oname)] = b
                rows.append(dict(field=field, object=oname, photons=photons, iters=0,
                                 method="blurred (no deconvolution)", **score(obj, b, photons),
                                 psf_rel_l2=np.nan))
                if oname == "membrane" and photons == max(a.photons):
                    panels[(field, "truth")] = obj
                    panels[(field, "blurred")] = b / photons
        del H_true

        def operators():
            for name in list(fields):                      # one operator in memory at a time
                psfs = fields.pop(name)
                yield name, sparse_ops(J.forward_matrix(psfs, shape), shape)
            for r in a.ranks:
                op = J.ProductConvolution(true_psfs, shape, rank=r)
                name = f"true field, rank-{r} product-convolution"
                errs[name] = float(np.sqrt(op.tail / (op.sigma**2).sum()))
                yield name, (op.matvec, op.rmatvec)

        for name, (fwd, adj) in operators():
            t0 = time.time()
            for (photons, oname), b in data.items():
                obj = objs[oname]
                for it, x in rl_checkpoints(fwd, adj, b).items():
                    rows.append(dict(field=field, object=oname, photons=photons, iters=it,
                                     method=name, **score(obj, x, photons), psf_rel_l2=errs[name]))
                    if oname == "membrane" and photons == max(a.photons) and it == 80:
                        panels[(field, name)] = x / photons
            print(f"  [{field}] {name}: {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2.csv", index=False)
    (OUT / "e2_summary.md").write_text(summary(df, a))
    plot(panels, list(dict.fromkeys(df.field)), OUT / "e2.png")
    print(summary(df, a))


def summary(df, a):
    lines = [f"# E2 summary: spatially varying RL on cells3d (slice 30), {a.beads} beads\n",
             "Values: PSNR (dB) vs object at **fixed 40 iterations** / **best over "
             f"{list(CHECKPOINTS)} iterations** (oracle stopping). `psf_err` = relative L2 error of "
             "the PSF field used (rank-r rows: sqrt of discarded SVD energy fraction).\n"]
    order = list(dict.fromkeys(df.method))
    for (field, obj), d in df.groupby(["field", "object"], sort=False):
        fixed = d[(d.iters == 40) | (d.iters == 0)].pivot_table(index="method", columns="photons", values="psnr")
        best = d.pivot_table(index="method", columns="photons", values="psnr", aggfunc="max")
        err = d.groupby("method").psf_rel_l2.first()
        tab = pd.DataFrame({f"{p:g} ph": fixed[p].map("{:.2f}".format) + " / " + best[p].map("{:.2f}".format)
                            for p in fixed.columns})
        tab["psf_err"] = err.map(lambda v: "" if np.isnan(v) else f"{v:.3f}")
        lines.append(f"\n## {field}, {obj}\n")
        lines.append(tab.loc[[m for m in order if m in tab.index]].to_markdown())
    return "\n".join(lines) + "\n"


def plot(panels, fields, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["truth", "blurred", "oracle (true H)", "mean (invariant)", "PCA+RBF (Gentile13)",
            "hybrid: Zquad + GP residual"]
    fig, axes = plt.subplots(len(fields), len(show), figsize=(2.6 * len(show), 2.8 * len(fields)))
    axes = np.atleast_2d(axes)
    for i, f in enumerate(fields):
        for j, name in enumerate(show):
            ax = axes[i, j]
            img = panels.get((f, name))
            if img is not None:
                ax.imshow(img[:128, :128], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{name}\n({f})", fontsize=7)
            ax.axis("off")
    fig.suptitle("E2: cells3d membrane, highest photon count, 80 RL iterations, top-left 128² corner", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
