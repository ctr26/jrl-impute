#!/usr/bin/env python3
"""E1: PSF-field recovery from sparse, noisy beads: literature baselines vs physics models.

Every method gets the SAME data: nb beads at random field positions, each a
noisy through-focus triplet (defocus -1, 0, +1 rad RMS; Poisson, background).
Non-parametric methods use the in-focus plane; parametric ones use all three.
Error is measured on a 24 x 24 grid of held-out field positions against the
noise-free truth.

    uv run experiments/e1_psf_field_recovery.py            # full sweep (~20-40 min)
    uv run experiments/e1_psf_field_recovery.py --quick    # smoke run (~1 min)

Outputs: results/e1.csv, results/e1.png, results/e1_summary.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import psf_fields as pf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def run_condition(args):
    warnings.filterwarnings("ignore")
    field, level, nb, seed, photons = args
    o = pf.Optics()
    afun = pf.truth_field(o, field, level, seed=100 + seed)
    rng = np.random.default_rng(seed)
    beads = pf.simulate_beads(o, afun, nb, rng, photons=photons)
    t = np.linspace(-1, 1, 24)
    Y = np.stack(np.meshgrid(t, t), -1).reshape(-1, 2)
    T = o.psf(afun(Y))
    rows = []

    def score(name, pred, secs):
        P = pred(Y)
        rows.append(dict(field=field, level=level, beads=nb, seed=seed, photons=photons,
                         method=name,
                         rel_l2=float(np.linalg.norm(P - T) / np.linalg.norm(T)),
                         mean_l1=float(np.abs(P - T).sum(axis=(-2, -1)).mean()),
                         seconds=secs))

    for name, est in pf.ESTIMATORS.items():
        if name == "hybrid: Zquad + GP residual":
            continue                                   # built from the Zquad fit below
        t0 = time.time()
        if name == "Zernike-quad-31 (physics)":
            basis = pf.FieldBasis("zpoly2", o, kind="zpoly", deg=2)
            theta = pf.fit_parametric(beads, o, basis)
            zq = lambda Yq, th=theta: o.psf(basis.coeffs(Yq, th))  # noqa: E731
            secs = time.time() - t0
            score(name, zq, secs)
            res = beads.infocus_psf() - zq(beads.Y)
            fres = pf.gp_residual(beads.Y, res)
            score("hybrid: Zquad + GP residual",
                  lambda Yq: np.clip(zq(Yq) + fres(Yq), 0, None), time.time() - t0)
            continue
        pred = est(beads, o)
        score(name, pred, time.time() - t0)
    return rows


def plot(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = df[["field", "level"]].drop_duplicates().values.tolist()
    methods = list(pf.ESTIMATORS)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (field, level) in zip(axes, conds):
        d = df[(df.field == field) & (df.level == level)]
        for i, m in enumerate(methods):
            g = d[d.method == m].groupby("beads").rel_l2
            mu, sd = g.mean(), g.std().fillna(0)
            style = "-" if m in pf.PARAMETRIC else "--"
            ax.errorbar(mu.index, mu.values, yerr=sd.values, label=m, color=cmap(i % 10),
                        ls=style, marker="o", ms=3, capsize=2, lw=1.4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("number of beads"); ax.set_title(f"{field} field, level {level:g}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("relative L2 error of PSF field (held-out positions)")
    axes[-1].legend(fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("E1: PSF-field recovery; solid = physics / hybrid, dashed = non-parametric", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def summarise(df: pd.DataFrame) -> str:
    lines = ["# E1 summary: relative L2 error of the PSF field, mean ± std over seeds (lower is better)\n"]
    order = [m for m in pf.ESTIMATORS if m in set(df.method)]
    for (field, level), d in df.groupby(["field", "level"], sort=False):
        g = d.groupby(["method", "beads"]).rel_l2
        tab = (g.mean().map("{:.4f}".format) + " ± " + g.std().fillna(0).map("{:.4f}".format)).unstack("beads")
        lines.append(f"\n## {field} field, level {level:g}\n")
        lines.append(tab.loc[order].to_markdown())
    t = df.groupby("method").seconds.median().round(2).loc[order]
    lines.append("\n## Median fit time (s)\n")
    lines.append(t.to_markdown())
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    if a.quick:
        grid = [(f, l, nb, s, 2e4) for f, l in [("centred", 1.0), ("decentred", 1.0)]
                for nb in (8, 32) for s in range(1)]
    else:
        grid = [(f, l, nb, s, 2e4)
                for f, l in [("centred", 1.0), ("decentred", 1.0), ("decentred", 2.0)]
                for nb in (8, 16, 32, 64, 128) for s in range(3)]
    grid.sort(key=lambda g: -g[2])                      # big jobs first
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run_condition, grid) for r in rs]
    df = pd.DataFrame(rows)
    tag = "_quick" if a.quick else ""
    df.to_csv(OUT / f"e1{tag}.csv", index=False)
    plot(df, OUT / f"e1{tag}.png")
    (OUT / f"e1{tag}_summary.md").write_text(summarise(df))
    print(summarise(df))
    print(f"{len(grid)} conditions in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
