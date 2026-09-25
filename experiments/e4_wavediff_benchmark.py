#!/usr/bin/env python3
"""E4: public benchmark: WaveDiff Euclid-like PSF field (Liaudat et al. 2023, Inverse Problems 39 035008).

Data (MIT, CosmoStat/wf-psf, ``data/coherent_euclid_dataset``): 2000 noisy
training stars at random focal-plane positions, 400 noiseless test stars at
other positions, 32 x 32 pixels, polychromatic (per-star SED). Position-only
methods are fit on the first N training stars and predict the test stars.

Metric as in the paper's Table 2 (x1 resolution): per-star pixel RMS
``RMS(A - Â)`` averaged over test stars (Err_abs), and the same relative to
``RMS(A)`` (Err_rel). Published values (Table 2, 2000 stars, x1):
PSFEx 69.2e-5 (9.5%), RCA 39.6e-5 (5.4%), MCCD 43.5e-5 (6.0%),
Zernike-40 32.6e-5 (4.4%), WaveDiff 6.4e-5 (0.86%).

    uv run experiments/e4_wavediff_benchmark.py

Download (51 MB) into data/wavediff/ first:
    https://raw.githubusercontent.com/CosmoStat/wf-psf/main/data/coherent_euclid_dataset/train_Euclid_res_2000_TrainStars_id_001.npy
    https://raw.githubusercontent.com/CosmoStat/wf-psf/main/data/coherent_euclid_dataset/test_Euclid_res_id_001.npy
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import psf_fields as pf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wavediff"
OUT = ROOT / "results"
warnings.filterwarnings("ignore")

PUBLISHED = {  # Table 2, x1 resolution, 2000 training stars: Err_abs (x1e-5), Err_rel (%)
    "PSFEx (published)": (69.2, 9.5), "RCA (published)": (39.6, 5.4),
    "MCCD (published)": (43.5, 6.0), "Zernike-15 (published)": (77.5, 10.6),
    "Zernike-40 (published)": (32.6, 4.4), "WaveDiff (published)": (6.4, 0.86),
}


class Stars:
    def __init__(self, Y, P):
        self.Y, self._P = Y, P

    def infocus_psf(self):
        return self._P


class Grid:
    k = 32


def main():
    tr = np.load(DATA / "train_Euclid_res_2000_TrainStars_id_001.npy", allow_pickle=True)[()]
    te = np.load(DATA / "test_Euclid_res_id_001.npy", allow_pickle=True)[()]
    to_unit = lambda p: p / 500.0 - 1.0  # noqa: E731  positions [0, 1000]^2 -> [-1, 1]^2
    Yte, Ate = to_unit(te["positions"]), te["stars"]
    rms = lambda d: np.sqrt((d**2).mean(axis=(-2, -1)))  # noqa: E731
    methods = {
        "mean (invariant)": pf.est_mean, "nearest star": pf.est_nn, "IDW kNN": pf.est_idw,
        "poly pixels (PSFEx-like)": pf.est_poly_pixels,
        "PCA8+poly (Jee07)": pf.est_pca_poly,
        "PCA8+RBF (Gentile13)": pf.est_pca_rbf, "PCA20+RBF": lambda b, o: pf.est_pca_rbf(b, o, r=20),
        "PCA8+GP (PIFF-like)": pf.est_pca_gp, "PCA20+GP": lambda b, o: pf.est_pca_gp(b, o, r=20),
    }
    rows = []
    for n in (200, 500, 1000, 2000):
        data = Stars(to_unit(tr["positions"][:n]), tr["noisy_stars"][:n])
        for name, est in methods.items():
            t0 = time.time()
            P = est(data, Grid)(Yte)
            P = P / P.sum(axis=(-2, -1), keepdims=True)          # unit flux, like the truth
            err = rms(P - Ate)
            rows.append(dict(train_stars=n, method=name, err_abs_1e5=float(err.mean() * 1e5),
                             err_rel_pct=float((err / rms(Ate)).mean() * 100),
                             seconds=time.time() - t0))
            print(f"n={n:<5} {name:<26} Err_abs={rows[-1]['err_abs_1e5']:6.1f}e-5  "
                  f"Err_rel={rows[-1]['err_rel_pct']:5.2f}%  ({rows[-1]['seconds']:.1f}s)", flush=True)
    # noise floor: the noiseless training stars at the training positions vs a perfect model is 0;
    # report the oracle "no-SED" floor: best position-only fit = noiseless star with mean SED is n/a.
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "e4.csv", index=False)
    pub = pd.DataFrame([dict(train_stars=2000, method=k, err_abs_1e5=v[0], err_rel_pct=v[1])
                        for k, v in PUBLISHED.items()])
    tab = pd.concat([df, pub]).pivot_table(index="method", columns="train_stars",
                                            values="err_rel_pct", sort=False).round(2)
    tab_abs = pd.concat([df, pub]).pivot_table(index="method", columns="train_stars",
                                                values="err_abs_1e5", sort=False).round(1)
    txt = ["# E4 summary: WaveDiff Euclid-like benchmark (400 test stars, x1 resolution)\n",
           "## Err_rel (%): mean over test stars of RMS(A−Â)/RMS(A)\n", tab.to_markdown(),
           "\n## Err_abs (×1e-5): mean over test stars of pixel RMS(A−Â)\n", tab_abs.to_markdown(),
           "\nPublished rows are Liaudat et al. 2023, Table 2 (2000 training stars). Ours use the "
           "first N stars of the 2000-star file; the paper used separate nested subsets.\n"]
    (OUT / "e4_summary.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
