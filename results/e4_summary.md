# E4 summary: WaveDiff Euclid-like benchmark (400 test stars, x1 resolution)

## Err_rel (%): mean over test stars of RMS(A−Â)/RMS(A)

| method                   |    200 |    500 |   1000 |   2000 |
|:-------------------------|-------:|-------:|-------:|-------:|
| mean (invariant)         |  22.46 |  22.43 |  22.35 |  22.34 |
| nearest star             |  27.93 |  28.06 |  27.08 |  27.55 |
| IDW kNN                  |  16.61 |  15.97 |  15.43 |  15.48 |
| poly pixels (PSFEx-like) |  10.95 |   9.5  |   9.18 |   9.13 |
| PCA8+poly (Jee07)        |  10.93 |   9.92 |   9.78 |   9.76 |
| PCA8+RBF (Gentile13)     |   8.67 |   6.76 |   6.25 |   6.04 |
| PCA20+RBF                |   9.89 |   6.9  |   5.22 |   4.31 |
| PCA8+GP (PIFF-like)      |   8.54 |   6.75 |   6.23 |   6.03 |
| PCA20+GP                 |   9.45 |   6.6  |   5.06 |   4.24 |
| PSFEx (published)        | nan    | nan    | nan    |   9.5  |
| RCA (published)          | nan    | nan    | nan    |   5.4  |
| MCCD (published)         | nan    | nan    | nan    |   6    |
| Zernike-15 (published)   | nan    | nan    | nan    |  10.6  |
| Zernike-40 (published)   | nan    | nan    | nan    |   4.4  |
| WaveDiff (published)     | nan    | nan    | nan    |   0.86 |

## Err_abs (×1e-5): mean over test stars of pixel RMS(A−Â)

| method                   |   200 |   500 |   1000 |   2000 |
|:-------------------------|------:|------:|-------:|-------:|
| mean (invariant)         | 163.8 | 163   |  162.3 |  162.3 |
| nearest star             | 212   | 213.6 |  205.1 |  209.2 |
| IDW kNN                  | 126.4 | 121.8 |  117.4 |  117.8 |
| poly pixels (PSFEx-like) |  81.8 |  70.2 |   67.6 |   67   |
| PCA8+poly (Jee07)        |  81   |  72.8 |   71.7 |   71.4 |
| PCA8+RBF (Gentile13)     |  64.3 |  49.7 |   45.8 |   44.2 |
| PCA20+RBF                |  74.5 |  51.6 |   38.8 |   32   |
| PCA8+GP (PIFF-like)      |  63.4 |  49.5 |   45.6 |   44.1 |
| PCA20+GP                 |  70.6 |  49.1 |   37.6 |   31.4 |
| PSFEx (published)        | nan   | nan   |  nan   |   69.2 |
| RCA (published)          | nan   | nan   |  nan   |   39.6 |
| MCCD (published)         | nan   | nan   |  nan   |   43.5 |
| Zernike-15 (published)   | nan   | nan   |  nan   |   77.5 |
| Zernike-40 (published)   | nan   | nan   |  nan   |   32.6 |
| WaveDiff (published)     | nan   | nan   |  nan   |    6.4 |

Published rows are Liaudat et al. 2023, Table 2 (2000 training stars). Ours use the first N stars of the 2000-star file; the paper used separate nested subsets.
