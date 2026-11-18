"""
Pulling the super-resolved bed back onto DEMOGORGN (shared by train_realism_v6.py
and make_realizations.py, so training and the final maps do exactly the same thing).

v2-v5: every 1 km square of the output averages to DEMOGORGN's 1 km average.
v6:    the same, plus every 500 m pixel averages to DEMOGORGN's own 500 m pixel
       wherever the radar pins DEMOGORGN down. There, all realizations agree, so
       DEMOGORGN's 500 m detail is real information. Far from the radar the
       realizations disagree (their 500 m detail is random), so the model stays free.

Trust (0-1) per 500 m pixel = 1 - (spread of its height across realizations)
                                  / (0.6 x the typical spread), clipped to 0-1:
1 where all realizations agree (data), 0 where they differ as much as usual (free).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_demogorgn_500(names, ny, nx, folder="demogorgn_real"):
    """DEMOGORGN realizations at 500 m on the model grid: (seeds, 2ny, 2nx).
    names are the stems saved in {r}_seeds.npz, e.g. 'pig_seed000'."""
    import rioxarray  # noqa: F401
    out = []
    for n in names:
        p = Path(folder) / f"{n}.tif"
        z = rioxarray.open_rasterio(p, masked=True).squeeze("band", drop=True).values
        out.append(z[:2 * ny, :2 * nx].astype("float32"))
    return np.stack(out)


def trust_weights(Z500):
    """How far each 500 m pixel of DEMOGORGN is pinned down by data (1) or free (0)."""
    if len(Z500) < 3:
        return np.zeros(Z500.shape[1:], dtype="float32")
    spread = np.nanstd(Z500, axis=0)          # how much this pixel's height differs between realizations
    ref = 0.6 * np.nanmedian(spread)          # most of the map is free; pinned pixels have ~0 spread
    w = np.clip(1 - spread / max(ref, 1e-6), 0, 1)
    return np.where(np.isfinite(w), w, 0).astype("float32")


def match(y, x1k, d500=None, w500=None, n=3):
    """y: (B,1,4n,4m) 250 m output covering 1 km cells x1k (B,1,n,m);
    d500, w500: (B,1,2n,2m) DEMOGORGN at 500 m and trust weights (or None)."""
    up = lambda z, f: F.interpolate(z, scale_factor=f, mode="bilinear", align_corners=False)
    for _ in range(n):                                   # 1 km averages (as in v2-v5)
        y = y + up(x1k - F.avg_pool2d(y, 4), 4)
    if d500 is None:
        return y
    ok = torch.isfinite(d500)
    w500 = torch.where(ok, w500, torch.zeros_like(w500))
    d500 = torch.where(ok, d500, torch.zeros_like(d500))
    # 500 m target: DEMOGORGN where trusted, the model's own 500 m averages where free
    target = w500 * d500 + (1 - w500) * F.avg_pool2d(y, 2)
    for _ in range(N500):                                # smooth corrections converge slowly at 500 m
        y = y + up(target - F.avg_pool2d(y, 2), 2)
    for _ in range(n):                                   # re-secure the 1 km averages
        y = y + up(x1k - F.avg_pool2d(y, 4), 4)
    return y


N500 = 12
