"""
Step 9: make your own realizations: sharpen every DEMOGORGN realization to 250 m.

Run in Colab (T4 GPU) with these files in the same folder:
    deepbedmap_torch.py
    realism_deepbedmap.pt          (from train_realism.py)
    pig_grids.npz, getz_grids.npz  (tiles/, from build_tiles.py)
    pig_seeds.npz, getz_seeds.npz  (tiles/, from prep_realism.py)

    python make_realizations.py              # everything
    python make_realizations.py --quick      # one realization, small area (a test)
    python make_realizations.py realism_v3.pt   # use the v3 model -> realizations_v3/
    python make_realizations.py realism_v4.pt   # use the v4 model -> realizations_v4/
    python make_realizations.py realism_v5.pt   # use the v5 model -> realizations_v5/
    python make_realizations.py realism_v6.pt   # use the v6 model -> realizations_v6/

Saves, in realizations/:
    {pig,getz}_seed000_250m.tif ... seed009     your super-resolved beds
    {pig,getz}_grounded_250m.tif                1 = grounded ice, where the model applies

Format matches DEMOGORGN as the englacial radar-sounder-simulator reads it:
EPSG:3031, metres, referenced to the EIGEN-6C4 geoid (the simulator adds the
BedMachine geoid itself), only finer: 250 m instead of 500 m.

The model was trained on grounded ice only, so on floating ice and open ocean
the file falls back to DEMOGORGN smoothly interpolated to 250 m.
The outer 1 km of each region has no output (the model needs 1 km of context).
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_origin

from deepbedmap_torch import Generator
from match500 import load_demogorgn_500, trust_weights, match

WEIGHTS = "realism_deepbedmap.pt"
CHUNK = 100       # km of output per piece
PAD = 40          # km of extra context per side; 40 km keeps seams under ~1 m (tested)
OUT = Path("realizations")
MATCH500 = False  # v6+: also keep DEMOGORGN's 500 m detail where the radar pins it down
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def sharpen(G, X, W1, W2, W3, Z5=None, T5=None):
    """Run the model over a whole region in overlapping pieces.
    X, W3: (ny, nx) at 1 km; W1: (10ny, 10nx) at 100 m; W2: (2, 2ny, 2nx) at 500 m.
    Returns (4(ny-2), 4(nx-2)) at 250 m covering 1 km cells 1 .. n-2."""
    ny, nx = X.shape
    out = np.full((4 * (ny - 2), 4 * (nx - 2)), np.nan, dtype="float32")
    for i0 in range(1, ny - 1, CHUNK):
        for j0 in range(1, nx - 1, CHUNK):
            i1, j1 = min(i0 + CHUNK, ny - 1), min(j0 + CHUNK, nx - 1)     # output cells [i0, i1)
            a0, a1 = max(i0 - 1 - PAD, 0), min(i1 + 1 + PAD, ny)            # input window
            b0, b1 = max(j0 - 1 - PAD, 0), min(j1 + 1 + PAD, nx)
            t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr, dtype="float32"))[None].to(device)
            y = G(t(X[None, a0:a1, b0:b1]), t(W1[None, 10 * a0:10 * a1, 10 * b0:10 * b1]),
                  t(W2[:, 2 * a0:2 * a1, 2 * b0:2 * b1]), t(W3[None, a0:a1, b0:b1]))[0, 0].cpu().numpy()
            # y covers input cells a0+1 .. a1-2; keep the part for output cells i0..i1
            ys, xs = 4 * (i0 - (a0 + 1)), 4 * (j0 - (b0 + 1))
            out[4 * (i0 - 1):4 * (i1 - 1), 4 * (j0 - 1):4 * (j1 - 1)] = \
                y[ys:ys + 4 * (i1 - i0), xs:xs + 4 * (j1 - j0)]
    # Pull onto DEMOGORGN exactly as in training: 1 km averages (all versions),
    # plus 500 m pixels where the radar pins DEMOGORGN down (v6 and later)
    t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr, dtype="float32"))[None, None]
    if Z5 is not None:
        d5, t5 = t(Z5[2:2 * (ny - 1), 2:2 * (nx - 1)]), t(T5[2:2 * (ny - 1), 2:2 * (nx - 1)])
    else:
        d5 = t5 = None
    return match(t(out), t(X[1:-1, 1:-1]), d5, t5)[0, 0].numpy()


def smooth_250(X):
    t = torch.from_numpy(X.astype("float32"))[None, None]
    up = F.interpolate(t, scale_factor=4, mode="bilinear", align_corners=False)[0, 0].numpy()
    return up[4:-4, 4:-4]


def write(path, arr, x0, y0):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:3031", compress="deflate",
                       transform=from_origin(x0 + 1000, y0 - 1000, 250, 250), nodata=np.nan) as dst:
        dst.write(arr.astype("float32"), 1)


def main():
    global WEIGHTS, OUT, MATCH500
    quick = "--quick" in sys.argv
    for a in sys.argv[1:]:
        if a.endswith(".pt"):
            WEIGHTS = a
            v = re.search(r"_v(\d+)\.pt$", a)          # realism_v5.pt -> realizations_v5/
            if v:
                OUT = Path(f"realizations_v{v.group(1)}")
                MATCH500 = int(v.group(1)) >= 6
    print(f"model: {WEIGHTS} -> {OUT}/")
    print(f"device: {device}")
    G = Generator().to(device).eval()
    G.load_state_dict(torch.load(WEIGHTS, map_location=device))
    OUT.mkdir(exist_ok=True)

    for r in ["pig", "getz"]:
        g, s = np.load(f"{r}_grids.npz"), np.load(f"{r}_seeds.npz")
        W1, W2, W3, grounded = g["W1"], g["W2"], g["W3"], g["grounded"]
        x0, y0 = float(g["x0"]), float(g["y0"])
        Xs, names = s["X"], s["seeds"]
        Z5s = load_demogorgn_500([str(n) for n in names], *Xs.shape[1:]) if MATCH500 else None
        T5 = trust_weights(Z5s) if MATCH500 else None
        if MATCH500:
            g500 = grounded[::2, ::2][:T5.shape[0], :T5.shape[1]]
            print(f"\n{r}: keeping DEMOGORGN's 500 m detail where the radar pins it down "
                  f"(trust > 0.5): {100 * (T5[g500] > 0.5).mean():.0f}% of grounded ice")
        if quick:
            n = 60
            Xs, names = Xs[:1, :n, :n], names[:1]
            if MATCH500:
                Z5s, T5 = Z5s[:1, :2 * n, :2 * n], T5[:2 * n, :2 * n]
            W1, W2, W3, grounded = W1[:10 * n, :10 * n], W2[:, :2 * n, :2 * n], W3[:n, :n], grounded[:4 * n, :4 * n]
        ny, nx = Xs.shape[1:]
        gmask = grounded[4:4 * (ny - 1), 4:4 * (nx - 1)]
        write(OUT / f"{r}_grounded_250m.tif", gmask.astype("float32"), x0, y0)
        print(f"\n{r}: {len(Xs)} realizations, output {4 * (ny - 2)} x {4 * (nx - 2)} at 250 m")

        for k, (X, name) in enumerate(zip(Xs, names)):
            sharp = sharpen(G, X, W1, W2, W3, Z5s[k] if MATCH500 else None, T5)
            bed = np.where(gmask, sharp, smooth_250(X))
            seed = str(name).split("seed")[-1]
            write(OUT / f"{r}_seed{seed}_250m.tif", bed, x0, y0)
            added = (sharp - smooth_250(X))[gmask]
            print(f"  seed {seed}: saved; detail added on grounded ice: "
                  f"RMS {np.sqrt(np.nanmean(added ** 2)):.1f} m")
    print(f"\nDone. Files are in {OUT}/")


if __name__ == "__main__":
    main()
