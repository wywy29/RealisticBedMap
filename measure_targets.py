"""
Step 11a: measure how rough the real bed is, from the radar, so training can aim at it.

Runs on your PC, in glyproject, next to roughness.py.
Needs: cropped/pig_radar_points.csv, cropped/getz_radar_points.csv,
       and (optional, for comparison) demogorgn_real/, realizations*/, cropped/*_bedmachine_bed.tif

Run:
    python measure_targets.py

Saves roughness_targets.json (the radar's roughness at each bump size; training
aims at it) and prints how rough DEMOGORGN, BedMachine and every version of
your beds are, as % of the radar, so you can see how far each is at each scale.

The radar is read along its flight lines in 9 km windows. Each map is read
straight off a 250 m grid, in a 9 x 9 km patch centred on each radar window,
along rows and columns: exactly how training measures roughness.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401

from roughness import rms_1d, psd_1d, psd_patches_np, wavelengths_km, K_LO, K_HI, N_SPEC

HERE = Path(__file__).resolve().parent
STEP = 250.0


def tracks(r):
    """Radar picks averaged into 250 m steps along each continuous stretch."""
    df = pd.read_csv(HERE / "cropped" / f"{r}_radar_points.csv")
    out = []
    for _, g in df.groupby("FILE", sort=False):
        x, y, z = g.x.values, g.y.values, g.bed.values
        for s in np.split(np.arange(len(x)), np.where(np.hypot(np.diff(x), np.diff(y)) > 500)[0] + 1):
            if len(s) < 2:
                continue
            d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x[s]), np.diff(y[s])))])
            agg = pd.DataFrame({"b": (d // STEP).astype(int), "x": x[s], "y": y[s], "z": z[s]}).groupby("b").mean()
            idx = agg.index.values
            for part in np.split(np.arange(len(idx)), np.where(np.diff(idx) > 1)[0] + 1):
                if len(part) >= 21:
                    out.append(agg.iloc[part][["x", "y", "z"]].to_numpy())
    return out


def tif(path):
    return rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)


def study_versions():
    out = []
    if (HERE / "realizations").exists():
        out.append(("Your beds v2", HERE / "realizations"))
    nums = sorted(int(m.group(1)) for p in HERE.glob("realizations_v*")
                  if p.is_dir() and (m := re.fullmatch(r"realizations_v(\d+)", p.name)))
    return out + [(f"Your beds v{n}", HERE / f"realizations_v{n}") for n in nums]


def patches(ref, arr, wins):
    """36 x 36 patch of a 250 m grid centred on each window; None if off the map or gappy."""
    x0, y0 = float(ref.x[0]), float(ref.y[0])
    ny, nx = arr.shape
    out = []
    for w in wins:
        ci = int(round((w[:, 0].mean() - x0) / STEP))
        ri = int(round((y0 - w[:, 1].mean()) / STEP))
        r0, c0 = ri - N_SPEC // 2, ci - N_SPEC // 2
        p = arr[r0:r0 + N_SPEC, c0:c0 + N_SPEC] if r0 >= 0 and c0 >= 0 else np.empty((0, 0))
        out.append(p if p.shape == (N_SPEC, N_SPEC) and np.all(np.isfinite(p)) else None)
    return out


def main():
    targets = {}
    versions = study_versions()
    for r, name in [("pig", "Pine Island"), ("getz", "Getz")]:
        t = tracks(r)
        fine, mid, n = rms_1d([s[:, 2] for s in t])
        wins = [s[k:k + N_SPEC] for s in t for k in range(0, len(s) - N_SPEC + 1, N_SPEC)]
        p_radar, nw = psd_1d([w[:, 2] for w in wins])
        targets[r] = {"fine_m": fine, "mid_m": mid, "psd": p_radar.tolist(), "n_windows": nw}
        print(f"\n{name}: {len(t)} radar stretches, {nw} windows of 9 km")

        rows = [("DEMOGORGN (seed 000)", sorted((HERE / "demogorgn_real").glob(f"{r}_seed*.tif"))[:1]),
                ("BedMachine", [HERE / "cropped" / f"{r}_bedmachine_bed.tif"])]
        rows += [(label, sorted(folder.glob(f"{r}_seed*_250m.tif"))[:1]) for label, folder in versions]
        rows = [(label, f[0]) for label, f in rows if f and f[0].exists()]
        if not versions or not rows:
            continue
        ref = tif(versions[-1][1] / f"{r}_grounded_250m.tif")      # the 250 m grid of your beds
        P = {label: patches(ref, tif(path).interp(x=ref.x, y=ref.y).values, wins) for label, path in rows}
        ok = [i for i in range(len(wins)) if all(P[label][i] is not None for label in P)]   # same windows for all
        pr, _ = psd_1d([wins[i][:, 2] for i in ok])
        wl = wavelengths_km()
        small = [k for k in range(K_LO, K_HI + 1) if wl[k - 1] < 1.0]
        large = [k for k in range(K_LO, K_HI + 1) if wl[k - 1] >= 1.0]
        print(f"  Bump size as % of the real (radar) bed; 100% = as rough as the real bed "
              f"({len(ok)} windows covered by every map)")
        print(f"  {'':22s} {'0.5-1 km':>9s} {'1-4.5 km':>9s}")
        for label in P:
            pw = psd_patches_np(np.stack([P[label][i] for i in ok]))
            a_s = 100 * np.sqrt(pw[small].sum() / pr[small].sum())
            a_l = 100 * np.sqrt(pw[large].sum() / pr[large].sum())
            print(f"  {label:22s} {a_s:8.0f}% {a_l:8.0f}%")
    (HERE / "roughness_targets.json").write_text(json.dumps(targets, indent=1))
    print("\nSaved roughness_targets.json")


if __name__ == "__main__":
    main()
