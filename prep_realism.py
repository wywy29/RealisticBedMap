"""
Step 8a: prepare data for realism training.

Needs (same folder):
    realbeds/pine_island_bay.tif, realbeds/getz_troughs.tif   (GMRT multibeam, topo-mask)
    demogorgn_real/{pig,getz}_seed*.tif                       (from get_demogorgn_realizations.py)
    tiles/pig_tiles.npz, tiles/{pig,getz}_grids.npz           (from build_tiles.py)

Run:
    python prep_realism.py

Makes:
    tiles/real_patches.npz     9 x 9 km patches of real glacier-carved seafloor at 250 m,
                               what the realism "judge" learns real beds look like
    tiles/pig_tiles_seeds.npz  the Pine Island training tiles, plus the bed input from
                               each DEMOGORGN realization (instead of only the mean)
    tiles/{pig,getz}_seeds.npz each realization on the 1 km model grid, for running
                               the finished model over the whole region

Seafloor patches are only kept from deep water (default deeper than 600 m). Shallower
seafloor is heavily scoured by icebergs, and we don't want the model to learn
iceberg scours as if they were subglacial bedforms.
"""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

HERE = Path(__file__).resolve().parent
TILES = HERE / "tiles"
CRS = "EPSG:3031"
RES = 250.0          # m, model output resolution
PATCH = 36           # 36 x 250 m = 9 km, same as a model output tile
STRIDE = 18          # patches overlap by half
MAX_DEPTH = -600.0   # keep only patches whose mean elevation is below this (m)
MIN_VALID = 0.98     # fraction of a patch that must have real multibeam data


def seafloor_patches(path):
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    da = da.astype("float32").rio.reproject(CRS, resolution=RES, resampling=Resampling.average)
    z = da.values
    out, kept_depth = [], 0
    ny, nx = z.shape
    for i in range(0, ny - PATCH + 1, STRIDE):
        for j in range(0, nx - PATCH + 1, STRIDE):
            p = z[i:i + PATCH, j:j + PATCH]
            ok = np.isfinite(p)
            if ok.mean() < MIN_VALID:
                continue
            if np.nanmean(p) > MAX_DEPTH:
                kept_depth += 1
                continue
            p = np.where(ok, p, np.nanmean(p))
            out.append(p)
    print(f"  {path.name}: {len(out)} deep patches kept, {kept_depth} shallow ones skipped")
    return out


def to_1km(z500):
    ny, nx = z500.shape[0] // 2, z500.shape[1] // 2
    return z500[:2 * ny, :2 * nx].reshape(ny, 2, nx, 2).mean(axis=(1, 3)).astype("float32")


def main():
    print("real seafloor patches")
    patches = []
    for name in ["pine_island_bay.tif", "getz_troughs.tif"]:
        p = HERE / "realbeds" / name
        if p.exists():
            patches += seafloor_patches(p)
        else:
            print(f"  {name} not found, skipping")
    if not patches:
        raise SystemExit("No seafloor patches; check the realbeds folder.")
    real = np.stack(patches)[:, None].astype("float32")
    np.savez_compressed(TILES / "real_patches.npz", real=real)
    print(f"  saved tiles/real_patches.npz ({len(real)} patches)")

    print("\nDEMOGORGN realizations on the 1 km model grid")
    for r in ["pig", "getz"]:
        files = sorted((HERE / "demogorgn_real").glob(f"{r}_seed*.tif"))
        if not files:
            raise SystemExit(f"No demogorgn_real/{r}_seed*.tif files")
        g = np.load(TILES / f"{r}_grids.npz")
        ny, nx = g["X"].shape
        Xs = []
        for f in files:
            z = rioxarray.open_rasterio(f, masked=True).squeeze("band", drop=True).values
            Xs.append(to_1km(z)[:ny, :nx])
        Xs = np.stack(Xs)
        np.savez_compressed(TILES / f"{r}_seeds.npz", X=Xs, seeds=np.array([f.stem for f in files]))
        print(f"  {r}: {len(files)} realizations, saved tiles/{r}_seeds.npz")
        if r == "pig":
            t = dict(np.load(TILES / "pig_tiles.npz"))
            Xt = np.stack([[Xs[k, i:i + 11, j:j + 11] for k in range(len(Xs))] for i, j in t["ij"]])
            t["X_seeds"] = Xt.astype("float32")       # (tiles, realizations, 11, 11)
            np.savez_compressed(TILES / "pig_tiles_seeds.npz", **t)
            print(f"  saved tiles/pig_tiles_seeds.npz ({len(Xt)} tiles x {len(Xs)} realizations)")


if __name__ == "__main__":
    main()
