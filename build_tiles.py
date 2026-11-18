"""
Step 6: build DeepBedMap training tiles.

Setup (once):
    pip install scipy

Needs (same folder as this script):
    demogorgn_{pig,getz}_mean.tif          from step 3
    cropped/{pig,getz}_radar_bed_250m.tif  from step 5
    cropped/{pig,getz}_bedmachine_mask.tif from step 4
    raw/rema_mosaic_100m_v2.0_filled_cop30_dem.tif
    raw/NSIDC-0484_2-.../*.nc              ice velocity
    amsr_accumulation_map.txt              snow accumulation

Run:
    python build_tiles.py

Makes, in tiles/:
    {pig,getz}_grids.npz   every input for the whole region on DeepBedMap's grids,
                           used later to run the model over the full area
    pig_tiles.npz          Pine Island training tiles, split into train/val/test

DeepBedMap's input format (one tile = 11 x 11 km):
    X  bed (DEMOGORGN)          1 km   11 x 11
    W1 surface (REMA)         100 m  110 x 110
    W2 velocity vx, vy        500 m   2 x 22 x 22
    W3 snow accumulation        1 km   11 x 11
    Y  radar bed (answer)     250 m   36 x 36  <- the central 9 x 9 km
    M  where Y has radar      250 m   36 x 36
Tiles step 9 km, so the answer areas never overlap. Only tiles with
enough radar inside are kept. Tiles are grouped into 45 x 45 km blocks and
whole blocks go to train / val / test, so test areas are never seen in training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from scipy.interpolate import griddata

HERE = Path(__file__).resolve().parent
RAW, CROP, OUT = HERE / "raw", HERE / "cropped", HERE / "tiles"
CRS = "EPSG:3031"

TILE = 11            # km, input size
STEP = 9             # km, tile step (= answer size, so answers don't overlap)
MIN_RADAR = 30       # keep tiles with at least this many 250 m radar cells (of 1296)
BLOCK = 5            # tiles per block side -> 45 km blocks for the split
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 0


def open_tif(path):
    return rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)


def force_crs(da):
    da = da.copy()
    da.attrs.pop("grid_mapping", None)
    da.encoding.pop("grid_mapping", None)
    for c in ("spatial_ref", "mapping", "coord_system", "crs", "polar_stereographic"):
        if c in da.coords:
            da = da.drop_vars(c)
    return da.rio.write_crs(CRS)


def template(x0, y0, res, nx, ny):
    """Empty raster with top-left corner (x0, y0), cell size res."""
    x = x0 + res * (np.arange(nx) + 0.5)
    y = y0 - res * (np.arange(ny) + 0.5)
    t = xr.DataArray(np.zeros((ny, nx), "float32"), dims=("y", "x"), coords={"y": y, "x": x})
    return t.rio.write_crs(CRS)


def onto(da, tmpl, how):
    xmin, ymin, xmax, ymax = tmpl.rio.bounds()
    pad = 5000
    da = da.rio.clip_box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    if da.rio.crs is None:
        da = force_crs(da)
    out = da.astype("float32").rio.write_crs(CRS).rio.reproject_match(tmpl, resampling=how)
    return out.values.astype("float32")


def accumulation_on(tmpl):
    """Interpolate the 25 km Arthern map onto a 1 km grid."""
    df = pd.read_csv(HERE / "amsr_accumulation_map.txt", sep=r"\s+", header=None, comment=None,
                     names=["lat", "lon", "x", "y", "acc", "err"], skiprows=_header_lines())
    df = df.dropna(subset=["acc"])
    xmin, ymin, xmax, ymax = tmpl.rio.bounds()
    pad = 100_000
    df = df[(df.x > xmin - pad) & (df.x < xmax + pad) & (df.y > ymin - pad) & (df.y < ymax + pad)]
    gx, gy = np.meshgrid(tmpl.x.values, tmpl.y.values)
    out = griddata((df.x.values, df.y.values), df.acc.values, (gx, gy), method="linear")
    return np.nan_to_num(out, nan=0.0).astype("float32")


def _header_lines():
    with open(HERE / "amsr_accumulation_map.txt") as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) == 6:
                try:
                    [float(p) for p in parts]
                    return i
                except ValueError:
                    pass
    raise SystemExit("Couldn't find where the numbers start in amsr_accumulation_map.txt")


def build_region(r, rema, vx, vy):
    print(f"\n{r}")
    dem = open_tif(HERE / f"demogorgn_{r}_mean.tif")
    x0, _, _, y0 = dem.rio.bounds()           # top-left corner shared by every grid
    nx = int(dem.sizes["x"] // 2)              # whole 1 km cells
    ny = int(dem.sizes["y"] // 2)

    # X: DEMOGORGN 500 m -> 1 km by averaging each 2 x 2 block
    d = dem.values[: 2 * ny, : 2 * nx].astype("float64")
    X = d.reshape(ny, 2, nx, 2).mean(axis=(1, 3)).astype("float32")
    print(f"  bed (DEMOGORGN)    1 km   {X.shape}")

    W1 = onto(rema, template(x0, y0, 100, 10 * nx, 10 * ny), Resampling.average)
    W1 = np.nan_to_num(W1, nan=0.0)
    print(f"  surface (REMA)   100 m  {W1.shape}")

    t500 = template(x0, y0, 500, 2 * nx, 2 * ny)
    W2 = np.stack([onto(vx, t500, Resampling.bilinear), onto(vy, t500, Resampling.bilinear)])
    W2 = np.nan_to_num(W2, nan=0.0)
    print(f"  velocity         500 m  {W2.shape}")

    W3 = accumulation_on(template(x0, y0, 1000, nx, ny))
    print(f"  accumulation       1 km   {W3.shape}")

    Y = open_tif(CROP / f"{r}_radar_bed_250m.tif").values[: 4 * ny, : 4 * nx].astype("float32")
    M = np.isfinite(Y)
    print(f"  radar bed        250 m  {Y.shape}  ({M.sum():,} cells with radar)")

    mask500 = open_tif(CROP / f"{r}_bedmachine_mask.tif").values[: 2 * ny, : 2 * nx]
    grounded = np.repeat(np.repeat(mask500 == 2, 2, 0), 2, 1)

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{r}_grids.npz", X=X, W1=W1, W2=W2, W3=W3, Y=Y, M=M,
                        grounded=grounded, x0=x0, y0=y0)
    print(f"  saved tiles/{r}_grids.npz")
    return X, W1, W2, W3, Y, M, grounded


def cut_tiles(X, W1, W2, W3, Y, M, grounded):
    ny, nx = X.shape
    rows, cols = range(0, ny - TILE + 1, STEP), range(0, nx - TILE + 1, STEP)
    rng = np.random.default_rng(SEED)
    nb = (len(rows) // BLOCK + 1) * (len(cols) // BLOCK + 1)
    order = rng.permutation(nb)
    cut1 = int(SPLIT["train"] * nb)
    cut2 = cut1 + int(SPLIT["val"] * nb)
    which = np.empty(nb, dtype="<U5")
    which[order[:cut1]], which[order[cut1:cut2]], which[order[cut2:]] = "train", "val", "test"
    ncb = len(cols) // BLOCK + 1

    keep = {k: [] for k in ["X", "W1", "W2", "W3", "Y", "M", "ij", "split"]}
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            ys, xs = slice(4 * (i + 1), 4 * (i + 1) + 36), slice(4 * (j + 1), 4 * (j + 1) + 36)
            m = M[ys, xs] & grounded[ys, xs]
            if m.sum() < MIN_RADAR:
                continue
            keep["X"].append(X[i:i + TILE, j:j + TILE])
            keep["W1"].append(W1[10 * i:10 * (i + TILE), 10 * j:10 * (j + TILE)])
            keep["W2"].append(W2[:, 2 * i:2 * (i + TILE), 2 * j:2 * (j + TILE)])
            keep["W3"].append(W3[i:i + TILE, j:j + TILE])
            keep["Y"].append(np.where(m, Y[ys, xs], 0.0))
            keep["M"].append(m)
            keep["ij"].append((i, j))
            keep["split"].append(which[(a // BLOCK) * ncb + (b // BLOCK)])

    if not keep["X"]:
        raise SystemExit("No tiles had enough radar; try a smaller MIN_RADAR.")
    out = {k: np.stack(v) for k, v in keep.items() if k != "split"}
    out["X"], out["W3"] = out["X"][:, None], out["W3"][:, None]
    out["W1"], out["Y"], out["M"] = out["W1"][:, None], out["Y"][:, None], out["M"][:, None]
    out["split"] = np.array(keep["split"])
    return out


def main():
    rema = open_tif(next(RAW.glob("rema*_dem.tif")))
    vel = xr.open_dataset(next(next(RAW.glob("NSIDC-0484*")).glob("*.nc")))
    pick = lambda n: next(vel[v] for v in vel.data_vars if v.lower() == n)
    vx = force_crs(pick("vx").squeeze(drop=True).rio.set_spatial_dims(x_dim="x", y_dim="y"))
    vy = force_crs(pick("vy").squeeze(drop=True).rio.set_spatial_dims(x_dim="x", y_dim="y"))

    grids = {r: build_region(r, rema, vx, vy) for r in ["pig", "getz"]}

    print("\ncutting Pine Island training tiles")
    t = cut_tiles(*grids["pig"])
    np.savez_compressed(OUT / "pig_tiles.npz", **t)
    for s in ["train", "val", "test"]:
        sel = t["split"] == s
        print(f"  {s:5s}: {sel.sum():5d} tiles, {int(t['M'][sel].sum()):,} radar cells")
    print("  saved tiles/pig_tiles.npz")


if __name__ == "__main__":
    main()
