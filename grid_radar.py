"""
Step 5: put the IceBridge radar bed onto the same 250 m grid as the inputs.

Folder layout it expects (same folder as this script):
    icebridge_pig/*.csv, icebridge_getz/*.csv   (IRMCR2 files from Earthdata)
    raw/NSIDC-0756_4-.../*.nc                    (BedMachine, for its geoid)
    cropped/{pig,getz}_rema_250m.tif             (from step 4, used as the grid)
    cropped/{pig,getz}_bedmachine_mask.tif       (from step 4)

Run:
    python grid_radar.py

Output, added to cropped/:
    {region}_radar_bed_250m.tif     radar bed elevation, averaged per 250 m cell
                                    (empty where no flight passed)
    {region}_radar_count_250m.tif   how many radar points went into each cell
    {region}_radar_points.csv       every kept point, for line-by-line testing

How bed elevation is computed (IRMCR2 user guide):
    BOTTOM is the *range* from the aircraft to the bed, not an elevation.
    bed (ellipsoid) = ELEVATION - BOTTOM
    bed (sea level) = bed (ellipsoid) - geoid      <- same reference as BedMachine
Points are kept only if they have a real bed pick (THICK > 0, BOTTOM != -9999),
QUALITY 1 or 2, and sit on grounded ice according to BedMachine's mask.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CROP = HERE / "cropped"
CRS = "EPSG:3031"
KEEP_QUALITY = {1, 2}  # 1 = high, 2 = medium, 3 = low confidence pick
GROUNDED = 2           # BedMachine mask: 0 ocean, 1 rock, 2 grounded ice, 3 floating


def load_csvs(folder):
    # rglob also finds files inside subfolders (Earthdata downloads often add one)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".csv")
    if not files:
        print(f"No .csv files in {folder}. What's in there:")
        for p in list(folder.rglob("*"))[:20]:
            print("   ", p.relative_to(folder))
        raise SystemExit(1)
    parts = []
    for f in files:
        df = pd.read_csv(f, sep=None, engine="python")  # works with commas or tabs
        df.columns = [c.strip().upper() for c in df.columns]
        df["FILE"] = f.name
        parts.append(df)
    return pd.concat(parts, ignore_index=True), len(files)


def sample(da, x, y):
    """Value of a raster at each point (nearest cell); NaN outside it."""
    xs, ys = da.x.values, da.y.values
    ix = np.round((x - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.round((y - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    out = np.full(len(x), np.nan)
    out[ok] = da.values[iy[ok], ix[ok]]
    return out


def load_geoid(bounds):
    nc = next(next(RAW.glob("NSIDC-0756*")).glob("*.nc"))
    ds = xr.open_dataset(nc)
    name = next((v for v in ds.data_vars if v.lower() == "geoid"), None)
    if name is None:
        raise SystemExit(f"No 'geoid' variable in {nc.name}. Variables: {list(ds.data_vars)}")
    g = ds[name]
    xmin, ymin, xmax, ymax = bounds
    ysl = slice(ymax, ymin) if g.y[0] > g.y[-1] else slice(ymin, ymax)
    return g.sel(x=slice(xmin, xmax), y=ysl).astype("float32").load()


def grid_points(x, y, v, tmpl):
    """Average point values into the template's cells."""
    xs, ys = tmpl.x.values, tmpl.y.values
    ix = np.round((x - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.round((y - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    flat = iy[ok] * len(xs) + ix[ok]
    n = len(xs) * len(ys)
    total = np.bincount(flat, weights=v[ok], minlength=n)
    count = np.bincount(flat, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    shape = (len(ys), len(xs))
    mk = lambda a: tmpl.copy(data=a.reshape(shape).astype("float32"))
    return mk(mean), mk(count.astype(float))


def main():
    to_xy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    for r in ["pig", "getz"]:
        print(f"\n{r}")
        tmpl = rioxarray.open_rasterio(CROP / f"{r}_rema_250m.tif", masked=True).squeeze("band", drop=True)
        mask = rioxarray.open_rasterio(CROP / f"{r}_bedmachine_mask.tif", masked=True).squeeze("band", drop=True)
        bm_bed = rioxarray.open_rasterio(CROP / f"{r}_bedmachine_bed.tif", masked=True).squeeze("band", drop=True)

        folder = next((p for p in HERE.iterdir() if p.is_dir()
                       and p.name.lower().replace("-", "_") == f"icebridge_{r}"), None)
        if folder is None:
            raise SystemExit(f"Can't find a folder named icebridge_{r} or icebridge-{r} in {HERE}")
        df, nfiles = load_csvs(folder)
        print(f"  read {len(df):,} points from {nfiles} files")

        df["x"], df["y"] = to_xy.transform(df["LON"].values, df["LAT"].values)
        xmin, ymin, xmax, ymax = tmpl.rio.bounds()
        df = df[(df.x >= xmin) & (df.x <= xmax) & (df.y >= ymin) & (df.y <= ymax)]
        print(f"  {len(df):,} inside the {r} area")

        df = df[(df.THICK > 0) & (df.BOTTOM != -9999) & (df.THICK != -9999)]
        df = df[df.QUALITY.isin(KEEP_QUALITY)]
        print(f"  {len(df):,} with a good bed pick (quality 1-2)")

        geoid = load_geoid((xmin - 1000, ymin - 1000, xmax + 1000, ymax + 1000))
        df = df.assign(geoid=sample(geoid, df.x.values, df.y.values))
        df["bed"] = df.ELEVATION - df.BOTTOM - df.geoid
        df["mask"] = sample(mask, df.x.values, df.y.values)
        df = df[(df["mask"] == GROUNDED) & df.bed.notna()]
        print(f"  {len(df):,} on grounded ice  <- these are the ones used")

        if df.empty:
            print("  nothing left to grid, skipping")
            continue

        # Sanity check: radar bed vs BedMachine at the same spots.
        diff = df.bed.values - sample(bm_bed, df.x.values, df.y.values)
        print(f"  radar minus BedMachine: median {np.nanmedian(diff):+.0f} m, "
              f"middle 50% between {np.nanpercentile(diff, 25):+.0f} and {np.nanpercentile(diff, 75):+.0f} m")

        bed, count = grid_points(df.x.values, df.y.values, df.bed.values, tmpl)
        bed.rio.write_crs(CRS).rio.to_raster(CROP / f"{r}_radar_bed_250m.tif", compress="deflate")
        count.rio.write_crs(CRS).rio.to_raster(CROP / f"{r}_radar_count_250m.tif", compress="deflate")
        df[["FILE", "FRAME", "LAT", "LON", "x", "y", "bed", "QUALITY"]].to_csv(
            CROP / f"{r}_radar_points.csv", index=False)

        filled = int((count.values > 0).sum())
        print(f"  {filled:,} of {count.size:,} grid cells have radar ({100 * filled / count.size:.2f}%)")
        print(f"  bed range {df.bed.min():.0f} to {df.bed.max():.0f} m")

    print(f"\nDone. New files are in {CROP}")


if __name__ == "__main__":
    main()
