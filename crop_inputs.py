"""
Step 4: crop REMA, ice velocity and BedMachine to the same areas as DEMOGORGN.

Setup (once):
    pip install netCDF4

Folder layout it expects (same folder as this script):
    demogorgn_pig_mean.tif, demogorgn_getz_mean.tif   (from step 3)
    raw/NSIDC-0484_2-.../   (ice velocity, a .nc file inside)
    raw/NSIDC-0756_4-.../   (BedMachine, a .nc file inside)
    raw/rema_mosaic_100m_v2.0_filled_cop30_dem.tif

Run:
    python crop_inputs.py

Output goes to cropped/:
    {region}_rema_250m.tif         surface elevation, 250 m
    {region}_vx_250m.tif, _vy_     ice velocity components, 250 m
    {region}_speed_250m.tif        ice speed, 250 m
    {region}_bedmachine_bed.tif    BedMachine bed, 500 m (same grid as DEMOGORGN)
    {region}_bedmachine_mask.tif   0 ocean, 1 rock, 2 grounded ice, 3 floating ice
    {region}_demogorgn_mean.tif / _std.tif   copied in so everything is in one place
where region is "pig" or "getz".

The 250 m grid lines up exactly with DEMOGORGN's 500 m grid (each 500 m
cell = 2 x 2 cells at 250 m), which is what the super-resolution model needs.
"""

import shutil
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "cropped"
CRS = "EPSG:3031"
REGIONS = ["pig", "getz"]
MARGIN = 5000  # m, read a bit extra around each region before resampling


def find_one(pattern):
    hits = sorted(RAW.glob(pattern))
    if not hits:
        raise SystemExit(f"Couldn't find {pattern} inside {RAW}")
    return hits[0]


def get_var(ds, name):
    """Find a variable regardless of upper/lower case."""
    for v in ds.data_vars:
        if v.lower() == name.lower():
            return ds[v]
    raise SystemExit(f"No variable '{name}'. Variables are: {list(ds.data_vars)}")


def force_crs(da):
    """Set EPSG:3031 explicitly, dropping any half-finished projection info.
    The velocity and BedMachine files are both in EPSG:3031 but don't
    always store it in a way rioxarray can read."""
    da = da.copy()
    da.attrs.pop("grid_mapping", None)
    da.encoding.pop("grid_mapping", None)
    for c in ("spatial_ref", "mapping", "coord_system", "crs", "polar_stereographic"):
        if c in da.coords:
            da = da.drop_vars(c)
    return da.rio.write_crs(CRS)


def prep(da):
    """Make a NetCDF variable look like a georeferenced raster."""
    da = da.squeeze(drop=True).rio.set_spatial_dims(x_dim="x", y_dim="y")
    return force_crs(da)


def grid_250(tmpl500):
    """A 250 m grid covering exactly the same area as the 500 m DEMOGORGN grid."""
    xmin, ymin, xmax, ymax = tmpl500.rio.bounds()
    x = np.arange(xmin + 125, xmax, 250.0)
    y = np.arange(ymax - 125, ymin, -250.0)
    g = xr.DataArray(np.zeros((len(y), len(x)), dtype="float32"),
                     dims=("y", "x"), coords={"y": y, "x": x})
    return g.rio.write_crs(CRS)


def clip(da, tmpl):
    xmin, ymin, xmax, ymax = tmpl.rio.bounds()
    out = da.rio.clip_box(xmin - MARGIN, ymin - MARGIN, xmax + MARGIN, ymax + MARGIN)
    return out if out.rio.crs is not None else force_crs(out)


def save(da, name):
    path = OUT / name
    da.astype("float32").rio.to_raster(path, compress="deflate")
    v = da.values
    print(f"  {name:32s} {da.shape}  min {np.nanmin(v):8.0f}  max {np.nanmax(v):8.0f}"
          f"  missing {100 * np.isnan(v).mean():4.1f}%")


def main():
    OUT.mkdir(exist_ok=True)

    rema = rioxarray.open_rasterio(find_one("rema*_dem.tif"), masked=True).squeeze("band", drop=True)
    vel = xr.open_dataset(next(find_one("NSIDC-0484*").glob("*.nc")))
    bm = xr.open_dataset(next(find_one("NSIDC-0756*").glob("*.nc")))
    vx, vy = prep(get_var(vel, "VX")), prep(get_var(vel, "VY"))
    bed, mask = prep(get_var(bm, "bed")), prep(get_var(bm, "mask"))

    for r in REGIONS:
        print(f"\n{r}")
        mean_src = HERE / f"demogorgn_{r}_mean.tif"
        if not mean_src.exists():
            raise SystemExit(f"Missing {mean_src.name}; run get_demogorgn.py first.")
        t500 = rioxarray.open_rasterio(mean_src, masked=True).squeeze("band", drop=True)
        t250 = grid_250(t500)

        for k in ("mean", "std"):
            src = HERE / f"demogorgn_{r}_{k}.tif"
            if src.exists():
                shutil.copy(src, OUT / f"{r}_demogorgn_{k}.tif")

        # REMA: 100 m -> 250 m, averaging the 100 m cells inside each 250 m cell
        save(clip(rema, t250).rio.reproject_match(t250, resampling=Resampling.average),
             f"{r}_rema_250m.tif")

        # Velocity: 450 m -> 250 m, smooth interpolation
        vx_r = clip(vx, t250).astype("float32").rio.reproject_match(t250, resampling=Resampling.bilinear)
        vy_r = clip(vy, t250).astype("float32").rio.reproject_match(t250, resampling=Resampling.bilinear)
        save(vx_r, f"{r}_vx_250m.tif")
        save(vy_r, f"{r}_vy_250m.tif")
        save(np.hypot(vx_r, vy_r), f"{r}_speed_250m.tif")

        # BedMachine: onto DEMOGORGN's exact 500 m grid
        save(clip(bed, t500).astype("float32").rio.reproject_match(t500, resampling=Resampling.bilinear),
             f"{r}_bedmachine_bed.tif")
        save(clip(mask, t500).astype("float32").rio.reproject_match(t500, resampling=Resampling.nearest),
             f"{r}_bedmachine_mask.tif")

    print(f"\nDone. Everything is in {OUT}")


if __name__ == "__main__":
    main()
