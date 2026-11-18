"""
Save individual DEMOGORGN realizations (not just the mean) for Pine Island and Getz.

Why: the mean of 100 realizations is much smoother than any single one, because
averaging cancels out their simulated roughness. For realism we want to sharpen
single realizations, which is also what the englacial radar-sounder-simulator
uses (one seed at a time).

Run:
    python get_demogorgn_realizations.py

Saves demogorgn_real/{pig,getz}_seed{000..009}.tif, geoid-referenced, EPSG:3031,
500 m, on the same grid as demogorgn_{pig,getz}_mean.tif.
"""

from pathlib import Path

import icechunk as ic
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

URL = "https://data.source.coop/englacial/demogorgn/icechunk/realizations.icechunk"
CRS = "EPSG:3031"
N_SEEDS = 10
HERE = Path(__file__).resolve().parent
OUT = HERE / "demogorgn_real"


def main():
    repo = ic.Repository.open(ic.http_storage(URL))
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    da = ds["realizations"] if "realizations" in ds else ds[list(ds.data_vars)[0]]
    rdim = [d for d in da.dims if d not in ("x", "y")][0]
    seeds = ds["seed_id"].values if "seed_id" in ds.variables else np.arange(da.sizes[rdim])
    print(f"{da.sizes[rdim]} realizations available, saving the first {N_SEEDS}")
    OUT.mkdir(exist_ok=True)

    for r in ["pig", "getz"]:
        ref = rioxarray.open_rasterio(HERE / f"demogorgn_{r}_mean.tif").squeeze("band", drop=True)
        xs, ys = ref.x.values, ref.y.values
        crop = da.sel(x=slice(xs.min(), xs.max()),
                      y=slice(ys.max(), ys.min()) if da.y[0] > da.y[-1] else slice(ys.min(), ys.max()))
        for k in range(N_SEEDS):
            one = crop.isel({rdim: k}).load()
            if one.y[0] < one.y[-1]:
                one = one.isel(y=slice(None, None, -1))
            one = one.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs(CRS)
            path = OUT / f"{r}_seed{int(seeds[k]):03d}.tif"
            one.astype("float32").rio.to_raster(path, compress="deflate")
            print(f"  {path.name}  {one.shape}  bed {float(one.min()):.0f} to {float(one.max()):.0f} m")
    print("Done.")


if __name__ == "__main__":
    main()
