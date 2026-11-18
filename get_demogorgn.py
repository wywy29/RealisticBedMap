"""
Step 3: get DEMOGORGN bed for Pine Island and Getz.

Setup (once, in a terminal):
    pip install -U icechunk zarr xarray rioxarray pyproj

Run:
    python get_demogorgn.py

It opens the DEMOGORGN cloud store, prints what's inside, then crops the
bed to Pine Island and Getz and saves:
    demogorgn_pig_mean.tif   demogorgn_pig_std.tif
    demogorgn_getz_mean.tif  demogorgn_getz_std.tif
Only the chunks inside the two boxes are downloaded, not all of Antarctica.

If it stops with an error, copy everything it printed and send it over.
"""

import icechunk as ic
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer

URL = "https://data.source.coop/englacial/demogorgn/icechunk/realizations.icechunk"
CRS = "EPSG:3031"

# Same boxes used for the IceBridge downloads (lat/lon).
BOXES = {
    "pig":  {"lat": (-78.0, -74.0), "lon": (-105.0, -85.0)},
    "getz": {"lat": (-77.0, -73.5), "lon": (-136.0, -114.0)},
}


def open_store():
    try:
        storage = ic.http_storage(URL)
    except AttributeError:
        raise SystemExit("Your icechunk is too old. Run: pip install -U icechunk")
    repo = ic.Repository.open(storage)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def latlon_box_to_xy(lat, lon):
    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = [], []
    for i in range(21):
        for la in lat:
            x, y = t.transform(lon[0] + (lon[1] - lon[0]) * i / 20, la)
            xs.append(x); ys.append(y)
        for lo in lon:
            x, y = t.transform(lo, lat[0] + (lat[1] - lat[0]) * i / 20)
            xs.append(x); ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def pick_bed_variable(ds):
    names = list(ds.data_vars)
    for n in names:
        if "bed" in n.lower():
            return n
    if len(names) == 1:
        return names[0]
    raise SystemExit(f"\nCan't tell which variable is the bed. Variables: {names}\n"
                     "Send this output over and the script can be pointed at the right one.")


def main():
    ds = open_store()
    print("=== DEMOGORGN contents ===")
    print(ds)
    print("==========================\n")

    var = pick_bed_variable(ds)
    da = ds[var]
    print(f"Using variable: {var}  dims={da.dims}")

    if "x" not in da.dims or "y" not in da.dims:
        raise SystemExit("No x/y dimensions found; send the printed contents over.")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    if da.rio.crs is None:
        da = da.rio.write_crs(CRS)

    # The realization dimension is whatever isn't x or y.
    extra = [d for d in da.dims if d not in ("x", "y")]
    print(f"Realization dimension(s): {extra or 'none'}")

    for name, b in BOXES.items():
        xmin, ymin, xmax, ymax = latlon_box_to_xy(b["lat"], b["lon"])
        ysl = slice(ymax, ymin) if da.y[0] > da.y[-1] else slice(ymin, ymax)
        crop = da.sel(x=slice(xmin, xmax), y=ysl)
        print(f"\n{name}: cropping to {crop.sizes} (downloading, may take a while)...")
        crop = crop.load()
        if extra:
            mean, std = crop.mean(extra), crop.std(extra)
            std.rio.write_crs(CRS).rio.to_raster(f"demogorgn_{name}_std.tif")
        else:
            mean = crop
        mean.rio.write_crs(CRS).rio.to_raster(f"demogorgn_{name}_mean.tif")
        print(f"  saved demogorgn_{name}_mean.tif" + (" and _std.tif" if extra else ""))

    print("\nDone.")


if __name__ == "__main__":
    main()
