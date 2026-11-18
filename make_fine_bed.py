"""
Step 15b: 10-20 m beds for the radar clutter simulation.

Takes your newest super-resolved beds (250 m, realizations_vN/) in a box you choose
and adds the bumps smaller than 500 m that a 250 m map can't hold:
  1. the 250 m bed is smoothly filled in to 10 or 20 m (nothing invented yet);
  2. random bumps narrower than 500 m are added, sized so that the bed's RMS height
     vs piece length follows the radar's trend (measure_fine_roughness.py) down to
     10-20 m; nothing wider is added, so the 250 m bed's hills, troughs and
     roughness are untouched;
  3. each of the 10 realizations gets its own random detail.
Run measure_fine_roughness.py first.

Runs on your PC, in glyproject.

Run (all settings optional, in any order):
    python make_fine_bed.py                                   # 20 m, 20 x 20 km box on fast ice, all realizations
    python make_fine_bed.py res=10 size=10 lat=-75.29 lon=-98.64
    python make_fine_bed.py region=getz res=20 size=30 x=-1500000 y=-600000 seeds=3

    res     cell size in metres (10-50; default 20)
    size    box width in km (default 20). Files: 20 km at 20 m ~ 4 MB, at 10 m ~ 16 MB each
    lat/lon or x/y   box centre (EPSG:3031 metres for x/y). Default: the fast-ice box used in the figures
    region  pig or getz (default pig)
    seeds   how many realizations (default all)

Saves, in fine_beds/:
    {region}_seed000_{res}m_{name}.tif ...    EPSG:3031, metres, same geoid reference as DEMOGORGN
    check_{region}_{res}m_{name}.png          shaded relief + roughness spectrum vs the radar
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from rasterio.transform import from_origin
from scipy.ndimage import map_coordinates

from make_figures import tif, pick_box, study_versions
from measure_fine_roughness import rms_vs_length

HERE = Path(__file__).resolve().parent
OUT = HERE / "fine_beds"
TAPER = (600.0, 400.0)       # m: no added detail above 600 m wavelength, full detail below 400 m


def args():
    a = {"res": 20.0, "size": 20.0, "region": "pig", "seeds": None}
    for s in sys.argv[1:]:
        k, _, v = s.partition("=")
        a[k.strip().lower()] = v.strip()
    a["res"], a["size"] = float(a["res"]), float(a["size"])
    if not 5 <= a["res"] <= 100:
        raise SystemExit("res should be between 5 and 100 m")
    return a


def centre(a, folder):
    if "lat" in a and "lon" in a:
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True).transform(
            float(a["lon"]), float(a["lat"]))
        return x, y
    if "x" in a and "y" in a:
        return float(a["x"]), float(a["y"])
    (x, y), _ = pick_box(a["region"], folder)
    return x, y


def fine_detail(n, res, H, rms1km, rng):
    """Random bumps (n x n, spacing res) narrower than ~500 m, with the radar's trend:
    RMS height of pieces of length L = rms1km * (L / 1 km)^H."""
    beta = 2 * H + 1                                             # matching slope of bumpiness by width
    k = np.fft.fftfreq(n, d=res / 1000.0)                       # cycles per km
    kk = np.hypot(k[:, None], k[None, :])
    kk[0, 0] = np.inf
    wl = 1000.0 / kk                                             # metres
    taper = np.clip((TAPER[0] - wl) / (TAPER[0] - TAPER[1]), 0, 1)
    taper = 0.5 - 0.5 * np.cos(np.pi * taper)
    amp = kk ** (-(beta + 1) / 2) * taper                        # 2D spectrum ~ k^-(beta+1)
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * amp))
    # scale so short pieces (where the 250 m bed adds almost nothing) follow the radar's trend
    Ls = np.array([L for L in (60, 80, 100, 150, 200) if L >= 4 * res], float)
    got = rms_vs_length(list(f[::4]) + list(f.T[::4]), res, Ls)
    want = rms1km * (Ls / 1000.0) ** H
    return f * np.median(want / got)


def write(path, arr, x0, y0, res):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:3031", transform=from_origin(x0, y0, res, res),
                       compress="deflate", nodata=np.nan) as dst:
        dst.write(arr.astype("float32"), 1)


def main():
    a = args()
    r, res, size = a["region"], a["res"], a["size"] * 1000
    fr = HERE / "fine_roughness.json"
    if not fr.exists():
        raise SystemExit("Run measure_fine_roughness.py first.")
    rough = json.loads(fr.read_text())[r]
    H, rms1km = rough["hurst"], rough["rms_1km_m"]
    label, folder = study_versions()[-1]
    cx, cy = centre(a, folder)
    files = sorted(folder.glob(f"{r}_seed*_250m.tif"))
    if a["seeds"]:
        files = files[:int(a["seeds"])]
    n = int(round(size / res))
    x0, y0 = cx - n * res / 2, cy + n * res / 2                  # top-left corner of the fine grid
    xs = x0 + res * (np.arange(n) + 0.5)
    ys = y0 - res * (np.arange(n) + 0.5)
    name = f"{cx / 1000:.0f}_{cy / 1000:.0f}km"
    OUT.mkdir(exist_ok=True)
    print(f"{label} -> {res:g} m, {size / 1000:g} x {size / 1000:g} km box centred at x={cx:.0f}, y={cy:.0f} "
          f"({n} x {n} cells, {len(files)} realizations)")
    print(f"radar trend: RMS = {rms1km:.1f} m x (piece length / 1 km)^{H:.2f}")

    first = None
    for f in files:
        bed = tif(f)
        sub = bed.sel(x=slice(cx - size / 2 - 2000, cx + size / 2 + 2000),
                      y=slice(cy + size / 2 + 2000, cy - size / 2 - 2000))
        if sub.size == 0:
            raise SystemExit("That box is outside the map.")
        z = sub.values.astype("float64")
        ok = np.isfinite(z)
        if not ok.all():
            z = np.where(ok, z, np.nanmean(z))
        # 1. smooth fill-in to the fine grid (cubic)
        ci = (xs - float(sub.x[0])) / 250.0
        ri = (float(sub.y[0]) - ys) / 250.0
        R, Cc = np.meshgrid(ri, ci, indexing="ij")
        base = map_coordinates(z, [R, Cc], order=3, mode="nearest")
        valid = map_coordinates(ok.astype(float), [R, Cc], order=1, mode="nearest") > 0.99
        # 2. random detail with the radar's roughness below 500 m (own random seed per realization)
        seed = int(str(f.stem).split("seed")[-1].split("_")[0])
        rng = np.random.default_rng([seed, int(abs(cx)) % 100000, int(abs(cy)) % 100000, int(res)])
        detail = fine_detail(n, res, H, rms1km, rng)
        out = np.where(valid, base + detail, np.nan)
        write(OUT / f"{r}_seed{seed:03d}_{res:g}m_{name}.tif", out, x0, y0, res)
        print(f"  seed {seed:03d}: saved; added detail RMS {np.std(detail):.1f} m")
        if first is None:
            first = (base, out)

    # check figure: shaded relief and roughness spectrum vs the radar
    base, out = first
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    zz = np.where(np.isfinite(out), out, np.nanmean(out))
    ax[0].imshow(LightSource(315, 40).shade(zz, cmap=plt.cm.gist_earth, vert_exag=2, dx=res, dy=res,
                                             blend_mode="soft"),
                 extent=(0, size / 1000, 0, size / 1000))
    ax[0].set_xlabel("km"); ax[0].set_ylabel("km")
    ax[0].set_title(f"{r} seed {files[0].stem.split('seed')[-1][:3]}, {res:g} m (shaded relief)")
    Ls = np.array([L for L in [40, 60, 80, 100, 150, 200, 300, 500, 700, 1000, 2000, 5000]
                   if 4 * res <= L <= size / 2], float)
    fill = lambda z: np.where(np.isfinite(z), z, np.nanmean(z))
    for z, lab, c in [(base, "250 m bed, filled in", "tab:blue"), (out, f"{res:g} m bed (this step)", "tab:red")]:
        zz = fill(z)
        ax[1].loglog(Ls, rms_vs_length(list(zz[::5]) + list(zz.T[::5]), res, Ls), "o-", color=c, ms=3,
                     lw=1.2, label=lab)
    Lr, rr = np.array(rough["lengths_m"]), np.array(rough["rms_m"])
    ax[1].loglog(Lr, rr, "s", color="black", ms=4, label="Radar picks")
    Lx = np.geomspace(4 * res, 5000, 100)
    ax[1].loglog(Lx, rms1km * (Lx / 1000) ** H, color="gray", ls="--", lw=1, label=f"Radar trend (H = {H:.2f})")
    ax[1].set_xlabel("Piece length (m)"); ax[1].set_ylabel("RMS height after removing slope (m)")
    ax[1].set_title("Roughness vs piece length")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"check_{r}_{res:g}m_{name}.png", dpi=150)
    plt.close(fig)
    print(f"\nDone. Files are in {OUT}/ (check the check_*.png: red should follow the radar's dashed trend)")


if __name__ == "__main__":
    main()
