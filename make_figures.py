"""
Realism figures and roughness numbers. Runs on your PC.

Needs (same folder, all made in earlier steps):
    realizations/ (v2) and any realizations_vN/ folders (v3, v4, v5, ...); all are picked up
    roughness.py
    demogorgn_real/{pig,getz}_seed*.tif
    cropped/{pig,getz}_bedmachine_bed.tif, cropped/{pig,getz}_radar_points.csv
    tiles/{pig,getz}_grids.npz          (ice-flow speed, for the 3D box)
    realbeds/pine_island_bay.tif        (real seafloor for the 3D comparison)

Run:
    python make_figures.py          # everything
    python make_figures.py 3d       # only redraw the 3D views (fast)

Saves in figures/:
    fig1_3d_{pig,getz}.png        3D views: DEMOGORGN, your newest bed, real seafloor
    fig2_profile_{pig,getz}.png   one flight line: radar vs DEMOGORGN, BedMachine, newest bed
    fig3_spectrum_{pig,getz}.png  roughness at each length scale near the radar lines
    fig4_roughness_percent.png    bump size (0.5-1 km and 1-4.5 km) as % of the real bed
    roughness_table.csv           the numbers

How roughness is measured (figs 3, 4):
    The RADAR is read along its flight lines: picks averaged into 250 m steps,
    cut into 20 km (fig 3) or 9 km (fig 4) windows.
    Each BED MAP is read straight off a 250 m grid (500 m maps are put on the
    same 250 m grid by interpolation), in a square patch centred on each radar
    window, along its rows and columns. No interpolation along diagonal flight
    lines, which would blur the finest bumps of any 250 m map by up to ~60%.
    This is exactly how training measures roughness, so training and the maps agree.
    Spectra are detrended and Hann-windowed; "% of the real bed" compares bump
    size (amplitude) with the radar's, over the same windows.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.colors import LightSource
from rasterio.enums import Resampling

from roughness import psd_1d, psd_patches_np, wavelengths_km, K_LO, K_HI, N_SPEC

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
STEP = 250.0          # m
WIN = 80              # 20 km windows for the spectrum figure
CUT_KM = 5.0
BOX = 20_000          # m, size of the 3D views
VEX = 10
NAMES = {"pig": "Pine Island", "getz": "Getz"}
BASE_COL = {"radar": "black", "DEMOGORGN": "tab:blue", "BedMachine": "tab:orange"}
OLD_SHADES = ["#f5c6c6", "#eb9c9c", "#dd7474", "#c95050", "#b33a3a"]


# ------------------------------------------------------------------ versions of your beds
def study_versions():
    """[(label, folder)] for v2 (realizations/) and every realizations_vN/, oldest first."""
    out = []
    if (HERE / "realizations").exists():
        out.append(("This study v2", HERE / "realizations"))
    nums = sorted(int(m.group(1)) for p in HERE.glob("realizations_v*")
                  if p.is_dir() and (m := re.fullmatch(r"realizations_v(\d+)", p.name)))
    out += [(f"This study v{n}", HERE / f"realizations_v{n}") for n in nums]
    return out


def colours(versions):
    col = dict(BASE_COL)
    for k, (label, _) in enumerate(versions):
        col[label] = "tab:red" if k == len(versions) - 1 else OLD_SHADES[k % len(OLD_SHADES)]
    return col


def tif(path):
    return rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)


def sample(da, x, y):
    return da.interp(x=xr.DataArray(x, dims="p"), y=xr.DataArray(y, dims="p")).values


# ------------------------------------------------------------------ radar lines
def radar_tracks(r, min_len):
    df = pd.read_csv(HERE / "cropped" / f"{r}_radar_points.csv")
    tracks = []
    for _, g in df.groupby("FILE", sort=False):
        x, y, z = g.x.values, g.y.values, g.bed.values
        for s in np.split(np.arange(len(x)), np.where(np.hypot(np.diff(x), np.diff(y)) > 500)[0] + 1):
            if len(s) < 2:
                continue
            d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x[s]), np.diff(y[s])))])
            agg = pd.DataFrame({"b": (d // STEP).astype(int), "x": x[s], "y": y[s], "z": z[s]}).groupby("b").mean()
            idx = agg.index.values
            for part in np.split(np.arange(len(idx)), np.where(np.diff(idx) > 1)[0] + 1):
                if len(part) >= min_len:
                    tracks.append(agg.iloc[part][["x", "y", "z"]].to_numpy())
    return tracks


def windows(tracks, n):
    return [t[k:k + n] for t in tracks for k in range(0, len(t) - n + 1, n)]


# ------------------------------------------------------------------ reading maps on the grid
class Grid:
    """The 250 m grid of your realizations; every map is read on it."""

    def __init__(self, ref):
        self.ref = ref
        self.x0, self.y0 = float(ref.x[0]), float(ref.y[0])
        self.ny, self.nx = ref.shape

    def values(self, da):
        return da.interp(x=self.ref.x, y=self.ref.y).values.astype("float32")

    def patches(self, arr, wins, n):
        """n x n patch centred on each window; None where it falls off the map or has gaps."""
        out = []
        for w in wins:
            cx, cy = w[:, 0].mean(), w[:, 1].mean()
            ci = int(round((cx - self.x0) / STEP))
            ri = int(round((self.y0 - cy) / STEP))
            r0, c0 = ri - n // 2, ci - n // 2
            if r0 < 0 or c0 < 0 or r0 + n > self.ny or c0 + n > self.nx:
                out.append(None)
                continue
            p = arr[r0:r0 + n, c0:c0 + n]
            out.append(p if np.all(np.isfinite(p)) else None)
        return out


def spectra_series(series):
    """Radar: mean spectrum of detrended, Hann-windowed 1D series (length WIN)."""
    t = np.arange(WIN)
    h = np.hanning(WIN)
    psd = []
    for s in series:
        s = s - np.polyval(np.polyfit(t, s, 1), t)
        psd.append(np.abs(np.fft.rfft(s * h)) ** 2 / (h ** 2).sum())
    return np.mean(psd, axis=0)


def spectra_patches(P):
    """Maps: the same spectrum, along every row and column of WIN x WIN patches."""
    h = np.hanning(WIN)
    tc = np.arange(WIN) - (WIN - 1) / 2

    def rows(Z):
        Z = Z.reshape(-1, WIN).astype("float64")
        Z = Z - Z.mean(1, keepdims=True)
        Z = Z - ((Z * tc).sum(1, keepdims=True) / (tc ** 2).sum()) * tc
        return (np.abs(np.fft.rfft(Z * h, axis=1)) ** 2 / (h ** 2).sum()).mean(0)

    P = np.asarray(P)
    return 0.5 * (rows(P) + rows(np.swapaxes(P, -1, -2)))


def rough(freq, psd):
    sel = freq > 1.0 / CUT_KM
    return np.sqrt(2 * psd[sel].sum() / WIN)


# ------------------------------------------------------------------ numbers + spectrum figure
def analyse(r, grid, products, col, latest):
    tracks80 = radar_tracks(r, WIN)
    w80 = windows(tracks80, WIN)
    w36 = windows(radar_tracks(r, N_SPEC), N_SPEC)
    print(f"  radar: {len(w80)} windows of 20 km, {len(w36)} windows of 9 km")

    # windows every product covers, so all are compared on the same ground
    firsts = {name: grid.values(layers[0]) for name, layers in products.items()}
    ok80 = [all(p is not None for p in ps) for ps in zip(*[grid.patches(a, w80, WIN) for a in firsts.values()])]
    ok36 = [all(p is not None for p in ps) for ps in zip(*[grid.patches(a, w36, N_SPEC) for a in firsts.values()])]
    w80 = [w for w, k in zip(w80, ok80) if k]
    w36 = [w for w, k in zip(w36, ok36) if k]
    print(f"  used (covered by every map): {len(w80)} x 20 km, {len(w36)} x 9 km")
    del firsts

    freq = np.fft.rfftfreq(WIN, d=STEP / 1000.0)
    p_radar = spectra_series([w[:, 2] for w in w80])
    pr36, _ = psd_1d([w[:, 2] for w in w36])
    wl36 = wavelengths_km()
    small = [k for k in range(K_LO, K_HI + 1) if wl36[k - 1] < 1.0]
    large = [k for k in range(K_LO, K_HI + 1) if wl36[k - 1] >= 1.0]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    wl = 1 / freq[1:]
    ax.loglog(wl, p_radar[1:], color=col["radar"], lw=2.8, label=f"Radar (IceBridge), {len(w80)} x 20 km")
    rows = []
    for name, layers in products.items():
        curves, a_s, a_l, a_all = [], [], [], []
        for da in layers:
            arr = grid.values(da)
            P80 = np.stack(grid.patches(arr, w80, WIN))
            P36 = np.stack(grid.patches(arr, w36, N_SPEC))
            p = spectra_patches(P80)
            curves.append(p)
            a_all.append(100 * rough(freq, p) / rough(freq, p_radar))
            p36 = psd_patches_np(P36)
            a_s.append(100 * np.sqrt(p36[small].sum() / pr36[small].sum()))
            a_l.append(100 * np.sqrt(p36[large].sum() / pr36[large].sum()))
            del arr, P80, P36
        curves = np.array(curves)
        is_study = name.startswith("This study")
        lw = 2.2 if (not is_study or name == latest) else 1.0
        ls = "-" if (not is_study or name == latest) else "--"
        if len(curves) > 1:
            ax.fill_between(wl, curves.min(0)[1:], curves.max(0)[1:], color=col[name], alpha=0.2, lw=0)
        ax.loglog(wl, curves.mean(0)[1:], color=col[name], lw=lw, ls=ls,
                  label=name + (f" ({len(layers)} realizations)" if len(layers) > 1 else ""))
        rows.append({"region": NAMES[r], "product": name,
                     "pct_0.5_to_1km": np.mean(a_s), "pct_0.5_to_1km_min": np.min(a_s),
                     "pct_0.5_to_1km_max": np.max(a_s),
                     "pct_1_to_4.5km": np.mean(a_l), "pct_under_5km": np.mean(a_all)})
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (km)  ← big bumps · small bumps →")
    ax.set_ylabel("Roughness power (m² per cycle/km)")
    ax.set_title(f"{NAMES[r]}: bed roughness by bump size, near radar lines")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig3_spectrum_{r}.png", dpi=150)
    plt.close(fig)
    return rows, tracks80


# ------------------------------------------------------------------ accuracy
def accuracy(r, products):
    """Typical difference from the radar picks (RMS, metres), at every 250 m radar
    point covered by all maps. Lower = closer to the real bed. DEMOGORGN was fitted
    to this radar, so it sets the bar; the aim is to stay close to it."""
    pts = np.concatenate(radar_tracks(r, 2))
    vals = {name: [sample(da, pts[:, 0], pts[:, 1]) for da in layers] for name, layers in products.items()}
    ok = np.all([np.isfinite(v).all(0) for v in vals.values()], axis=0)
    out = {name: float(np.mean([np.sqrt(np.mean((v[ok] - pts[ok, 2]) ** 2)) for v in vs]))
           for name, vs in vals.items()}
    print(f"  accuracy: {ok.sum():,} radar points (250 m steps) covered by every map")
    return out


# ------------------------------------------------------------------ other figures
def fig_profile(r, tracks, products, col, latest):
    t = max(tracks, key=len)[:240]
    d = np.arange(len(t)) * STEP / 1000
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d, t[:, 2], ".", color=col["radar"], ms=3, label="Radar (IceBridge)")
    for name in ["DEMOGORGN", "BedMachine", latest]:
        s = np.array([sample(da, t[:, 0], t[:, 1]) for da in products[name]])
        if len(s) > 1:
            ax.fill_between(d, s.min(0), s.max(0), color=col[name], alpha=0.25, lw=0)
        ax.plot(d, s[0], color=col[name], lw=1.5, label=name + (" (seed 000 + range)" if len(s) > 1 else ""))
    ax.set_xlabel("Distance along flight line (km)")
    ax.set_ylabel("Bed elevation (m above sea level)")
    ax.set_title(f"{NAMES[r]}: bed along one IceBridge flight line")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig2_profile_{r}.png", dpi=150)
    plt.close(fig)


def pick_box(r, folder):
    g = np.load(HERE / "tiles" / f"{r}_grids.npz")
    x0, y0, W2 = float(g["x0"]), float(g["y0"]), g["W2"]
    speed = np.hypot(W2[0], W2[1])
    mask = tif(folder / f"{r}_grounded_250m.tif")
    best, best_c = -1, None
    for cx in np.arange(float(mask.x.min()) + BOX, float(mask.x.max()) - BOX, BOX / 2):
        for cy in np.arange(float(mask.y.min()) + BOX, float(mask.y.max()) - BOX, BOX / 2):
            m = mask.sel(x=slice(cx - BOX / 2, cx + BOX / 2), y=slice(cy + BOX / 2, cy - BOX / 2))
            if m.size == 0 or float(m.min()) < 1:
                continue
            i, j = int((y0 - cy) // 500), int((cx - x0) // 500)
            s = speed[max(i - 20, 0):i + 20, max(j - 20, 0):j + 20].mean()
            if s > best:
                best, best_c = s, (cx, cy)
    return best_c, best


def detrended(z):
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ok = np.isfinite(z)
    A = np.c_[np.ones(ok.sum()), xx[ok], yy[ok]]
    coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    return z - (coef[0] + coef[1] * xx + coef[2] * yy)


def seafloor_box():
    p = HERE / "realbeds" / "pine_island_bay.tif"
    if not p.exists():
        return None
    da = tif(p)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    z = da.astype("float32").rio.reproject("EPSG:3031", resolution=250, resampling=Resampling.average).values
    n, cands = int(BOX / 250), []
    for i in range(0, z.shape[0] - n, n // 2):
        for j in range(0, z.shape[1] - n, n // 2):
            w = z[i:i + n, j:j + n]
            if np.isfinite(w).mean() < 0.99 or np.nanmean(w) > -600:
                continue
            w = detrended(np.where(np.isfinite(w), w, np.nanmean(w)))
            gy, gx = np.gradient(w)
            T = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
            ev = np.linalg.eigvalsh(T)
            cands.append((np.std(w), ev[1] / max(ev[0], 1e-9), w))
    if not cands:
        return None
    relief = np.array([c[0] for c in cands])
    lo, hi = np.percentile(relief, [25, 75])
    mid = [c for c in cands if lo <= c[0] <= hi] or cands
    return max(mid, key=lambda c: c[1])[2]


def surface(ax, z, res, zr, title):
    X, Y = np.meshgrid(np.arange(z.shape[1]) * res / 1000, np.arange(z.shape[0]) * res / 1000)
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(z, cmap=plt.cm.gist_earth, vert_exag=VEX, dx=res, dy=res, blend_mode="soft", vmin=-zr, vmax=zr)
    ax.plot_surface(X, Y, z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, VEX * 2 * zr / BOX))
    ax.set_zlim(-zr, zr)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("km", fontsize=7); ax.set_ylabel("km", fontsize=7); ax.set_zlabel("m", fontsize=7)
    ax.tick_params(labelsize=6)


def fig_3d(r, sea, latest, folder):
    c, sp = pick_box(r, folder)
    if c is None:
        print("  3D: no fully grounded 20 km box found, skipped")
        return
    cx, cy = c
    cut = lambda da: da.sel(x=slice(cx - BOX / 2, cx + BOX / 2), y=slice(cy + BOX / 2, cy - BOX / 2))
    sr_da = cut(tif(sorted(folder.glob(f"{r}_seed*_250m.tif"))[0]))
    # DEMOGORGN drawn on the same 250 m grid, smoothly filled in between its 500 m
    # points (the same way the roughness numbers read it), so both panels are
    # drawn with the same number of points
    dem_da = tif(sorted((HERE / "demogorgn_real").glob(f"{r}_seed*.tif"))[0])
    dem = detrended(dem_da.interp(x=sr_da.x, y=sr_da.y).values)
    sr = detrended(sr_da.values)
    panels = [(dem, 250, "DEMOGORGN realization (500 m, smoothly filled in to 250 m)"),
              (sr, 250, f"{latest}, super-resolved (250 m)")]
    if sea is not None:
        panels.append((detrended(sea), 250, "Real seafloor, Pine Island Bay (250 m)"))
    zr = max(np.nanpercentile(np.abs(p[0]), 99) for p in panels)
    # second row: only the small bumps (what's left after removing a 1 km-wide smooth
    # version), which are only ~10 m tall and so invisible next to the 300 m hills above
    from scipy.ndimage import gaussian_filter
    small = [z - gaussian_filter(np.where(np.isfinite(z), z, np.nanmean(z)), 2.0) for z, _, _ in panels]
    sr_lim = max(np.nanpercentile(np.abs(q[4:-4, 4:-4]), 99) for q in small)
    n = len(panels)
    fig = plt.figure(figsize=(4.2 * n, 8.0))
    for k, (z, res, title) in enumerate(panels):
        surface(fig.add_subplot(2, n, k + 1, projection="3d"), z, res, zr, title)
    for k, (q, (_, res, title)) in enumerate(zip(small, panels)):
        ax = fig.add_subplot(2, n, n + k + 1)
        im = ax.imshow(q[4:-4, 4:-4], cmap="RdBu_r", vmin=-sr_lim, vmax=sr_lim,
                       extent=(1, BOX / 1000 - 1, 1, BOX / 1000 - 1))
        rms = np.nanstd(q[4:-4, 4:-4])
        ax.set_title(f"small bumps only (< ~1.5 km): RMS {rms:.1f} m", fontsize=9)
        ax.set_xlabel("km", fontsize=7); ax.set_ylabel("km", fontsize=7); ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=fig.axes[n:], fraction=0.02, pad=0.01)
    cb.set_label("height above / below the smoothed bed (m)", fontsize=8)
    fig.suptitle(f"{NAMES[r]}, fast-flowing ice (~{sp:.0f} m/yr) · 20 × 20 km · regional tilt removed · "
                 f"vertical exaggeration {VEX}×", fontsize=10)
    fig.savefig(OUT / f"fig1_3d_{r}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_bars(table, col):
    regions = list(table.region.unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(1.3 * table["product"].nunique() * len(regions) + 2, 4.4),
                             sharey=True, squeeze=False)
    for ax, reg in zip(axes[0], regions):
        t = table[table.region == reg]
        x = np.arange(len(t))
        wbar = 0.38
        for k, (c, hatch) in enumerate([("pct_0.5_to_1km", "//"), ("pct_1_to_4.5km", "")]):
            xs = x + (k - 0.5) * wbar
            ax.bar(xs, t[c], wbar, color=[col[p] for p in t["product"]], hatch=hatch, edgecolor="white")
            for xi, v in zip(xs, t[c]):
                ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(100, color="black", ls="--", lw=1)
        ax.set_xticks(x, [p.replace("This study ", "This study\n") for p in t["product"]], fontsize=8)
        ax.set_title(reg)
    axes[0][0].set_ylabel("Bump size, % of the real (radar) bed")
    h = [plt.Rectangle((0, 0), 1, 1, facecolor="gray", hatch="//", edgecolor="white"),
         plt.Rectangle((0, 0), 1, 1, facecolor="gray")]
    axes[0][-1].legend(h, ["bumps 0.5-1 km", "bumps 1-4.5 km"], fontsize=8, frameon=False, loc="upper left")
    ymax = float(np.nanmax(table[["pct_0.5_to_1km", "pct_1_to_4.5km"]].values))
    axes[0][0].set_ylim(0, max(125, ymax * 1.15))
    fig.suptitle("How much of the real (radar) bed roughness each map reproduces (100% = as rough as the real bed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_roughness_percent.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    versions = study_versions()
    if not versions:
        raise SystemExit("No realizations folders found.")
    col = colours(versions)
    latest, latest_folder = versions[-1]
    print(f"your bed versions: {', '.join(v for v, _ in versions)} (newest: {latest})")
    print("real seafloor box for 3D views ...")
    sea = seafloor_box()
    if "3d" in [a.lower() for a in sys.argv[1:]]:          # python make_figures.py 3d
        for r in ["pig", "getz"]:
            fig_3d(r, sea, latest, latest_folder)
        print(f"3D figures saved in {OUT}")
        return
    rows = []
    for r in ["pig", "getz"]:
        print(f"\n{NAMES[r]}")
        products = {
            "DEMOGORGN": [tif(p) for p in sorted((HERE / "demogorgn_real").glob(f"{r}_seed*.tif"))],
            "BedMachine": [tif(HERE / "cropped" / f"{r}_bedmachine_bed.tif")],
        }
        for label, folder in versions:
            files = sorted(folder.glob(f"{r}_seed*_250m.tif"))
            if files:
                products[label] = [tif(p) for p in files]
        grid = Grid(tif(latest_folder / f"{r}_grounded_250m.tif"))
        new_rows, tracks80 = analyse(r, grid, products, col, latest)
        err = accuracy(r, products)
        for row in new_rows:
            row["error_vs_radar_m"] = err[row["product"]]
        rows += new_rows
        fig_profile(r, tracks80, products, col, latest)
        fig_3d(r, sea, latest, latest_folder)
    table = pd.DataFrame(rows)
    table.round(1).to_csv(OUT / "roughness_table.csv", index=False)
    fig_bars(table, col)
    print("\nRealism: bump size as % of the real (radar) bed (100% = as rough as the real bed)")
    print("Accuracy: typical difference from the radar picks, in metres (lower = closer to the real bed)")
    print(table[["region", "product", "pct_0.5_to_1km", "pct_1_to_4.5km", "error_vs_radar_m"]]
          .round(1).to_string(index=False))
    print(f"\nFigures saved in {OUT}")


if __name__ == "__main__":
    main()
