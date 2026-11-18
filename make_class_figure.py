"""
Class figure: DEMOGORGN vs this study, in 3D, Pine Island Glacier.
    a  DEMOGORGN realization (500 m), 3D
    b  this study: the same realization super-resolved (250 m), 3D

Runs on your PC, in glyproject (same folder as make_figures.py).
Needs cartopy once:   pip install cartopy

Run:
    python make_class_figure.py

Saves figures/class_figure.png (600 dpi) and figures/class_figure.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap

from make_figures import HERE, BOX, tif, pick_box, detrended, study_versions

LABEL, FOLDER = study_versions()[-1]          # your newest version
OUT = HERE / "figures"
PS = ccrs.SouthPolarStereo(true_scale_latitude=-71)      # = EPSG:3031
PC = ccrs.PlateCarree()
CMAP = LinearSegmentedColormap.from_list("bed", plt.cm.gist_earth(np.linspace(0.18, 0.92, 256)))
OCEAN, SHELF, ROCK = "#9ecae1", "#eef4f8", "#8c8c8c"
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
VEX = 8          # vertical exaggeration of the 3D views

# place names (lat, lon); positions approximate, nudge if a label sits off its feature
PLACES = [("Pine Island\nGlacier", -75.35, -98.0), ("Pine Island\nIce Shelf", -75.05, -101.2),
          ("Pine Island\nBay", -74.55, -103.0), ("Thwaites\nGlacier", -75.5, -106.75)]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.titlesize": 8.5, "pdf.fonttype": 42})


def panel_map(fig, ax, bx, by, speed):
    ref = tif(FOLDER / "pig_grounded_250m.tif")
    xs, ys = ref.x.values[::2], ref.y.values[::2]                 # 500 m is plenty for this view
    X, Y = xr.DataArray(xs, dims="x"), xr.DataArray(ys, dims="y")
    bed = tif(sorted(FOLDER.glob("pig_seed*_250m.tif"))[0]).interp(x=X, y=Y).values
    mpath = HERE / "cropped" / "pig_bedmachine_mask.tif"
    if mpath.exists():      # BedMachine: 0 ocean, 1 ice-free land, 2 grounded, 3 floating
        m = tif(mpath).interp(x=X, y=Y, method="nearest").values
    else:
        m = np.where(tif(FOLDER / "pig_grounded_250m.tif").interp(x=X, y=Y).values > 0.5, 2, 0)
    g = m == 2
    shade = LightSource(315, 40).hillshade(np.where(np.isfinite(bed), bed, np.nanmean(bed)),
                                           vert_exag=3, dx=500, dy=500)
    img = np.empty(bed.shape + (3,))
    img[:] = matplotlib.colors.to_rgb(OCEAN)
    img[m == 3] = matplotlib.colors.to_rgb(SHELF)
    img[m == 1] = matplotlib.colors.to_rgb(ROCK)
    grey = 0.55 + 0.4 * shade
    img[g] = np.stack([grey, grey, grey], -1)[g]
    ext = (xs[0] - 250, xs[-1] + 250, ys[-1] - 250, ys[0] + 250)
    ax.set_extent(ext, PS)
    ax.imshow(img, extent=ext, origin="upper", transform=PS, interpolation="nearest")
    ax.contour(xs, ys, g.astype(float), levels=[0.5], colors="black", linewidths=0.8, transform=PS)
    ax.contour(xs, ys, (m >= 2).astype(float), levels=[0.5], colors="#4a6b82", linewidths=0.4, transform=PS)
    for ec, lw in [("white", 2.2), ("#d7301f", 1.2)]:
        ax.add_patch(mpatches.Rectangle((bx - BOX / 2, by - BOX / 2), BOX, BOX, transform=PS, fill=False,
                                        ec=ec, lw=lw, zorder=5))
    ax.text(bx + BOX / 2 + 5e3, by, "b, c", transform=PS, fontsize=7, fontweight="bold", color="#d7301f",
            va="center", zorder=6, path_effects=HALO)
    for txt, lat, lon in PLACES:
        px, py = PS.transform_point(lon, lat, PC)
        if ext[0] < px < ext[1] and ext[2] < py < ext[3]:
            ax.text(lon, lat, txt, transform=PC, fontsize=5.8, ha="center", va="center", style="italic",
                    zorder=6, path_effects=HALO)
    gl = ax.gridlines(crs=PC, draw_labels=True, x_inline=False, y_inline=False, color="#333333", lw=0.3,
                      alpha=0.6, linestyle="--", xlocs=np.arange(-180, 181, 4), ylocs=np.arange(-85, -60, 1))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6}
    x0, y0 = ext[0] + 0.07 * (ext[1] - ext[0]), ext[2] + 0.05 * (ext[3] - ext[2])
    ax.plot([x0, x0 + 50e3], [y0, y0], color="black", lw=2, transform=PS, solid_capstyle="butt", zorder=6)
    ax.text(x0 + 25e3, y0 + 4e3, "50 km", transform=PS, ha="center", va="bottom", fontsize=6, zorder=6,
            path_effects=HALO)
    ax.set_title("a  Location", loc="left", fontweight="bold")

    # Antarctica inset
    p = ax.get_position()
    s = 0.36 * p.width
    W, H = fig.get_size_inches()
    ins = fig.add_axes([p.x1 - s - 0.003, p.y1 - s * W / H - 0.003, s, s * W / H], projection=PS)
    ins.set_extent([-180, 180, -90, -64], PC)
    ins.set_facecolor(OCEAN)
    ins.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m"), facecolor="white", edgecolor="none")
    ins.add_feature(cfeature.NaturalEarthFeature("physical", "antarctic_ice_shelves_polys", "50m"),
                    facecolor=SHELF, edgecolor="#7a8a96", lw=0.2)
    ins.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "50m"), facecolor="none",
                    edgecolor="#333333", lw=0.3)
    ins.add_patch(mpatches.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], transform=PS,
                                     fill=False, ec="#d7301f", lw=1.0))
    handles = [mpatches.Patch(fc="#bbbbbb", label="Grounded ice (bed relief)"),
               mpatches.Patch(fc=SHELF, ec="#4a6b82", lw=0.4, label="Floating ice shelf"),
               mpatches.Patch(fc=OCEAN, label="Open ocean"),
               mlines.Line2D([], [], color="black", lw=0.8, label="Grounding line (BedMachine)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.1), fontsize=5.8, frameon=False,
              ncol=1, handlelength=1.4)


def surface(ax, z, norm, title, sub):
    n = z.shape[0]
    Xk, Yk = np.meshgrid(np.arange(z.shape[1]) * 0.25, np.arange(n)[::-1] * 0.25)
    rgb = LightSource(azdeg=315, altdeg=35).shade(z, cmap=CMAP, norm=norm, vert_exag=VEX, dx=250, dy=250,
                                                   blend_mode="soft")
    ax.plot_surface(Xk, Yk, z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)
    zr = max(abs(norm.vmin), abs(norm.vmax))
    ax.set_zlim(-zr, zr)
    t = float(np.round(zr / 2, -1 if zr < 300 else -2))
    ax.set_zticks([-t, 0, t])
    ax.set_box_aspect((1, 1, VEX * 2 * zr / BOX))
    ax.view_init(elev=32, azim=-62)
    ax.set_xlabel("km", fontsize=6.5, labelpad=-6)
    ax.set_ylabel("km", fontsize=6.5, labelpad=-6)
    ax.set_zlabel("m", fontsize=6.5, labelpad=-6)
    ax.tick_params(labelsize=5.5, pad=-2)
    ax.text2D(0.04, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold")
    ax.text2D(0.04, 0.90, sub, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#333333")


def main():
    OUT.mkdir(exist_ok=True)
    print(f"loading ({LABEL}) ...")
    (bx, by), speed = pick_box("pig", FOLDER)
    sr = tif(sorted(FOLDER.glob("pig_seed*_250m.tif"))[0])
    sr = sr.sel(x=slice(bx - BOX / 2, bx + BOX / 2), y=slice(by + BOX / 2, by - BOX / 2))
    dem = tif(sorted((HERE / "demogorgn_real").glob("pig_seed*.tif"))[0]).interp(x=sr.x, y=sr.y)
    z_dem, z_sr = detrended(dem.values), detrended(sr.values)
    lim = max(np.nanpercentile(np.abs(z), 99.5) for z in (z_dem, z_sr))
    norm = Normalize(-lim, lim)
    lon, lat = PC.transform_point(bx, by, PS)

    fig = plt.figure(figsize=(8.0, 3.6))
    axb = fig.add_axes([0.03, 0.0, 0.42, 0.88], projection="3d")
    surface(axb, z_dem, norm, "a  DEMOGORGN (500 m)", "realization, smoothly filled in to 250 m")
    axc = fig.add_axes([0.46, 0.0, 0.42, 0.88], projection="3d")
    surface(axc, z_sr, norm, "b  This study (250 m)", "the same realization, super-resolved")
    cax = fig.add_axes([0.905, 0.22, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, extend="both")
    cb.set_label("Bed height, overall slope\nof the box removed (m)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("Bed under Pine Island Glacier: DEMOGORGN vs. this study", fontsize=9.5, fontweight="bold",
                 y=0.985)
    fig.text(0.5, 0.925, f"20 × 20 km box centred {abs(lat):.2f}°S, {abs(lon):.2f}°W · grounded ice flowing "
             f"~{speed:.0f} m/yr · overall slope removed · vertical exaggeration {VEX}×", ha="center",
             fontsize=6.8, color="#333333")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"class_figure.{ext}", dpi=600)
    plt.close(fig)
    print("Saved figures/class_figure.png and figures/class_figure.pdf")


if __name__ == "__main__":
    main()
