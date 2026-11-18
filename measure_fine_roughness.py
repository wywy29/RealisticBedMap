"""
Step 15a: how rough is the real bed BELOW 250 m? Measured from the raw radar picks.

The IceBridge picks are ~15 m apart along each flight line, so along the lines we
can see the bed in fine detail. Method (roughness vs piece length):
  1. cut the flight lines into pieces of one length L (e.g. 5 km, 2 km, 1 km, ... 100 m);
  2. remove each piece's overall slope, then measure how much the bed still wiggles
     up and down (RMS height, in metres);
  3. repeat for every length. Longer pieces contain bigger bumps, so they wiggle more.
On a log-log plot the points fall on a straight line:  RMS(L) = RMS(1 km) x (L / 1 km)^H
H (the Hurst exponent) says how fast roughness shrinks with scale. Extending the line
down to 10-20 m tells make_fine_bed.py how bumpy to make the fine detail.

Second check: the roughness spectrum (bumpiness by bump width) of the same picks;
its slope should be about 2H + 1. If the two agree, the trend is solid.

Runs on your PC, in glyproject.  Needs: cropped/{pig,getz}_radar_points.csv

Run:
    python measure_fine_roughness.py

Saves:
    fine_roughness.json                       the fitted line per region (used by make_fine_bed.py)
    figures/fine_roughness_{pig,getz}.png     RMS vs piece length, and the spectrum check

Note: below ~100 m the picks are affected by radar resolution and pick noise, so the
fit uses pieces of 150 m to 2 km; shorter pieces are shown but not fitted.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DS = 20.0                                                     # m, along-track resampling step
LENGTHS = np.array([80, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000], float)   # m
FIT = (150.0, 2000.0)                                         # m, piece lengths used for the fit
NWIN = 256                                                    # spectrum check: 5.12 km windows
NAMES = {"pig": "Pine Island", "getz": "Getz"}


def rms_vs_length(series, ds, lengths):
    """RMS height (m) after removing each piece's slope, for pieces of each length."""
    out = []
    for L in lengths:
        n = int(round(L / ds))
        if n < 4:
            out.append(np.nan)
            continue
        t = np.arange(n) - (n - 1) / 2
        tt = (t ** 2).sum()
        ss, cnt = 0.0, 0
        for s in series:
            m = len(s) // n
            if m == 0:
                continue
            P = np.asarray(s[:m * n], float).reshape(m, n)
            P = P - P.mean(1, keepdims=True)
            P = P - ((P * t).sum(1, keepdims=True) / tt) * t
            ss += (P ** 2).sum()
            cnt += P.size
        out.append(np.sqrt(ss / cnt) if cnt else np.nan)
    return np.array(out)


def fit_line(lengths, rms):
    sel = (lengths >= FIT[0]) & (lengths <= FIT[1]) & np.isfinite(rms)
    H, logc = np.polyfit(np.log(lengths[sel] / 1000.0), np.log(rms[sel]), 1)
    return float(H), float(np.exp(logc))                      # RMS(L) = rms1km * (L / 1 km)^H


def psd_series(series, ds):
    """Mean power spectrum of equal-length series (second check). Returns (wavelength_m, psd)."""
    n = len(series[0])
    t = np.arange(n)
    h = np.hanning(n)
    out = []
    for s in series:
        s = np.asarray(s, float)
        s = s - np.polyval(np.polyfit(t, s, 1), t)
        out.append(np.abs(np.fft.rfft(s * h)) ** 2)
    k = np.fft.rfftfreq(n, d=ds / 1000.0)
    p = np.mean(out, 0) * (ds / 1000.0) / (h ** 2).sum()
    return 1000.0 / k[1:], p[1:]


def radar_profiles(r):
    """Continuous stretches of radar picks, resampled every DS metres."""
    df = pd.read_csv(HERE / "cropped" / f"{r}_radar_points.csv")
    prof = []
    for _, g in df.groupby("FILE", sort=False):
        x, y, z = g.x.values, g.y.values, g.bed.values
        step = np.hypot(np.diff(x), np.diff(y))
        for s in np.split(np.arange(len(x)), np.where((step > 100) | (step < 0.01))[0] + 1):
            if len(s) < 10:
                continue
            d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x[s]), np.diff(y[s])))])
            if d[-1] < 1000:
                continue
            u = np.arange(0, d[-1], DS)
            prof.append(np.interp(u, d, z[s]))
    return prof


def main():
    out = {}
    (HERE / "figures").mkdir(exist_ok=True)
    for r in ["pig", "getz"]:
        prof = radar_profiles(r)
        if not prof:
            print(f"{NAMES[r]}: no radar stretches long enough")
            continue
        rms = rms_vs_length(prof, DS, LENGTHS)
        H, rms1km = fit_line(LENGTHS, rms)
        wins = [p[k:k + NWIN] for p in prof for k in range(0, len(p) - NWIN + 1, NWIN)]
        wl, psd = psd_series(wins, DS)
        sel = (wl <= 1000) & (wl >= 150)
        beta = -np.polyfit(np.log(1000 / wl[sel]), np.log(psd[sel]), 1)[0]
        out[r] = {"hurst": H, "rms_1km_m": rms1km, "fit_lengths_m": list(FIT),
                  "lengths_m": LENGTHS.tolist(), "rms_m": rms.tolist(),
                  "spectrum_slope": float(beta), "wavelength_m": wl.tolist(), "psd": psd.tolist()}
        print(f"\n{NAMES[r]}: {len(prof)} radar stretches, {sum(len(p) for p in prof) * DS / 1000:,.0f} km of line")
        print("  piece length -> RMS height after removing its slope:")
        for L, v in zip(LENGTHS, rms):
            print(f"    {L:6.0f} m  {v:6.2f} m")
        print(f"  trend: RMS = {rms1km:.1f} m x (length / 1 km)^{H:.2f}   (Hurst exponent H = {H:.2f})")
        print(f"  extended: 20 m pieces {rms1km * 0.02 ** H:.2f} m, 10 m pieces {rms1km * 0.01 ** H:.2f} m")
        print(f"  check: spectrum slope {beta:.2f}, expected about 2H + 1 = {2 * H + 1:.2f}")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        ax[0].loglog(LENGTHS, rms, "o", color="black", label="Radar picks")
        Lx = np.geomspace(10, 5000, 100)
        ax[0].loglog(Lx, rms1km * (Lx / 1000) ** H, "--", color="tab:red", lw=1.2,
                     label=f"Trend, H = {H:.2f} (extended to 10 m)")
        Lf = np.geomspace(*FIT, 50)
        ax[0].loglog(Lf, rms1km * (Lf / 1000) ** H, color="tab:red", lw=2.4, label="Fitted range")
        ax[0].axvspan(10, 150, color="#bbbbbb", alpha=0.25, lw=0)
        ax[0].set_xlabel("Piece length (m)")
        ax[0].set_ylabel("RMS height after removing slope (m)")
        ax[0].set_title(f"{NAMES[r]}: roughness vs piece length")
        ax[0].legend(fontsize=7, frameon=False)
        ax[1].loglog(wl, psd, color="black", lw=1.6, label="Radar picks")
        ax[1].loglog(wl[sel], np.exp(np.polyval(np.polyfit(np.log(1000 / wl[sel]), np.log(psd[sel]), 1),
                                                np.log(1000 / wl[sel]))),
                     color="tab:red", lw=2, label=f"Slope {beta:.2f} (2H+1 = {2 * H + 1:.2f})")
        ax[1].invert_xaxis()
        ax[1].set_xlabel("Bump width (m)")
        ax[1].set_ylabel("Roughness power (m² per cycle/km)")
        ax[1].set_title("Check: roughness by bump width")
        ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(HERE / "figures" / f"fine_roughness_{r}.png", dpi=150)
        plt.close(fig)
    (HERE / "fine_roughness.json").write_text(json.dumps(out))
    print("\nSaved fine_roughness.json and figures/fine_roughness_*.png")


if __name__ == "__main__":
    main()
