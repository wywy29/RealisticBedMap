"""
Shared roughness measure, used both on radar flight lines (1D) and on model
output (2D, along rows and columns), so the two are directly comparable.

For a series sampled every 250 m:
    fine = what's left after removing a 5-point (1.25 km) running mean
           -> bumps smaller than about 1 km
    mid  = what a 21-point (5.25 km) running mean removes, minus the fine part
           -> bumps about 1-5 km across
Roughness = RMS of each (metres).
"""

import numpy as np


def bands_1d(s):
    """fine and mid residuals of one 1D series (numpy), aligned to the same points."""
    k5 = np.ones(5) / 5
    k21 = np.ones(21) / 21
    r5 = s[2:-2] - np.convolve(s, k5, mode="valid")
    r21 = s[10:-10] - np.convolve(s, k21, mode="valid")
    fine = r5[8:-8]
    return fine, r21 - fine


def rms_1d(series_list):
    """Pooled fine / mid RMS over many 1D series of length >= 21."""
    f2 = m2 = n = 0.0
    for s in series_list:
        if len(s) < 21 or not np.all(np.isfinite(s)):
            continue
        f, m = bands_1d(np.asarray(s, dtype="float64"))
        f2 += (f ** 2).sum()
        m2 += (m ** 2).sum()
        n += len(f)
    return np.sqrt(f2 / n), np.sqrt(m2 / n), int(n)


def bands_2d_torch(y):
    """Same measure on a batch of 2D patches (B, 1, H, W) in torch, along rows
    and along columns. Returns (fine_rms, mid_rms) as differentiable tensors."""
    import torch.nn.functional as F

    def along_last(z):
        ma5 = F.avg_pool2d(z, (1, 5), stride=1)
        ma21 = F.avg_pool2d(z, (1, 21), stride=1)
        r5 = z[..., 2:-2] - ma5
        r21 = z[..., 10:-10] - ma21
        fine = r5[..., 8:-8]
        return fine, r21 - fine

    f1, m1 = along_last(y)
    f2, m2 = along_last(y.transpose(-1, -2))
    fine = ((f1 ** 2).mean() + (f2 ** 2).mean()).div(2).sqrt()
    mid = ((m1 ** 2).mean() + (m2 ** 2).mean()).div(2).sqrt()
    return fine, mid


# ---------------------------------------------------------------- spectra (v4)
# Power spectrum of 36-sample (9 km) series at 250 m, detrended and Hann-windowed.
# The SAME function is used on radar flight lines (numpy) and on model output rows
# and columns (torch), so the two spectra are directly comparable, wavelength by
# wavelength. Bins used: k = 2..18 -> wavelengths 4.5 km down to 0.5 km.
N_SPEC = 36
K_LO, K_HI = 2, 18


def _hann_np():
    return np.hanning(N_SPEC + 2)[1:-1]


def psd_1d(series_list):
    """Mean power at each wavelength over many 36-sample series (numpy)."""
    t = np.arange(N_SPEC)
    h = _hann_np()
    out = []
    for s in series_list:
        s = np.asarray(s, dtype="float64")
        if len(s) != N_SPEC or not np.all(np.isfinite(s)):
            continue
        s = s - np.polyval(np.polyfit(t, s, 1), t)
        out.append(np.abs(np.fft.rfft(s * h)) ** 2)
    return np.mean(out, axis=0), len(out)


def wavelengths_km():
    return N_SPEC * 0.25 / np.arange(1, N_SPEC // 2 + 1)


def psd_2d_torch(y):
    """Mean power at each wavelength along rows and columns of (B,1,36,36) patches."""
    import torch
    n = y.shape[-1]
    t = torch.arange(n, dtype=y.dtype, device=y.device)
    tc = t - t.mean()
    h = torch.from_numpy(_hann_np()).to(y.dtype).to(y.device)

    def rows(z):
        z = z.reshape(-1, n)
        z = z - z.mean(1, keepdim=True)
        slope = (z * tc).sum(1, keepdim=True) / (tc ** 2).sum()
        z = z - slope * tc
        return (torch.fft.rfft(z * h).abs() ** 2).mean(0)

    return 0.5 * (rows(y) + rows(y.transpose(-1, -2)))


def highpass_torch(y, k=5):
    """Fine texture only: each patch minus a k x k running mean (edges cropped)."""
    import torch.nn.functional as F
    p = k // 2
    return y[..., p:-p, p:-p] - F.avg_pool2d(y, k, stride=1)


def psd_patches_np(patches):
    """Mean power at each wavelength along the rows and columns of many 36 x 36
    patches read straight off a 250 m grid (numpy). Identical maths to
    psd_2d_torch, which training uses, so map checks and training agree."""
    P = np.asarray(patches, dtype="float64")
    n = P.shape[-1]
    tc = np.arange(n) - (n - 1) / 2
    h = _hann_np()

    def rows(Z):
        Z = Z.reshape(-1, n)
        Z = Z - Z.mean(1, keepdims=True)
        Z = Z - ((Z * tc).sum(1, keepdims=True) / (tc ** 2).sum()) * tc
        return (np.abs(np.fft.rfft(Z * h, axis=1)) ** 2).mean(0)

    return 0.5 * (rows(P) + rows(np.swapaxes(P, -1, -2)))
