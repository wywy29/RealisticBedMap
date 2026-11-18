"""
Step 13: realism training v5 (runs on your PC's GPU).

Why v5: v3/v4 said ~100% roughness during training, but the final maps came out
smooth. The reason, tested on the original DeepBedMap network:
  * the network's output at a point depends on everything within ~30 km of it
    (its last layers reach out up to ~35 km);
  * training used 11 km tiles, so every graded point was 1-10 km from a tile
    edge, and the network makes fake texture near edges;
  * with full surroundings (as in the final maps, 40 km per side) the same
    network made almost no fine texture.
So v2-v4 learned to be rough only next to tile edges, which the maps don't have.

v5 trains on exactly what make_realizations.py produces:
  * each training window is a 36 x 36 km area with 40 km of REAL surroundings on
    every side (118 x 118 km input), exactly as make_realizations.py runs, so the
    graded area is computed exactly as in the final maps (checked: identical);
  * the output is pulled onto DEMOGORGN's 1 km averages, as in the maps;
  * the graded area is cut into sixteen 9 x 9 km patches; patches on grounded
    ice are graded by the spectrum rule (match the radar's roughness wavelength
    by wavelength, 4.5 km down to 0.5 km) and by the judge (real seafloor);
    radar points inside the area keep a light radar term.
The "% of radar" numbers printed each epoch are therefore what the maps will have.

Needs (same folder):
    deepbedmap_torch.py, roughness.py
    finetuned_deepbedmap.pt            (starting point)
    pig_tiles_seeds.npz                (where the radar tiles are, and the train/val split)
    pig_grids.npz, pig_seeds.npz       (whole-region inputs)
    real_patches.npz                   (real seafloor)
    roughness_targets.json             (from measure_targets.py)

Run:
    python train_realism_v5.py

Saves:
    realism_v5.pt, realism_v5_log.csv, v5_samples_epochXXX.png
"""

import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from deepbedmap_torch import Generator, lrelu
from roughness import psd_2d_torch, highpass_torch, wavelengths_km, K_LO, K_HI

START = "finetuned_deepbedmap.pt"
OUT_NAME = "realism_v5.pt"
CTX = 40                  # km of real surroundings on each side, same as make_realizations.py (PAD)
GK = 36                   # km, graded area per window (4 x 4 patches of 9 km)
EPOCHS = 40
STEPS = 60                # training steps per epoch
BATCH = 4                 # windows per step (each = 16 patches); lowered automatically if GPU memory runs out
N_VAL = 24                # validation windows
LR_G, LR_D = 1e-5, 4e-5
W_TOPO, W_RADAR, W_ADV, W_SPEC = 1.0, 0.02, 2.0, 5.0   # topo/radar in units of 10 m
MIN_GROUNDED = 0.95       # a 9 km patch is graded if at least this much of it is grounded ice
WARMUP = 2
SEED = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

TARGETS = json.load(open("roughness_targets.json"))["pig"]
if "psd" not in TARGETS:
    raise SystemExit("roughness_targets.json has no spectrum: rerun measure_targets.py first")
P_RADAR = torch.tensor(TARGETS["psd"], dtype=torch.float32)
KS = list(range(K_LO, K_HI + 1))
_WL = wavelengths_km()
SMALL = [k for k in KS if _WL[k - 1] < 1.0]      # 0.5-1 km
LARGE = [k for k in KS if _WL[k - 1] >= 1.0]     # 1-4.5 km
N_IN = GK + 2 + 2 * CTX                          # input window, km (118)


# ------------------------------------------------------------------ judge
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, s):
            return [nn.Conv2d(i, o, 4 if s == 2 else 3, s, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            *blk(64, 64, 2), *blk(64, 128, 1), *blk(128, 128, 2),
            *blk(128, 256, 1), *blk(256, 256, 2), *blk(256, 512, 1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 100), nn.LeakyReLU(0.2), nn.Linear(100, 1))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ the map-maker, as in make_realizations.py
def generate(G, x, w1, w2, w3, grad=True):
    """DeepBedMap on a 118 x 118 km window, returning the graded 36 x 36 km at 250 m
    (144 x 144) exactly as make_realizations.py gives it (same 40 km surroundings,
    same pull onto DEMOGORGN's 1 km averages). Checked: identical to the full map."""
    a0 = G.input_block(x, w1, w2, w3)
    a1 = lrelu(G.pre_residual_conv_layer(a0))
    a2 = a1
    for blk in G.residual_network:      # recompute instead of store, to fit in GPU memory
        a2 = checkpoint(blk, a2, use_reentrant=False) if (grad and torch.is_grad_enabled()) else blk(a2)
    a3 = G.post_residual_conv_layer(a2) + a1
    a4 = lrelu(G.post_upsample_conv_layer_1(F.interpolate(a3, scale_factor=2, mode="nearest")))
    a4 = lrelu(G.post_upsample_conv_layer_2(F.interpolate(a4, scale_factor=2, mode="nearest")))
    y = G.final_conv_layer2(lrelu(G.final_conv_layer1(a4)))
    xc = x[..., 1:-1, 1:-1]
    for _ in range(3):
        y = y + F.interpolate(xc - F.avg_pool2d(y, 4), scale_factor=4, mode="bilinear", align_corners=False)
    c = 4 * CTX                          # output row 4(k-1) <-> input cell k; graded cells start at CTX+1
    return y[..., c:c + 4 * GK, c:c + 4 * GK]


def to_patches(z):
    """(B, 1, 144, 144) -> (B * 16, 1, 36, 36)."""
    B = z.shape[0]
    return z.reshape(B, 1, 4, 36, 4, 36).permute(0, 2, 4, 1, 3, 5).reshape(B * 16, 1, 36, 36)


def spec_loss(p):
    pr = P_RADAR.to(p.device)
    return ((torch.log(p[KS]) - torch.log(pr[KS])) ** 2).mean()


def amp_pct(p, ks):
    pr = P_RADAR.to(p.device)
    return 100 * float(torch.sqrt(p[ks].sum() / pr[ks].sum()))


def texture(z):
    return highpass_torch(z) / 10.0


def rotate_velocity(w2, k):
    vx, vy = w2[:, 0], w2[:, 1]
    for _ in range(k % 4):
        vx, vy = -vy, vx
    return torch.stack([vx, vy], 1)


# ------------------------------------------------------------------ data
class Data:
    """Windows are placed so the graded 36 km covers a radar tile (from
    build_tiles.py) and its neighbours, on the same 9 km tile grid."""

    def __init__(self):
        t = np.load("pig_tiles_seeds.npz")
        g = np.load("pig_grids.npz")
        self.Xs = np.load("pig_seeds.npz")["X"].astype("float32")      # (seeds, ny, nx) 1 km
        self.W1, self.W2, self.W3 = (g[k].astype("float32") for k in ("W1", "W2", "W3"))
        self.Y = g["Y"].astype("float32")
        self.M = g["M"] & g["grounded"]
        self.ground = g["grounded"].astype("float32")
        ny, nx = self.Xs.shape[1:]
        wins = {"train": set(), "val": set()}
        for (i, j), s in zip(t["ij"].astype(int), t["split"]):
            if s not in wins:
                continue
            # the tile's cells start at i+1; the graded area starts at i+1-9a, a = 0..3
            slots = [(a, b) for a in (1, 2, 0, 3) for b in (1, 2, 0, 3)]
            for a, b in slots:
                r0, c0 = i - 9 * a - CTX, j - 9 * b - CTX            # input window start
                if r0 >= 0 and c0 >= 0 and r0 + N_IN <= ny and c0 + N_IN <= nx:
                    wins[s].add((r0, c0))
                    if s == "val":
                        break                                         # one window per val tile
        # keep windows with at least 4 of their 16 patches on grounded ice
        self.windows = {k: sorted(w for w in v if self.n_grounded(*w) >= 4) for k, v in wins.items()}
        rng = np.random.default_rng(0)
        val = self.windows["val"]
        self.windows["val"] = [val[k] for k in sorted(rng.permutation(len(val))[:N_VAL])]
        print(f"windows ({N_IN} x {N_IN} km input, {GK} x {GK} km graded): "
              f"{len(self.windows['train'])} train, {len(self.windows['val'])} val")
        if not self.windows["train"] or not self.windows["val"]:
            raise SystemExit("Not enough room for the windows around the radar tiles.")

    def n_grounded(self, r0, c0):
        g0, h0 = 4 * (r0 + CTX + 1), 4 * (c0 + CTX + 1)
        g = self.ground[g0:g0 + 4 * GK, h0:h0 + 4 * GK].reshape(4, 36, 4, 36).mean((1, 3))
        return int((g >= MIN_GROUNDED).sum())

    def batch(self, wins, aug, seed=None):
        xs, w1s, w2s, w3s, ys, ms, gs = [], [], [], [], [], [], []
        for r0, c0 in wins:
            r1, c1 = r0 + N_IN, c0 + N_IN
            k = np.random.randint(len(self.Xs)) if seed is None else seed
            xs.append(self.Xs[k, r0:r1, c0:c1])
            w1s.append(self.W1[10 * r0:10 * r1, 10 * c0:10 * c1])
            w2s.append(self.W2[:, 2 * r0:2 * r1, 2 * c0:2 * c1])
            w3s.append(self.W3[r0:r1, c0:c1])
            g0, h0 = 4 * (r0 + CTX + 1), 4 * (c0 + CTX + 1)         # graded area on the 250 m grids
            sl = (slice(g0, g0 + 4 * GK), slice(h0, h0 + 4 * GK))
            ys.append(np.nan_to_num(self.Y[sl]))
            ms.append(self.M[sl])
            gs.append(self.ground[sl])
        T = lambda a: torch.from_numpy(np.stack(a))
        x, w1, w3, y, m, gr = (T(a)[:, None] for a in (xs, w1s, w3s, ys, ms, gs))
        w2 = T(w2s)
        if aug:
            k = int(torch.randint(4, (1,)))
            x, w1, w2, w3, y, m, gr = [z.rot90(k, (-2, -1)) for z in (x, w1, w2, w3, y, m, gr)]
            w2 = rotate_velocity(w2, k)
        return [z.contiguous().to(device) for z in (x, w1, w2, w3, y, m, gr)]


# ------------------------------------------------------------------ main
def main():
    print(f"device: {device}")
    D_ = Data()
    real = torch.from_numpy(np.load("real_patches.npz")["real"].astype("float32"))
    print(f"real seafloor patches: {len(real)}")

    G = Generator().to(device)
    G.load_state_dict(torch.load(START, map_location=device))
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    BS = [BATCH]                         # windows per step (halved if the GPU runs out of memory)

    def graded(out, gr):
        """the 9 km patches on grounded ice"""
        keep = to_patches(gr).mean((1, 2, 3)) >= MIN_GROUNDED
        return to_patches(out)[keep]

    @torch.no_grad()
    def validate():
        G.eval()
        topo = rad = n = 0.0
        pats = []
        va = D_.windows["val"]
        for s in range(0, len(va), BS[0]):
            x, w1, w2, w3, y, m, gr = D_.batch(va[s:s + BS[0]], aug=False, seed=0)
            out = generate(G, x, w1, w2, w3, grad=False)
            xc = x[..., CTX + 1:CTX + 1 + GK, CTX + 1:CTX + 1 + GK]
            topo += (F.avg_pool2d(out, 4) - xc).abs().mean().item() * len(out)
            rad += (out - y)[m].abs().sum().item()
            n += m.sum().item()
            pats.append(graded(out, gr))
        G.train()
        pats = torch.cat(pats)
        return topo / len(va), rad / max(n, 1), psd_2d_torch(pats), len(pats)

    t, r, p0, npat = validate()
    print(f"validation: {npat} patches of 9 x 9 km on grounded ice, each with full surroundings")
    print(f"\nbefore training (= what the maps would look like now): topo err {t:4.1f} m  radar MAE {r:5.1f} m  "
          f"bumps 0.5-1 km {amp_pct(p0, SMALL):4.0f}%  1-4.5 km {amp_pct(p0, LARGE):4.0f}% of radar\n")

    def train_step(wins, gen_step):
        x, w1, w2, w3, y, m, gr = D_.batch(wins, aug=True)
        # --- judge step
        with torch.no_grad():
            fake = graded(generate(G, x, w1, w2, w3, grad=False), gr)
        if len(fake) < 2:
            return None
        rb = real[torch.randint(len(real), (len(fake),))].rot90(int(torch.randint(4, (1,))), (-2, -1)).to(device)
        d_loss = bce(D(texture(rb)), torch.full((len(rb), 1), 0.9, device=device)) + \
                 bce(D(texture(fake)), torch.zeros(len(fake), 1, device=device))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        del fake
        if not gen_step:
            return d_loss.item()
        # --- generator step
        out = generate(G, x, w1, w2, w3)
        xc = x[..., CTX + 1:CTX + 1 + GK, CTX + 1:CTX + 1 + GK]
        l_topo = (F.avg_pool2d(out, 4) - xc).abs().mean() / 10.0
        l_rad = (out - y)[m].abs().mean() / 10.0 if m.any() else out.sum() * 0
        pats = graded(out, gr)
        l_adv = bce(D(texture(pats)), torch.ones(len(pats), 1, device=device))
        l_spec = spec_loss(psd_2d_torch(pats))
        g_loss = W_TOPO * l_topo + W_RADAR * l_rad + W_ADV * l_adv + W_SPEC * l_spec
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        return d_loss.item()

    log, best = [], None
    tr = D_.windows["train"]
    d_loss = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        done = 0
        while done < STEPS:
            wins = [tr[k] for k in np.random.randint(len(tr), size=BS[0])]
            try:
                d = train_step(wins, gen_step=epoch > WARMUP)
            except torch.cuda.OutOfMemoryError:
                opt_g.zero_grad(); opt_d.zero_grad()
                torch.cuda.empty_cache()
                if BS[0] == 1:
                    raise SystemExit("Out of GPU memory even with 1 window per step.")
                BS[0] //= 2
                print(f"  (GPU memory full: now {BS[0]} windows per step)")
                continue
            done += 1
            if d is not None:
                d_loss = d

        t_err, r_err, p_v, _ = validate()
        a_s, a_l = amp_pct(p_v, SMALL), amp_pct(p_v, LARGE)
        score = float(spec_loss(p_v))
        log.append((epoch, d_loss, t_err, r_err, a_s, a_l, score))
        flag = ""
        if epoch > WARMUP and t_err < 5.0 and (best is None or score < best):
            best = score
            torch.save(G.state_dict(), OUT_NAME)
            flag = "  <- saved"
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  judge {d_loss:4.2f}  topo err {t_err:4.1f} m  radar MAE {r_err:5.1f} m  "
              f"bumps 0.5-1 km {a_s:4.0f}%  1-4.5 km {a_l:4.0f}% of radar  ({dt:.0f}s){flag}")
        if epoch == WARMUP + 1:
            print(f"           (about {dt * (EPOCHS - WARMUP) / 60:.0f} minutes in total)")
        if epoch % 10 == 0:
            save_samples(G, D_, real, epoch)

    with open("realism_v5_log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "judge_loss", "topo_err_m", "radar_mae_m",
                    "pct_radar_0.5-1km", "pct_radar_1-4.5km", "spectrum_misfit"])
        w.writerows(log)
    if best is None:
        torch.save(G.state_dict(), OUT_NAME)
        print("topo error never stayed under 5 m; saved the final epoch instead")
    print(f"\nSaved {OUT_NAME}, realism_v5_log.csv")


@torch.no_grad()
def save_samples(G, D_, real, epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G.eval()
    nc = min(3, len(D_.windows["val"]))
    x, w1, w2, w3, y, m, gr = D_.batch(D_.windows["val"][:nc], aug=False, seed=0)
    out = generate(G, x, w1, w2, w3, grad=False).cpu()
    xc = x[..., CTX + 1:CTX + 1 + GK, CTX + 1:CTX + 1 + GK].cpu()
    up = F.interpolate(xc, scale_factor=4, mode="bilinear", align_corners=False)
    fig, ax = plt.subplots(2, nc + 1, figsize=(3.3 * (nc + 1), 6.6), squeeze=False)
    for c in range(nc):
        for r, (img, title) in enumerate([(up[c, 0], "DEMOGORGN (smooth)"), (out[c, 0], "v5 output")]):
            img = img.numpy()
            yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
            A = np.c_[np.ones(img.size), xx.ravel(), yy.ravel()]
            img = img - (A @ np.linalg.lstsq(A, img.ravel(), rcond=None)[0]).reshape(img.shape)
            ax[r, c].imshow(img, cmap="terrain")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if c == 0:
                ax[r, c].set_ylabel(title)
    for r in range(2):
        img = real[(r * 11) % len(real), 0].numpy()
        ax[r, nc].imshow(img - img.mean(), cmap="terrain")
        ax[r, nc].set_xticks([]); ax[r, nc].set_yticks([])
        ax[r, nc].set_title("real seafloor, 9 x 9 km", fontsize=8)
    fig.suptitle(f"v5 epoch {epoch}: 36 x 36 km areas exactly as in the final maps (tilt removed)")
    fig.tight_layout()
    fig.savefig(f"v5_samples_epoch{epoch:03d}.png", dpi=80)
    plt.close(fig)
    G.train()


if __name__ == "__main__":
    main()
