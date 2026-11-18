"""
Step 7: fine-tune DeepBedMap on the Pine Island tiles.

Needs (same folder):
    deepbedmap_torch.py
    srgan_generator_model_weights.npz
    pig_tiles.npz            (from tiles/, made by build_tiles.py)

Run (in Colab: Runtime > Change runtime type > T4 GPU, then):
    python train_deepbedmap.py

What it does
  1. Loads the pretrained DeepBedMap.
  2. Scores two baselines on the validation tiles, so you can tell whether
     training helped: DEMOGORGN smoothly upsampled to 250 m, and the
     pretrained DeepBedMap with no fine-tuning.
  3. Trains, grading the model only on 250 m cells that have radar.
     Early layers are frozen so 913 tiles can't wipe out what it already knows.
     Tiles are randomly flipped/rotated (with velocity directions turned to match)
     so it sees more variety.
  4. Keeps the version with the lowest validation error, stops when it stops improving.
  5. Reports error on the untouched test tiles.

Saves:
    finetuned_deepbedmap.pt   the fine-tuned weights
    train_log.csv             error after each epoch (for a training curve figure)
"""

import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from deepbedmap_torch import load_pretrained

# ------------------------------------------------------------------ settings
TILES = "pig_tiles.npz"
WEIGHTS = "srgan_generator_model_weights.npz"
EPOCHS = 150
PATIENCE = 20            # stop after this many epochs without improvement
BATCH = 32
LR = 2e-5                # small: nudge the weights, don't overwrite them
FREEZE_FIRST = 8         # freeze this many of the 12 residual blocks (0 = train everything)
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ data
def load_split(d, name):
    sel = d["split"] == name
    return [torch.from_numpy(d[k][sel].astype("float32")) for k in ("X", "W1", "W2", "W3", "Y")] + \
           [torch.from_numpy(d["M"][sel])]


def augment(x, w1, w2, w3, y, m):
    """Random flip / transpose applied to every layer of a batch the same way.
    Arrays are (row, col) with rows pointing south (-y), so:
      flip columns  -> vx changes sign
      flip rows     -> vy changes sign
      transpose     -> (vx, vy) becomes (-vy, -vx)"""
    if torch.rand(1) < 0.5:
        x, w1, w2, w3, y, m = [t.flip(-1) for t in (x, w1, w2, w3, y, m)]
        w2 = torch.stack([-w2[:, 0], w2[:, 1]], 1)
    if torch.rand(1) < 0.5:
        x, w1, w2, w3, y, m = [t.flip(-2) for t in (x, w1, w2, w3, y, m)]
        w2 = torch.stack([w2[:, 0], -w2[:, 1]], 1)
    if torch.rand(1) < 0.5:
        x, w1, w2, w3, y, m = [t.transpose(-1, -2) for t in (x, w1, w2, w3, y, m)]
        w2 = torch.stack([-w2[:, 1], -w2[:, 0]], 1)
    return x, w1, w2, w3, y, m


def batches(data, shuffle, aug=False):
    n = len(data[0])
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, BATCH):
        b = [t[idx[s:s + BATCH]] for t in data]
        if aug:
            b = augment(*b)
        yield [t.to(device) for t in b]


def demogorgn_upsampled(x):
    """Baseline: DEMOGORGN smoothly interpolated to 250 m, central 9 km."""
    return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)[..., 4:40, 4:40]


def masked_errors(pred, y, m):
    d = (pred - y)[m]
    return d.abs().sum().item(), (d ** 2).sum().item(), m.sum().item()


@torch.no_grad()
def score(model, data, use_model=True):
    if use_model:
        model.eval()
    ae = se = n = 0.0
    for x, w1, w2, w3, y, m in batches(data, shuffle=False):
        pred = model(x, w1, w2, w3) if use_model else demogorgn_upsampled(x)
        a, s, k = masked_errors(pred, y, m)
        ae, se, n = ae + a, se + s, n + k
    return ae / n, (se / n) ** 0.5   # MAE, RMSE in metres


# ------------------------------------------------------------------ main
def main():
    print(f"device: {device}")
    d = np.load(TILES)
    train, val, test = (load_split(d, s) for s in ("train", "val", "test"))
    print(f"tiles: {len(train[0])} train, {len(val[0])} val, {len(test[0])} test")

    model = load_pretrained(WEIGHTS).to(device)

    mae, rmse = score(model, val, use_model=False)
    print(f"\nbaseline on validation tiles (error at radar cells):")
    print(f"  DEMOGORGN upsampled      MAE {mae:6.1f} m   RMSE {rmse:6.1f} m")
    mae, rmse = score(model, val)
    print(f"  DeepBedMap, not tuned    MAE {mae:6.1f} m   RMSE {rmse:6.1f} m\n")

    for i, blk in enumerate(model.residual_network):
        if i < FREEZE_FIRST:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in trainable):,} of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.Adam(trainable, lr=LR)

    best, best_epoch, log = float("inf"), 0, []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        ae = n = 0.0
        for x, w1, w2, w3, y, m in batches(train, shuffle=True, aug=True):
            pred = model(x, w1, w2, w3)
            loss = (pred - y)[m].abs().mean()          # mean absolute error at radar cells
            opt.zero_grad()
            loss.backward()
            opt.step()
            ae += loss.item() * m.sum().item()
            n += m.sum().item()
        tr_mae = ae / n
        va_mae, va_rmse = score(model, val)
        log.append((epoch, tr_mae, va_mae, va_rmse))
        flag = ""
        if va_rmse < best:
            best, best_epoch = va_rmse, epoch
            torch.save(model.state_dict(), "finetuned_deepbedmap.pt")
            flag = "  <- best so far, saved"
        print(f"epoch {epoch:3d}  train MAE {tr_mae:6.1f}  val MAE {va_mae:6.1f}  "
              f"val RMSE {va_rmse:6.1f}  ({time.time() - t0:.0f}s){flag}")
        if epoch - best_epoch >= PATIENCE:
            print(f"no improvement for {PATIENCE} epochs, stopping")
            break

    with open("train_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mae_m", "val_mae_m", "val_rmse_m"])
        w.writerows(log)

    model.load_state_dict(torch.load("finetuned_deepbedmap.pt", map_location=device))
    print(f"\nbest epoch {best_epoch}, val RMSE {best:.1f} m")
    print("test tiles (never used in training):")
    for name, use in [("DEMOGORGN upsampled", False), ("DeepBedMap fine-tuned", True)]:
        mae, rmse = score(model, test, use_model=use)
        print(f"  {name:22s}  MAE {mae:6.1f} m   RMSE {rmse:6.1f} m")
    print("\nsaved finetuned_deepbedmap.pt and train_log.csv")


if __name__ == "__main__":
    main()
