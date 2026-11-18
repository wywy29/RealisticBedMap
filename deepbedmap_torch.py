"""
DeepBedMap generator rebuilt in PyTorch, so the original trained weights
(srgan_generator_model_weights.npz from Zenodo, made with the old Chainer
library) can be loaded and fine-tuned without installing Chainer.

Architecture and settings follow Leong & Horgan (2020), DeepBedMap v1.1.0:
12 residual-in-residual dense blocks, residual scaling 0.2.

Inputs (all in raw units, no normalisation, same as the original):
    x  bed elevation,        1 km,  shape (N, 1, 11, 11)    in metres
    w1 surface elevation,  100 m,  shape (N, 1, 110, 110)  in metres
    w2 ice velocity vx,vy, 500 m,  shape (N, 2, 22, 22)    in m/yr
    w3 snow accumulation,    1 km,  shape (N, 1, 11, 11)
Output: bed elevation at 250 m, shape (N, 1, 36, 36) for the 11x11 input,
covering the central 9x9 km of the input tile. Bigger inputs work too:
an input of k x k km gives (k-2)*4 pixels per side.

Usage:
    from deepbedmap_torch import load_pretrained
    model = load_pretrained("srgan_generator_model_weights.npz")
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def lrelu(x):
    return F.leaky_relu(x, 0.2)


class InputBlock(nn.Module):
    """Brings the four inputs onto a common 1 km grid and stacks them."""

    def __init__(self, out_channels=32):
        super().__init__()
        self.conv_on_X = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)
        self.conv_on_W1 = nn.Conv2d(1, out_channels, 30, stride=10, padding=0)
        self.conv_on_W2 = nn.Conv2d(2, out_channels, 6, stride=2, padding=0)
        self.conv_on_W3 = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)

    def forward(self, x, w1, w2, w3):
        return torch.cat([self.conv_on_X(x), self.conv_on_W1(w1),
                          self.conv_on_W2(w2), self.conv_on_W3(w3)], dim=1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, ch=64, inter=32, scaling=0.2):
        super().__init__()
        self.scaling = scaling
        self.conv_layer1 = nn.Conv2d(ch, inter, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(ch + inter, inter, 3, padding=1)
        self.conv_layer3 = nn.Conv2d(ch + 2 * inter, inter, 3, padding=1)
        self.conv_layer4 = nn.Conv2d(ch + 3 * inter, inter, 3, padding=1)
        self.conv_layer5 = nn.Conv2d(ch + 4 * inter, ch, 3, padding=1)

    def forward(self, a0):
        a1 = lrelu(self.conv_layer1(a0))
        a2 = lrelu(self.conv_layer2(torch.cat([a0, a1], 1)))
        a3 = lrelu(self.conv_layer3(torch.cat([a0, a1, a2], 1)))
        a4 = lrelu(self.conv_layer4(torch.cat([a0, a1, a2, a3], 1)))
        a5 = self.conv_layer5(torch.cat([a0, a1, a2, a3, a4], 1))
        return a5 * self.scaling + a0


class ResInResDenseBlock(nn.Module):
    def __init__(self, scaling=0.2):
        super().__init__()
        self.scaling = scaling
        self.residual_dense_block1 = ResidualDenseBlock(scaling=scaling)
        self.residual_dense_block2 = ResidualDenseBlock(scaling=scaling)
        self.residual_dense_block3 = ResidualDenseBlock(scaling=scaling)

    def forward(self, x):
        a = self.residual_dense_block3(self.residual_dense_block2(self.residual_dense_block1(x)))
        return a * self.scaling + x


class DeformableConv(nn.Module):
    """Chainer's DeformableConvolution2D: a normal conv predicts sampling
    offsets, then a deformable conv samples the input at the shifted spots."""

    def __init__(self, in_ch, out_ch, k=3, pad=1):
        super().__init__()
        self.k, self.pad = k, pad
        self.offset_conv = nn.Conv2d(in_ch, 2 * k * k, k, padding=pad)
        self.deform_weight = nn.Parameter(torch.zeros(out_ch, in_ch, k, k))
        self.deform_bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.deform_weight)

    def forward(self, x):
        off = self.offset_conv(x)
        kk = self.k * self.k
        # Chainer stores all x-offsets first, then all y-offsets.
        # torchvision wants (y, x) pairs interleaved per kernel position.
        ox, oy = off[:, :kk], off[:, kk:]
        off = torch.stack([oy, ox], dim=2).flatten(1, 2)
        return deform_conv2d(x, off, self.deform_weight, self.deform_bias,
                             padding=(self.pad, self.pad))


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=12, residual_scaling=0.2):
        super().__init__()
        self.input_block = InputBlock()
        self.pre_residual_conv_layer = nn.Conv2d(128, 64, 3, padding=1)
        self.residual_network = nn.Sequential(
            *[ResInResDenseBlock(residual_scaling) for _ in range(num_residual_blocks)])
        self.post_residual_conv_layer = nn.Conv2d(64, 64, 3, padding=1)
        self.post_upsample_conv_layer_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.post_upsample_conv_layer_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.final_conv_layer1 = DeformableConv(64, 64)
        self.final_conv_layer2 = DeformableConv(64, 1)

    def forward(self, x, w1, w2, w3):
        a0 = self.input_block(x, w1, w2, w3)
        a1 = lrelu(self.pre_residual_conv_layer(a0))
        a2 = self.residual_network(a1)
        a3 = self.post_residual_conv_layer(a2) + a1
        a4 = lrelu(self.post_upsample_conv_layer_1(F.interpolate(a3, scale_factor=2, mode="nearest")))
        a4 = lrelu(self.post_upsample_conv_layer_2(F.interpolate(a4, scale_factor=2, mode="nearest")))
        a5 = lrelu(self.final_conv_layer1(a4))
        return self.final_conv_layer2(a5)


def _chainer_to_torch_name(key):
    parts = key.split("/")
    leaf = parts[-1]
    if "deform_conv" in parts:
        parts = parts[:-2] + ["deform_weight" if leaf == "W" else "deform_bias"]
    else:
        parts[-1] = {"W": "weight", "b": "bias"}[leaf]
    return ".".join(parts)


def load_pretrained(npz_path, num_residual_blocks=12, residual_scaling=0.2, verbose=True):
    """Build the generator and fill it with DeepBedMap's trained weights."""
    model = Generator(num_residual_blocks, residual_scaling)
    state = model.state_dict()
    raw = np.load(npz_path)
    keys = [k for k in raw.files if k.split("/")[-1] in ("W", "b")]

    # Strip any wrapper prefix (e.g. "updater/model:main/predictor/")
    # by finding where the model's own layer names start.
    loaded = {}
    for k in keys:
        parts = k.split("/")
        for i in range(len(parts)):
            name = _chainer_to_torch_name("/".join(parts[i:]))
            if name in state:
                loaded[name] = torch.from_numpy(np.asarray(raw[k], dtype=np.float32))
                break

    missing = sorted(set(state) - set(loaded))
    if missing:
        raise ValueError(f"{len(missing)} layers not found in {npz_path}, e.g. {missing[:5]}. "
                         f"File has keys like {keys[:5]}")
    for name, t in loaded.items():
        if t.shape != state[name].shape:
            raise ValueError(f"Shape mismatch for {name}: file {tuple(t.shape)} vs model {tuple(state[name].shape)}")
    model.load_state_dict(loaded)
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded {len(loaded)} weight arrays ({n:,} parameters) from {npz_path}")
    return model


if __name__ == "__main__":
    import sys
    m = load_pretrained(sys.argv[1]) if len(sys.argv) > 1 else Generator()
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 1, 11, 11), torch.zeros(1, 1, 110, 110),
              torch.zeros(1, 2, 22, 22), torch.zeros(1, 1, 11, 11))
    print("output shape:", tuple(y.shape))
