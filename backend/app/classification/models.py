"""The classifiers. All take a batch of point clouds shaped (B, N, 3) and
return unnormalised class scores (logits) shaped (B, 10).

A "shared" layer applies the same weights to every point independently. It is
written as ``nn.Conv1d(in, out, kernel_size=1)`` over the (B, channels, N)
layout, which is exactly a per-point ``nn.Linear`` and lets BatchNorm1d
normalise each channel over the batch and all points. A symmetric pooling over
the N points makes the result independent of point order.

HW2: ``shallow`` and ``mini_pointnet`` (global max pooling).
HW3: two Mini-PointNet variants that differ from ``mini_pointnet`` only in the
pooling. With ``m = max_N h`` and ``a = mean_N h`` (per channel), every pooling
here is ``z = m + g * (a - m)``:

- ``mini_pointnet``             g = 0 (max pooling, the HW2 model)
- ``mini_pointnet_fixed_dual``  g = 0.5, i.e. ``z = 0.5 m + 0.5 a`` (no parameters)
- ``mini_pointnet_gated_dual``  g = sigmoid(W2 relu(W1 [m; a] + b1) + b2), learned
  per channel and per object (Residual Gated Dual Pooling, docs/hw3-design.md)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}
# Model name -> pooling of the Mini-PointNet family.
MINI_POINTNET_POOLING = {
    "mini_pointnet": "max",
    "mini_pointnet_fixed_dual": "fixed_dual",
    "mini_pointnet_gated_dual": "gated_dual",
}
MODEL_NAMES = ("shallow", *MINI_POINTNET_POOLING)
GATE_HIDDEN = 16     # SENet's reduction ratio 16 applied to the 256 pooled channels
GATE_BIAS = -2.0     # sigmoid(-2) ~ 0.119: start close to max pooling, outside saturation


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; choose one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class ShallowPointNet(nn.Module):
    """Benchmark: one shared 3→h layer, activation, global max pool, linear classifier."""

    def __init__(self, num_classes: int = 10, hidden_dim: int = 64, activation: str = "relu"):
        super().__init__()
        self.point_layer = nn.Conv1d(3, hidden_dim, kernel_size=1)
        self.activation = make_activation(activation)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, points: Tensor) -> Tensor:
        x = points.transpose(1, 2)                  # (B, 3, N)
        x = self.activation(self.point_layer(x))    # (B, h, N): one hidden layer, per point
        x = x.max(dim=2).values                     # (B, h): global max pool over points
        return self.classifier(x)                   # (B, classes): logits


class GlobalMaxPool(nn.Module):
    """HW2 pooling: ``z_c = max_i h_{i,c}``. Gradient reaches one point per channel."""

    def forward(self, features: Tensor) -> Tensor:   # (B, C, N) -> (B, C)
        return features.max(dim=2).values


class FixedDualPool(nn.Module):
    """E1 control: ``z = 0.5 * max + 0.5 * mean`` with no trainable parameters."""

    mix = 0.5

    def forward(self, features: Tensor) -> Tensor:
        return 0.5 * features.max(dim=2).values + 0.5 * features.mean(dim=2)


class ResidualGatedDualPool(nn.Module):
    """E2: ``z = m + g * (a - m)`` with an SE-style channel gate computed from ``[m; a]``.

    ``m`` (max) is the identity path, ``g * (a - m)`` a learned per-channel correction
    toward the mean (ResNet's correction idea); ``g`` is a bottleneck MLP with a
    sigmoid over global descriptors (SENet's squeeze-and-excitation idea).
    The last gate layer starts at weight 0 and bias ``gate_bias`` so that, at
    initialisation, ``g = sigmoid(gate_bias)`` for every channel and object.
    """

    def __init__(self, channels: int, gate_hidden: int = GATE_HIDDEN, gate_bias: float = GATE_BIAS):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * channels, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, channels),
        )
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, gate_bias)

    def gate_values(self, maximum: Tensor, mean: Tensor) -> Tensor:
        return torch.sigmoid(self.gate(torch.cat([maximum, mean], dim=1)))   # (B, C) in (0, 1)

    def forward(self, features: Tensor) -> Tensor:
        maximum, mean = features.max(dim=2).values, features.mean(dim=2)
        return maximum + self.gate_values(maximum, mean) * (mean - maximum)


def make_pooling(name: str, channels: int, gate_hidden: int = GATE_HIDDEN, gate_bias: float = GATE_BIAS) -> nn.Module:
    if name == "max":
        return GlobalMaxPool()
    if name == "fixed_dual":
        return FixedDualPool()
    if name == "gated_dual":
        # Created inside a forked RNG state: for a given seed, the shared layers built
        # before it and the global random stream used afterwards by dropout are
        # exactly those of the max-pooling model, so only the pooling differs.
        with torch.random.fork_rng(devices=[]):
            return ResidualGatedDualPool(channels, gate_hidden, gate_bias)
    raise ValueError(f"Unknown pooling {name!r}; choose one of max, fixed_dual, gated_dual")


class MiniPointNet(nn.Module):
    """Shared MLP 3→h→2h→4h with BatchNorm, symmetric pooling, head 4h→2h→classes.

    With the default ``hidden_dim=64`` this is 3→64→128→256, pool, 256→128→10.
    ``pooling="max"`` is the HW2 model; the HW3 variants change only the pooling.
    """

    def __init__(self, num_classes: int = 10, hidden_dim: int = 64, activation: str = "relu",
                 dropout: float = 0.3, pooling: str = "max", gate_hidden: int = GATE_HIDDEN,
                 gate_bias: float = GATE_BIAS):
        super().__init__()
        widths = [3, hidden_dim, 2 * hidden_dim, 4 * hidden_dim]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Conv1d(c_in, c_out, kernel_size=1, bias=False), nn.BatchNorm1d(c_out), make_activation(activation)]
        self.point_mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=False),
            nn.BatchNorm1d(2 * hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )
        # Built last, so the shared layers above are initialised exactly as in HW2.
        self.pooling = pooling
        self.pool = make_pooling(pooling, 4 * hidden_dim, gate_hidden, gate_bias)

    def point_features(self, points: Tensor) -> Tensor:
        return self.point_mlp(points.transpose(1, 2))  # (B, 4h, N): three hidden layers, per point

    def forward(self, points: Tensor) -> Tensor:
        x = self.point_features(points)
        x = self.pool(x)                               # (B, 4h): global feature
        return self.head(x)                            # (B, classes): logits

    @torch.no_grad()
    def pooling_diagnostics(self, points: Tensor) -> dict[str, Tensor]:
        """Max, mean, the mixing weight ``g`` and the pooled feature, without changing ``forward``.

        ``g`` is 0 for max pooling, 0.5 for the fixed dual pooling and the learned gate otherwise.
        """
        features = self.point_features(points)
        maximum, mean = features.max(dim=2).values, features.mean(dim=2)
        if isinstance(self.pool, ResidualGatedDualPool):
            gate = self.pool.gate_values(maximum, mean)
        else:
            gate = torch.full_like(maximum, getattr(self.pool, "mix", 0.0))
        pooled = self.pool(features)
        return {"max": maximum, "mean": mean, "gate": gate, "pooled": pooled, "logits": self.head(pooled)}


def build_model(name: str, num_classes: int = 10, hidden_dim: int = 64, activation: str = "relu",
                dropout: float = 0.3, gate_hidden: int = GATE_HIDDEN, gate_bias: float = GATE_BIAS) -> nn.Module:
    """``gate_hidden``/``gate_bias`` are used by ``mini_pointnet_gated_dual`` only."""
    if name == "shallow":
        return ShallowPointNet(num_classes, hidden_dim, activation)
    if name in MINI_POINTNET_POOLING:
        return MiniPointNet(num_classes, hidden_dim, activation, dropout, MINI_POINTNET_POOLING[name],
                            gate_hidden, gate_bias)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")


def model_kwargs(name: str, hidden_dim: int, activation: str, dropout: float,
                 gate_hidden: int = GATE_HIDDEN, gate_bias: float = GATE_BIAS) -> dict:
    """The constructor arguments stored in a checkpoint (``build_model(name, classes, **kwargs)``)."""
    kwargs = {"hidden_dim": hidden_dim, "activation": activation, "dropout": dropout}
    if MINI_POINTNET_POOLING.get(name) == "gated_dual":
        kwargs.update(gate_hidden=gate_hidden, gate_bias=gate_bias)
    return kwargs


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
