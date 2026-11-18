"""Parameter spaces and Pearson pattern atlas for Gray–Scott RD systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class GrayScottParams:
    """Gray–Scott kinetic and diffusion parameters.

    Continuous system (Pearson 1993):
        ∂u/∂t = Du ∇²u − u v² + F (1 − u)
        ∂v/∂t = Dv ∇²v + u v² − (F + k) v

    where F is feed rate and k is kill rate.
    """

    Du: float = 0.16
    Dv: float = 0.08
    F: float = 0.035
    k: float = 0.065

    def __post_init__(self) -> None:
        if self.Du <= 0 or self.Dv <= 0:
            raise ValueError("Diffusion coefficients must be positive")
        if self.F < 0 or self.k < 0:
            raise ValueError("F and k must be non-negative")
        if self.Du < self.Dv:
            # Classic Turing requires inhibitor (often v) to diffuse faster;
            # Gray–Scott uses Du > Dv with u as substrate. Warn only via ratio.
            pass

    @property
    def diffusion_ratio(self) -> float:
        return self.Du / self.Dv

    @property
    def kill_plus_feed(self) -> float:
        return self.F + self.k


@dataclass(frozen=True)
class GridSpec:
    """Spatial discretization of the unit torus [0, L)²."""

    n: int = 128
    L: float = 1.0
    dx: float | None = None

    def __post_init__(self) -> None:
        if self.n < 8:
            raise ValueError("Grid resolution n must be >= 8")
        if self.L <= 0:
            raise ValueError("Domain length L must be positive")
        object.__setattr__(self, "dx", self.L / self.n if self.dx is None else self.dx)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.n)


@dataclass(frozen=True)
class PearsonPattern:
    """Named (F, k) regime from Pearson's 1993 pattern zoo."""

    code: str
    name: str
    F: float
    k: float
    description: str
    morphology: str  # spots | stripes | worms | labyrinth | chaos | waves


# Pearson (1993) / classic Gray–Scott pattern atlas (Du=0.16, Dv=0.08 canonical).
PEARSON_ATLAS: Dict[str, PearsonPattern] = {
    "alpha": PearsonPattern(
        "alpha", "Blue spots on red", 0.014, 0.053,
        "Isolated blue spots on red substrate; slow self-replication.",
        "spots",
    ),
    "beta": PearsonPattern(
        "beta", "Solitons / mitosis", 0.026, 0.051,
        "Self-replicating spots that undergo mitotic splitting.",
        "spots",
    ),
    "gamma": PearsonPattern(
        "gamma", "Stripes", 0.026, 0.055,
        "Labyrinthine stripes and worm-like filaments.",
        "stripes",
    ),
    "delta": PearsonPattern(
        "delta", "Worms", 0.042, 0.059,
        "Transient worms that meander then stabilize.",
        "worms",
    ),
    "epsilon": PearsonPattern(
        "epsilon", "Moving spots", 0.018, 0.055,
        "Moving and interacting localized spots.",
        "spots",
    ),
    "zeta": PearsonPattern(
        "zeta", "Negatons", 0.026, 0.059,
        "Negative (red) spots in blue field.",
        "spots",
    ),
    "eta": PearsonPattern(
        "eta", "Spots and worms", 0.034, 0.060,
        "Mixed morphology: spots coexist with short worms.",
        "worms",
    ),
    "theta": PearsonPattern(
        "theta", "Network", 0.030, 0.057,
        "Interconnected network / coral-like structure.",
        "labyrinth",
    ),
    "iota": PearsonPattern(
        "iota", "Time-dependent", 0.046, 0.0594,
        "Spatiotemporal chaos near Hopf boundary.",
        "chaos",
    ),
    "kappa": PearsonPattern(
        "kappa", "U-skate world", 0.050, 0.063,
        "Exotic U-skate patterns and slow oscillations.",
        "waves",
    ),
    "lambda": PearsonPattern(
        "lambda", "Spots with holes", 0.026, 0.061,
        "Spots that develop internal holes (ring precursors).",
        "spots",
    ),
    "mu": PearsonPattern(
        "mu", "Stripes (fine)", 0.046, 0.065,
        "Fine labyrinthine stripes.",
        "stripes",
    ),
    "xi": PearsonPattern(
        "xi", "Chaos", 0.010, 0.041,
        "Spatiotemporal chaos / turbulence-like fields.",
        "chaos",
    ),
    "pi": PearsonPattern(
        "pi", "Mitosis classic", 0.037, 0.060,
        "Canonical self-replicating spot mitosis regime.",
        "spots",
    ),
    "rho": PearsonPattern(
        "rho", "Stable spots", 0.035, 0.065,
        "Stable non-replicating spot lattice (common demo default).",
        "spots",
    ),
    "sigma": PearsonPattern(
        "sigma", "Worms stable", 0.054, 0.063,
        "Long stable worms / filaments.",
        "worms",
    ),
}
