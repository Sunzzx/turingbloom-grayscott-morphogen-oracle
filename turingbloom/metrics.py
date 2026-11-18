"""Pattern diagnostics for Gray–Scott fields."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .solver import SimulationState


def mass_totals(state: SimulationState, dx: float) -> Tuple[float, float]:
    """Integrated mass ∫ u dA and ∫ v dA over the torus."""
    cell = dx * dx
    return float(np.sum(state.u) * cell), float(np.sum(state.v) * cell)


def concentration_bounds(state: SimulationState) -> dict:
    """Min/max of each field — used for positivity & boundedness invariants."""
    return {
        "u_min": float(np.min(state.u)),
        "u_max": float(np.max(state.u)),
        "v_min": float(np.min(state.v)),
        "v_max": float(np.max(state.v)),
    }


def pattern_entropy(field: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of the empirical concentration histogram (nats).

    Higher entropy ⇒ more spatially disordered / mixed field.
    """
    hist, _ = np.histogram(field.ravel(), bins=bins, density=False)
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def spot_count_estimate(v: np.ndarray, threshold: float | None = None) -> int:
    """Rough connected-component count of high-v regions (spot proxy).

    Uses a simple 4-connected flood fill above a threshold.
    """
    if threshold is None:
        threshold = float(0.5 * (np.mean(v) + np.max(v)))
    mask = v >= threshold
    n = mask.shape[0]
    visited = np.zeros_like(mask, dtype=bool)
    count = 0

    def neighbors(i: int, j: int):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yield (i + di) % n, (j + dj) % n

    for i in range(n):
        for j in range(n):
            if not mask[i, j] or visited[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                ci, cj = stack.pop()
                for ni, nj in neighbors(ci, cj):
                    if mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        stack.append((ni, nj))
    return count


def mean_gradient_energy(field: np.ndarray, dx: float) -> float:
    """(1/|Ω|) ∫ |∇f|² dA via central differences — roughness measure."""
    fx = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dx)
    fy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dx)
    return float(np.mean(fx * fx + fy * fy))
