"""Pearson pattern lookup and timestep heuristics."""

from __future__ import annotations

from typing import List

from .config import PEARSON_ATLAS, GrayScottParams, GridSpec, PearsonPattern


def list_patterns() -> List[str]:
    return sorted(PEARSON_ATLAS.keys())


def get_pattern(code: str) -> PearsonPattern:
    key = code.strip().lower()
    if key not in PEARSON_ATLAS:
        known = ", ".join(list_patterns())
        raise KeyError(f"Unknown pattern '{code}'. Known: {known}")
    return PEARSON_ATLAS[key]


def params_from_pattern(
    code: str,
    Du: float = 0.16,
    Dv: float = 0.08,
) -> GrayScottParams:
    pat = get_pattern(code)
    return GrayScottParams(Du=Du, Dv=Dv, F=pat.F, k=pat.k)


def recommend_dt(grid: GridSpec, params: GrayScottParams, safety: float = 0.9) -> float:
    """Recommend an explicit FTCS timestep under the 2-D diffusion CFL."""
    dx = grid.dx  # type: ignore[union-attr]
    assert dx is not None
    dmax = max(params.Du, params.Dv)
    return safety * (dx * dx) / (4.0 * dmax)
