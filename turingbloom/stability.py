"""Linear Turing stability analysis for the Gray–Scott system."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import GrayScottParams


def homogeneous_steady_states(params: GrayScottParams) -> list[Tuple[float, float]]:
    """Analytic homogeneous equilibria of Gray–Scott kinetics (no diffusion).

    Always includes the trivial state (u, v) = (1, 0).
    Nontrivial roots solve a quadratic when the discriminant is positive.
    """
    F, k = params.F, params.k
    states: list[Tuple[float, float]] = [(1.0, 0.0)]

    # From ∂v/∂t = 0 with v≠0: u = (F+k)/v  wait — actually:
    # uv² − (F+k)v = 0 ⇒ u v = (F+k) if v≠0 ⇒ u = (F+k)/v
    # From ∂u/∂t = 0: −u v² + F(1−u) = 0
    # Substitute: −(F+k) v + F − F(F+k)/v = 0
    # Multiply by v: −(F+k) v² + F v − F(F+k) = 0
    # (F+k) v² − F v + F(F+k) = 0
    a = F + k
    if a <= 0:
        return states
    b = -F
    c = F * a
    disc = b * b - 4 * a * c
    if disc < 0:
        return states
    sqrt_d = np.sqrt(disc)
    for sign in (+1.0, -1.0):
        v = (-b + sign * sqrt_d) / (2 * a)
        if v <= 0:
            continue
        u = a / v  # u = (F+k)/v from uv = F+k
        # Also check consistency with feed equation: u = 1 - u v² / F when F>0
        if 0.0 <= u <= 1.0 + 1e-9:
            states.append((float(u), float(v)))
    # Deduplicate
    unique: list[Tuple[float, float]] = []
    for s in states:
        if not any(abs(s[0] - t[0]) < 1e-9 and abs(s[1] - t[1]) < 1e-9 for t in unique):
            unique.append(s)
    return unique


def jacobian_reaction(u: float, v: float, params: GrayScottParams) -> np.ndarray:
    """Jacobian of the reaction vector field R(u,v) at (u,v).

    R_u = −u v² + F(1 − u)
    R_v =  u v² − (F + k) v

    J = [[ −v² − F,  −2 u v ],
         [  v²,     2 u v − (F+k) ]]
    """
    F, k = params.F, params.k
    juu = -(v * v) - F
    juv = -2.0 * u * v
    jvu = v * v
    jvv = 2.0 * u * v - (F + k)
    return np.array([[juu, juv], [jvu, jvv]], dtype=float)


def turing_dispersion(
    params: GrayScottParams,
    u0: float,
    v0: float,
    q2: np.ndarray | float,
) -> np.ndarray:
    """Dispersion relation λ(q²) for Fourier mode wave-number squared q².

    Characteristic polynomial of J − q² D, D = diag(Du, Dv):
        det(J − q² D − λ I) = 0

    Returns the eigenvalue with largest real part (growth rate).
    """
    J = jacobian_reaction(u0, v0, params)
    Du, Dv = params.Du, params.Dv
    q2_arr = np.asarray(q2, dtype=float)
    scalar = q2_arr.ndim == 0
    q2_flat = np.atleast_1d(q2_arr).ravel()

    growth = np.empty(q2_flat.shape, dtype=float)
    for i, q2i in enumerate(q2_flat):
        M = J - q2i * np.diag([Du, Dv])
        eigs = np.linalg.eigvals(M)
        growth[i] = float(np.max(np.real(eigs)))
    if scalar:
        return growth[0]
    return growth.reshape(q2_arr.shape)


def is_turing_unstable(
    params: GrayScottParams,
    u0: float | None = None,
    v0: float | None = None,
    q2_grid: np.ndarray | None = None,
) -> bool:
    """True if some spatial mode has positive linear growth (Turing instability)."""
    if u0 is None or v0 is None:
        states = homogeneous_steady_states(params)
        # Prefer nontrivial steady state if present
        u0, v0 = states[-1]

    # Diffusion-free stability of the homogeneous state is required for classical Turing
    # (stable without diffusion, unstable with). We report spatial instability broadly.
    if q2_grid is None:
        q2_grid = np.linspace(0.0, 500.0, 256)
    growth = turing_dispersion(params, u0, v0, q2_grid)
    # Require some q² > 0 with growth > 0 (spatial mode)
    return bool(np.any((q2_grid > 1e-12) & (growth > 1e-8)))


def linear_growth_rate(
    params: GrayScottParams,
    u0: float,
    v0: float,
    q2: float,
) -> float:
    """Scalar max Re(λ) at a single q²."""
    return float(turing_dispersion(params, u0, v0, q2))
