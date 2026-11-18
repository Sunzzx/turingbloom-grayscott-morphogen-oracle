"""Gray–Scott reaction–diffusion time integrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import GrayScottParams, GridSpec
from .laplacian import discrete_laplacian


@dataclass
class SimulationState:
    """Snapshot of the RD fields and clock."""

    u: np.ndarray
    v: np.ndarray
    t: float = 0.0
    step: int = 0

    def copy(self) -> "SimulationState":
        return SimulationState(self.u.copy(), self.v.copy(), self.t, self.step)


@dataclass
class GrayScottSimulator:
    """Forward-Euler / FTCS integrator for the Gray–Scott system on a torus.

    ∂u/∂t = Du ∇²u − u v² + F (1 − u)
    ∂v/∂t = Dv ∇²v + u v² − (F + k) v
    """

    params: GrayScottParams = field(default_factory=GrayScottParams)
    grid: GridSpec = field(default_factory=GridSpec)
    dt: float = 1.0
    seed: Optional[int] = 42
    _rng: np.random.Generator = field(init=False, repr=False)
    state: SimulationState = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._validate_cfl()
        self.state = self.initial_state()

    def _validate_cfl(self) -> None:
        """Explicit diffusion CFL: dt <= dx² / (4 max(Du,Dv)) for 2-D 5-point stencil."""
        dx = self.grid.dx  # type: ignore[union-attr]
        assert dx is not None
        dmax = max(self.params.Du, self.params.Dv)
        dt_max = (dx * dx) / (4.0 * dmax + 1e-15)
        if self.dt > dt_max * 1.01:
            raise ValueError(
                f"dt={self.dt} violates diffusion CFL (max≈{dt_max:.6f} for dx={dx})"
            )

    def initial_state(
        self,
        noise: float = 0.02,
        seed_block: float = 0.25,
        u_bg: float = 1.0,
        v_bg: float = 0.0,
        u_seed: float = 0.50,
        v_seed: float = 0.25,
    ) -> SimulationState:
        """Pearson-style initial condition: uniform (1,0) + central square seed + noise."""
        n = self.grid.n
        u = np.full((n, n), u_bg, dtype=np.float64)
        v = np.full((n, n), v_bg, dtype=np.float64)

        half = max(1, int(n * seed_block / 2))
        c = n // 2
        sl = slice(c - half, c + half)
        u[sl, sl] = u_seed
        v[sl, sl] = v_seed

        if noise > 0:
            u += noise * self._rng.standard_normal((n, n))
            v += noise * self._rng.standard_normal((n, n))
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)

        return SimulationState(u=u, v=v, t=0.0, step=0)

    def reset(self, **kwargs) -> SimulationState:
        self._rng = np.random.default_rng(self.seed)
        self.state = self.initial_state(**kwargs)
        return self.state

    def step_once(self) -> SimulationState:
        """One explicit FTCS step with concentration clamping to [0, ∞) physically."""
        p = self.params
        dx = self.grid.dx  # type: ignore[union-attr]
        assert dx is not None
        u, v = self.state.u, self.state.v
        dt = self.dt

        lap_u = discrete_laplacian(u, dx)
        lap_v = discrete_laplacian(v, dx)
        uvv = u * v * v

        u_new = u + dt * (p.Du * lap_u - uvv + p.F * (1.0 - u))
        v_new = v + dt * (p.Dv * lap_v + uvv - (p.F + p.k) * v)

        # Physical non-negativity (concentrations cannot go negative)
        np.maximum(u_new, 0.0, out=u_new)
        np.maximum(v_new, 0.0, out=v_new)
        # Substrate u saturates near 1 under feed; soft-cap for numerical safety
        np.minimum(u_new, 1.5, out=u_new)

        self.state.u = u_new
        self.state.v = v_new
        self.state.t += dt
        self.state.step += 1
        return self.state

    def run(self, steps: int, record_every: int = 0) -> list[SimulationState]:
        """Integrate `steps` timesteps. Optionally record snapshots every `record_every`."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        history: list[SimulationState] = []
        if record_every > 0:
            history.append(self.state.copy())
        for i in range(steps):
            self.step_once()
            if record_every > 0 and (i + 1) % record_every == 0:
                history.append(self.state.copy())
        return history

    def reaction_rates(self) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise reaction terms (no diffusion) at current state."""
        p = self.params
        u, v = self.state.u, self.state.v
        uvv = u * v * v
        ru = -uvv + p.F * (1.0 - u)
        rv = uvv - (p.F + p.k) * v
        return ru, rv
