"""Numerical and mathematical invariant tests for TuringBloom."""

from __future__ import annotations

import numpy as np
import pytest

from turingbloom.atlas import get_pattern, list_patterns, params_from_pattern, recommend_dt
from turingbloom.config import GrayScottParams, GridSpec, PEARSON_ATLAS
from turingbloom.laplacian import (
    apply_laplacian_spectral,
    discrete_laplacian,
    spectral_laplacian_eigenvalues,
)
from turingbloom.metrics import (
    concentration_bounds,
    mass_totals,
    mean_gradient_energy,
    pattern_entropy,
    spot_count_estimate,
)
from turingbloom.solver import GrayScottSimulator
from turingbloom.stability import (
    homogeneous_steady_states,
    is_turing_unstable,
    jacobian_reaction,
    linear_growth_rate,
    turing_dispersion,
)


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def test_laplacian_constant_is_zero():
    field = np.ones((32, 32))
    lap = discrete_laplacian(field, dx=1.0 / 32)
    assert np.allclose(lap, 0.0, atol=1e-12)


def test_laplacian_matches_spectral_on_sine():
    n, L = 64, 2.0 * np.pi
    dx = L / n
    x = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    field = np.sin(2 * X) * np.cos(3 * Y)
    # Analytic: ∇² = -(2²+3²) field = -13 field
    analytic = -13.0 * field
    stencil = discrete_laplacian(field, dx)
    spectral = apply_laplacian_spectral(field, L=L)
    # Spectral is exact for these modes on the grid
    assert np.allclose(spectral, analytic, atol=1e-10)
    # 5-point stencil is second-order accurate
    rel = np.linalg.norm(stencil - analytic) / np.linalg.norm(analytic)
    assert rel < 0.05


def test_spectral_eigenvalues_negative_semidefinite():
    eig = spectral_laplacian_eigenvalues(32, L=1.0)
    assert np.all(eig <= 1e-12)
    assert abs(eig[0, 0]) < 1e-12  # zero mode


# ---------------------------------------------------------------------------
# Parameters & atlas
# ---------------------------------------------------------------------------

def test_atlas_nonempty_and_unique():
    codes = list_patterns()
    assert len(codes) >= 10
    assert len(codes) == len(set(codes))
    for c in codes:
        p = get_pattern(c)
        assert p.F >= 0 and p.k >= 0
        assert p.morphology in {"spots", "stripes", "worms", "labyrinth", "chaos", "waves"}


def test_params_validation():
    with pytest.raises(ValueError):
        GrayScottParams(Du=-0.1, Dv=0.08)
    with pytest.raises(ValueError):
        GridSpec(n=4)


def test_recommend_dt_respects_cfl():
    grid = GridSpec(n=64, L=1.0)
    params = GrayScottParams()
    dt = recommend_dt(grid, params, safety=0.9)
    dx = grid.dx
    dt_max = (dx * dx) / (4.0 * max(params.Du, params.Dv))
    assert dt <= dt_max + 1e-15
    # Simulator accepts recommended dt
    GrayScottSimulator(params=params, grid=grid, dt=dt)


def test_cfl_violation_raises():
    grid = GridSpec(n=32, L=1.0)
    params = GrayScottParams()
    with pytest.raises(ValueError, match="CFL"):
        GrayScottSimulator(params=params, grid=grid, dt=1.0)


# ---------------------------------------------------------------------------
# Stability / equilibria
# ---------------------------------------------------------------------------

def test_trivial_steady_state_always_present():
    for code in ("rho", "pi", "mu", "xi"):
        params = params_from_pattern(code)
        states = homogeneous_steady_states(params)
        assert any(abs(u - 1.0) < 1e-9 and abs(v) < 1e-9 for u, v in states)


def test_jacobian_at_trivial_state():
    params = GrayScottParams(F=0.035, k=0.065)
    J = jacobian_reaction(1.0, 0.0, params)
    # At (1,0): Juu=-F, Juv=0, Jvu=0, Jvv=-(F+k)
    assert abs(J[0, 0] + params.F) < 1e-12
    assert abs(J[0, 1]) < 1e-12
    assert abs(J[1, 0]) < 1e-12
    assert abs(J[1, 1] + (params.F + params.k)) < 1e-12


def test_trivial_state_stable_without_spatial_modes():
    """At q²=0 the trivial state (1,0) has both eigenvalues negative."""
    params = GrayScottParams(F=0.035, k=0.065)
    g0 = linear_growth_rate(params, 1.0, 0.0, q2=0.0)
    assert g0 < 0


def test_dispersion_shape():
    params = params_from_pattern("rho")
    states = homogeneous_steady_states(params)
    u0, v0 = states[-1]
    q2 = np.linspace(0, 200, 50)
    growth = turing_dispersion(params, u0, v0, q2)
    assert growth.shape == q2.shape
    assert np.isfinite(growth).all()


# ---------------------------------------------------------------------------
# Solver invariants
# ---------------------------------------------------------------------------

def test_positivity_preserved():
    params = params_from_pattern("rho")
    grid = GridSpec(n=48, L=2.0)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=11)
    sim.run(500)
    b = concentration_bounds(sim.state)
    assert b["u_min"] >= -1e-12
    assert b["v_min"] >= -1e-12
    assert b["u_max"] < 2.0
    assert np.isfinite(sim.state.u).all() and np.isfinite(sim.state.v).all()


def test_mass_finite_and_substrate_bounded():
    params = params_from_pattern("pi")
    grid = GridSpec(n=32, L=2.0)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=2)
    mu0, mv0 = mass_totals(sim.state, grid.dx)
    sim.run(800)
    mu1, mv1 = mass_totals(sim.state, grid.dx)
    assert np.isfinite(mu1) and np.isfinite(mv1)
    # Total domain area = L²
    area = grid.L * grid.L
    assert 0 <= mu1 <= area * 1.6
    assert 0 <= mv1 <= area * 1.6
    # Mass should not explode
    assert mu1 < 10 * max(mu0, 1e-6) + area


def test_homogeneous_equilibrium_stays_put():
    """Exact (1,0) with zero noise remains at equilibrium under the PDE."""
    params = GrayScottParams(F=0.04, k=0.06)
    grid = GridSpec(n=24, L=1.0)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=0)
    sim.state.u[:] = 1.0
    sim.state.v[:] = 0.0
    sim.run(200)
    assert np.allclose(sim.state.u, 1.0, atol=1e-10)
    assert np.allclose(sim.state.v, 0.0, atol=1e-10)


def test_seed_breaks_homogeneity_and_grows_structure():
    """Central seed + noise should sustain spatial structure (spots / mitosis)."""
    params = params_from_pattern("beta")  # self-replicating spots (robust)
    grid = GridSpec(n=64, L=64.0)  # Pearson dx=1
    dt = recommend_dt(grid, params, safety=0.5)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=5)
    # Stronger localized seed, mild noise
    sim.reset(noise=0.01, seed_block=0.20, u_seed=0.50, v_seed=0.25)
    e0 = mean_gradient_energy(sim.state.v, grid.dx)
    sim.run(2000)
    e1 = mean_gradient_energy(sim.state.v, grid.dx)
    # Pattern must persist (v not collapsed to homogeneous zero)
    assert sim.state.v.max() > 0.1
    assert spot_count_estimate(sim.state.v) >= 1 or e1 > 0.005
    assert sim.state.v.max() > sim.state.v.mean()


def test_reaction_only_conservation_relation():
    """At each point, Ru + Rv = F(1-u) − (F+k)v (diffusion-free identity)."""
    params = params_from_pattern("mu")
    grid = GridSpec(n=16, L=1.0)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=9)
    ru, rv = sim.reaction_rates()
    expected = params.F * (1.0 - sim.state.u) - (params.F + params.k) * sim.state.v
    assert np.allclose(ru + rv, expected, atol=1e-12)


def test_record_history_length():
    params = params_from_pattern("rho")
    grid = GridSpec(n=16, L=1.0)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=1)
    hist = sim.run(100, record_every=25)
    # initial + 4 snapshots
    assert len(hist) == 5
    assert hist[-1].step == 100


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_entropy_bounds():
    uniform = np.ones((20, 20)) * 0.5
    H_u = pattern_entropy(uniform, bins=16)
    noise = np.random.default_rng(0).random((20, 20))
    H_n = pattern_entropy(noise, bins=16)
    assert H_u < H_n
    assert H_u >= 0 and H_n >= 0


def test_spot_count_on_synthetic_blobs():
    field = np.zeros((40, 40))
    field[5:10, 5:10] = 1.0
    field[25:32, 20:28] = 1.0
    field[15:18, 30:35] = 1.0
    assert spot_count_estimate(field, threshold=0.5) == 3


# ---------------------------------------------------------------------------
# End-to-end multi-pattern smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["rho", "pi", "mu", "beta", "sigma"])
def test_pattern_smoke_runs(code):
    params = params_from_pattern(code)
    grid = GridSpec(n=32, L=2.0)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=42)
    sim.run(300)
    b = concentration_bounds(sim.state)
    assert b["v_min"] >= -1e-12
    assert np.isfinite(sim.state.u).all()


def test_demo_simulate_cli_importable():
    from turingbloom.demo import main

    rc = main(["simulate", "--pattern", "rho", "--n", "24", "--steps", "100"])
    assert rc == 0
    rc = main(["info"])
    assert rc == 0
    rc = main(["patterns"])
    assert rc == 0


def test_version_exported():
    import turingbloom

    assert isinstance(turingbloom.__version__, str)
    assert len(PEARSON_ATLAS) == len(list_patterns())
