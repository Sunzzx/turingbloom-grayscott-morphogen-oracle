"""Interactive ANSI demo for TuringBloom Gray–Scott morphogenesis."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .atlas import get_pattern, list_patterns, params_from_pattern, recommend_dt
from .config import GrayScottParams, GridSpec
from .metrics import concentration_bounds, mass_totals, pattern_entropy, spot_count_estimate
from .solver import GrayScottSimulator
from .stability import homogeneous_steady_states, is_turing_unstable


# ANSI 256-color ramp (dark → bright cyan/magenta for activator v)
_RAMP = " .:-=+*#%@"


def _ansi_frame(v: np.ndarray, width: int = 64) -> str:
    """Downsample v field to ANSI block art."""
    n = v.shape[0]
    # Target rows ≈ width/2 for roughly square terminal cells
    rows = max(8, width // 2)
    cols = width
    ys = np.linspace(0, n - 1, rows).astype(int)
    xs = np.linspace(0, n - 1, cols).astype(int)
    sub = v[np.ix_(ys, xs)]
    vmin, vmax = float(sub.min()), float(sub.max())
    span = max(vmax - vmin, 1e-12)
    norm = (sub - vmin) / span
    lines = []
    for r in range(rows):
        chars = []
        for c in range(cols):
            idx = int(norm[r, c] * (len(_RAMP) - 1))
            ch = _RAMP[idx]
            # Color by intensity
            level = 232 + int(norm[r, c] * 23)  # grayscale 232-255
            # Mix teal for high activator
            if norm[r, c] > 0.55:
                color = 51  # cyan-ish
            elif norm[r, c] > 0.3:
                color = 39
            else:
                color = level
            chars.append(f"\033[38;5;{color}m{ch}\033[0m")
        lines.append("".join(chars))
    return "\n".join(lines)


def run_demo(
    pattern: str = "rho",
    n: int = 96,
    steps: int = 4000,
    frame_every: int = 100,
    width: int = 72,
    live: bool = True,
) -> int:
    pat = get_pattern(pattern)
    params = params_from_pattern(pattern)
    grid = GridSpec(n=n, L=2.5)
    dt = recommend_dt(grid, params, safety=0.85)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=7)

    states = homogeneous_steady_states(params)
    turing = is_turing_unstable(params)

    header = (
        f"TuringBloom · Gray–Scott Morphogen Oracle\n"
        f"Pattern: {pat.code} — {pat.name} ({pat.morphology})\n"
        f"F={params.F}, k={params.k}, Du={params.Du}, Dv={params.Dv}\n"
        f"Grid {n}×{n}, dt={dt:.5f}, steps={steps}\n"
        f"Steady states: {states}\n"
        f"Turing-unstable (linear): {turing}\n"
        f"{pat.description}\n"
        + "─" * width
    )
    print(header)

    t0 = time.perf_counter()
    for s in range(0, steps, frame_every):
        sim.run(frame_every)
        st = sim.state
        mu, mv = mass_totals(st, grid.dx)  # type: ignore[arg-type]
        bounds = concentration_bounds(st)
        H = pattern_entropy(st.v)
        spots = spot_count_estimate(st.v)
        frame = _ansi_frame(st.v, width=width)
        status = (
            f"t={st.t:8.1f}  step={st.step:6d}  "
            f"∫u={mu:.4f} ∫v={mv:.4f}  "
            f"v∈[{bounds['v_min']:.3f},{bounds['v_max']:.3f}]  "
            f"H(v)={H:.3f}  spots≈{spots}"
        )
        if live and sys.stdout.isatty():
            # Clear screen + home
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(header + "\n")
            sys.stdout.write(frame + "\n")
            sys.stdout.write(status + "\n")
            sys.stdout.flush()
        else:
            if s == 0 or s + frame_every >= steps:
                print(frame)
                print(status)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s ({steps/elapsed:.0f} steps/s)")
    return 0


def run_benchmark(n: int = 128, steps: int = 2000, pattern: str = "rho") -> int:
    params = params_from_pattern(pattern)
    grid = GridSpec(n=n, L=2.5)
    dt = recommend_dt(grid, params)
    sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=1)
    t0 = time.perf_counter()
    sim.run(steps)
    elapsed = time.perf_counter() - t0
    cells = n * n
    print(
        f"benchmark pattern={pattern} n={n} steps={steps}\n"
        f"  wall={elapsed:.3f}s  steps/s={steps/elapsed:.1f}  "
        f"cell-updates/s={cells*steps/elapsed:.2e}"
    )
    return 0


def run_info() -> int:
    print("TuringBloom — Gray–Scott Reaction–Diffusion Morphogen Oracle")
    print()
    print("PDE system:")
    print("  ∂u/∂t = Du ∇²u − u v² + F (1 − u)")
    print("  ∂v/∂t = Dv ∇²v + u v² − (F + k) v")
    print()
    print("Pearson atlas patterns:")
    for code in list_patterns():
        p = get_pattern(code)
        print(f"  {p.code:8s} F={p.F:.4f} k={p.k:.4f}  [{p.morphology:9s}] {p.name}")
    print()
    print("Complexity: O(N) per step for N = n² grid cells (5-point stencil).")
    print("Humanity impact: computational morphogenesis for developmental biology,")
    print("tissue patterning, synthetic biology design, and materials self-assembly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="turingbloom",
        description="Gray–Scott Turing morphogenesis oracle",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Live ANSI morphogenesis visualization")
    p_demo.add_argument("--pattern", default="rho", choices=list_patterns())
    p_demo.add_argument("--n", type=int, default=96)
    p_demo.add_argument("--steps", type=int, default=4000)
    p_demo.add_argument("--frame-every", type=int, default=100)
    p_demo.add_argument("--width", type=int, default=72)
    p_demo.add_argument("--no-live", action="store_true")

    p_bench = sub.add_parser("benchmark", help="Throughput benchmark")
    p_bench.add_argument("--pattern", default="rho", choices=list_patterns())
    p_bench.add_argument("--n", type=int, default=128)
    p_bench.add_argument("--steps", type=int, default=2000)

    sub.add_parser("info", help="Theory & atlas summary")
    sub.add_parser("patterns", help="List Pearson atlas codes")

    p_sim = sub.add_parser("simulate", help="Headless run + final metrics JSON-ish")
    p_sim.add_argument("--pattern", default="pi", choices=list_patterns())
    p_sim.add_argument("--n", type=int, default=64)
    p_sim.add_argument("--steps", type=int, default=3000)

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        return run_demo(
            pattern=args.pattern,
            n=args.n,
            steps=args.steps,
            frame_every=args.frame_every,
            width=args.width,
            live=not args.no_live,
        )
    if args.cmd == "benchmark":
        return run_benchmark(n=args.n, steps=args.steps, pattern=args.pattern)
    if args.cmd == "info":
        return run_info()
    if args.cmd == "patterns":
        for c in list_patterns():
            p = get_pattern(c)
            print(f"{p.code}\tF={p.F}\tk={p.k}\t{p.morphology}\t{p.name}")
        return 0
    if args.cmd == "simulate":
        params = params_from_pattern(args.pattern)
        grid = GridSpec(n=args.n, L=2.5)
        dt = recommend_dt(grid, params)
        sim = GrayScottSimulator(params=params, grid=grid, dt=dt, seed=3)
        sim.run(args.steps)
        st = sim.state
        mu, mv = mass_totals(st, grid.dx)  # type: ignore[arg-type]
        b = concentration_bounds(st)
        print(
            f"pattern={args.pattern} steps={args.steps} t={st.t:.2f} "
            f"mass_u={mu:.6f} mass_v={mv:.6f} "
            f"u=[{b['u_min']:.4f},{b['u_max']:.4f}] "
            f"v=[{b['v_min']:.4f},{b['v_max']:.4f}] "
            f"entropy_v={pattern_entropy(st.v):.4f} "
            f"spots={spot_count_estimate(st.v)}"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
