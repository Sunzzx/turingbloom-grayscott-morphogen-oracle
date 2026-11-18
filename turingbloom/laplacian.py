"""Discrete Laplacian operators on the periodic torus."""

from __future__ import annotations

import numpy as np


def discrete_laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """5-point stencil Laplacian with periodic (toroidal) boundaries.

    ∇²f ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} − 4 f_{i,j}) / dx²
    """
    if field.ndim != 2:
        raise ValueError("field must be 2-D")
    if dx <= 0:
        raise ValueError("dx must be positive")

    # np.roll is O(N) and enforces periodic BC without halo buffers.
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def spectral_laplacian_eigenvalues(n: int, L: float = 1.0) -> np.ndarray:
    """Eigenvalues of the continuous Laplacian on [0,L)² with periodic BC.

    For wave numbers kx = 2π mx / L, ky = 2π my / L:
        λ(mx, my) = −(kx² + ky²)

    Returned as an (n, n) array indexed by integer modes in FFT order.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    freqs = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
    return -(kx * kx + ky * ky)


def apply_laplacian_spectral(field: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Exact (to grid) Laplacian via FFT for verification against stencil."""
    n = field.shape[0]
    eig = spectral_laplacian_eigenvalues(n, L)
    return np.fft.ifft2(np.fft.fft2(field) * eig).real
