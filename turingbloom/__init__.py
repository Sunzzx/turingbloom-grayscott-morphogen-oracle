"""TuringBloom — Gray–Scott reaction–diffusion morphogen oracle.

Implements the cubic autocatalytic Gray–Scott PDE system that produces
Turing patterns (spots, stripes, worms, labyrinths, self-replicating
spots) — the mathematical backbone of developmental morphogenesis.
"""

from .config import GrayScottParams, GridSpec, PearsonPattern, PEARSON_ATLAS
from .laplacian import discrete_laplacian, spectral_laplacian_eigenvalues
from .solver import GrayScottSimulator, SimulationState
from .stability import turing_dispersion, is_turing_unstable, linear_growth_rate
from .metrics import mass_totals, pattern_entropy, spot_count_estimate, concentration_bounds
from .atlas import list_patterns, get_pattern, recommend_dt

__version__ = "0.1.0"
__all__ = [
    "GrayScottParams",
    "GridSpec",
    "PearsonPattern",
    "PEARSON_ATLAS",
    "discrete_laplacian",
    "spectral_laplacian_eigenvalues",
    "GrayScottSimulator",
    "SimulationState",
    "turing_dispersion",
    "is_turing_unstable",
    "linear_growth_rate",
    "mass_totals",
    "pattern_entropy",
    "spot_count_estimate",
    "concentration_bounds",
    "list_patterns",
    "get_pattern",
    "recommend_dt",
    "__version__",
]
