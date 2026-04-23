from .model       import HeartEMSystem, DEFAULT_IC
from .lyapunov    import compute_lyapunov, lyapunov_vs_g1, find_gcrit
from .bifurcation import bifurcation_diagram, poincare_section
from .scaling_law import compute_scaling_law, gcrit_polynomial, SCALING_COEFFS

__version__ = "1.0.0"
__author__  = "Chur Chin"
__email__   = "tpotaoai@gmail.com"
