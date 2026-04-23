"""
scaling_law.py
Chaos-onset scaling law: g_crit(w0) ~ a*w0^2 + b*w0 + c  (paper Eq. 13)
Best-fit: a=0.312, b=-0.621, c=0.624  (R^2 > 0.97)
"""

import numpy as np
from .lyapunov import find_gcrit

SCALING_COEFFS = (0.312, -0.621, 0.624)   # published best-fit (a, b, c)


def gcrit_polynomial(w0, coeffs=SCALING_COEFFS):
    """Evaluate g_crit(w0) = a*w0^2 + b*w0 + c."""
    a, b, c = coeffs
    return a * w0**2 + b * w0 + c


def compute_scaling_law(w0_values=None, g1_min=0.05, g1_max=1.8,
                        n_g1=30, verbose=True, **lyapunov_kwargs):
    """
    Numerically compute g_crit for each w0 and fit polynomial.

    Returns
    -------
    w0_arr   : array of w0 values
    gcrit_arr: numerically computed g_crit values
    coeffs   : best-fit (a, b, c)
    """
    if w0_values is None:
        w0_values = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.3])

    gcrit_list = []
    for w0 in w0_values:
        gc = find_gcrit(w0, g1_min=g1_min, g1_max=g1_max,
                        n_points=n_g1, **lyapunov_kwargs)
        gcrit_list.append(gc)
        if verbose:
            print(f"  w0={w0:.2f}  ->  g_crit={gc:.4f}")

    gcrit_arr = np.array(gcrit_list)
    coeffs    = tuple(np.polyfit(w0_values, gcrit_arr, 2))

    if verbose:
        a, b, c = coeffs
        print(f"\nFit: g_crit = {a:.4f}*w0^2 + {b:.4f}*w0 + {c:.4f}")

    return w0_values, gcrit_arr, coeffs


def save_scaling_data(w0_arr, gcrit_arr, path="data/gcrit_vs_w0.csv"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.column_stack([w0_arr, gcrit_arr]),
               delimiter=',', header="w0_rad_per_s,gcrit", comments='')
    print(f"Saved -> {path}")
