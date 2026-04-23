"""
lyapunov.py
Maximal Lyapunov exponent via two-trajectory divergence method.
Ref: Benettin et al. (1980), Meccanica 15, 9-30.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .model import HeartEMSystem


def compute_lyapunov(system, T=300.0, dt=0.05, eps=1e-6,
                     transient_frac=1/3, rtol=1e-5, atol=1e-7):
    """
    Compute maximal Lyapunov exponent lambda_max.

    lambda_max > 0  -->  chaotic
    lambda_max <= 0 -->  regular

    Parameters
    ----------
    system        : HeartEMSystem instance
    T             : integration time [s]
    dt            : output timestep [s]
    eps           : initial perturbation size
    transient_frac: fraction of T to discard as transient

    Returns
    -------
    lam : float
    """
    y0  = list(system.ic)
    y0p = list(y0)
    y0p[0] += eps          # perturb E by eps

    t_eval = np.arange(0.0, T, dt)

    def _run(ic):
        return solve_ivp(system.rhs, (0.0, T), ic,
                         t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)

    sol  = _run(y0)
    solp = _run(y0p)

    diff = np.sqrt(np.sum((sol.y - solp.y) ** 2, axis=0))
    diff = np.maximum(diff, 1e-300)

    skip = int(len(t_eval) * transient_frac)
    return float(np.mean(np.log(diff[skip:] / eps)) / dt)


def lyapunov_vs_g1(w0, g1_values, **kwargs):
    """Sweep g1 at fixed w0, return array of lambda_max."""
    return np.array([compute_lyapunov(HeartEMSystem(w0=w0, g1=g), **kwargs)
                     for g in g1_values])


def find_gcrit(w0, g1_min=0.05, g1_max=1.8, n_points=30,
               lam_threshold=0.0, **kwargs):
    """
    Find chaos onset threshold g_crit for given w0.
    Returns the first g1 where lambda_max crosses lam_threshold.
    """
    g1_arr = np.linspace(g1_min, g1_max, n_points)
    lams   = lyapunov_vs_g1(w0, g1_arr, **kwargs)

    for j in range(len(lams) - 1):
        if lams[j] > lam_threshold and lams[j + 1] <= lam_threshold:
            return float((g1_arr[j] + g1_arr[j + 1]) / 2)

    return float(g1_arr[np.argmax(lams)])
