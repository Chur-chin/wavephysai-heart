"""
bifurcation.py
Bifurcation diagram and Poincare section for the heart-EM system.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from .model import HeartEMSystem


def bifurcation_diagram(w0=1.0, g1_values=None,
                        T=300.0, transient=200.0, dt=0.05,
                        n_peaks=50, rtol=1e-5, atol=1e-7):
    """
    Compute bifurcation diagram: local maxima of x(t) vs g1.

    Returns
    -------
    g_out   : array of g1 values (one per peak)
    x_peaks : corresponding x peak values
    """
    if g1_values is None:
        g1_values = np.linspace(0.05, 1.5, 100)

    t_eval = np.arange(transient, T, dt)
    g_out, x_peaks = [], []

    for g1 in g1_values:
        sys = HeartEMSystem(w0=w0, g1=g1)
        sol = solve_ivp(sys.rhs, (0.0, T), sys.ic,
                        t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)
        if not sol.success:
            continue
        peaks, _ = find_peaks(sol.y[2], height=0, distance=4)
        if len(peaks) == 0:
            continue
        vals = sol.y[2][peaks][-n_peaks:]
        g_out.extend([g1] * len(vals))
        x_peaks.extend(vals)

    return np.array(g_out), np.array(x_peaks)


def poincare_section(system, T=350.0, transient=150.0, dt=0.05):
    """
    Poincare section in (x, E) plane.
    Condition: upward zero-crossings of x-dot.

    Returns
    -------
    x_cross, E_cross : arrays
    """
    sol  = system.integrate(T=T, dt=dt)
    mask = sol.t > transient

    x_arr  = sol.y[2, mask]
    dx_arr = sol.y[3, mask]
    E_arr  = sol.y[0, mask]

    cross = np.where((dx_arr[:-1] < 0) & (dx_arr[1:] >= 0))[0]
    return x_arr[cross], E_arr[cross]
