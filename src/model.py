"""
model.py
Core coupled ODE system for the biomorphic Physical AI heart model.

Equations (paper Sec. II.C):
    E'' = -wE^2 * E - alpha * E^3 + g1 * x
    x'' = -w0^2 * x - mu*(x^2-1)*x' + g1*E + g2*p
    p'' = -wp^2 * p + g2 * x

State vector: y = [E, E', x, x', p, p']
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field


DEFAULT_IC = [0.2, 0.0, 0.5, 0.0, 0.1, 0.0]


@dataclass
class HeartEMSystem:
    """
    Three-component multi-scale coupled oscillator.

    Parameters
    ----------
    wE    : EM field frequency (rad/s)
    w0    : Heart oscillator frequency (rad/s)
    wp    : Vascular wave frequency (rad/s)
    mu    : Van der Pol nonlinearity
    alpha : Duffing cubic coefficient
    g1    : EM-heart coupling  <-- primary bifurcation parameter
    g2    : Heart-vascular coupling
    ic    : Initial conditions [E, E', x, x', p, p']
    """
    wE:    float = 2.0
    w0:    float = 1.0
    wp:    float = 1.5
    mu:    float = 0.8
    alpha: float = 0.05
    g1:    float = 0.8
    g2:    float = 0.2
    ic:    list  = field(default_factory=lambda: list(DEFAULT_IC))

    def rhs(self, t, y):
        E, dE, x, dx, p, dp = y
        ddE = -self.wE**2 * E - self.alpha * E**3 + self.g1 * x
        ddx = -self.w0**2 * x - self.mu * (x**2 - 1.0) * dx + self.g1 * E + self.g2 * p
        ddp = -self.wp**2 * p + self.g2 * x
        return [dE, ddE, dx, ddx, dp, ddp]

    def integrate(self, T=300.0, dt=0.05, rtol=1e-5, atol=1e-7, ic=None):
        y0 = ic if ic is not None else self.ic
        t_eval = np.arange(0.0, T, dt)
        sol = solve_ivp(self.rhs, (0.0, T), y0,
                        t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return sol

    def steady_state(self, T=300.0, transient=0.5, **kwargs):
        """Return solution with transient discarded."""
        sol = self.integrate(T=T, **kwargs)
        skip = int(sol.t.size * transient)
        return sol.t[skip:], sol.y[:, skip:]
