"""
test_model.py  —  run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.model       import HeartEMSystem, DEFAULT_IC
from src.lyapunov    import compute_lyapunov
from src.scaling_law import gcrit_polynomial, SCALING_COEFFS


def test_rhs_length():
    assert len(HeartEMSystem().rhs(0.0, DEFAULT_IC)) == 6

def test_rhs_decoupled():
    sys = HeartEMSystem(g1=0.0, g2=0.0)
    E,dE,x,dx,p,dp = DEFAULT_IC
    out = sys.rhs(0.0, DEFAULT_IC)
    expected = -sys.wE**2 * E - sys.alpha * E**3
    assert abs(out[1] - expected) < 1e-12

def test_integrate_success():
    sol = HeartEMSystem().integrate(T=20.0, dt=0.1)
    assert sol.success
    assert sol.y.shape == (6, 200)

def test_lyapunov_returns_float():
    lam = compute_lyapunov(HeartEMSystem(g1=0.5), T=80)
    assert isinstance(lam, float)

def test_gcrit_polynomial():
    a, b, c = SCALING_COEFFS
    assert abs(gcrit_polynomial(1.0) - (a + b + c)) < 1e-12

def test_gcrit_positive():
    for w0 in [0.3, 1.0, 2.0]:
        assert gcrit_polynomial(w0) > 0
