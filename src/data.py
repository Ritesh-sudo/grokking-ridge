"""Teacher-student data generation for over-parameterised ridge regression.

Setup (paper Sec. 2 / 5.1 / Remark A.14): features phi(x_i) ~ N(0, I_m / m) i.i.d.
(identity feature map, so Phi is the n x m design matrix), noiseless labels
y = Phi @ theta_star, ||theta_star|| = 1 (or the zero teacher), student init
theta(0) ~ N(0, nu2 * I_m).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    Phi: np.ndarray          # (n, m) training features, rows ~ N(0, I/m)
    y: np.ndarray            # (n,) labels  Phi @ theta_star
    theta_star: np.ndarray   # (m,)
    theta0: np.ndarray       # (m,) initialisation
    X_test: np.ndarray       # (n_test, m) fresh held-out features
    y_test: np.ndarray       # (n_test,)
    n: int
    m: int
    nu2: float


def make_problem(n, m, nu2=1.0, seed=0, teacher="unit", n_test=10_000, noise_std=0.0):
    """teacher: 'unit' -> theta* ~ N(0,I) normalised to norm 1 (project default,
    the paper does not say how theta* is drawn); 'zero' -> theta* = 0."""
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((n, m)) / np.sqrt(m)
    if teacher == "zero":
        ts = np.zeros(m)
    elif teacher == "unit":
        ts = rng.standard_normal(m)
        ts /= np.linalg.norm(ts)
    else:
        raise ValueError(teacher)
    theta0 = np.sqrt(nu2) * rng.standard_normal(m)
    X_test = rng.standard_normal((n_test, m)) / np.sqrt(m)
    y = Phi @ ts
    if noise_std > 0:
        y = y + noise_std * rng.standard_normal(n)
    return Problem(Phi, y, ts, theta0, X_test, X_test @ ts, n, m, nu2)


def test_loss_exact(theta, theta_star, m):
    """Population loss E[(<theta-theta*, phi>)^2] = ||theta-theta*||^2 / m for phi ~ N(0, I/m)."""
    d = theta - theta_star
    return float(d @ d) / m
