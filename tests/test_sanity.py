import numpy as np
from src.data import make_problem
from src.train import (run_explicit_gd, SpectralGD, ridge_closed_form, min_norm_interpolant,
                       log_steps, critical_times)


def _prob(seed=0, n=30, m=200, **kw):
    return make_problem(n, m, nu2=kw.pop("nu2", 1.0), seed=seed, n_test=500, **kw)


def test_lambda_zero_converges_to_min_norm_plus_init_orthogonal_part():
    """lam=0: GD interpolates and the data-subspace component converges to the pinv solution.
    From theta0 = 0 the whole iterate equals the minimum-norm interpolant; from random init the
    iterate = pinv solution + the (unchanged) component of theta0 orthogonal to row-space(Phi)."""
    p = _prob()
    p0 = _prob(); p0.theta0 = np.zeros(p0.m)
    out = run_explicit_gd(p0, eta=1.0, lam=0.0, steps=np.array([0, 40000]))
    t_mn = min_norm_interpolant(p0.Phi, p0.y)
    assert np.linalg.norm(out["theta_final"] - t_mn) / np.linalg.norm(t_mn) < 1e-6
    out = run_explicit_gd(p, 1.0, 0.0, np.array([0, 40000]))
    V = np.linalg.svd(p.Phi, full_matrices=False)[2].T
    perp = p.theta0 - V @ (V.T @ p.theta0)
    expect = min_norm_interpolant(p.Phi, p.y) + perp
    assert np.linalg.norm(out["theta_final"] - expect) / np.linalg.norm(expect) < 1e-6


def test_lambda_positive_converges_to_closed_form_ridge():
    p = _prob()
    lam = 1e-2
    out = run_explicit_gd(p, 1.0, lam, np.array([0, 20000]))
    th = ridge_closed_form(p.Phi, p.y, lam)
    # the closed form is also the minimiser of the paper's objective: gradient must vanish
    g = p.Phi.T @ (p.Phi @ th - p.y) / p.n + lam * th
    assert np.linalg.norm(g) < 1e-10
    assert np.linalg.norm(out["theta_final"] - th) / np.linalg.norm(th) < 1e-6


def test_regularised_objective_decreases_monotonically():
    """Full objective L_n(theta;lam) decreases monotonically for eta < 1/(lam + lmax/n).
    (The *unregularised* train loss L_n is not guaranteed monotone with weight decay; we check it
    separately in the lam=0 case.)"""
    from src.theory import stability_limit
    p = _prob()
    lam = 1e-3
    eta = 0.9 * stability_limit(p.Phi, lam)
    out = run_explicit_gd(p, eta, lam, np.arange(0, 2000))
    assert np.all(np.diff(out["obj_reg"]) <= 1e-15)
    out0 = run_explicit_gd(p, eta, 0.0, np.arange(0, 2000))
    assert np.all(np.diff(out0["train"]) <= 1e-15)


def test_spectral_matches_explicit_gd():
    p = _prob(teacher="unit")
    eta, lam = 1.0, 1e-3
    steps = log_steps(3000, 40)
    ex = run_explicit_gd(p, eta, lam, steps)
    sp = SpectralGD(p, eta, lam).run(steps)
    np.testing.assert_allclose(sp["train"], ex["train"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(sp["test"], ex["test"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(sp["test_mc"], ex["test_mc"], rtol=1e-8, atol=1e-14)


def test_critical_times_definition():
    p = _prob(n=20, m=400, teacher="unit")
    sp = SpectralGD(p, 1.0, 1e-3)
    t1, t2 = critical_times(sp, 0.01, 0.01, 200000)
    assert t1 < t2
    tr, te = sp.losses(np.array([int(t1), int(t1) + 1, int(t2) - 1, int(t2)]))
    assert tr[0] >= 0.01 > tr[1]
    assert te[2] > 0.01 >= te[3]
