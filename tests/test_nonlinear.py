import numpy as np
from src.appendix import linear_curves
from src.nonlinear import times


def test_spectral_gd_matches_explicit_gd_on_arbitrary_features():
    """linear_curves (exact SVD iterates, used for random-features / Fourier-features runs) vs a literal GD loop."""
    rng = np.random.default_rng(0)
    n, m, nt = 12, 60, 40
    Phi, Pt = np.maximum(rng.standard_normal((n, m)), 0) / np.sqrt(m), np.maximum(rng.standard_normal((nt, m)), 0) / np.sqrt(m)
    y, yt, th0 = rng.standard_normal(n), rng.standard_normal(nt), rng.standard_normal(m)
    eta, lam, T = 0.5, 1e-3, 3000
    out = linear_curves(Phi, y, Pt, yt, th0, eta, lam, T, k=25)
    want = set(int(s) for s in out["step"])
    th, got = th0.copy(), {}
    for t in range(T + 1):
        r = Phi @ th - y
        if t in want:
            got[t] = (0.5 * np.mean(r ** 2), np.mean((Pt @ th - yt) ** 2))
        th = (1 - eta * lam) * th - (eta / n) * (Phi.T @ r)
    for t, tr, te in zip(out["step"], out["train"], out["test"]):
        assert abs(tr - got[int(t)][0]) < 1e-10 and abs(te - got[int(t)][1]) < 1e-10


def test_times_definitions():
    """t1 = last logged step with train >= eps; t2 = first logged step with test <= c."""
    r = dict(step=np.array([0, 1, 2, 3, 4]), train=np.array([1.0, 0.5, 0.02, 0.005, 0.001]), test=np.array([1.0, 1.0, 0.9, 0.5, 0.005]))
    assert times(r, eps=0.01, c=0.01) == (2.0, 4.0)
