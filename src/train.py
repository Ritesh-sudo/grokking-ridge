"""Full-batch gradient descent with weight decay for the linear (ridge) student.

Two interchangeable engines for the *same* recursion
    theta_{t+1} = (1 - eta*lam) theta_t - (eta/n) Phi^T (Phi theta_t - y)
(= GD on L_n(theta;lam) = 1/(2n)||Phi theta - y||^2 + lam/2 ||theta||^2, paper Eq. 1/2):

* `run_explicit_gd`  : literal loop of GD steps (reference implementation).
* `SpectralGD`       : exact closed-form iterate theta_t at any t, via the SVD of Phi.
                       Used for very long horizons (t ~ 1e6) and sweeps; tests/ verify it
                       matches the explicit loop.
"""
import json
import logging
import numpy as np
from .data import test_loss_exact
from .theory import stability_limit

log = logging.getLogger("grokking")


def log_steps(T, k=200):
    """~k log-spaced integer steps in [1, T], plus step 0."""
    s = np.unique(np.round(np.geomspace(1, T, k)).astype(np.int64))
    return np.concatenate([[0], s])


def check_stable(Phi, eta, lam, name=""):
    """Never silently change eta: report whether eta is inside the stability region.
    ok_gd      : eta < 2/(lam + lmax/n)   (GD on the quadratic converges)
    ok_theorem : eta < 1/(lam + lmax/n)   (the paper's Thm 4.4 assumption)"""
    lim = stability_limit(Phi, lam)  # = 1/(lam + lmax/n)
    info = dict(name=name, eta=eta, lam=lam, theorem_limit=lim, gd_limit=2 * lim,
                ok_theorem=bool(eta < lim), ok_gd=bool(eta < 2 * lim))
    if not info["ok_gd"]:
        log.warning("UNSTABLE config %s: eta=%g >= GD limit %g", name, eta, 2 * lim)
    elif not info["ok_theorem"]:
        log.warning("config %s: eta=%g violates theorem condition (< %g) but GD is stable", name, eta, lim)
    return info


def _record(prob, theta, r=None):
    if r is None:
        r = prob.Phi @ theta - prob.y
    te = test_loss_exact(theta, prob.theta_star, prob.m)
    mc = float(np.mean((prob.X_test @ theta - prob.y_test) ** 2))
    return 0.5 * float(np.mean(r ** 2)), te, mc


def run_explicit_gd(prob, eta, lam, steps):
    """Literal full-batch GD. `steps` = sorted int array (may contain 0). Returns dict of logged arrays.
    train = L_n(theta) = 1/(2n) sum (N - N*)^2 (paper convention, unregularised, with the 1/2).
    test  = population L(theta) = E (N - N*)^2 (paper convention, no 1/2), exact closed form.
    test_mc = same, Monte-Carlo on the n_test fresh held-out samples.
    test_half = 0.5 * test (logged as requested, for like-for-like comparison with train)."""
    Phi, y = prob.Phi, prob.y
    theta = prob.theta0.copy()
    n = prob.n
    out = {k: [] for k in ("step", "train", "test", "test_mc", "obj_reg", "norm2")}
    j, T = 0, int(steps[-1])
    want = set(int(s) for s in steps)
    for t in range(T + 1):
        r = Phi @ theta - y
        if t in want:
            tr, te, mc = _record(prob, theta, r)
            out["step"].append(t); out["train"].append(tr); out["test"].append(te); out["test_mc"].append(mc)
            out["obj_reg"].append(tr + 0.5 * lam * float(theta @ theta)); out["norm2"].append(float(theta @ theta))
        theta = (1 - eta * lam) * theta - (eta / n) * (Phi.T @ r)
    out = {k: np.array(v) for k, v in out.items()}
    out["test_half"] = 0.5 * out["test"]
    out["theta_final"] = theta
    return out


class SpectralGD:
    """Exact iterates of the GD recursion. With Phi = U S V^T (thin SVD) and a_i = 1 - eta*lam - eta*s_i^2/n:
        V^T theta_t   = a^t V^T theta_0 + (1-a^t) * s/(n lam + s^2) * U^T y       (data subspace)
        theta_t_perp  = (1 - eta lam)^t theta_0_perp                                (orthogonal complement)
    Requires |a_i| < 1 and |1-eta*lam| < 1 (checked)."""

    def __init__(self, prob, eta, lam):
        self.p, self.eta, self.lam = prob, eta, lam
        n, m = prob.n, prob.m
        U, s, Vt = np.linalg.svd(prob.Phi, full_matrices=False)
        self.U, self.s, self.V = U, s, Vt.T
        self.a = 1 - eta * lam - eta * s ** 2 / n
        self.rho = 1 - eta * lam
        if np.any(np.abs(self.a) >= 1) or abs(self.rho) >= 1:
            raise ValueError("unstable / non-contractive GD recursion")
        self.c0 = self.V.T @ prob.theta0
        self.uty = U.T @ prob.y
        self.fix = s / (n * lam + s ** 2) * self.uty          # ridge solution coordinates
        self.th0_perp = prob.theta0 - self.V @ self.c0
        cs = self.V.T @ prob.theta_star
        self.cs = cs
        self.ts_perp = prob.theta_star - self.V @ cs
        self.th0_perp_n2 = float(self.th0_perp @ self.th0_perp)
        self.cross = float(self.th0_perp @ self.ts_perp)
        self.ts_perp_n2 = float(self.ts_perp @ self.ts_perp)

    def _pw(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.power(self.a[None, :], t[:, None]), np.power(self.rho, t)

    def coords(self, t):
        at, rt = self._pw(t)
        c = at * self.c0[None, :] + (1 - at) * self.fix[None, :]
        return c, rt

    def losses(self, steps):
        """Vectorised exact train / test (closed form) at the given steps."""
        n, m = self.p.n, self.p.m
        c, rt = self.coords(steps)
        train = 0.5 / n * np.sum((self.s[None, :] * c - self.uty[None, :]) ** 2, axis=1)
        d_par = np.sum((c - self.cs[None, :]) ** 2, axis=1)
        d_perp = rt ** 2 * self.th0_perp_n2 - 2 * rt * self.cross + self.ts_perp_n2
        test = (d_par + d_perp) / m
        return train, test

    def norm2(self, steps):
        """Exact ||theta_t||^2."""
        c, rt = self.coords(steps)
        return np.sum(c ** 2, axis=1) + rt ** 2 * self.th0_perp_n2

    def theta_at(self, t):
        c, rt = self.coords([t])
        return self.V @ c[0] + rt[0] * self.th0_perp

    def run(self, steps, mc=True):
        steps = np.asarray(steps)
        train, test = self.losses(steps)
        out = dict(step=steps, train=train, test=test, test_half=0.5 * test)
        if mc:
            out["test_mc"] = np.array([float(np.mean((self.p.X_test @ self.theta_at(t) - self.p.y_test) ** 2)) for t in steps])
        return out


def critical_times(spec, eps, c, t_max, n_grid=4000):
    """Empirical t1 = last step with L_n >= eps, t2 = first step with L <= c (paper Sec. 3 definitions).
    Dense log grid then integer bisection between adjacent grid points. Returns (t1, t2); NaN if the
    event is not located within t_max (e.g. t2 not reached)."""
    grid = np.unique(np.concatenate([np.arange(0, 50), np.round(np.geomspace(1, t_max, n_grid)).astype(np.int64)]))
    tr, te = spec.losses(grid)
    # t1: last grid index with train >= eps
    idx = np.nonzero(tr >= eps)[0]
    t1 = np.nan
    if len(idx) and idx[-1] < len(grid) - 1:
        lo, hi = int(grid[idx[-1]]), int(grid[idx[-1] + 1])      # train(lo)>=eps, train(hi)<eps
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if spec.losses([mid])[0][0] >= eps: lo = mid
            else: hi = mid
        t1 = lo
    idx = np.nonzero(te <= c)[0]
    t2 = np.nan
    if len(idx):
        i = idx[0]
        if i == 0: t2 = 0
        else:
            lo, hi = int(grid[i - 1]), int(grid[i])               # test(lo)>c, test(hi)<=c
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if spec.losses([mid])[1][0] <= c: hi = mid
                else: lo = mid
            t2 = hi
    return t1, t2


def ridge_closed_form(Phi, y, lam):
    """argmin of L_n(theta;lam): (Phi^T Phi + n lam I)^{-1} Phi^T y  (computed in the dual, n x n)."""
    n = Phi.shape[0]
    return Phi.T @ np.linalg.solve(Phi @ Phi.T + n * lam * np.eye(n), y)


def min_norm_interpolant(Phi, y):
    return np.linalg.pinv(Phi) @ y


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return super().default(o)
