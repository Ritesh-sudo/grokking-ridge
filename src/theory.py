"""Theoretical critical times from Xu, Vardi, Safran (arXiv:2601.19791v4).

All functions are *bounds*: t1 <= t1_bound, t2 >= t2_bound.
"""
import numpy as np


def lam_plus_min(Phi):
    """Smallest positive eigenvalue of Phi^T Phi (un-normalised), via the n x n Gram matrix."""
    ev = np.linalg.eigvalsh(Phi @ Phi.T)
    return float(ev[ev > 1e-12 * ev[-1]].min())


def lam_max_over_n(Phi):
    return float(np.linalg.eigvalsh(Phi @ Phi.T)[-1]) / Phi.shape[0]


def stability_limit(Phi, lam):
    """GD on L_n(theta;lam) is stable iff eta < 2/(lam + lam_max(Phi^T Phi)/n);
    the paper's theorems require the stronger eta < 1/(lam + lam_max/n) (Thm 4.4)."""
    return 1.0 / (lam + lam_max_over_n(Phi))


def t1_bound_eq6(n, b2, theta0_norm2, eps, eta, lpm):
    """Eq. (6): t1 <= n ln(6 b^2 ||theta0||^2 / eps) / (2 eta lam+_min(Phi^T Phi))."""
    return n * np.log(6 * b2 * theta0_norm2 / eps) / (2 * eta * lpm)


def t2_bound_eq7(m, n, nu2, c, lam_min_Sigma, theta_star_norm, eta, lam, const=4.0):
    """Eq. (7): t2 >= ln( (m-n) nu2/2 * (sqrt(c/lmin(Sigma)) + ||theta*||)^-2 ) / (4 eta lam).
    `const` = 4 is the generic Eq. 7; 2.02 is the tighter constant of Remark A.14 (valid for eta*lam <= 0.01)."""
    arg = (m - n) * nu2 / 2 * (np.sqrt(c / lam_min_Sigma) + theta_star_norm) ** -2
    return np.log(arg) / (const * eta * lam)


def t1_bound_eq8(n, m, nu2, eps, eta, lpm):
    """Eq. (8), Gaussian features N(0,I/m): t1 <= n ln(14 m nu2/eps) / (2 eta lam+_min(Phi^T Phi))."""
    return n * np.log(14 * m * nu2 / eps) / (2 * eta * lpm)


def t2_bound_eq8(n, m, nu2, eps, eta, lam):
    """Eq. (8), Gaussian features N(0,I/m), c = eps, eta*lam <= 0.01:
    t2 >= ln((m-n) nu2 / (8 m eps)) / (2.02 eta lam)."""
    if eta * lam > 0.01:
        raise ValueError("Eq. 8 requires eta*lam <= 0.01")
    return np.log((m - n) * nu2 / (8 * m * eps)) / (2.02 * eta * lam)


def regime_tags(n, m, nu2, lam, eps, c, eta, theta_star_norm, b2=1.5, delta=0.05, C=1.0):
    """Tag whether a configuration satisfies the *literal* sufficient conditions of Thm 4.2.
    The paper hides numerical constants in Omega/O; we set them to C=1 (so the tags are 'strict-ish', not
    definitive).  b2 = 3/2 is the paper's whp bound on ||phi||^2 for phi ~ N(0, I/m); delta=0.05.
      eq3 : n >= C b^4 ||theta*||^4 / eps^2 * log(1/delta)
      eq4 : m - n >= C max{ log(1/delta), ||theta*||^2/nu2, c/(lmin(Sigma) nu2) },  lmin(Sigma)=1/m
      eq5 : lam <= C min{ eps/||theta*||^2, b sqrt(eps)/||theta*|| }
      eta_thm42 : eta < 1/(lam + b^2)   (step-size condition of Thm 4.2)
      eq8_ok : conditions under which Eq. (8) is derived (eta*lam <= 0.01, m >= ||theta*||^2/eps)
      t2_nonvacuous : (m-n) nu2 / (8 m eps) > 1  (otherwise the Eq. 8 t2 bound is <= 0)
    in_regime = eq3 & eq4 & eq5 & eta_thm42."""
    ts = float(theta_star_norm)
    b = np.sqrt(b2)
    eq3 = (ts == 0) or (n >= C * b2 ** 2 * ts ** 4 / eps ** 2 * np.log(1 / delta))
    eq4 = (m - n) >= C * max(np.log(1 / delta), ts ** 2 / nu2, c * m / nu2)
    eq5 = (ts == 0) or (lam <= C * min(eps / ts ** 2, b * np.sqrt(eps) / ts))
    eta42 = eta < 1.0 / (lam + b2)
    return dict(eq3=bool(eq3), eq4=bool(eq4), eq5=bool(eq5), eta_thm42=bool(eta42),
                eq8_ok=bool(eta * lam <= 0.01 and m >= ts ** 2 / eps),
                t2_nonvacuous=bool((m - n) * nu2 / (8 * m * eps) > 1),
                in_regime_eq345=bool(eq3 and eq4 and eq5),
                in_regime=bool(eq3 and eq4 and eq5 and eta42))
