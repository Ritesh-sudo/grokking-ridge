"""Paper App. B experiments (Figs. 5-7). All are linear in the trained weights -> exact spectral GD.
Choices not given by the paper are flagged in RESULTS.md. Usage: python -m src.appendix"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .train import log_steps, NumpyEncoder
from .nonlinear import times

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def linear_curves(Phi, y, Pt, yt, th0, eta, lam, T, eps=None, k=150):
    """Exact GD iterates for 1/(2n)||Phi th - y||^2 + lam/2 ||th||^2; test = MC mean (Pt th - yt)^2."""
    n = Phi.shape[0]
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    A, rho = 1 - eta * lam - eta * s ** 2 / n, 1 - eta * lam
    assert np.all(np.abs(A) < 1), "GD unstable"
    c0, fix = Vt @ th0, s / (n * lam + s ** 2) * (U.T @ y)
    perp = th0 - Vt.T @ c0
    steps = log_steps(T, k)
    out = dict(step=steps, train=[], test=[], acc=[])
    for t in steps:
        at = A ** t
        th = Vt.T @ (at * c0 + (1 - at) * fix) + rho ** t * perp
        out["train"].append(0.5 * np.mean((Phi @ th - y) ** 2))
        e2 = (Pt @ th - yt) ** 2
        out["test"].append(np.mean(e2))
        out["acc"].append(np.mean(e2 <= eps) if eps else np.nan)
    return {k_: np.array(v) for k_, v in out.items()}


def _rf_feats(g, n, m, d, nu2, ntest):
    X, Xt = g.standard_normal((n, d)), g.standard_normal((ntest, d))
    W = g.standard_normal((m, d)) * np.sqrt(nu2 / (d * m))
    return np.maximum(X @ W.T, 0), np.maximum(Xt @ W.T, 0)


def fig5(T=10_000_000):
    """Surrogate accuracy P_x[N^2 <= eps] vs steps. Paper curves (Fig. 5 loss panel) match the random-features model with a
    ZERO teacher (see RESULTS.md), so that is what is used: m=1e4, d=100, n=100, eta=1, lam=1e-5, eps=0.01, single run (as the paper)."""
    n, m, d, lam, eta, eps = 100, 10000, 100, 1e-5, 1.0, 0.01
    g = np.random.default_rng(0)
    P, Pt = _rf_feats(g, n, m, d, 1.0, 5000)
    # accuracy needs per-sample errors: P_x[(N-N*)^2<=eps]; zero teacher => N*=0
    r = linear_curves(P, np.zeros(n), Pt, np.zeros(len(Pt)), g.standard_normal(m), eta, lam, T, eps)
    # training accuracy on the n training points would need per-sample residuals; reuse the same exact iterates
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].semilogx(r["step"][1:], r["acc"][1:]); ax[0].set_title(f"test accuracy P[N^2<={eps}]"); ax[0].set_ylim(0, 1.02)
    ax[1].semilogx(r["step"][1:], r["train"][1:], "--", label="train"); ax[1].semilogx(r["step"][1:], r["test"][1:], label="test"); ax[1].legend()
    for a_ in ax: a_.set_xlabel("GD steps")
    fig.tight_layout(); fig.savefig(f"{RES}/app_fig5_accuracy.png", dpi=130); plt.close(fig)
    ix = lambda t: int(np.searchsorted(r["step"], t))
    return dict(acc_at={str(t): float(r["acc"][ix(t)]) for t in (1, 10, 100, 1000, 10000, 100000)},
                test_at={str(t): float(r["test"][ix(t)]) for t in (1, 10, 100, 1000, 10000, 100000)},
                acc_final=float(r["acc"][-1]), paper_acc_test="0.12 0.15 0.15 0.18 0.19 0.45 (1e0..1e5), ->1.0 at ~2e5")


def fig6(T=10_000_000, seeds=10):
    """Label noise: random ReLU features, d=50, m=5000, n=100, eta=1, lam=1e-5, zero teacher, labels = N(0,std^2) noise;
    test loss measured against the clean (zero) teacher. Matches the paper's curves (identity features did not)."""
    n, m, d, lam, eta = 100, 5000, 50, 1e-5, 1.0
    stds = [0.01, 0.05, 0.1]
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); summ = {}
    for sd, col in zip(stds, ("tab:blue", "tab:red", "tab:orange")):
        rs = []
        for s in range(seeds):
            g = np.random.default_rng(s)
            P, Pt = _rf_feats(g, n, m, d, 1.0, 3000)
            rs.append(linear_curves(P, sd * g.standard_normal(n), Pt, np.zeros(len(Pt)), g.standard_normal(m), eta, lam, T))
        mean = {k: np.mean([r[k] for r in rs], 0) for k in ("step", "train", "test")}
        st = mean["step"][1:]
        ax.semilogx(st, mean["train"][1:], "--", color=col); ax.semilogx(st, mean["test"][1:], color=col, label=f"std={sd}")
        ts_ = np.array([times(r, 0.01, 0.01) for r in rs], dtype=float)
        ix = lambda t: int(np.searchsorted(mean["step"], t))
        summ[f"std={sd}"] = dict(t1=float(np.nanmean(ts_[:, 0])), t2=float(np.nanmean(ts_[:, 1])), t2_sd=float(np.nanstd(ts_[:, 1])),
                                 test_at={str(t): float(mean["test"][ix(t)]) for t in (1, 10, 100, 1000, 10000, 100000)},
                                 train_at={str(t): float(mean["train"][ix(t)]) for t in (1, 10, 100, 1000)},
                                 final_test=float(mean["test"][-1]))
        print("fig6", sd, summ[f"std={sd}"], flush=True)
    ax.set_ylim(0, 0.5); ax.set_xlabel("training steps"); ax.set_ylabel("square loss"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{RES}/app_fig6_label_noise.png", dpi=130); plt.close(fig)
    summ["paper_test_std0.01"] = "0.385 0.28 0.18 0.145 0.12 0.02 at 1e0..1e5"
    return summ


def fig7(T=10_000_000, seeds=10, c=0.01, eps=0.01):
    """Random Fourier features phi_j = sqrt(2/m) cos(<w_j,x>+b_j), x~N(0,I_d), teacher = one feature, theta* = e_1
    (literal reading; target = sqrt(2/m) cos(.), so the zero predictor already has test loss ~2e-4 << c: generalisation here is
    weight decay removing the init noise, as in the realizable ridge setting). m=5000, d=10, n=100, eta=1, lam=1e-4."""
    n, m, d, lam, eta = 100, 5000, 10, 1e-4, 1.0
    rs = []
    for s in range(seeds):
        g = np.random.default_rng(s)
        W, b = g.standard_normal((m, d)), g.uniform(0, 2 * np.pi, m)
        f = lambda X: np.sqrt(2 / m) * np.cos(X @ W.T + b)
        X, Xt = g.standard_normal((n, d)), g.standard_normal((5000, d))
        ts = np.zeros(m); ts[0] = 1.0
        rs.append(linear_curves(f(X), f(X) @ ts, f(Xt), f(Xt) @ ts, g.standard_normal(m), eta, lam, T))
    mean = {k: np.mean([r[k] for r in rs], 0) for k in ("step", "train", "test")}
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); st = mean["step"][1:]
    for k, ls in (("train", "--"), ("test", "-")):
        A = np.array([r[k][1:] for r in rs]); ax.semilogx(st, A.mean(0), ls, label=k); ax.fill_between(st, A.min(0), A.max(0), alpha=0.2, lw=0)
    ax.set_ylim(0, 1.05); ax.set_xlabel("training steps"); ax.set_title("RFF, theta*=e1 (amplitude 1)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{RES}/app_fig7_fourier.png", dpi=130); plt.close(fig)
    ts_ = np.array([times(r, eps, c) for r in rs], dtype=float)
    ix = lambda t: int(np.searchsorted(mean["step"], t))
    return dict(seeds=seeds, test_at={str(t): float(mean["test"][ix(t)]) for t in (1, 10, 100, 1000, 10000, 100000)},
                train_at={str(t): float(mean["train"][ix(t)]) for t in (1, 10, 100, 1000)},
                t1=float(np.nanmean(ts_[:, 0])), t2=float(np.nanmean(ts_[:, 1])), t2_sd=float(np.nanstd(ts_[:, 1])),
                paper_test="0.95 0.93 0.885 0.877 0.735 0.12 at 1e0..1e5 (t2 ~2e5)")


if __name__ == "__main__":
    res = dict(fig5=fig5(), fig6=fig6(), fig7=fig7())
    print(json.dumps(res, indent=1, cls=NumpyEncoder))
    json.dump(res, open(f"{RES}/app_summary.json", "w"), cls=NumpyEncoder)
