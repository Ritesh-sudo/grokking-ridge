"""Diagnostics (NOT part of the official results):
 (1) t1 diagnosis: which convention of Eq. 8's t1 bound reproduces the paper's dashed t1 line in Fig. 2?
     "diagnosis, not a fit": the literal Eq. 8 stays the reported bound everywhere else.
 (2) Late rise of the unregularised train loss (weight decay pulling theta from the interpolant toward the ridge solution).
"""
import itertools, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import make_problem
from .train import SpectralGD, ridge_closed_form, min_norm_interpolant, log_steps
from . import theory

RES = os.path.join(os.path.dirname(__file__), "..", "results")

# Values READ OFF the paper's Fig. 2 (dashed blue 't1 theory' markers) from the log2 axes of the PNGs in the arXiv source.
# Eyeballed, uncertainty about +-0.3 octave.  Defaults n=100, m=1000, nu2=1, lam=1e-4, eps=0.01, eta=1.
PAPER_T1_LOG2 = {
    ("n", 10): 5.3, ("n", 20): 6.5, ("n", 40): 7.7, ("n", 80): 9.1, ("n", 160): 10.5,
    ("m", 1000): 9.25, ("m", 1500): 9.1, ("m", 2000): 8.95, ("m", 2500): 8.9, ("m", 3000): 8.85,
    ("lam", "default"): 9.5,   # lambda panel: flat at ~2^9.5 for all five lambdas, n=100, m=1000
}


def t1_candidate(n, m, nu2, eps, eta, lpm_mode, eps_factor, th0_mode, b2, p):
    lpm = theory.lam_plus_min(p.Phi) if lpm_mode == "emp" else 1.0
    th0 = {"emp_||th0||^2": float(p.theta0 @ p.theta0), "E=m*nu2": m * nu2, "nu2 (init var nu2/m)": nu2}[th0_mode]
    return n * np.log(6 * b2 * th0 / (eps_factor * eps)) / (2 * eta * lpm)


def diagnose_t1(seeds=10):
    cands = list(itertools.product(("emp", "1"), (1, 2), ("emp_||th0||^2", "E=m*nu2", "nu2 (init var nu2/m)"), (1.5, 1.0)))
    rows = []
    for (lm, ef, tm, b2) in cands:
        errs = {}
        for (param, v), tgt in PAPER_T1_LOG2.items():
            n, m = (v, 1000) if param == "n" else (100, v) if param == "m" else (100, 1000)
            vals = [t1_candidate(n, m, 1.0, 0.01, 1.0, lm, ef, tm, b2, make_problem(n, m, 1.0, seed=s, n_test=0)) for s in range(seeds)]
            errs[(param, v)] = float(np.log2(np.mean(vals)) - tgt)
        e = np.array(list(errs.values()))
        shape = np.array([errs[k] for k in errs if k[0] in ("n", "m")])
        rows.append(dict(lam_plus_min=lm, eps_factor=ef, theta0=tm, b2=b2, rms_log2_err=float(np.sqrt(np.mean(e ** 2))),
                         mean_offset_log2=float(e.mean()), shape_rms_log2=float(np.sqrt(np.mean((shape - shape.mean()) ** 2))),
                         per_point={f"{k[0]}={k[1]}": round(v, 2) for k, v in errs.items()}))
    rows.sort(key=lambda r: r["rms_log2_err"])
    return rows


def late_rise_figure(cfg_n=100, m=1000, nu2=1.0, lam=1e-4, eta=1.0, eps=0.01, seeds=range(10)):
    """Pick the first seed whose unregularised train loss is non-monotone and show train, test and ||theta||^2."""
    chosen = None
    steps = log_steps(1_200_000, 400)
    for s in seeds:
        p = make_problem(cfg_n, m, nu2, seed=s, n_test=0)
        sp = SpectralGD(p, eta, lam)
        tr, te = sp.losses(steps)
        if np.any(np.diff(tr) > 1e-15): chosen = (s, p, sp, tr, te); break
    if chosen is None:
        s = 0; p = make_problem(cfg_n, m, nu2, seed=s, n_test=0); sp = SpectralGD(p, eta, lam); tr, te = sp.losses(steps); chosen = (s, p, sp, tr, te)
    s, p, sp, tr, te = chosen
    nrm = sp.norm2(steps)
    th_r = ridge_closed_form(p.Phi, p.y, lam); th_mn = min_norm_interpolant(p.Phi, p.y)
    r_tr = 0.5 * float(np.mean((p.Phi @ th_r - p.y) ** 2)); r_te = float(np.sum((th_r - p.theta_star) ** 2)) / m
    i_min = int(np.argmin(tr[1:])) + 1
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.2), sharex=True)
    x = steps[1:]
    ax[0].plot(x, tr[1:], "tab:red"); ax[0].axhline(r_tr, color="tab:red", ls=":", label=f"ridge solution train loss = {r_tr:.2e}")
    ax[0].plot(x[i_min - 1], tr[i_min], "ko", ms=5, label=f"minimum at t≈{steps[i_min]:.2g}")
    ax[0].set_yscale("log"); ax[0].set_title("train loss $L_n$ (unregularised, with ½)"); ax[0].legend(fontsize=7)
    ax[1].plot(x, te[1:], "tab:blue"); ax[1].axhline(r_te, color="tab:blue", ls=":", label=f"ridge solution test loss = {r_te:.2e}")
    ax[1].set_yscale("log"); ax[1].set_title("test loss $L$"); ax[1].legend(fontsize=7)
    ax[2].plot(x, nrm[1:], "k"); ax[2].axhline(float(th_r @ th_r), color="k", ls=":", label=r"$\|\theta_{ridge}\|^2$")
    ax[2].axhline(float(th_mn @ th_mn), color="gray", ls="--", label=r"$\|\theta_{min-norm}\|^2$ (interpolant)")
    ax[2].set_yscale("log"); ax[2].set_title(r"$\|\theta(t)\|^2$"); ax[2].legend(fontsize=7)
    rel_rise = tr[1:] / np.minimum.accumulate(tr[1:]) - 1
    ax[3].plot(x, np.maximum(rel_rise, 1e-16), "tab:red"); ax[3].set_yscale("log"); ax[3].set_ylim(1e-8, None)
    ax[3].set_title(r"late rise: $L_n(t)/\min_{s\leq t}L_n(s) - 1$")
    for a in ax: a.set_xscale("log"); a.set_xlabel("GD steps"); a.grid(alpha=0.3, which="both")
    fig.suptitle(f"Late rise of the train loss (seed {s}, n={cfg_n}, m={m}, λ={lam:g}, ν²={nu2:g}, η={eta:g})", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(RES, f"late_rise_train_loss.{ext}"), dpi=140)
    plt.close(fig)
    # size of the rise across all 10 seeds
    rises = []
    for s2 in seeds:
        p2 = make_problem(cfg_n, m, nu2, seed=s2, n_test=0); t2_ = SpectralGD(p2, eta, lam).losses(steps)[0][1:]
        rises.append(float(np.max(t2_ / np.minimum.accumulate(t2_) - 1)))
    return dict(null_predictor_test=float(p.theta_star @ p.theta_star) / m, ridge_test_over_null=r_te / (float(p.theta_star @ p.theta_star) / m),
                max_relative_rise_per_seed=rises, seed=s, min_train=float(tr[i_min]), step_of_min=int(steps[i_min]), final_train=float(tr[-1]), ridge_train=r_tr,
                final_test=float(te[-1]), ridge_test=r_te, norm2_final=float(nrm[-1]), norm2_ridge=float(th_r @ th_r),
                norm2_mn=float(th_mn @ th_mn), norm2_init=float(nrm[0]))


def main():
    os.makedirs(RES, exist_ok=True)
    rows = diagnose_t1()
    lr = late_rise_figure()
    with open(os.path.join(RES, "t1_diagnosis.json"), "w") as f: json.dump(dict(paper_readoffs_log2=[[list(map(str, k)), v] for k, v in PAPER_T1_LOG2.items()], candidates=rows), f, indent=1)
    with open(os.path.join(RES, "late_rise.json"), "w") as f: json.dump(lr, f, indent=1)
    return rows, lr


if __name__ == "__main__":
    rows, lr = main()
    for r in rows[:8]: print({k: v for k, v in r.items() if k != "per_point"})
    print([r for r in rows if r["lam_plus_min"] == "emp" and r["eps_factor"] == 1 and r["theta0"] == "E=m*nu2" and r["b2"] == 1.5][0])
    print(lr)
