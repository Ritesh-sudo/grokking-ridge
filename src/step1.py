"""Step 1: main figure (train/test loss vs steps, log x) with theoretical critical times."""
import argparse, json, logging, os, time
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import make_problem
from .train import (SpectralGD, run_explicit_gd, log_steps, critical_times, check_stable, NumpyEncoder)
from . import theory

log = logging.getLogger("grokking")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def run_config(cfg):
    n, m, nu2, eta, lam = cfg["n"], cfg["m"], cfg["nu2"], cfg["eta"], cfg["lam"]
    eps, c = cfg["eps"], cfg["c"]
    t2_b = theory.t2_bound_eq8(n, m, nu2, eps, eta, lam)
    T = int(cfg["budget_factor"] * t2_b)
    steps = log_steps(T, cfg["n_log"])
    seeds = []
    for seed in range(cfg["seeds"]):
        p = make_problem(n, m, nu2, seed=seed, teacher=cfg["teacher"], n_test=cfg["n_test"])
        stab = check_stable(p.Phi, eta, lam, name=f"{cfg['name']}/seed{seed}")
        if not stab["ok_gd"]:
            seeds.append(dict(seed=seed, stability=stab, skipped=True)); continue
        sp = SpectralGD(p, eta, lam)
        out = sp.run(steps)
        t1, t2 = critical_times(sp, eps, c, T)
        lpm = theory.lam_plus_min(p.Phi)
        tstar = float(np.linalg.norm(p.theta_star))
        seeds.append(dict(
            seed=seed, stability=stab, lam_plus_min=lpm, t1_emp=t1, t2_emp=t2,
            t1_eq8=theory.t1_bound_eq8(n, m, nu2, eps, eta, lpm),
            t2_eq8=t2_b,
            t2_eq7_generic=theory.t2_bound_eq7(m, n, nu2, c, 1.0 / m, tstar, eta, lam),
            theta0_norm2=float(p.theta0 @ p.theta0),
            train=out["train"], test=out["test"], test_half=out["test_half"], test_mc=out["test_mc"]))
    return dict(cfg=cfg, steps=steps, T=T, seeds=seeds)


def crosscheck(cfg, res):
    """Explicit GD loop vs exact spectral iterates on seed 0 (horizon cfg['crosscheck_steps'])."""
    H = int(cfg["crosscheck_steps"])
    p = make_problem(cfg["n"], cfg["m"], cfg["nu2"], seed=0, teacher=cfg["teacher"], n_test=cfg["n_test"])
    steps = log_steps(H, 60)
    t0 = time.time()
    ex = run_explicit_gd(p, cfg["eta"], cfg["lam"], steps)
    sp = SpectralGD(p, cfg["eta"], cfg["lam"]).run(steps)
    # relative error is only meaningful above the float64 noise floor of the explicit loop (~1e-14 absolute
    # for losses of order 1); below that we report absolute error instead.
    def rel(a, b, floor=1e-12):
        mask = np.abs(b) > floor
        return float(np.max(np.abs(a - b)[mask] / np.abs(b)[mask])) if mask.any() else 0.0
    ab = lambda a, b: float(np.max(np.abs(a - b)))
    r = dict(horizon=H, max_rel_err_train=rel(sp["train"], ex["train"]), max_abs_err_train=ab(sp["train"], ex["train"]),
             max_rel_err_test=rel(sp["test"], ex["test"]), max_abs_err_test=ab(sp["test"], ex["test"]),
             max_rel_err_test_mc=rel(sp["test_mc"], ex["test_mc"]), rel_err_floor=1e-12, explicit_seconds=time.time() - t0)
    log.info("crosscheck %s", r)
    return r


def aggregate(res, key):
    arr = np.array([s[key] for s in res["seeds"] if not s.get("skipped")])
    return arr.mean(0), arr.std(0)


def plot(res, path_base):
    cfg, steps = res["cfg"], res["steps"]
    ok = [s for s in res["seeds"] if not s.get("skipped")]
    x = steps[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ylog in zip(axes, (False, True)):
        for key, col, lab in (("train", "tab:red", r"train $L_n$ (with ½)"), ("test", "tab:blue", r"test $L$ (paper convention)")):
            mu, sd = aggregate(res, key)
            ax.plot(x, mu[1:], color=col, label=lab)
            lo = np.maximum(mu - sd, 1e-12) if ylog else mu - sd
            ax.fill_between(x, lo[1:], (mu + sd)[1:], color=col, alpha=0.25, lw=0)
        mu, _ = aggregate(res, "test_half")
        ax.plot(x, mu[1:], color="tab:blue", ls=":", lw=1.2, label=r"test $L/2$ (like-for-like with train)")
        t1b = np.mean([s["t1_eq8"] for s in ok]); t2b = ok[0]["t2_eq8"]
        t1e = np.nanmean([s["t1_emp"] for s in ok]); t2e = np.nanmean([s["t2_emp"] for s in ok])
        ax.axvline(t1b, color="tab:red", ls="--", label=f"Eq.8 $t_1$ bound ≈ {t1b:.3g}")
        ax.axvline(t2b, color="tab:blue", ls="--", label=f"Eq.8 $t_2$ bound ≈ {t2b:.3g}")
        ax.axvline(t1e, color="tab:red", ls="-.", alpha=0.6, label=f"measured $t_1$ ≈ {t1e:.3g}")
        ax.axvline(t2e, color="tab:blue", ls="-.", alpha=0.6, label=f"measured $t_2$ ≈ {t2e:.3g}")
        ax.axhline(cfg["eps"], color="gray", lw=0.8, ls=":")
        ax.set_xscale("log")
        if ylog: ax.set_yscale("log"); ax.set_ylim(1e-6, 3)
        ax.set_xlabel("GD steps"); ax.set_ylabel("squared loss")
        ax.set_title(("log-y" if ylog else "linear-y (paper style)"))
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{cfg['name']}: n={cfg['n']}, m={cfg['m']}, λ={cfg['lam']:g}, ν²={cfg['nu2']:g}, η={cfg['eta']:g}; "
                 f"mean ± std over {len(ok)} seeds", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)
    plt.close(fig)


def save(res, cc):
    name = res["cfg"]["name"]
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{name}.json"), "w") as f:
        json.dump(dict(res, crosscheck=cc), f, cls=NumpyEncoder)
    ok = [s for s in res["seeds"] if not s.get("skipped")]
    mu = {k: aggregate(res, k) for k in ("train", "test", "test_half", "test_mc")}
    with open(os.path.join(RES, f"{name}_mean_curves.csv"), "w") as f:
        f.write("step,train_mean,train_std,test_mean,test_std,test_half_mean,test_mc_mean,test_mc_std\n")
        for i, t in enumerate(res["steps"]):
            f.write(f"{t},{mu['train'][0][i]:.8g},{mu['train'][1][i]:.8g},{mu['test'][0][i]:.8g},{mu['test'][1][i]:.8g},"
                    f"{mu['test_half'][0][i]:.8g},{mu['test_mc'][0][i]:.8g},{mu['test_mc'][1][i]:.8g}\n")
    summ = dict(
        name=name, n_seeds=len(ok), skipped=[s["seed"] for s in res["seeds"] if s.get("skipped")],
        t1_emp=[s["t1_emp"] for s in ok], t2_emp=[s["t2_emp"] for s in ok],
        t1_eq8=[s["t1_eq8"] for s in ok], t2_eq8=ok[0]["t2_eq8"],
        t1_valid_all=bool(all(s["t1_emp"] <= s["t1_eq8"] for s in ok)),
        t2_valid_all=bool(all(s["t2_emp"] >= s["t2_eq8"] for s in ok)),
        t1_bound_over_measured=[s["t1_eq8"] / s["t1_emp"] for s in ok],
        t2_bound_over_measured=[s["t2_eq8"] / s["t2_emp"] for s in ok],
        initial_train=float(mu["train"][0][0]), initial_test=float(mu["test"][0][0]),
        stability=[s["stability"] for s in ok][:1])
    with open(os.path.join(RES, f"{name}_summary.json"), "w") as f:
        json.dump(summ, f, cls=NumpyEncoder, indent=1)
    return summ


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    t0 = time.time()
    res = run_config(cfg)
    cc = crosscheck(cfg, res)
    summ = save(res, cc)
    plot(res, os.path.join(RES, cfg["name"]))
    log.info("%s done in %.1fs: %s", cfg["name"], time.time() - t0, {k: summ[k] for k in ("t1_valid_all", "t2_valid_all")})
    return summ


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
