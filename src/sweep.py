"""Step 2: hyperparameter sweeps testing the paper's bounds.  `python -m src.sweep --all` reproduces everything.

Three families (see README):
  F1_realizable_paper : unit-norm teacher, paper's Fig. 2 defaults (eps=c=0.01).  Out of regime for Eq. 3.
  F2_realizable_inregime : unit-norm teacher, eps=c=0.5, nu2=100, eta=0.5 -> satisfies Eq. 3/4/5 and eta<1/(lam+b^2) with C=1.
  F3_zero_teacher : zero teacher (Eq. 3 vacuous), eps=c=0.01.
Measured t1/t2 follow the paper's definitions (Sec. 3):  t1 = last step with L_n >= eps, t2 = first step with L <= c.
Bounds are the *literal* Eq. 8 (Gaussian features); empirical lam+_min(Phi^T Phi) of each sampled dataset is used.
"""
import warnings
import argparse, json, logging, os, time, itertools
import numpy as np
import yaml
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import make_problem
from .train import SpectralGD, critical_times, check_stable, NumpyEncoder
from . import theory

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected (undefined t1/t2) and handled
log = logging.getLogger("grokking")
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
SEEDS = 10

P2 = lambda lo, hi: [2.0 ** k for k in range(lo, hi + 1)]
FAMILIES = {
    "F1_realizable_paper": dict(
        teacher="unit", base=dict(n=100, m=1000, nu2=1.0, lam=1e-4, eta=1.0, eps=0.01),
        sweeps=dict(lam=P2(-19, 0), nu2=[0.01, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6],
                    n=[10, 20, 40, 80, 160, 320, 640, 900], m=[1000, 1500, 2000, 2500, 3000, 5000, 10000],
                    eta=[0.25, 0.5, 1, 2, 4, 8])),
    "F2_realizable_inregime": dict(
        teacher="unit", base=dict(n=100, m=1000, nu2=100.0, lam=1e-4, eta=0.5, eps=0.5),
        sweeps=dict(lam=P2(-19, 0), nu2=[2, 5, 10, 1e2, 1e3, 1e4, 1e5, 1e6],
                    n=[10, 20, 40, 80, 160, 320, 640, 900], m=[1000, 1500, 2000, 2500, 3000, 5000, 10000],
                    eta=[0.0625, 0.125, 0.25, 0.5])),
    "F3_zero_teacher": dict(
        teacher="zero", base=dict(n=100, m=1000, nu2=1.0, lam=1e-4, eta=1.0, eps=0.01),
        sweeps=dict(lam=P2(-19, 0), nu2=[0.01, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6],
                    n=[10, 20, 40, 80, 160, 320, 640, 900], m=[1000, 1500, 2000, 2500, 3000, 5000, 10000],
                    eta=[0.25, 0.5, 1, 2, 4, 8])),
}


def run_point(teacher, hp, seeds=SEEDS):
    n, m, nu2, lam, eta, eps = (hp[k] for k in ("n", "m", "nu2", "lam", "eta", "eps"))
    c = eps
    ts_norm = 0.0 if teacher == "zero" else 1.0
    tags = theory.regime_tags(n, m, nu2, lam, eps, c, eta, ts_norm)
    t2_ok = tags["eq8_ok"] and tags["t2_nonvacuous"]
    t2_b = theory.t2_bound_eq8(n, m, nu2, eps, eta, lam) if t2_ok else float("nan")
    T = int(max(10 * (t2_b if t2_ok else 0), 20 / (eta * lam)))
    recs = []
    for seed in range(seeds):
        p = make_problem(n, m, nu2, seed=seed, teacher=teacher, n_test=0)
        stab = check_stable(p.Phi, eta, lam, name=f"{teacher}/{hp}/seed{seed}")
        if not stab["ok_gd"]:
            recs.append(dict(seed=seed, skipped=True, stability=stab)); continue
        sp = SpectralGD(p, eta, lam)
        t1, t2 = critical_times(sp, eps, c, T)
        lpm = theory.lam_plus_min(p.Phi)
        recs.append(dict(seed=seed, skipped=False, stability=stab, t1=t1, t2=t2, lpm=lpm,
                         t1_eq8=theory.t1_bound_eq8(n, m, nu2, eps, eta, lpm), t2_eq8=t2_b, T=T))
    return dict(hp=hp, tags=tags, recs=recs)


def run_family(name):
    fam = FAMILIES[name]
    out = {}
    for param, values in fam["sweeps"].items():
        pts = []
        for v in values:
            hp = dict(fam["base"]); hp[param] = v
            if param == "m" and hp["m"] <= hp["n"]: continue
            pts.append(run_point(fam["teacher"], hp))
        out[param] = pts
        log.info("%s sweep %s done (%d points)", name, param, len(pts))
    return out


# ----------------------------------------------------------------------------- analysis
def arr(pt, key):
    return np.array([r[key] for r in pt["recs"] if not r["skipped"]], dtype=float)


def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (float("nan"),) * 3
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def slope_tests(fam_name, res):
    """Pre-registered fit ranges (fixed before looking at results):
       lam: points with eta*lam <= 1e-3, fit log t2 vs log(1/(eta lam))                (expect slope 1)
       eta: all eta points,               fit log t2 vs log(1/(eta lam))                (expect slope 1)
       nu2: points with nu2 >= 1 (and bound non-vacuous), fit t vs ln(nu2); theory slope t2: 1/(2.02 eta lam)
            [from Eq. 8], t1: n/(2 eta lam+_min) [from Eq. 6/8]; report measured/theory.
       n  : fit log t1 vs log n and log t2 vs log n.   m: fit t vs ln m (small effect; Eq. 8 predicted shift reported)."""
    out = {}
    def logfit(pts, xfun, key):
        sl = []
        for s in range(SEEDS):
            xs, ys = [], []
            for pt in pts:
                r = [q for q in pt["recs"] if q["seed"] == s and not q["skipped"]]
                if r and np.isfinite(r[0][key]) and r[0][key] > 0:
                    xs.append(xfun(pt["hp"])); ys.append(np.log(r[0][key]))
            if len(xs) >= 3: sl.append(np.polyfit(xs, ys, 1)[0])
        return ci95(sl)
    lam_pts = [p for p in res["lam"] if p["hp"]["eta"] * p["hp"]["lam"] <= 1e-3]
    out["t2_vs_1/(eta*lam)__lam_sweep"] = logfit(lam_pts, lambda h: np.log(1 / (h["eta"] * h["lam"])), "t2")
    out["t1_vs_1/(eta*lam)__lam_sweep"] = logfit(lam_pts, lambda h: np.log(1 / (h["eta"] * h["lam"])), "t1")
    out["t2_vs_1/(eta*lam)__eta_sweep"] = logfit(res["eta"], lambda h: np.log(1 / (h["eta"] * h["lam"])), "t2")
    out["t1_vs_1/(eta*lam)__eta_sweep"] = logfit(res["eta"], lambda h: np.log(1 / (h["eta"] * h["lam"])), "t1")
    out["t1_loglog_slope_vs_n"] = logfit(res["n"], lambda h: np.log(h["n"]), "t1")
    out["t2_loglog_slope_vs_n"] = logfit(res["n"], lambda h: np.log(h["n"]), "t2")
    # nu2: slope of t vs ln nu2 in steps per e-fold, compare with theory
    nu_pts = [p for p in res["nu2"] if p["hp"]["nu2"] >= 1 and p["tags"]["t2_nonvacuous"]]
    for key in ("t1", "t2"):
        sl, ratio = [], []
        for s in range(SEEDS):
            xs, ys, lp = [], [], []
            for pt in nu_pts:
                r = [q for q in pt["recs"] if q["seed"] == s and not q["skipped"]]
                if r and np.isfinite(r[0][key]):
                    xs.append(np.log(pt["hp"]["nu2"])); ys.append(r[0][key]); lp.append(r[0]["lpm"])
            if len(xs) >= 3:
                b = np.polyfit(xs, ys, 1)[0]; h = nu_pts[0]["hp"]
                th = h["n"] / (2 * h["eta"] * lp[0]) if key == "t1" else 1 / (2.02 * h["eta"] * h["lam"])
                sl.append(b); ratio.append(b / th)
        out[f"{key}_slope_vs_ln_nu2_steps_per_efold"] = ci95(sl)
        out[f"{key}_slope_vs_ln_nu2__measured_over_theory"] = ci95(ratio)
    # m: measured shift between smallest and largest m vs Eq. 8 prediction
    mp = res["m"]
    if len(mp) >= 2:
        a, b = mp[0], mp[-1]
        for key in ("t1", "t2"):
            d_meas = np.nanmean(arr(b, key)) - np.nanmean(arr(a, key))
            d_th = np.nanmean(arr(b, key + "_eq8")) - np.nanmean(arr(a, key + "_eq8"))
            out[f"{key}_shift_m{a['hp']['m']}_to_m{b['hp']['m']}__measured_vs_eq8"] = (float(d_meas), float(d_th))
    return out


def validity_table(all_res):
    """Bound validity (t1<=bound, t2>=bound in *every seed*), split by family and by regime tag."""
    rows = []
    for fam, res in all_res.items():
        for param, pts in res.items():
            for pt in pts:
                for key, b, ok in (("t1", "t1_eq8", lambda e, bd: e <= bd), ("t2", "t2_eq8", lambda e, bd: e >= bd)):
                    e, bd = arr(pt, key), arr(pt, b)
                    mask = np.isfinite(e) & np.isfinite(bd) & (e > 0)   # t1 = 0 (train never >= eps after step 0) gives an infinite ratio: excluded
                    if not mask.any(): continue
                    valid = np.array([ok(x, y) for x, y in zip(e[mask], bd[mask])])
                    rows.append(dict(family=fam, param=param, value=pt["hp"][param], bound=key,
                                     n_runs=int(mask.sum()), n_valid=int(valid.sum()),
                                     ratio_median=float(np.median(bd[mask] / e[mask])),
                                     ratio_min=float(np.min(bd[mask] / e[mask])), ratio_max=float(np.max(bd[mask] / e[mask])),
                                     **{f"tag_{k}": v for k, v in pt["tags"].items()}))
    return rows


def regime_summary(rows):
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        for key in ("t1", "t2"):
          for tag in ("in_regime", "in_regime_eq345"):
            for reg in (True, False):
                sel = [r for r in rows if r["family"] == fam and r["bound"] == key and r["tag_" + tag] == reg]
                if sel:
                    out[f"{fam}|{key}|{tag}={reg}"] = dict(
                        runs=sum(r["n_runs"] for r in sel), valid=sum(r["n_valid"] for r in sel),
                        points=len(sel), points_all_valid=sum(r["n_valid"] == r["n_runs"] for r in sel),
                        ratio_median=float(np.median([r["ratio_median"] for r in sel])),
                        ratio_range=[min(r["ratio_min"] for r in sel), max(r["ratio_max"] for r in sel)])
    return out


# ----------------------------------------------------------------------------- plots
PARAM_LABEL = dict(lam=r"weight decay $\lambda$", nu2=r"init scale $\nu^2$", n="sample size $n$", m="width $m$", eta=r"step size $\eta$")


def plot_family(name, res):
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.8))
    for ax, (param, pts) in zip(axes, res.items()):
        x = np.array([p["hp"][param] for p in pts])
        for key, col in (("t1", "tab:blue"), ("t2", "tab:red")):
            mu = np.array([np.nanmean(arr(p, key)) for p in pts]); sd = np.array([np.nanstd(arr(p, key)) for p in pts])
            th = np.array([np.nanmean(arr(p, key + "_eq8")) if np.isfinite(arr(p, key + "_eq8")).any() else np.nan for p in pts])
            ax.errorbar(x, mu, yerr=sd, color=col, marker="o", ms=4, label=f"{key} measured")
            ax.plot(x, th, color=col, ls="--", marker="x", ms=4, label=f"{key} Eq.8 bound")
        inreg = [p["tags"]["in_regime"] for p in pts]
        ax.set_xscale("log", base=2 if param == "lam" else 10); ax.set_yscale("log")
        ax.set_xlabel(PARAM_LABEL[param]); ax.set_ylabel("steps"); ax.grid(alpha=0.3, which="both")
        if all(inreg): ax.set_title("all points in-regime", fontsize=9)
        elif not any(inreg): ax.set_title("out-of-regime", fontsize=9)
        else: ax.set_title("mixed regime", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{name}: measured (solid, ±std over {SEEDS} seeds) vs literal Eq.8 bound (dashed)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(RES, f"sweep_{name}.{ext}"), dpi=140)
    plt.close(fig)


def plot_scatter(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key in zip(axes, ("t1", "t2")):
        for fam, mk in zip(all_res, ("o", "s", "^")):
            for inreg, col in ((True, "tab:green"), (False, "tab:orange")):
                xs, ys = [], []
                for pts in all_res[fam].values():
                    for pt in pts:
                        if pt["tags"]["in_regime"] != inreg: continue
                        b, e = arr(pt, key + "_eq8"), arr(pt, key)
                        mk_ = np.isfinite(b) & np.isfinite(e)
                        xs += list(b[mk_]); ys += list(e[mk_])
                if xs: ax.scatter(xs, ys, s=10, marker=mk, c=col, alpha=0.6, label=f"{fam} ({'in' if inreg else 'out of'}-regime)")
        lim = ax.get_xlim() + ax.get_ylim(); lo, hi = min(lim), max(lim)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="measured = bound")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(f"literal Eq.8 {key} bound (steps)"); ax.set_ylabel(f"measured {key} (steps)")
        ax.set_title(f"{key}: bound should lie {'above' if key=='t1' else 'below'} the diagonal-y=x (measured {'≤' if key=='t1' else '≥'} bound)", fontsize=9)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(RES, f"sweep_pred_vs_measured.{ext}"), dpi=140)
    plt.close(fig)


def plot_amplify_eliminate(all_res):
    """t2/t1 (grokking ratio) vs lam, nu2, n.  Grokking 'eliminated' where t2 <= t1 (no generalisation delay)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, param in zip(axes, ("lam", "nu2", "n")):
        for fam, col in zip(all_res, ("tab:red", "tab:green", "tab:purple")):
            pts = all_res[fam][param]
            x = np.array([p["hp"][param] for p in pts])
            r = np.array([np.nanmedian(arr(p, "t2") / arr(p, "t1")) for p in pts])
            r[~np.isfinite(r)] = np.nan      # t1 = 0 -> ratio undefined (infinite); not drawn
            ax.plot(x, r, marker="o", color=col, label=fam)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("log", base=2 if param == "lam" else 10); ax.set_yscale("log")
        ax.set_xlabel(PARAM_LABEL[param]); ax.set_ylabel("$t_2 / t_1$  (median over seeds)"); ax.grid(alpha=0.3, which="both")
        ax.text(0.02, 0.04, "below dotted line (≤1): grokking eliminated", transform=ax.transAxes, fontsize=7)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(RES, f"sweep_amplified_vs_eliminated.{ext}"), dpi=140)
    plt.close(fig)


def amplify_table(all_res):
    rows = []
    for fam, res in all_res.items():
        for param in ("lam", "nu2", "n"):
            for pt in res[param]:
                t1, t2 = arr(pt, "t1"), arr(pt, "t2")
                rows.append(dict(family=fam, param=param, value=pt["hp"][param], t1=float(np.nanmean(t1)),
                                 t2=float(np.nanmean(t2)) if np.isfinite(t2).any() else None,
                                 gap_ratio_median=float(np.nanmedian(t2 / t1)) if np.isfinite(t2).any() else None,
                                 n_t2_not_reached=int(np.sum(~np.isfinite(t2))),
                                 eliminated=bool(np.nanmedian(t2 / t1) <= 1) if np.isfinite(t2).any() else None))
    return rows


def main_sweeps():
    os.makedirs(RES, exist_ok=True)
    all_res = {}
    for name in FAMILIES:
        t0 = time.time(); all_res[name] = run_family(name); log.info("%s: %.1fs", name, time.time() - t0)
    with open(os.path.join(RES, "sweeps_raw.json"), "w") as f: json.dump(all_res, f, cls=NumpyEncoder)
    rows = validity_table(all_res)
    with open(os.path.join(RES, "sweeps_validity.json"), "w") as f: json.dump(dict(rows=rows, summary=regime_summary(rows)), f, cls=NumpyEncoder, indent=1)
    slopes = {fam: slope_tests(fam, res) for fam, res in all_res.items()}
    with open(os.path.join(RES, "sweeps_slopes.json"), "w") as f: json.dump(slopes, f, cls=NumpyEncoder, indent=1)
    with open(os.path.join(RES, "sweeps_amplify.json"), "w") as f: json.dump(amplify_table(all_res), f, cls=NumpyEncoder, indent=1)
    # flat CSV of all points
    with open(os.path.join(RES, "sweeps_points.csv"), "w") as f:
        f.write("family,param,value,n,m,nu2,lam,eta,eps,in_regime,in_regime_eq345,eq3,eq4,eq5,eta_thm42,t2_nonvacuous,t1_mean,t1_std,t2_mean,t2_std,t1_eq8_mean,t2_eq8,n_seeds,n_t2_missing\n")
        for fam, res in all_res.items():
            for param, pts in res.items():
                for pt in pts:
                    h, tg = pt["hp"], pt["tags"]; t1, t2 = arr(pt, "t1"), arr(pt, "t2")
                    f.write(",".join(map(str, [fam, param, h[param], h["n"], h["m"], h["nu2"], h["lam"], h["eta"], h["eps"], tg["in_regime"], tg["in_regime_eq345"], tg["eq3"], tg["eq4"], tg["eq5"], tg["eta_thm42"], tg["t2_nonvacuous"],
                                               np.nanmean(t1), np.nanstd(t1), np.nanmean(t2) if np.isfinite(t2).any() else "nan", np.nanstd(t2) if np.isfinite(t2).any() else "nan",
                                               np.nanmean(arr(pt, "t1_eq8")), np.nanmean(arr(pt, "t2_eq8")) if np.isfinite(arr(pt, "t2_eq8")).any() else "nan", len(t1), int(np.sum(~np.isfinite(t2)))])) + "\n")
    for name, res in all_res.items(): plot_family(name, res)
    plot_scatter(all_res); plot_amplify_eliminate(all_res)
    return all_res


def main_all():
    from . import step1, diagnose
    for cfgf in sorted(os.listdir(os.path.join(ROOT, "configs"))):
        if cfgf.startswith("fig1_"): step1.main(os.path.join(ROOT, "configs", cfgf))
    diagnose.main()
    main_sweeps()
    from . import report
    report.build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); ap.add_argument("--sweeps-only", action="store_true")
    a = ap.parse_args()
    if a.sweeps_only: main_sweeps()
    else: main_all()
