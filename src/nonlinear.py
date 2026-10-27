"""Step 3: non-linear students (paper Sec. 5.2/5.3, Fig. 1 right, Fig. 3).

Network N(x) = sum_j a_j relu(<w_j, x>) (no 1/sqrt(m) factor, as in the paper).
Input distribution is not given by the paper: x ~ N(0, I_d) (project choice).

* rf : random features (only a trained, W fixed), teacher relu(<w*,x>), ||w*||=1, a_j~N(0,1), w_j~N(0,nu2/(d m) I).
       Linear in a, so GD is evaluated exactly with the SVD of the feature matrix (same engine idea as train.SpectralGD).
* full: both layers trained with explicit full-batch GD + weight decay, zero teacher,
        a_j~N(0,1/m), w_j~N(0,nu2/d I). Train L_n = 1/(2n) sum N^2, test = E N^2 (MC on fresh x).
Usage: python -m src.nonlinear --all
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .train import log_steps, NumpyEncoder

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
torch.set_default_dtype(torch.float64)


def rf_run(n, m, d, lam, nu2, eta, T, seed, n_test=5000, k=150, teacher="relu"):
    rng = np.random.default_rng(seed)
    ws = rng.standard_normal(d); ws /= np.linalg.norm(ws)
    X, Xt = rng.standard_normal((n, d)), rng.standard_normal((n_test, d))
    W = rng.standard_normal((m, d)) * np.sqrt(nu2 / (d * m))
    a0 = rng.standard_normal(m)
    Phi, Pt = np.maximum(X @ W.T, 0), np.maximum(Xt @ W.T, 0)
    y, yt = np.maximum(X @ ws, 0), np.maximum(Xt @ ws, 0)
    if teacher == "zero":
        y, yt = 0 * y, 0 * yt
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    lmax = s[0] ** 2 / n
    stable = eta < 2 / (lam + lmax)
    A = 1 - eta * lam - eta * s ** 2 / n
    rho = 1 - eta * lam
    c0, uty = Vt @ a0, U.T @ y
    fix = s / (n * lam + s ** 2) * uty
    perp = a0 - Vt.T @ c0
    steps = np.unique(np.concatenate([log_steps(T, k), 10 ** np.arange(0, int(np.log10(T)) + 1)]))
    tr, te, nrm = [], [], []
    for t in steps:
        at = A ** t
        c = at * c0 + (1 - at) * fix
        a = Vt.T @ c + rho ** t * perp
        tr.append(0.5 * np.mean((Phi @ a - y) ** 2)); te.append(np.mean((Pt @ a - yt) ** 2)); nrm.append(a @ a)
    return dict(step=steps, train=np.array(tr), test=np.array(te), norm2=np.array(nrm),
                stable=bool(stable), eta_limit=float(2 / (lam + lmax)),
                null_test=float(np.mean(yt ** 2)))


def full_run(n, m, d, lam, nu2, eta, T, seed, n_test=5000, k=150):
    g = torch.Generator().manual_seed(seed)
    X, Xt = torch.randn(n, d, generator=g), torch.randn(n_test, d, generator=g)
    W = (torch.randn(m, d, generator=g) * np.sqrt(nu2 / d)).requires_grad_()
    a = (torch.randn(m, generator=g) / np.sqrt(m)).requires_grad_()
    steps = log_steps(T, k); want = set(int(s) for s in steps)
    out = {kk: [] for kk in ("step", "train", "test")}
    for t in range(int(T) + 1):
        N = torch.relu(X @ W.T) @ a
        loss = 0.5 * torch.mean(N ** 2)
        if t in want:
            with torch.no_grad():
                out["step"].append(t); out["train"].append(loss.item())
                out["test"].append(torch.mean((torch.relu(Xt @ W.T) @ a) ** 2).item())
        gW, ga = torch.autograd.grad(loss, (W, a))
        with torch.no_grad():
            W -= eta * (gW + lam * W); a -= eta * (ga + lam * a)
    return {kk: np.array(v) for kk, v in out.items()}


def times(r, eps=0.01, c=0.01):
    """t1 = last logged step with train >= eps, t2 = first logged step with test <= c (log-grid resolution)."""
    i = np.nonzero(r["train"] >= eps)[0]
    t1 = float(r["step"][i[-1]]) if len(i) and i[-1] < len(r["step"]) - 1 else np.nan
    j = np.nonzero(r["test"] <= c)[0]
    t2 = float(r["step"][j[0]]) if len(j) else np.nan
    return t1, t2


def agg(rs):
    """Mean over seeds plus min/max band for train and test (log-scale friendly)."""
    out = {"step": rs[0]["step"]}
    for k in ("train", "test"):
        A = np.array([r[k] for r in rs])
        out[k] = A.mean(0); out[k + "_lo"] = A.min(0); out[k + "_hi"] = A.max(0)
    return out


def tstats(rs):
    ts = np.array([times(r) for r in rs], dtype=float)
    return dict(t1=float(np.nanmean(ts[:, 0])), t1_sd=float(np.nanstd(ts[:, 0])),
                t2=float(np.nanmean(ts[:, 1])), t2_sd=float(np.nanstd(ts[:, 1])), n_seeds=len(rs))


def panel(ax, runs, labels, title):
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    for r, lb, col in zip(runs, labels, cm):
        ax.loglog(r["step"][1:], r["train"][1:], "--", color=col)
        ax.loglog(r["step"][1:], r["test"][1:], "-", color=col, label=lb)
        if "test_lo" in r:
            for k in ("train", "test"):
                ax.fill_between(r["step"][1:], np.maximum(r[k + "_lo"][1:], 1e-30), np.maximum(r[k + "_hi"][1:], 1e-30), color=col, alpha=0.2, lw=0)
    ax.set_title(title); ax.set_xlabel("GD steps"); ax.legend(fontsize=7)


def _seed_job(args):
    torch.set_num_threads(1)
    cfg, T, seed = args
    return seed, full_run(T=T, seed=seed, **cfg)


def exp_full(seeds, T):
    """Fig. 1-right analogue; independent dataset + init per seed (seeds run in parallel processes)."""
    from multiprocessing import Pool
    cfg = dict(n=50, m=1000, d=50, lam=0.1, nu2=1.0, eta=1e-4)
    with Pool(min(seeds, os.cpu_count())) as pool:
        res = dict(pool.imap_unordered(_seed_job, [(cfg, T, s) for s in range(seeds)]))
    runs = [res[s] for s in range(seeds)]
    m = agg(runs)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    st = m["step"][1:]
    for key, ls in (("train", "--"), ("test", "-")):
        ax.loglog(st, m[key][1:], ls, label=key)
        ax.fill_between(st, np.maximum(m[key + "_lo"][1:], 1e-30), np.maximum(m[key + "_hi"][1:], 1e-30), alpha=0.2, lw=0)
    ax.set_xlabel("GD steps"); ax.set_title(f"Fig.1-right analogue ({seeds} runs)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{RES}/nl_fig1_right.png", dpi=130); plt.close(fig)
    st_ = tstats(runs); print("fig1 right", st_, flush=True)
    json.dump(dict(cfg=cfg, T=T, seeds=seeds, runs=runs, times=[times(r) for r in runs], stats=st_),
              open(f"{RES}/nl_fig1_right.json", "w"), cls=NumpyEncoder)


def exp_rf(seeds, T):
    base = dict(n=100, m=10000, d=100, lam=1e-5, nu2=1.0, eta=1.0)
    grids = {"lam": [1e-4, 1e-5, 1e-6], "n": [50, 100, 200], "m": [2000, 5000, 10000], "nu2": [0.25, 1.0, 4.0]}
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8)); summ = {}
    for ax, (k, vals) in zip(axs, grids.items()):
        runs = []
        for v in vals:
            rs = [rf_run(T=T, seed=s, **{**base, k: v}) for s in range(seeds)]
            mean = agg(rs)
            summ[f"{k}={v}"] = dict(**tstats(rs), stable=all(r["stable"] for r in rs), final_test=mean["test"][-1],
                                    final_test_sd=float(np.std([r["test"][-1] for r in rs])),
                                    min_test=float(mean["test"].min()), null_test=float(np.mean([r["null_test"] for r in rs])))
            print(k, v, summ[f"{k}={v}"], flush=True); runs.append(mean)
        panel(ax, runs, [f"{k}={v}" for v in vals], f"vary {k}")
    fig.tight_layout(); fig.savefig(f"{RES}/nl_fig3_random_features.png", dpi=130); plt.close(fig)
    json.dump(dict(base=base, T=T, seeds=seeds, summary=summ), open(f"{RES}/nl_fig3_summary.json", "w"), cls=NumpyEncoder)


PAPER_FIG1_LEFT = {1: (0.39, 0.40), 10: (0.27, 0.32), 100: (0.07, 0.24), 1000: (0.01, 0.20), 10000: (None, 0.17), 100000: (None, 0.03)}


def _rf_zero_job(args):
    cfg, T, seed = args
    return seed, rf_run(T=T, seed=seed, teacher="zero", **cfg)


def exp_fig1_left_rf(seeds, T):
    """Paper Fig. 1 left, run with the random ReLU features model (only a trained), ZERO teacher.
    Settings: n=100, m=1e4, d=100 (d is my pick: it is the Fig. 3 value), lam=1e-5, nu2=1, eta=1, 50 runs.
    Evidence for this model: matches the paper's curve readings (PAPER_FIG1_LEFT, read off the figure by eye)."""
    from multiprocessing import Pool
    cfg = dict(n=100, m=10000, d=100, lam=1e-5, nu2=1.0, eta=1.0)
    with Pool(min(seeds, os.cpu_count())) as pool:
        res = dict(pool.imap_unordered(_rf_zero_job, [(cfg, T, s) for s in range(seeds)]))
    runs = [res[s] for s in range(seeds)]
    m = agg(runs); st = m["step"]
    cmp = {}
    for t, (ptr, pte) in PAPER_FIG1_LEFT.items():
        i = int(np.searchsorted(st, t)); cmp[t] = dict(my_train_half=float(m["train"][i]), my_test=float(m["test"][i]), paper_train=ptr, paper_test=pte)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for key, col, lb in (("train", "red", "training loss"), ("test", "blue", "test loss")):
        ax.semilogx(st[1:], m[key][1:], color=col, label=lb)
        ax.fill_between(st[1:], m[key + "_lo"][1:], m[key + "_hi"][1:], color=col, alpha=0.2, lw=0)
    ax.plot(list(PAPER_FIG1_LEFT), [v[1] for v in PAPER_FIG1_LEFT.values()], "k^", label="paper (test, read off)")
    ax.plot([t for t, v in PAPER_FIG1_LEFT.items() if v[0]], [v[0] for v in PAPER_FIG1_LEFT.values() if v[0]], "kv", label="paper (train, read off)")
    ax.set_ylim(0, 0.65); ax.set_xlim(0.7, 1e6); ax.set_xlabel("training steps"); ax.set_ylabel("square loss"); ax.legend(fontsize=7)
    ax.set_title(f"Fig.1-left, random ReLU features + zero teacher ({seeds} runs)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{RES}/nl_fig1_left_rf.png", dpi=130); plt.close(fig)
    ts = np.array([times(r) for r in runs], dtype=float)
    stats = dict(**tstats(runs), final_train_max=float(max(r["train"][-1] for r in runs)), final_test_max=float(max(r["test"][-1] for r in runs)),
                 stable=all(r["stable"] for r in runs))
    print("fig1 left rf", stats, flush=True)
    for t, v in cmp.items(): print(t, v, flush=True)
    json.dump(dict(cfg=cfg, T=T, seeds=seeds, stats=stats, comparison=cmp, mean=m), open(f"{RES}/nl_fig1_left_rf.json", "w"), cls=NumpyEncoder)


def _fig3p_job(args):
    key, kw, T, seed, nt = args
    return key, seed, rf_run(T=T, seed=seed, teacher="zero", n_test=nt, **kw)


def exp_fig3_paper(seeds, T):
    """Fig. 3 with the paper's panel values (read from the figure legends) and a ZERO teacher.
    The relu-teacher version (exp_rf) does not match the paper (floor 0.2, t2 never reached); the zero teacher does
    (t2 = 1.5e4/1.5e5/1.5e6 for lam=1e-4/1e-5/1e-6, as in the paper)."""
    from multiprocessing import Pool
    base = dict(n=100, m=10000, d=100, lam=1e-5, nu2=1.0, eta=1.0)
    grids = {"lam": [1e-4, 1e-5, 1e-6], "n": [100, 200, 500], "m": [1000, 10000, 100000], "nu2": [1.0, 5.0, 10.0]}
    jobs, seen = [], set()
    for k, vals in grids.items():
        for v in vals:
            kw = {**base, k: v}
            for sd in range(seeds):
                key = (tuple(sorted(kw.items())), sd)
                if key not in seen:
                    seen.add(key); jobs.append((key[0], kw, T, sd, 500 if kw["m"] > 20000 else 3000))
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = {(k, sd): r for k, sd, r in pool.imap_unordered(_fig3p_job, jobs)}
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8)); summ = {}
    for ax, (k, vals) in zip(axs, grids.items()):
        runs = []
        for v in vals:
            kw = {**base, k: v}; rs = [res[(tuple(sorted(kw.items())), sd)] for sd in range(seeds)]
            mean = agg(rs); i = lambda t: int(np.searchsorted(mean["step"], t))
            summ[f"{k}={v}"] = dict(**tstats(rs), test_at={str(t): float(mean["test"][i(t)]) for t in (1, 100, 1000, 10000, 100000)},
                                    train_at={str(t): float(mean["train"][i(t)]) for t in (1, 100, 1000)})
            print(k, v, summ[f"{k}={v}"], flush=True); runs.append(mean)
        panel(ax, runs, [f"{k}={v}" for v in vals], f"vary {k}")
        ax.set_xscale("log"); ax.set_yscale("linear"); ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(f"{RES}/nl_fig3_zero_teacher_paper_grids.png", dpi=130); plt.close(fig)
    json.dump(dict(base=base, T=T, seeds=seeds, summary=summ), open(f"{RES}/nl_fig3_zero_teacher_summary.json", "w"), cls=NumpyEncoder)


def _full_job(args):
    torch.set_num_threads(1)
    key, kw, T, seed = args
    return key, full_run(T=T, seed=seed, **kw)


def exp_fig4(seeds, T):
    """Fig. 4 analogue: fully-trained net, zero teacher; vary lam, n, m, nu2 one at a time (grids are mine)."""
    from multiprocessing import Pool
    base = dict(n=50, m=1000, d=50, lam=0.05, nu2=1.0, eta=1e-4)
    grids = {"lam": [0.025, 0.05, 0.1], "n": [25, 50, 100], "m": [500, 1000, 2000], "nu2": [0.5, 1.0, 2.0]}
    jobs, seen = [], set()
    for k, vals in grids.items():
        for v in vals:
            for s in range(seeds):
                kw = {**base, k: v}; key = (tuple(sorted(kw.items())), s)
                if key not in seen: seen.add(key); jobs.append((key, kw, T, s))
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = dict(pool.imap_unordered(_full_job, jobs))
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8)); summ = {}
    for ax, (k, vals) in zip(axs, grids.items()):
        runs = []
        for v in vals:
            kw = {**base, k: v}
            rs = [res[(tuple(sorted(kw.items())), s)] for s in range(seeds)]
            mean = agg(rs)
            summ[f"{k}={v}"] = dict(**tstats(rs), final_train=float(mean["train"][-1]), final_test=float(mean["test"][-1]))
            print(k, v, summ[f"{k}={v}"], flush=True); runs.append(mean)
        panel(ax, runs, [f"{k}={v}" for v in vals], f"vary {k}")
    fig.tight_layout(); fig.savefig(f"{RES}/nl_fig4_full_sweeps.png", dpi=130); plt.close(fig)
    json.dump(dict(base=base, T=T, seeds=seeds, summary=summ), open(f"{RES}/nl_fig4_summary.json", "w"), cls=NumpyEncoder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true"); ap.add_argument("--rf", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--fig3p", action="store_true"); ap.add_argument("--fig1l-rf", action="store_true"); ap.add_argument("--fig4", action="store_true"); ap.add_argument("--T-fig4", type=int, default=1_000_000)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--T-full", type=int, default=1_000_000)
    ap.add_argument("--T-rf", type=int, default=10_000_000)
    a = ap.parse_args()
    if a.all or a.rf: exp_rf(a.seeds, a.T_rf)
    if a.fig3p: exp_fig3_paper(a.seeds, a.T_rf)
    if a.fig1l_rf: exp_fig1_left_rf(a.seeds, a.T_rf)
    if a.fig4: exp_fig4(a.seeds, a.T_fig4)
    if a.all or a.full: exp_full(a.seeds, a.T_full)
