"""Generate results/results_tables.md from the JSON/CSV result files (no hand-copied numbers) and splice it into RESULTS.md
between <!--TABLES-->/<!--/TABLES--> markers."""
import json, os, re
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
J = lambda n: json.load(open(os.path.join(RES, n)))
g = lambda x: f"{x:.3g}"


def step1_table():
    L = ["| config | measured t1 (mean±std) | Eq.8 t1 bound | bound/measured | measured t2 | Eq.8 t2 bound | bound/measured | gap t2−t1 measured | gap lower bound (t2_b−t1_b) | valid (t1≤b, t2≥b) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("fig1_realizable", "fig1_left_zero_teacher"):
        s = J(f"{n}_summary.json")
        t1, t2, b1 = np.array(s["t1_emp"]), np.array(s["t2_emp"]), np.array(s["t1_eq8"]); b2 = s["t2_eq8"]
        L.append(f"| {n} | {g(t1.mean())}±{g(t1.std())} | {g(b1.mean())} | {g(np.mean(b1/t1))} | {g(t2.mean())}±{g(t2.std())} | {g(b2)} | {g(b2/t2.mean())} | {g((t2-t1).mean())} | {g(b2-b1.mean())} | {s['t1_valid_all'] and s['t2_valid_all']} ({s['n_seeds']}/{s['n_seeds']} seeds) |")
    return "\n".join(L)


def validity_table():
    v = J("sweeps_validity.json")
    rows = v["rows"]
    L = ["| family | tag | bound | measured runs | bound holds | median bound/measured | bound/measured range |", "|---|---|---|---|---|---|---|"]
    for fam in sorted({r["family"] for r in rows}):
        for tag in ("in_regime", "in_regime_eq345"):
            for key in ("t1", "t2"):
                for reg in (True, False):
                    sel = [r for r in rows if r["family"] == fam and r["bound"] == key and r["tag_" + tag] == reg]
                    if not sel: continue
                    runs, ok = sum(r["n_runs"] for r in sel), sum(r["n_valid"] for r in sel)
                    med = np.median([r["ratio_median"] for r in sel]); lo = min(r["ratio_min"] for r in sel); hi = max(r["ratio_max"] for r in sel)
                    L.append(f"| {fam} | {tag}={reg} | {key} | {runs} | {ok}/{runs} | {g(med)} | {g(lo)} – {g(hi)} |")
    tot = sum(r["n_runs"] for r in rows); ok = sum(r["n_valid"] for r in rows)
    return "\n".join(L) + f"\n\nOverall: {ok}/{tot} (run, bound) checks satisfy the bound."


def slopes_table():
    s = J("sweeps_slopes.json")
    names = {
        "t2_vs_1/(eta*lam)__lam_sweep": ("log-log slope, t2 vs 1/(ηλ), λ-sweep (ηλ≤1e-3)", "1"),
        "t2_vs_1/(eta*lam)__eta_sweep": ("log-log slope, t2 vs 1/(ηλ), η-sweep", "1"),
        "t1_vs_1/(eta*lam)__lam_sweep": ("log-log slope, t1 vs 1/(ηλ), λ-sweep", "0 (λ has no effect on the t1 bound)"),
        "t1_vs_1/(eta*lam)__eta_sweep": ("log-log slope, t1 vs 1/(ηλ), η-sweep", "1 (t1 ∝ 1/η)"),
        "t1_loglog_slope_vs_n": ("log-log slope, t1 vs n", "≥1 (n/λ⁺min(n))"),
        "t2_loglog_slope_vs_n": ("log-log slope, t2 vs n", "≈0"),
        "t2_slope_vs_ln_nu2__measured_over_theory": ("t2 slope vs ln ν² / Eq.8 slope 1/(2.02ηλ)", "≈1"),
        "t1_slope_vs_ln_nu2__measured_over_theory": ("t1 slope vs ln ν² / Eq.8 slope n/(2ηλ⁺min)", "≈1"),
    }
    L = ["| family | quantity | fitted (mean over seeds) | 95% CI (t, 10 seeds) | theory expects |", "|---|---|---|---|---|"]
    for fam, d in s.items():
        for k, (lab, exp) in names.items():
            m, lo, hi = d[k]
            L.append(f"| {fam} | {lab} | {m:.4f} | [{lo:.4f}, {hi:.4f}] | {exp} |")
        for k in d:
            if k.startswith(("t1_shift", "t2_shift")):
                L.append(f"| {fam} | {k.split('__')[0]} (steps; measured vs Eq.8-predicted) | {d[k][0]:.1f} vs {d[k][1]:.1f} | – | same sign & size |")
    return "\n".join(L)


def diagnosis_table():
    d = J("t1_diagnosis.json")["candidates"]
    L = ["| rank | λ⁺min | ε factor | ‖θ(0)‖² used | b² | RMS log2 error vs paper's read-off line | mean offset (octaves) | shape-only RMS (octaves) |", "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(d[:6], 1):
        L.append(f"| {i} | {r['lam_plus_min']} | {r['eps_factor']} | {r['theta0']} | {r['b2']} | {r['rms_log2_err']:.2f} | {r['mean_offset_log2']:+.2f} | {r['shape_rms_log2']:.2f} |")
    lit = [(i, r) for i, r in enumerate(d, 1) if r["lam_plus_min"] == "emp" and r["eps_factor"] == 1 and r["theta0"] == "E=m*nu2" and r["b2"] == 1.5][0]
    r = lit[1]
    L.append(f"| {lit[0]} (**literal Eq. 8**) | emp | 1 | E=m·ν² | 1.5 | {r['rms_log2_err']:.2f} | {r['mean_offset_log2']:+.2f} | {r['shape_rms_log2']:.2f} |")
    l1 = [(i, r) for i, r in enumerate(d, 1) if r["lam_plus_min"] == "1" and r["eps_factor"] == 1 and r["theta0"] == "E=m*nu2" and r["b2"] == 1.5][0]
    r = l1[1]
    L.append(f"| {l1[0]} (λ⁺min=1 approx., otherwise literal) | 1 | 1 | E=m·ν² | 1.5 | {r['rms_log2_err']:.2f} | {r['mean_offset_log2']:+.2f} | {r['shape_rms_log2']:.2f} |")
    return "\n".join(L)


def amplify_table():
    a = J("sweeps_amplify.json")
    L = ["| family | sweep | value | t1 | t2 | median t2/t1 | t2 not reached (of 10) |", "|---|---|---|---|---|---|---|"]
    for fam in sorted({x["family"] for x in a}):
        for par, picks in (("lam", "ends+min"), ("nu2", "ends"), ("n", "ends")):
            sel = [x for x in a if x["family"] == fam and x["param"] == par]
            fin = [x for x in sel if x["gap_ratio_median"] is not None and np.isfinite(x["gap_ratio_median"])]
            idx = {0, len(sel) - 1}
            if picks == "ends+min" and fin: idx.add(sel.index(min(fin, key=lambda x: x["gap_ratio_median"])))
            for i in sorted(idx):
                x = sel[i]; r = x["gap_ratio_median"]
                L.append(f"| {fam} | {par} | {g(x['value'])} | {g(x['t1'])} | {g(x['t2']) if x['t2'] is not None else '–'} | {g(r) if r is not None and np.isfinite(r) else '∞/undefined'} | {x['n_t2_not_reached']} |")
    return "\n".join(L)


def late_rise_text():
    l = J("late_rise.json")
    r = l["max_relative_rise_per_seed"]
    return (f"Across the 10 seeds of the realizable default config the largest rise of L_n above its running minimum is "
            f"{', '.join(f'{x:.1e}' for x in r)} (relative); {sum(x > 0 for x in r)}/10 seeds rise at all. In seed {l['seed']}: min L_n = {l['min_train']:.4e} at t≈{l['step_of_min']}, "
            f"final L_n = {l['final_train']:.4e} = ridge-solution train loss {l['ridge_train']:.4e}; ‖θ‖² goes {l['norm2_init']:.0f} → {l['norm2_final']:.4f} "
            f"(ridge {l['norm2_ridge']:.4f} vs interpolant {l['norm2_mn']:.4f}). Final test loss {l['final_test']:.3e} vs null predictor (θ=0) {l['null_predictor_test']:.1e}: ratio {l['ridge_test_over_null']:.3f}.")


def build():
    parts = {
        "STEP1": step1_table(), "VALIDITY": validity_table(), "SLOPES": slopes_table(),
        "DIAG": diagnosis_table(), "AMPLIFY": amplify_table(), "LATE": late_rise_text()}
    md = "\n\n".join(f"<!--{k}-->\n{v}\n<!--/{k}-->" for k, v in parts.items())
    open(os.path.join(RES, "results_tables.md"), "w").write(md)
    rp = os.path.join(ROOT, "RESULTS.md")
    if os.path.exists(rp):
        s = open(rp).read()
        for k, v in parts.items():
            s = re.sub(rf"<!--{k}-->.*?<!--/{k}-->", lambda m: f"<!--{k}-->\n{v}\n<!--/{k}-->", s, flags=re.S)
        open(rp, "w").write(s)
    return parts


if __name__ == "__main__":
    for k, v in build().items(): print("###", k); print(v); print()
