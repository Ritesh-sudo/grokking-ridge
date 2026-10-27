# PAPER_NOTES — "To Grok Grokking: Provable Grokking in Ridge Regression"

Xu, Vardi, Safran. ICML 2026 (PMLR 306). arXiv:2601.19791, **v4 (16 Jul 2026)**.
Source: `paper.pdf` (downloaded from arxiv.org/pdf/2601.19791), text extracted to `paper.txt`.

**Revision 2 (after LaTeX source + figures):** I downloaded the arXiv LaTeX source (`src_tex/`) and read all 19 plot PNGs
visually (poppler installed). Eq. 1, Thm 4.2 (Eq. 3–7), Thm 4.5 and Remark A.14 (Eq. 8) were re-verified **verbatim against the
TeX** — all formulas in §2 below are confirmed correct, including the 4ηλ in Eq. 7 and the 2.02ηλ / 8mε / 14mν² constants in Eq. 8.
The source bundle contains **no code, no data, no scripts and no code link** (only .tex/.bib/.sty + PNGs), so nothing can be
checked against the authors' implementation. Items marked UNSPECIFIED remain so unless §6 says otherwise.

---

## 1. Problem setup (Sections 2, 4, 5.1)

| Item | Paper's specification |
|---|---|
| Data | x_i ~ D_x, labels y_i = N*(x_i) **exactly (noiseless)**, i = 1..n |
| Feature map | φ: R^d → R^m fixed. **Section 5.1 uses identity: φ(x) = x ∈ R^m** |
| Student | N(x;θ) = ⟨θ, φ(x)⟩, θ ∈ R^m |
| Teacher (realizable) | N*(x) = ⟨θ*, φ(x)⟩, **‖θ*‖₂ = 1** in experiments. Zero teacher (θ* = 0) used as warm-up (Thm 4.1) and for Fig 1 and Fig 4 |
| Features in experiments | φ(x_1..x_n) i.i.d. **N(0, (1/m) I_m)** (Sec 5.1, Remark A.14) |
| Init | θ(0) ~ N(0, ν² I_m) |
| Loss (Eq. 1) | L_n(θ;λ) = (1/2n) Σ (⟨θ−θ*, φ(x_i)⟩)² + (λ/2)‖θ‖² |
| Update (Eq. 2) | θ(t+1) = θ(t) − (η/n) Σ ⟨θ(t)−θ*, φ(x_i)⟩ φ(x_i) − ηλ θ(t) (full-batch GD) |
| Train loss reported | L_n(θ) = (1/2n) Σ (N−N*)² (**note the 1/2 factor**, unregularized) |
| Test loss | L(θ) = E_x[(N(x;θ) − N*(x))²] (population, **no 1/2 factor**). For φ~N(0,I/m): L = ‖θ−θ*‖²/m |
| Notation | Φ ∈ R^{n×m} empirical feature matrix; L̄ = ‖Φ‖_F²/n; Σ = E[φφᵀ]; λ⁺_min(ΦᵀΦ) = smallest *positive* eigenvalue |
| Thresholds | c = ε = 0.01 in **all** experiments |
| Default hyperparameters (Sec 5.1) | η = 1, n = 100, m = 1000, ν² = 1, λ = 1e-4 |

Implications I derived (not stated verbatim in the paper; flagged as my derivations):
- Gradient is (1/n)Φᵀ(Φθ − y) + λθ, so the closed-form minimizer is θ*_λ = (ΦᵀΦ + nλ I)⁻¹ Φᵀ y — matches the formula in the task brief.
- With φ ~ N(0, I/m), ‖φ‖² ≈ 1, so L̄ ≈ 1; the stability condition η < 1/(λ + λ_max(ΦᵀΦ)/n) with η = 1 requires λ_max(ΦᵀΦ)/n < 1 − λ. λ_max(ΦᵀΦ)/n ≈ (1/m)(√m+√n)²/n ≈ (1+√(n/m))²/n·… — must be **checked numerically**, since η = 1 is claimed to be used.

## 2. Main theorem and critical times

### Definitions (Sec 3, Thm 4.2)
- t₁(ε) := max{ t : L_n(θ(t)) ≥ ε }  — *last* step at which train loss is ≥ ε (overfitting time)
- t₂(c) := min{ t : L(θ(t)) ≤ c }   — *first* step at which test loss is ≤ c (generalization time)
- Grokking time = t₂ − t₁.

### Theorem 4.2 (realizable teacher)
Assume ‖φ(x)‖₂ ≤ b ∀x; θ(0) ~ N(0, ν²I_m); ε > 0, c ≥ ε arbitrary; δ ∈ (0,1).
- (3) n = Ω( b⁴‖θ*‖⁴ / ε² · log(1/δ) )
- (4) m = n + Ω( max{ log(1/δ), ‖θ*‖²/ν², c/(λ_min(Σ) ν²) } )
- (5) λ = O( min{ ε/‖θ*‖², b√ε/‖θ*‖ } )
- η < 1/(λ + b²)

Then with prob ≥ 1−δ:
- **(6) Early critical time (upper bound on t₁):**
  t₁ ≤ n · ln( 6 b² ‖θ(0)‖² / ε ) / ( 2 η λ⁺_min(ΦᵀΦ) )
- **(7) Late critical time (lower bound on t₂):**
  t₂ ≥ ln( ((m−n) ν² / 2) · ( √(c/λ_min(Σ)) + ‖θ*‖ )⁻² ) / (4 η λ)
- and L(θ(t)) ≤ ε for all sufficiently large t.

Both are **one-sided bounds** (t₁ ≤ …, t₂ ≥ …), not exact predictions. The paper nevertheless plots them against
empirical values (Fig 2).

### Specialization for Gaussian features (Eq. 8, Remark A.14) — what Fig 2 actually plots
Needs φ_i ~ N(0, I/m), c = ε, ηλ ≤ 0.01, m = Ω(‖θ*‖²/ε). Uses b² = 3/2, λ_min(Σ) = 1/m, mν²/2 ≤ ‖θ(0)‖² ≤ 3mν²/2 (whp), and ln(1−ηλ) ≥ −1.01ηλ:
- **t₂ ≥ ln( (m−n) ν² / (8 m ε) ) / (2.02 η λ)**
- **t₁ ≤ n · ln( 14 m ν² / ε ) / ( 2 η λ⁺_min(ΦᵀΦ) )**

### Component theorems
- **Thm 4.1 (zero teacher, η ≤ 2/(L̄+2λ)):**
  (i) L_n(θ(t)) ≤ (L̄/2)(1 − ηλ⁺_min(ΦᵀΦ)/n − ηλ)^{2t} ‖θ(0)‖²;
  (ii) w.p. ≥ 1−2e^{−(m−n)/32}: L(θ(t)) ≥ λ_min(Σ)(1−ηλ)^{2t}(m−n)ν²/2;
  (iii) ‖θ(t)‖² ≤ (1−ηλ)^{2t}‖θ(0)‖².
- **Thm 4.4:** train loss ≤ ε for t ≥ ½ ln(ε/(6L̄‖θ(0)‖²)) / ln(1 − ηλ⁺_min/n − ηλ) [requires η < 1/(λ+λ_max(ΦᵀΦ)/n), λ = O(min{ε/‖θ*‖², √(L̄ε)/‖θ*‖})].
- **Thm 4.5:** L(θ(t)) ≥ c for all t ≤ log_{1−ηλ}( √(2/((m−n)ν²)) (√(c/λ_min(Σ)) + ‖θ*‖) ) [requires η < 1/λ]. (This is what gives Eq. 7.)
- **Thm 4.6:** GD converges to θ*_λ with L(θ*_λ) ≤ 2L_n(θ*_λ) + ε (Rademacher uniform convergence; η ≤ 1/(λ+b²)).

### Qualitative predictions about hyperparameters (Sec 4.2)
- **λ:** t₂ ∝ 1/λ; no effect on t₁ bound; (t₂−t₁) → ∞ as λ → 0. For large λ the t₁ bound is loose (should have ηλ⁺_min/n + ηλ in denominator).
- **n, m:** no clean theory for t₁ (no quantitative bound on λ⁺_min(ΦᵀΦ); only Marchenko–Pastur asymptotics λ⁺_min → γ⁻¹(1−√γ)² as m/n → γ). Paper claims empirically t₁ follows Eq. (6)/(8). Decreasing n **amplifies** grokking (speeds up train convergence, lowers t₁). Larger m has little effect on t₁, t₂.
- **ν²:** t₁, t₂ ∝ ln(ν²), with slope of t₂ larger than slope of t₁ ⇒ larger ν² **amplifies** grokking.
- **Eliminating** grokking = large λ (outside the paper's focus; bounds not tight there) and/or small ν²/large n so t₂ − t₁ shrinks. The paper does not run an explicit "elimination" experiment that I can find in the text — UNSPECIFIED (maybe visible in Fig 2 curves).

## 3. Experiments and figures

| Fig | What | Hyperparameters stated in text |
|---|---|---|
| **1 Left** | Ridge regression, **zero teacher**, train vs test MSE, log x-axis, **50 independent runs** (independent datasets and inits) | n=100, m=10000, λ=1e-5, ν²=1, η=1. (d not stated; features are φ(x) i.i.d. Gaussian per Sec 5.1 presumably) |
| **1 Right** | Two-layer ReLU net, both layers trained, **zero teacher**, 50 runs | d=50, n=50, m=1000, λ=0.1, ν²=1, η=1e-4 |
| **2** (4 panels) | Hyperparameter effects, realizable ridge regression, solid=empirical t₁,t₂; dashed=Eq. (8) | Defaults: η=1, n=100, m=1000, ν²=1, λ=1e-4, ‖θ*‖=1, c=ε=0.01. Panels: (UL) vary λ (t₂∝1/λ); (UR) vary n; (LL) vary m; (LR) vary ν² |
| **3** (4 panels) | Two-layer **random-features** ReLU net (train output layer only), learn **single ReLU neuron teacher** σ(⟨w*,x⟩), ‖w*‖=1 (non-realizable); train/test curves | d=100, η=1, defaults n=100, λ=1e-5, m=10000, ν²=1. Init a_j~N(0,1), w_j~N(0, ν²/(dm) I_d), w fixed. Panels: vary λ, n, m, ν² |
| **4** (4 panels) | Two-layer ReLU net, **both layers trained**, **zero teacher**; t₁,t₂ vs hyperparameters | η=1e-4, d=50, defaults n=50, m=1000, ν²=1, λ=0.05. Init a_j~N(0,1/m), w_j~N(0,ν²/d I_d) |
| **5** (App. B) | Surrogate "accuracy" P_x[(N−N*)² ≤ ε] vs steps — recovers a plateau. **Single simulation** | m=10000, d=100, n=100, η=1, λ=1e-5, ε=0.01 |
| **6** (App. B) | Ridge regression with **label noise** N(0, std²). Left: grokking with noise; Right: different noise levels barely change grokking time | Left: m=5000, d=50, n=100, η=1, std=0.1, λ=1e-5. Right: same but varied std (values UNSPECIFIED) |
| **7** (App. B) | Random Fourier features, realizable teacher = one random Fourier feature; "requires careful hyperparameter tuning" | m=5000, d=10, n=100, η=1, σ²=1, λ=1e-4. φ(x)=√(2/m) cos(Wx+b), rows of W ~ N(0,σ²I), b ~ U[0,2π] |

## 4. UNSPECIFIED / inconsistent in the text (do not invent)

1. **Sweep values** for every panel of Figs 2, 3, 4 (which λ, n, m, ν² values) — only in figure images. UNSPECIFIED.
2. **Number of seeds** for Figs 2–7 (only Fig 1 says 50; Fig 5 says single run). UNSPECIFIED. Whether Fig 2 points average over seeds or are single runs: UNSPECIFIED.
3. **Total number of GD steps / x-axis range** and **logging grid** for every figure. UNSPECIFIED.
4. **How θ\* is drawn** (direction random? fixed?) given ‖θ*‖=1. UNSPECIFIED.
5. **Test set size / how L(θ) is evaluated** (closed form vs Monte-Carlo). UNSPECIFIED. (Closed form is available for Gaussian features; for ReLU nets a sample is needed.)
6. **Input distribution D_x** for Figs 1-right, 3, 4 (d-dimensional inputs, Gaussian assumed?) — UNSPECIFIED; for Sec 5.1 the paper specifies features directly as φ~N(0,I/m) while also stating "d" in other figures (Fig 1 left lists no d).
7. **Feature map in Fig 1 Left**: "ridge regression" with m=10000; whether x∈R^m identity features — assumed from Sec 5.1 but not stated for Fig 1. UNSPECIFIED.
8. **Empirical t₁/t₂ procedure**: Sec 3 defines t₁ (last step with L_n ≥ ε) and t₂ (first step with L ≤ c), but the maximum step budget and how runs that never reach c are handled are UNSPECIFIED.
9. **Test-loss normalization:** train loss has a 1/2 factor, test loss does not, so "c = ε = 0.01" is compared across slightly different normalizations. Paper doesn't comment. (A real wrinkle for reproduction — see README later.)
10. **Teacher in Fig 3 (non-realizable):** w* direction / how many steps / label scaling UNSPECIFIED.
11. **Figure 4 / Fig 1-right init scale of `a`**: stated (a_j~N(0,1/m)); but the **output-scaling (e.g. 1/√m factor)** of the network, and bias terms, are UNSPECIFIED (formula N = Σ a_j σ(⟨w_j,x⟩), no bias).
12. **Fig 1 Right λ = 0.1** vs Sec 5.3 **default λ = 0.05** — different values, both stated; not an error, but sweeps in Fig 4 are centered at 0.05.
13. **Fig 3 caption** labels two panels "Left Lower" (typo; presumably Right Lower is ν²).
14. **Theorem constants** (Ω, O hide numerical constants in Eqs 3–5) — only the Gaussian specialization (Eq. 8) has explicit constants.
15. **Eq. 7 vs Eq. 8 consistency:** Eq. 7 has 4ηλ in the denominator; Eq. 8 uses 2.02ηλ (tighter ln(1−x) ≥ −1.01x applied to a (1−ηλ)^{2t}... the factor-2 bookkeeping is the paper's; I'll implement Eq. 8 as the "theory line" for Gaussian features, and Eq. 7 only as the generic version).
16. **Eq. 8 assumes c = ε and m ≥ ‖θ*‖²/ε**: with m=1000, ε=0.01 this is m ≥ 100, fine. For m=100 sweeps in "vary m", m must stay ≫ n for (m−n) > 0.

## 5. Things to verify numerically before trusting the reproduction
- Whether η = 1 is stable with φ~N(0,I/m) at the paper's n, m (check λ_max(ΦᵀΦ)/n vs 1/η − λ).
- That the t₂ ≥ Eq. 8 bound is actually respected by simulation (it's a lower bound; a violation would indicate a normalization mismatch, e.g. the 1/2 factor).
- That the t₁ ≤ Eq. 8 bound is respected, using the *realized* λ⁺_min(ΦᵀΦ) of each sampled dataset.

---

## 6. Revision 2 — findings from LaTeX source, figure legends, and numerical checks

### 6.1 Verification of formulas against TeX
Confirmed verbatim: Eq. 1 (L_n(θ;λ) = 1/(2n)Σ(...)² + (λ/2)‖θ‖²), Eq. 6 (t₁ bound), Eq. 7 (t₂ bound, 4ηλ), Thm 4.5 (log_{1−ηλ} form),
Remark A.14 / Eq. 8 (t₂ ≥ ln((m−n)ν²/(8mε))/(2.02ηλ); t₁ ≤ n·ln(14mν²/ε)/(2ηλ⁺_min(ΦᵀΦ))).
Notation check: the paper writes λ⁺_min(ΦᵀΦ) for the smallest positive eigenvalue of the **un-normalized** ΦᵀΦ (n·(1/n)): GD contraction per step is
1 − ηλ⁺_min(ΦᵀΦ)/n − ηλ, hence the factor n in the t₁ bound.

### 6.2 Feature normalization — RESOLVED
- Remark A.14 states explicitly: **φ(x_i) ~ N(0, (1/m) I_m)** (so b²≈3/2, λ_min(Σ)=1/m). Sec 5.1 says the same ("drawn i.i.d. from N(0, (1/m)I_m)").
- Your estimate is right for unit-variance features: measured λ_max(ΦᵀΦ)/n = 17.0 at (n,m)=(100,1000) ⇒ η=1 diverges (needs η < 2/17).
- With the paper's 1/m scaling it is 0.0172 ⇒ η=1 is extremely stable (needs < 2). Numerical results (one draw each, η=1):

| n | m | λ_max/n (N(0,1/m)) | λ⁺_min/n (N(0,1/m)) | λ_max/n if unit variance |
|---|---|---|---|---|
| 100 | 1000 | 0.0172 | 0.00487 | 17.0 ✗ |
| 10 | 1000 | 0.114 | 0.0837 | 116.7 ✗ |
| 160 | 1000 | 0.0121 | 0.00233 | 12.1 ✗ |
| 100 | 3000 | 0.0140 | 0.00675 | 41.2 ✗ |
| 100 | 10000 | 0.0120 | 0.00802 | 119.9 ✗ |

- **Decision:** use φ ~ N(0, I/m) exactly as the paper states; keep η = 1 (never change η). Every config will be checked with the
  stability condition η < 1/(λ + λ_max(ΦᵀΦ)/n) *before* running, and unstable configs are logged and skipped, not silently altered.
- Side effect: with this scaling, the per-step contraction on the data subspace is only ≈ ηλ⁺_min/n ≈ 5e-3 (n=100,m=1000), which is why t₁ ~ 10²–10³ steps.

### 6.3 Figure → file mapping and read-off sweep values (approximate where read from tick positions)
| Paper fig | File | Content |
|---|---|---|
| Fig 1 left | multitest_grokking_rf.png | ridge, zero teacher, 50 runs, mean ± band (std presumably; band type UNSPECIFIED) |
| Fig 1 right | multitest_grokking_relu.png | 2-layer ReLU, zero teacher, 50 runs |
| Fig 2 UL/UR/LL/LR | ridge_regularization / ridge_sample_size / ridge_width / ridge_initialization | t₁,t₂ empirical (solid, with error bars ⇒ **averaged over ≥2 runs; count UNSPECIFIED**) vs Eq. 8 (dashed) |
| Fig 3 | regularization / sample_size / width / initialization | random-features loss curves |
| Fig 4 | full_trained_{regularization,sample_size,width,initialization} | 2-layer ReLU, t₁,t₂ vs hyperparameter |
| Fig 5 | plateau_loss / plateau_accu | surrogate accuracy |
| Fig 6 | noise / noise_hyperparameter | label noise |
| Fig 7 | rff | random Fourier features |

**Fig 2 sweeps (read from axes, all other params at defaults n=100,m=1000,ν²=1,λ=1e-4,η=1):**
- λ: 5 values, powers of two **2⁻¹⁹ … 2⁻¹⁵** (note: *much* smaller than the 1e-4 default). Empirical t₂ ≈ 2^19.6 at 2⁻¹⁹ … 2^16.1 at 2⁻¹⁵; theory (dashed) ≈ 2^19.2 … 2^15.2. t₁ ≈ 2^8 flat (theory ≈ 2^9.2).
- n: 5 values, ≈ **10, 20, 40, 80, 160** (geometric; approximate from tick positions). t₁ rises 2^5→2^9.7, t₂ flat ≈ 2^14.5; theory t₁ 2^5.3→2^10.5.
- m: 5 values **1000, 1500, 2000, 2500, 3000**. t₁ ≈ 2^8 slowly decreasing, t₂ ≈ 2^14.3 flat.
- ν²: 7 values **10⁰ … 10⁶** (decades). t₂ emp. ≈ 2.3e4 → 9.2e4 (linear in log ν² ✓.), t₁ ≈ flat/slow.
- **Observations (eyeballed from the log₂ axes, so only good to ~±0.3 octave):** in every Fig 2 panel the empirical t₂ is above the dashed t₂ theory
  (bound valid; gap ≈ 0.4–1 octave, i.e. theory ≈ 0.3–0.75× measured) and empirical t₁ is below the dashed t₁ theory (bound valid; gap ≈ 1–1.3 octaves at defaults, ≈1.5 octaves at n=160).
  So the paper's "match" is slope agreement with a constant-factor gap, not equality.
- **My independent check of Eq. 8** at λ=2⁻¹⁹, n=100, m=1000, ν²=1, ε=0.01: ln(900/80)/(2.02·2⁻¹⁹) ≈ 6.3e5 ≈ 2^19.3 ✓ (matches the dashed line ≈2^19.2).
- Fig 2's t₂ at default λ=1e-4 would be ≈ ln(11.25)/(2.02e-4) ≈ 1.2e4 (matches the ν² panel's ν²=1 theory ≈1.2e4 ✓).

**Fig 3 (random features) legends:** λ ∈ {1e-4, 1e-5, 1e-6}; n ∈ {100, 200, 500}; m ∈ {1e3, 1e4, 1e5}; ν² ∈ {1, 5, 10}. x-axis 10⁰–10⁶, y = square loss, dashed = train, solid = test.
**Fig 4 (full-trained ReLU, t₁/t₂ vs hp):** λ: 6 values doubling from ≈0.05 to ≈1.6 (ticks 2⁻⁴…2⁰; approximate); n: ≈10…160 (5 values); m: 2000…4000 step 500; ν²: 7 values ≈ 2⁻³…2² (approximate).
  t₂ ∝ 1/λ visibly (2^17→2^12 over λ range), t₁ flat; t₁ increases with n, t₂ mildly decreases; t₂ ≈ linear in log ν².
**Fig 6 (label noise):** std ∈ {0.01, 0.05, 0.1}. **Fig 1** curves are drawn with only ≈7 log-spaced points (10⁰,10¹,…,10⁵,~1.5e5,…) joined by straight lines.

### 6.4 Remaining discrepancies / UNSPECIFIED after the revision
1. **Initial test loss in Fig 1 left (≈0.40 at step 1)** is *incompatible* with the stated setup: zero teacher, φ~N(0,I/m), ν²=1 gives
   L(θ(0)) = ‖θ(0)‖²/m ≈ 0.97 (measured 0.972, and 0.972 after one step), and train L_n ≈ 0.52 (paper ≈0.39). Possible causes (unverified): different input
   model (e.g. features through x ∈ R^d + a nonlinear map), different init, or averaging/plot artefacts. I will NOT tune to match; I'll report my initial values and flag the gap.
2. Fig 1 uses only ≈7 plotted points, so the paper's "plateau" is coarser than what a ~200-point log grid will show.
3. Still UNSPECIFIED: number of seeds for Fig 2–4 (error bars exist), what the Fig 1 band is (std vs min/max), exact test-set size, step budgets, how θ* is drawn, D_x for ReLU nets, bias terms, ReLU net output scaling, noise-level sweep beyond {0.01,0.05,0.1}.
4. No explicit "eliminated grokking" experiment appears in the figures I read. Paper's claim is theoretical: large λ (or tiny ν², large n) shrinks t₂−t₁. My Step 2 will test this myself and label it as my extension.
5. The theory's validity conditions (Eq. 3–5) are asymptotic (hidden constants) and are NOT satisfied literally at n=100, ε=0.01 (Eq. 3 would need n ≳ b⁴/ε² = 2.25e4 with ‖θ*‖=1). Eq. 8 is therefore a heuristic bound at those sizes; test (a) validity may fail for the *t₁* bound for small n in the sweeps — to be reported as found.

### 6.5 Implementation plan implied by the notes (Step 1)
- Because GD is linear, θ(t) = (I−ηA)^t θ(0) + … can be evaluated **exactly at arbitrary t via the eigendecomposition of ΦᵀΦ/n** (cheap for n ≪ m through the n×n Gram matrix); explicit iterative GD is used for the main figure runs and as a cross-check against the closed form on short horizons. Both reported. Float64 for the NumPy linear model.
- Test loss: exact closed form L(θ)=‖θ−θ*‖²/m (and also a 10k-sample Monte-Carlo estimate as requested), plus the ½-scaled variant logged alongside.
- Defaults per your instruction: 10 seeds, θ* ~ N(0,I) normalized to ‖θ*‖=1, 10k test samples, step budget ≥ 10× t₂ bound, ≈200 log-spaced log points.

### 6.6 Step 1 findings that touch the notes (added after running the code)
- **t₂ matches the paper quantitatively.** Realizable defaults (n=100,m=1000,ν²=1,λ=1e-4): measured t₂ ≈ 2.3e4 (paper's Fig 2 ν²=1 solid ≈ 2.3e4 ✓), Eq. 8 t₂ bound = 1.198e4 (paper dashed ≈ 1.2e4 ✓).
- **t₁ theory does NOT match the paper's dashed line**: my literal Eq. 8 gives ≈1.48e3 (λ⁺_min(ΦᵀΦ) ≈ 0.49, ln(14mν²/ε)=14.15, n=100) whereas the paper's dashed t₁ at the same setting reads ≈ 2^9.2 ≈ 5.9e2
  (factor ≈2.1–2.5 smaller; the same factor appears at n=10 and n=160). Measured t₁ ≈ 2.2e2 agrees with the paper's solid line (≈2^8). I could not find which convention gives their line
  (no code released); possibilities: a different constant/normalisation of λ⁺_min in their plotting code. Reported as a discrepancy; my bound is still valid (larger).
- **Initial losses differ from Fig 1 left**: I get train ≈ 0.49, test ≈ 1.00 at step 1 (zero teacher, ν²=1), the paper's plot starts at ≈0.39 / ≈0.40. Unresolved; one hypothesis (unverified) is that the paper's test curve also carries a ½ factor and uses a slightly different init scale.
- **Monotonicity (your sanity check):** the *regularised* objective L_n(θ;λ) decreases monotonically (tested in `tests/`, η<1/(λ+λ_max/n)). The *unregularised* train loss L_n is **not** monotone for λ>0 in 4/10 realizable seeds
  (it creeps up slightly late in training as weight decay pulls θ toward the ridge solution, whose train loss is >0); it is monotone in all 10 zero-teacher seeds and in the λ=0 test. This is expected, not a bug.
