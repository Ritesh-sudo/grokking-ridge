<!--STEP1-->
| config | measured t1 (mean±std) | Eq.8 t1 bound | bound/measured | measured t2 | Eq.8 t2 bound | bound/measured | gap t2−t1 measured | gap lower bound (t2_b−t1_b) | valid (t1≤b, t2≥b) |
|---|---|---|---|---|---|---|---|---|---|
| fig1_realizable | 218±13.1 | 1.48e+03 | 6.81 | 2.3e+04±221 | 1.2e+04 | 0.521 | 2.28e+04 | 1.05e+04 | True (10/10 seeds) |
| fig1_left_zero_teacher | 194±7.96 | 1.01e+03 | 5.2 | 2.3e+05±589 | 1.25e+05 | 0.542 | 2.3e+05 | 1.24e+05 | True (50/50 seeds) |
<!--/STEP1-->

<!--VALIDITY-->
| family | tag | bound | measured runs | bound holds | median bound/measured | bound/measured range |
|---|---|---|---|---|---|---|
| F1_realizable_paper | in_regime=False | t1 | 480 | 480/480 | 6.69 | 2.01 – 1.25e+03 |
| F1_realizable_paper | in_regime=False | t2 | 420 | 420/420 | 0.52 | 0.0494 – 0.877 |
| F1_realizable_paper | in_regime_eq345=False | t1 | 480 | 480/480 | 6.69 | 2.01 – 1.25e+03 |
| F1_realizable_paper | in_regime_eq345=False | t2 | 420 | 420/420 | 0.52 | 0.0494 – 0.877 |
| F2_realizable_inregime | in_regime=True | t1 | 430 | 430/430 | 5.81 | 2.33 – 919 |
| F2_realizable_inregime | in_regime=False | t1 | 40 | 40/40 | 211 | 3.35 – 1.09e+03 |
| F2_realizable_inregime | in_regime=True | t2 | 380 | 380/380 | 0.592 | 0.051 – 0.851 |
| F2_realizable_inregime | in_regime=False | t2 | 20 | 20/20 | 0.599 | 0.585 – 0.608 |
| F2_realizable_inregime | in_regime_eq345=True | t1 | 440 | 440/440 | 5.81 | 2.33 – 919 |
| F2_realizable_inregime | in_regime_eq345=False | t1 | 30 | 30/30 | 4.29 | 3.35 – 1.09e+03 |
| F2_realizable_inregime | in_regime_eq345=True | t2 | 380 | 380/380 | 0.592 | 0.051 – 0.851 |
| F2_realizable_inregime | in_regime_eq345=False | t2 | 20 | 20/20 | 0.599 | 0.585 – 0.608 |
| F3_zero_teacher | in_regime=True | t1 | 20 | 20/20 | 7.06 | 6.54 – 7.55 |
| F3_zero_teacher | in_regime=False | t1 | 460 | 460/460 | 7.08 | 2.01 – 1.24e+03 |
| F3_zero_teacher | in_regime=True | t2 | 20 | 20/20 | 0.535 | 0.523 – 0.548 |
| F3_zero_teacher | in_regime=False | t2 | 400 | 400/400 | 0.535 | 0.0511 – 0.884 |
| F3_zero_teacher | in_regime_eq345=True | t1 | 480 | 480/480 | 7.07 | 2.01 – 1.24e+03 |
| F3_zero_teacher | in_regime_eq345=True | t2 | 420 | 420/420 | 0.535 | 0.0511 – 0.884 |

Overall: 2670/2670 (run, bound) checks satisfy the bound.
<!--/VALIDITY-->

<!--SLOPES-->
| family | quantity | fitted (mean over seeds) | 95% CI (t, 10 seeds) | theory expects |
|---|---|---|---|---|
| F1_realizable_paper | log-log slope, t2 vs 1/(ηλ), λ-sweep (ηλ≤1e-3) | 1.0000 | [1.0000, 1.0000] | 1 |
| F1_realizable_paper | log-log slope, t2 vs 1/(ηλ), η-sweep | 1.0001 | [1.0000, 1.0001] | 1 |
| F1_realizable_paper | log-log slope, t1 vs 1/(ηλ), λ-sweep | 0.0151 | [0.0145, 0.0157] | 0 (λ has no effect on the t1 bound) |
| F1_realizable_paper | log-log slope, t1 vs 1/(ηλ), η-sweep | 1.0128 | [1.0105, 1.0151] | 1 (t1 ∝ 1/η) |
| F1_realizable_paper | log-log slope, t1 vs n | 1.0630 | [1.0483, 1.0778] | ≥1 (n/λ⁺min(n)) |
| F1_realizable_paper | log-log slope, t2 vs n | -0.0890 | [-0.0922, -0.0858] | ≈0 |
| F1_realizable_paper | t2 slope vs ln ν² / Eq.8 slope 1/(2.02ηλ) | 1.0099 | [1.0099, 1.0100] | ≈1 |
| F1_realizable_paper | t1 slope vs ln ν² / Eq.8 slope n/(2ηλ⁺min) | 0.8213 | [0.7969, 0.8457] | ≈1 |
| F1_realizable_paper | t1_shift_m1000_to_m10000 (steps; measured vs Eq.8-predicted) | -24.8 vs -474.6 | – | same sign & size |
| F1_realizable_paper | t2_shift_m1000_to_m10000 (steps; measured vs Eq.8-predicted) | 10.6 vs 471.8 | – | same sign & size |
| F2_realizable_inregime | log-log slope, t2 vs 1/(ηλ), λ-sweep (ηλ≤1e-3) | 1.0000 | [1.0000, 1.0000] | 1 |
| F2_realizable_inregime | log-log slope, t2 vs 1/(ηλ), η-sweep | 1.0000 | [1.0000, 1.0000] | 1 |
| F2_realizable_inregime | log-log slope, t1 vs 1/(ηλ), λ-sweep | 0.0257 | [0.0248, 0.0265] | 0 (λ has no effect on the t1 bound) |
| F2_realizable_inregime | log-log slope, t1 vs 1/(ηλ), η-sweep | 1.0014 | [1.0012, 1.0015] | 1 (t1 ∝ 1/η) |
| F2_realizable_inregime | log-log slope, t1 vs n | 1.0968 | [1.0845, 1.1091] | ≥1 (n/λ⁺min(n)) |
| F2_realizable_inregime | log-log slope, t2 vs n | -0.0745 | [-0.0769, -0.0722] | ≈0 |
| F2_realizable_inregime | t2 slope vs ln ν² / Eq.8 slope 1/(2.02ηλ) | 1.0100 | [1.0100, 1.0100] | ≈1 |
| F2_realizable_inregime | t1 slope vs ln ν² / Eq.8 slope n/(2ηλ⁺min) | 0.7442 | [0.7208, 0.7676] | ≈1 |
| F2_realizable_inregime | t1_shift_m1000_to_m10000 (steps; measured vs Eq.8-predicted) | -71.7 vs -1009.5 | – | same sign & size |
| F2_realizable_inregime | t2_shift_m1000_to_m10000 (steps; measured vs Eq.8-predicted) | 838.0 vs 943.7 | – | same sign & size |
| F3_zero_teacher | log-log slope, t2 vs 1/(ηλ), λ-sweep (ηλ≤1e-3) | 1.0000 | [1.0000, 1.0000] | 1 |
| F3_zero_teacher | log-log slope, t2 vs 1/(ηλ), η-sweep | 1.0001 | [1.0001, 1.0001] | 1 |
| F3_zero_teacher | log-log slope, t1 vs 1/(ηλ), λ-sweep | 0.0145 | [0.0139, 0.0151] | 0 (λ has no effect on the t1 bound) |
| F3_zero_teacher | log-log slope, t1 vs 1/(ηλ), η-sweep | 1.0144 | [1.0122, 1.0165] | 1 (t1 ∝ 1/η) |
| F3_zero_teacher | log-log slope, t1 vs n | 1.0866 | [1.0672, 1.1060] | ≥1 (n/λ⁺min(n)) |
| F3_zero_teacher | log-log slope, t2 vs n | -0.0875 | [-0.0909, -0.0841] | ≈0 |
| F3_zero_teacher | t2 slope vs ln ν² / Eq.8 slope 1/(2.02ηλ) | 1.0099 | [1.0099, 1.0100] | ≈1 |
| F3_zero_teacher | t1 slope vs ln ν² / Eq.8 slope n/(2ηλ⁺min) | 0.7975 | [0.7707, 0.8243] | ≈1 |
| F3_zero_teacher | t1_shift_m1000_to_m10000 (steps; measured vs Eq.8-predicted) | -16.0 vs -474.6 | – | same sign & size |
| F3_zero_teacher | t2_shift_m1000_to_m10000 (steps; measured vs Eq.8-predicted) | 565.4 vs 471.8 | – | same sign & size |
<!--/SLOPES-->

<!--DIAG-->
| rank | λ⁺min | ε factor | ‖θ(0)‖² used | b² | RMS log2 error vs paper's read-off line | mean offset (octaves) | shape-only RMS (octaves) |
|---|---|---|---|---|---|---|---|
| 1 | emp | 1 | nu2 (init var nu2/m) | 1.0 | 0.09 | -0.02 | 0.09 |
| 2 | emp | 1 | nu2 (init var nu2/m) | 1.5 | 0.11 | +0.07 | 0.09 |
| 3 | emp | 2 | nu2 (init var nu2/m) | 1.5 | 0.13 | -0.09 | 0.09 |
| 4 | emp | 2 | nu2 (init var nu2/m) | 1.0 | 0.21 | -0.19 | 0.09 |
| 5 | 1 | 2 | E=m*nu2 | 1.0 | 0.42 | +0.22 | 0.35 |
| 6 | 1 | 2 | emp_||th0||^2 | 1.0 | 0.42 | +0.22 | 0.35 |
| 23 (**literal Eq. 8**) | emp | 1 | E=m·ν² | 1.5 | 1.11 | +1.11 | 0.12 |
| 11 (λ⁺min=1 approx., otherwise literal) | 1 | 1 | E=m·ν² | 1.5 | 0.49 | +0.34 | 0.35 |
<!--/DIAG-->

<!--AMPLIFY-->
| family | sweep | value | t1 | t2 | median t2/t1 | t2 not reached (of 10) |
|---|---|---|---|---|---|---|
| F1_realizable_paper | lam | 1.91e-06 | 221 | 1.21e+06 | 5.5e+03 | 0 |
| F1_realizable_paper | lam | 0.125 | 13.4 | 18 | 1.34 | 0 |
| F1_realizable_paper | lam | 1 | 0 | 1 | ∞/undefined | 0 |
| F1_realizable_paper | nu2 | 0.01 | nan | 126 | ∞/undefined | 0 |
| F1_realizable_paper | nu2 | 1e+06 | 1.4e+03 | 9.21e+04 | 66.1 | 0 |
| F1_realizable_paper | n | 10 | 18.6 | 2.36e+04 | 1.25e+03 | 0 |
| F1_realizable_paper | n | 900 | 2.02e+03 | 1.4e+04 | 6.99 | 0 |
| F2_realizable_inregime | lam | 1.91e-06 | 538 | 2.73e+06 | 5.09e+03 | 0 |
| F2_realizable_inregime | lam | 0.25 | 16.4 | 20.2 | 1.21 | 0 |
| F2_realizable_inregime | lam | 1 | 3 | 4 | 1.33 | 0 |
| F2_realizable_inregime | nu2 | 2 | 73.9 | 1.29e+04 | 187 | 0 |
| F2_realizable_inregime | nu2 | 1e+06 | 2.07e+03 | 1.44e+05 | 69.7 | 0 |
| F2_realizable_inregime | n | 10 | 45.8 | 5.32e+04 | 1.16e+03 | 0 |
| F2_realizable_inregime | n | 900 | 5.76e+03 | 3.41e+04 | 5.96 | 0 |
| F3_zero_teacher | lam | 1.91e-06 | 211 | 1.18e+06 | 5.53e+03 | 0 |
| F3_zero_teacher | lam | 0.25 | 6 | 8.2 | 1.33 | 0 |
| F3_zero_teacher | lam | 1 | 0 | 1 | ∞/undefined | 0 |
| F3_zero_teacher | nu2 | 0.01 | nan | 9.5 | ∞/undefined | 0 |
| F3_zero_teacher | nu2 | 1e+06 | 1.35e+03 | 9.15e+04 | 67.5 | 0 |
| F3_zero_teacher | n | 10 | 15.7 | 2.31e+04 | 1.44e+03 | 0 |
| F3_zero_teacher | n | 900 | 2e+03 | 1.36e+04 | 6.76 | 0 |
<!--/AMPLIFY-->

<!--LATE-->
Across the 10 seeds of the realizable default config the largest rise of L_n above its running minimum is 0.0e+00, 0.0e+00, 4.0e-04, 1.4e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 6.7e-03, 1.2e-02 (relative); 4/10 seeds rise at all. In seed 2: min L_n = 5.6127e-08 at t≈2024, final L_n = 5.6150e-08 = ridge-solution train loss 5.6150e-08; ‖θ‖² goes 982 → 0.1136 (ridge 0.1136 vs interpolant 0.1159). Final test loss 8.841e-04 vs null predictor (θ=0) 1.0e-03: ratio 0.884.
<!--/LATE-->