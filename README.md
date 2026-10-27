# Grokking in ridge regression and two-layer networks

Experiments on grokking (train loss collapses early, test loss follows much later) in over-parameterised ridge regression and two-layer ReLU networks, following Xu, Vardi & Safran, [*To Grok Grokking*](https://arxiv.org/abs/2601.19791) (ICML 2026). NumPy / PyTorch, CPU only. Full tables and caveats: [`RESULTS.md`](RESULTS.md).

## Key insights

- **Grokking comes from weight decay.** Train loss reaches ε=0.01 within a few hundred steps, while test loss only drops once weight decay shrinks the initialisation noise, at a time t2 ∝ 1/(ηλ). The delay t2 ∝ 1/λ holds essentially exactly; the overfitting time t1 does not depend on λ.
- **What amplifies it:** small λ, small sample size n, large initialisation scale ν². The width m has little effect on t2.
- **The theory's bounds hold.** The paper's Eq. 8 bounds on t1 and t2 were satisfied in all 2670 (run, bound) checks, including settings outside the theorem's stated regime. The t2 bound is within ≈2× of the truth; the t1 bound is ≈5–7× loose.
- **Which model matches which figure matters.** The paper's Figs. 1-left, 3, 5 and 6 are matched by *random ReLU features with a zero teacher*, not by the plain Gaussian-feature ridge model (which matches Fig. 2). A single-ReLU teacher, as the text describes for Fig. 3, leaves a test-loss floor of ≈0.2 and no t2.
- **It carries over to a fully trained network.** A two-layer ReLU net (both layers trained) shows the same pattern: t1 ≈ 2×10³, t2 ≈ 7.7×10⁴ steps over 50 runs.

## Selected results

Random ReLU features, zero teacher, 50 runs (black markers = values read off the paper's figure):

![Fig. 1 left](results/nl_fig1_left_rf.png)

Two-layer ReLU net, both layers trained, 50 runs:

![Fig. 1 right](results/nl_fig1_right.png)

Random features, varying λ / n / m / ν² (t2 = 1.5×10⁴ / 1.6×10⁵ / 1.6×10⁶ for λ = 10⁻⁴ / 10⁻⁵ / 10⁻⁶):

![Fig. 3](results/nl_fig3_zero_teacher_paper_grids.png)

Ridge regression sweeps: measured t1, t2 (solid) vs the Eq. 8 bounds (dashed):

![Ridge sweeps](results/sweep_F1_realizable_paper.png)

## How it compares with the paper

Paper values were read off its figures by eye (≈ ±10–15 %), so agreement means similar shape and rough size, not exact numbers. Differences that remain: the Fig. 7 test-loss drop comes ≈10× earlier than in the paper; the paper's dashed t1 line is ≈2.5× below the literal Eq. 8; train loss in Fig. 1 starts lower than the paper's. Several settings are my own choices (input distribution, `d` for Fig. 1, sweep grids, seed counts for most figures). Details in [`RESULTS.md`](RESULTS.md).

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m src.sweep --all                      # ridge experiments, sweeps, bound checks
python -m src.nonlinear --fig1l-rf --seeds 50  # random features, zero teacher (Fig. 1 left)
python -m src.nonlinear --fig3p --seeds 5      # random-features sweeps (Fig. 3)
python -m src.nonlinear --full --seeds 50 --T-full 600000   # two-layer net (Fig. 1 right, slow)
python -m src.appendix                         # Figs. 5-7
```

Code is in `src/` (`train.py` has an exact spectral GD engine checked against explicit GD in `tests/`), configs in `configs/`, outputs in `results/`.
