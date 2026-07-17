# DecisionFlow — clean reference implementation

Exact sequential sampling from an Ising Gibbs distribution
`p(σ) ∝ exp(-E(σ))` via the Decision Flow (DF) correction of a proposal
prior (paper: *Decision Flow*, Behjoo, Chertkov & Ahn).

This package consolidates and supersedes the legacy scripts
(`alg1_sample.py`, `alg1_sweep.py`, `alg1_sweep_with_lbp.py`,
`analytic_df*.py`, `df_*.py`, `*_ising.py`, `priors.py`). See
`MIGRATION.md` for the old → new map.

## Layout

```
decisionflow/
    ising.py     IsingModel (grid / glassy grid / dense) + exact 2^N enumeration
    solvers.py   naive mean-field and loopy BP singleton marginals q_i(σ_i)
    priors.py    ProductPrior (PLP, Eqs. 20–22) and LocalBoltzmannPrior (LBP)
    df.py        exact_sample (ancestral / K→∞ limit) and alg1_sample (Alg. 1)
    mcmc.py      single-spin-flip Metropolis baseline (sweep units)
    metrics.py   Δ1, Δ2 (relative, paper Sec. 5), KL proxy (Eq. 23), NPLL (Eqs. 24–25)
experiments/
    run_sk_sweep.py   the S = K accuracy–effort sweep (2×2 figure + CSV)
tests/
    test_df.py        9 deterministic tests vs exact enumeration
```

## Quickstart

```python
import numpy as np
from decisionflow import (IsingModel, ProductPrior, LocalBoltzmannPrior,
                          exact_sample, alg1_sample, metrics)

model = IsingModel.grid(3, 3, seed=42)     # or glassy_grid / random_dense
enum  = model.enumerate()                  # exact references (small N)

prior = ProductPrior.from_bp(model)        # uniform / from_mean_field / from_bp
# prior = LocalBoltzmannPrior(model, order=list(range(model.N)))

K = S = 20000
paths, log_w = prior.sample(K, np.random.default_rng(0))
samples, info = alg1_sample(paths, log_w, S, np.random.default_rng(1),
                            order=prior.order if prior.requires_fixed_order else None,
                            enum=enum)
print(metrics.evaluate(samples, model, enum))   # delta1, delta2, kl, npll
```

Run the tests and the sweep:

```bash
python tests/test_df.py
python experiments/run_sk_sweep.py
```

## Theory ↔ code map

| Paper | Code |
|---|---|
| Energy / target, Eqs. (1), (19) | `IsingModel.energy`, `Enumeration.probs` |
| Theorem 1, exact correction Eq. (4) | `exact_sample` (ancestral evaluation of the same weights) |
| PLP prior, Eqs. (20)–(22) | `ProductPrior` (π_T = ∏ q_a; prior factors cancel in Eq. (4)) |
| Algorithm 1 (K-path empirical) | `Prior.sample` + `alg1_sample` |
| LBP prior & weight identity | `LocalBoltzmannPrior` (log w = Σ_t log 2 cosh h_loc) |
| Δ1, Δ2 (Sec. 5) | `metrics.delta1`, `metrics.delta2` |
| KL proxy, Eq. (23) | `metrics.kl_proxy` |
| NPLL, Eqs. (24)–(25) | `metrics.npll` |

## Key facts that must be respected

1. **Importance weights are `log w = -E(σ) - log π_T(σ)`** — mind the sign
   of the energy (the legacy `generate_prior_paths` had it flipped; see the
   review report).
2. **LBP requires a fixed node order shared by prior and sampler.** The
   weight identity w(σ) = ∏ 2 cosh(h_loc) is only valid along the
   generation order; `alg1_sample(..., order=prior.order)` enforces it.
3. For PLP priors the exact DF limit is **prior-independent** (q cancels in
   Eq. (4), Appendix B/C); the proposal only sets the finite-K variance.
   Measured ESS/K: LBP 0.88/0.77/0.59, BP ≈ uniform ≈ 0.09/0.005-0.02,
   MF ≈ 3·10⁻⁴–6·10⁻⁴ on the 3×3/4×4/5×5 instances — naive MF marginals
   are overconfident and make the worst proposal, and all product priors
   degrade with N while LBP holds up.
4. `alg1_sample` aggregates duplicate paths (exact identity) and tracks the
   residual weight (one gemv per node-step): ~200× faster than the legacy
   vectorized sampler.
5. If no active prior path survives conditioning (IS starvation), the
   sampler falls back to exact child weights (`enum`, small N), or to the
   local Gibbs conditional (`on_starvation="local_gibbs"`, any N; biased at
   finite K, reported via `info["n_starved"]`), or raises
   (`on_starvation="raise"`). Empirically starvation essentially never
   happens in the swept regimes (size-biased prefix steering).

## System-size roadmap

| regime | reference | fallback | status |
|---|---|---|---|
| N ≤ 16 (≤ 4×4) | `Enumeration` | exact | default; full test suite |
| N ≤ 26 (≤ 5×5) | `ExactReference` (bit-decoded, ~1.2 GB at N=25) | local-Gibbs | verified; 5×5 sweep in `experiments/run_5x5.py` |
| N > 26 | none (MCMC / self-consistency) | local-Gibbs / raise | sampler + NPLL work as-is; needs NN-amortized Green functions (paper Sec. 6) for honest large-N claims |
