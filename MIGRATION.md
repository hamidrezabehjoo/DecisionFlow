# Migration map: legacy files → clean package

| Legacy file | Fate | Replacement |
|---|---|---|
| `alg1_sweep.py` | superseded | `experiments/run_sk_sweep.py` (+ `decisionflow/*`) |
| `alg1_sweep_with_lbp.py` | superseded | `experiments/run_sk_sweep.py` |
| `alg1_sample.py` | dropped | stack-A experiment; approach superseded by direct IS (`alg1_sample`) |
| `analytic_df.py`, `analytic_df_sweep.py` | dropped | O(4^N) Green-function dicts; `exact_sample` covers small-N exactly |
| `df_exact.py` | dropped | `exact_sample` / `alg1_sample` |
| `df_bp.py` | dropped | biased IS (completion order ≠ proposal scoring order); BP now only builds the PLP prior via `bp_marginals` |
| `bp_ising.py` | consolidated | `solvers.bp_marginals` (mind the spin order: legacy beliefs were [P(+1), P(−1)], package q is [P(−1), P(+1)]) |
| `mf_ising.py` | consolidated | `solvers.mean_field_marginals` |
| `lbp_ising.py` | consolidated | `priors.LocalBoltzmannPrior` (fixed-order mode; the random-order A₀ subset-DP mode is dropped — same sign bug in its fallback weights and no sampler support) |
| `mh_ising.py` | consolidated | `mcmc.mh_sample` (thinning now in sweep units, matching the legacy `sample_mcmc`) |
| `ising_utils.py` | consolidated | `ising.py` + `metrics.py` (`compute_metrics_bp` interp hack dropped) |
| `priors.py` | consolidated | `priors.ProductPrior` / `priors.LocalBoltzmannPrior` |

## Behavioural differences to be aware of

1. **Corrected importance-weight sign.** Legacy `generate_prior_paths`
   returned `log w = +E − log π_T` (see review report §1). All PLP-prior
   Algorithm 1 results produced with it sampled the inverted-temperature
   model. The package returns the correct `-E − log π_T`.
2. **Metrics are the paper's relative Δ1/Δ2** (normalized by |m_ref|,
   |c_ref|) everywhere. The legacy stack A used absolute errors in
   `ising_utils.compute_metrics`.
3. **Spin column convention** in q arrays is unified: column 0 = P(σ=−1),
   column 1 = P(σ=+1).
4. **MCMC burn-in/thin are in sweeps** (N single-spin flips), not single
   flips as in legacy `mh_ising.py`.
5. Legacy `alg1_sample.py` mixed an empirical per-prefix prior estimate
   p̂ (a point mass — K paths over 2^t prefixes means each prefix is seen
   at most once) with theoretical G/π_T. The clean implementation uses the
   IS form of the same correction, which needs no per-prefix transition
   estimates at all.
