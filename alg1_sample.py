import numpy as np
from itertools import product
from scipy.special import logsumexp as scipy_logsumexp

np.seterr(divide='ignore', invalid='ignore')

from bp_ising import BPIsingSolver
from mf_ising import MFIsingSolver
from priors import prior_uniform_step, prior_bp_step_simple, prior_mf_step_simple
from ising_utils import energy_sigma, sample_prior_path, estimate_empirical_prior, compute_metrics
from df_exact import compute_pi_T_and_G, sample_df_alg1_exact

# ==========================================
# Main Experiment
# ==========================================
def run_experiment(N=9, K=2000, S=2000, seed=42):
    rng = np.random.default_rng(seed)

    # Random symmetric Ising coupling matrix
    J = rng.normal(0, 0.3, size=(N, N))
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0)
    h = np.zeros(N)

    assigned_order = list(range(N))
    bp_solver = BPIsingSolver(J, h)
    mf_solver = MFIsingSolver(J, h)

    all_sigmas = np.array(list(product([-1, 1], repeat=N)))
    logw = -np.array([energy_sigma(s, J, h) for s in all_sigmas])  # log(unnormalized target weights)

    # Exact target distribution
    probs_true = np.exp(logw - scipy_logsumexp(logw))

    # Define priors
    priors = {
        "uniform": lambda s: prior_uniform_step(s),
        "mf": lambda s: prior_mf_step_simple(s, mf_solver, assigned_order),
        "bp": lambda s: prior_bp_step_simple(s, bp_solver, assigned_order),
    }

    results = {}

    for name, prior_func in priors.items():
        print(f"\n--- Running DF with '{name}' prior ---")
        
        # Step 1: Generate prior paths
        paths = [sample_prior_path(prior_func, N, rng) for _ in range(K)]
        
        # Step 2: Estimate empirical prior
        p_hat = estimate_empirical_prior(paths)
        
        # Step 3: Compute π_T and Green's functions (exact from theoretical prior)
        pi_T, G = compute_pi_T_and_G(prior_func, N)
        
        # Step 4: Generate DF-corrected samples
        samples = [sample_df_alg1_exact(p_hat, prior_func, pi_T, G, all_sigmas, logw, N, rng) for _ in range(S)]
        
        # Evaluate
        d1, d2, kl, npll_val = compute_metrics(samples, probs_true, all_sigmas, N, J=J, h=h)
        results[name] = {"Δ1": d1, "Δ2": d2, "Cross-Entropy": kl, "NPLL": npll_val}
        print(f"Δ1: {d1:.4f} | Δ2: {d2:.4f} | Cross-Entropy: {kl:.4f} | NPLL: {npll_val:.4f}")

    return results

if __name__ == "__main__":
    res = run_experiment(N=9, K=10000, S=10000)
    print("\n=== Final Results ===")
    for k, v in res.items():
        print(f"{k}: {v}")
