import numpy as np
from collections import defaultdict
from scipy.special import logsumexp as scipy_logsumexp


def logsumexp_stable(a):
    """Numerically stable log-sum-exp."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = np.max(a)
    if np.isinf(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(a - m)))


def energy_sigma(sigma, J, h=None):
    """Compute Ising energy: E(σ) = -0.5 * σ^T J σ - h^T σ."""
    sigma = np.array(sigma, dtype=np.float64)
    E = -0.5 * sigma @ J @ sigma
    if h is not None:
        E -= np.dot(h, sigma)
    return E


def npll(samples, J, h=None):
    """Negative pseudo-log-likelihood (NPLL) for an Ising model.

    For each sample σ^(s), uses:
      log p(σ_i | σ_~i) = σ_i * (h_i + Σ_j J_ij σ_j) - log(2 cosh(h_i + Σ_j J_ij σ_j))
    """
    samples_arr = np.asarray(samples, dtype=np.float64)
    if samples_arr.ndim == 1:
        samples_arr = samples_arr[None, :]

    N = samples_arr.shape[1]
    h_vec = np.zeros(N, dtype=np.float64) if h is None else np.asarray(h, dtype=np.float64)

    # field_{s,i} = h_i + Σ_j J_ij σ_j^(s)
    fields = h_vec + samples_arr @ np.asarray(J, dtype=np.float64).T

    # log(2 cosh(x)) = logaddexp(x, -x)
    log_norm = np.logaddexp(fields, -fields)
    log_cond = samples_arr * fields - log_norm
    return float(-np.mean(np.sum(log_cond, axis=1)))


def sample_prior_path(p_prior_step, N, rng):
    """Sample a sequential path σ_1..σ_N from a step-wise prior."""
    s = []
    path = [tuple(s)]
    for _ in range(N):
        probs = p_prior_step(s)
        vals = list(probs.keys())
        ps = np.array(list(probs.values()), dtype=np.float64)
        ps /= ps.sum()
        s.append(vals[rng.choice(len(vals), p=ps)])
        path.append(tuple(s))
    return path


def estimate_empirical_prior(paths):
    """Estimate step-wise prior transition probabilities from sampled paths."""
    counts = defaultdict(lambda: defaultdict(int))
    for path in paths:
        for t in range(len(path) - 1):
            counts[path[t]][path[t + 1][-1]] += 1
    return {
        s_t: {k: v / sum(d.values()) for k, v in d.items()}
        for s_t, d in counts.items()
    }


def compute_metrics(samples, probs_true, all_sigmas, N, J=None, h=None):
    samples_arr = np.array(samples)
    m_ref = np.array([np.sum(p * s[i] for p, s in zip(probs_true, all_sigmas)) for i in range(N)])
    c_ref = np.array(
        [[np.sum(p * s[i] * s[j] for p, s in zip(probs_true, all_sigmas)) for j in range(N)] for i in range(N)]
    )
    m_est = np.mean(samples_arr, axis=0)
    c_est = np.mean(samples_arr[:, :, None] * samples_arr[:, None, :], axis=0)
    delta1 = np.sum(np.abs(m_est - m_ref)) / N
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)
    delta2 = np.sum(np.abs(c_est[mask] - c_ref[mask])) / (N * (N - 1) / 2)
    sigma_to_idx = {tuple(s): i for i, s in enumerate(all_sigmas)}
    logp_true = np.array([np.log(p + 1e-15) for p in probs_true])
    ce = -np.mean([logp_true[sigma_to_idx[tuple(s)]] for s in samples])
    npll_val = None if J is None else npll(samples_arr, J, h=h)
    return delta1, delta2, ce, npll_val


def compute_metrics_bp(samples, J, h, N, n_ref_samples=20000, rng=None):
    """Reference via Gibbs sampling; returns (Δ1, Δ2, CE-proxy, NPLL)."""
    if rng is None:
        rng = np.random.default_rng(42)

    sigma = rng.choice([-1, 1], size=N)
    for _ in range(5000):
        for i in rng.permutation(N):
            field = h[i] + np.sum(J[i, :] * sigma)
            sigma[i] = 1 if rng.random() < 1 / (1 + np.exp(-2 * field)) else -1

    ref_samples = []
    for _ in range(n_ref_samples):
        for i in rng.permutation(N):
            field = h[i] + np.sum(J[i, :] * sigma)
            sigma[i] = 1 if rng.random() < 1 / (1 + np.exp(-2 * field)) else -1
        ref_samples.append(tuple(sigma))

    ref_arr = np.array(ref_samples)
    m_ref = np.mean(ref_arr, axis=0)
    c_ref = np.mean(ref_arr[:, :, None] * ref_arr[:, None, :], axis=0)

    samples_arr = np.array(samples)
    m_est = np.mean(samples_arr, axis=0)
    c_est = np.mean(samples_arr[:, :, None] * samples_arr[:, None, :], axis=0)

    delta1 = np.sum(np.abs(m_est - m_ref)) / N
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)
    delta2 = np.sum(np.abs(c_est[mask] - c_ref[mask])) / (N * (N - 1) / 2)

    energies_df = np.array([energy_sigma(s, J, h) for s in samples])
    energies_ref = np.array([energy_sigma(s, J, h) for s in ref_samples])
    logp_ref = -energies_ref - scipy_logsumexp(-energies_ref)

    # np.interp requires ascending x
    x_ref = -energies_ref
    order = np.argsort(x_ref)
    x_ref_sorted = x_ref[order]
    logp_ref_sorted = logp_ref[order]

    ce_proxy = -np.mean([np.interp(-e, x_ref_sorted, logp_ref_sorted) for e in energies_df])
    npll_val = npll(samples_arr, J, h=h)
    return delta1, delta2, ce_proxy, npll_val

