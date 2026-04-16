import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from ising_utils import logsumexp_stable


def compute_pi_T_and_G(p_prior_func, N):
    pi_prior = [{} for _ in range(N + 1)]
    pi_prior[0][()] = 1.0
    for t in range(N):
        for s_t, prob in pi_prior[t].items():
            if prob < 1e-15:
                continue
            for val, p_val in p_prior_func(list(s_t)).items():
                s_next = s_t + (val,)
                pi_prior[t + 1][s_next] = pi_prior[t + 1].get(s_next, 0.0) + prob * p_val
    pi_T = pi_prior[N]

    G = [{} for _ in range(N + 1)]
    for s_T in pi_T:
        G[N][s_T] = {s_T: 1.0}
    for t in range(N - 1, -1, -1):
        G[t] = {}
        for s_t in pi_prior[t]:
            trans = p_prior_func(list(s_t))
            G[t][s_t] = {}
            for s_T in pi_T:
                g_val = sum(
                    p_val * G[t + 1].get(s_t + (val,), {}).get(s_T, 0.0)
                    for val, p_val in trans.items()
                )
                if g_val > 1e-15:
                    G[t][s_t][s_T] = g_val
    return pi_T, G


def sample_df_alg1_exact(p_hat, p_prior_func, pi_T, G, all_sigmas, logw, N, rng):
    s = []
    for t in range(N):
        unnorm_log_probs = []
        for val in [+1, -1]:
            s_new = tuple(s + [val])
            mask = np.ones(len(all_sigmas), dtype=bool)
            for i, v in enumerate(s_new):
                mask &= all_sigmas[:, i] == v
            if not np.any(mask):
                unnorm_log_probs.append(-np.inf)
                continue

            terms = [
                logw[idx]
                + np.log(G[t + 1].get(s_new, {}).get(tuple(all_sigmas[idx]), 0.0) + 1e-15)
                - np.log(pi_T.get(tuple(all_sigmas[idx]), 1e-15) + 1e-15)
                for idx in np.where(mask)[0]
            ]
            log_corr = logsumexp_stable(terms)
            p_prior = p_hat.get(tuple(s), {}).get(val, p_prior_func(s).get(val, 1e-12))
            unnorm_log_probs.append(np.log(p_prior + 1e-12) + log_corr)

        log_probs = np.array(unnorm_log_probs, dtype=np.float64)
        probs = np.exp(log_probs - scipy_logsumexp(log_probs))
        s.append(rng.choice([+1, -1], p=probs))
    return tuple(s)

