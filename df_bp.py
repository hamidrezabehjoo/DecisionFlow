import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from ising_utils import energy_sigma


def sample_df_alg1_bp(
    p_hat,
    bp_solver,
    assigned_order,
    J,
    h,
    N,
    rng,
    n_is_samples=500,
    p_prior_fallback=None,
):
    s = []
    for t in range(N):
        unnorm_log_probs = []
        for val in [+1, -1]:
            s_new = s + [val]
            current_nodes = assigned_order[: t + 1]
            log_weights = []
            for _ in range(n_is_samples):
                s_T_list = bp_solver.sample_completion_given_partial(rng, s_new, current_nodes)
                s_T = tuple(s_T_list)
                log_target = -energy_sigma(s_T, J, h)
                log_proposal = 0.0
                for step in range(t + 1, N):
                    node = assigned_order[step]
                    prev_vals = [s_T[assigned_order[i]] for i in range(step)]
                    cond = bp_solver.get_conditional(assigned_order[:step], prev_vals)
                    log_proposal += np.log(cond.get(node, {}).get(s_T[node], 1e-15) + 1e-15)
                log_weights.append(log_target - log_proposal)

            log_corr = scipy_logsumexp(log_weights) - np.log(len(log_weights)) if log_weights else -np.inf
            s_curr = tuple(s)
            if p_prior_fallback is None:
                fallback = (
                    bp_solver.get_conditional(assigned_order[:t], s)
                    .get(assigned_order[t], {})
                    .get(val, 0.5)
                )
            else:
                fallback = p_prior_fallback(s).get(val, 0.5)
            p_prior = p_hat.get(s_curr, {}).get(val, fallback)
            unnorm_log_probs.append(np.log(p_prior + 1e-12) + log_corr)

        log_probs = np.array(unnorm_log_probs, dtype=np.float64)
        probs = np.exp(log_probs - scipy_logsumexp(log_probs))
        s.append(rng.choice([+1, -1], p=probs))
    return tuple(s)

