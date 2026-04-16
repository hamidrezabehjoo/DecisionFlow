import numpy as np
from scipy.special import logsumexp as scipy_logsumexp


class BPIsingSolver:
    """Loopy belief propagation for pairwise Ising models."""

    def __init__(self, J, h=None, max_iter=100, tol=1e-6, damping=0.5):
        self.N = J.shape[0]
        self.J = np.array(J, dtype=np.float64)
        self.h = np.zeros(self.N) if h is None else np.array(h, dtype=np.float64)
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.neighbors = {
            i: [j for j in range(self.N) if j != i and abs(self.J[i, j]) > 1e-10]
            for i in range(self.N)
        }
        self._run_bp()

    def _run_bp(self):
        """Run loopy BP to convergence for pairwise Ising:
        m_{i->j}(σ_j) ∝ Σ_{σ_i} exp(J_ij σ_i σ_j + h_i σ_i) Π_{k∈N(i)\\{j}} m_{k->i}(σ_i)
        """
        states = np.array([+1, -1], dtype=np.int8)

        # messages[(i,j)] is an array over σ_j in [+1,-1] order
        self.messages = {}
        for i in range(self.N):
            for j in self.neighbors[i]:
                self.messages[(i, j)] = np.array([0.5, 0.5], dtype=np.float64)

        for _ in range(self.max_iter):
            max_diff = 0.0
            new_messages = {}

            for i in range(self.N):
                for j in self.neighbors[i]:
                    # incoming product to i for each σ_i
                    log_in_prod = np.zeros(2, dtype=np.float64)  # index is σ_i
                    for k in self.neighbors[i]:
                        if k == j:
                            continue
                        m_ki = self.messages[(k, i)]
                        log_in_prod += np.log(m_ki + 1e-300)

                    # compute m_{i->j}(σ_j)
                    msg_ij = np.zeros(2, dtype=np.float64)  # index is σ_j
                    Jij = self.J[i, j]
                    hi = self.h[i]

                    for idx_j, sigma_j in enumerate(states):
                        log_terms = np.array(
                            [
                                (Jij * (+1) * sigma_j) + (hi * (+1)) + log_in_prod[0],
                                (Jij * (-1) * sigma_j) + (hi * (-1)) + log_in_prod[1],
                            ],
                            dtype=np.float64,
                        )
                        msg_ij[idx_j] = np.exp(scipy_logsumexp(log_terms))

                    msg_ij /= (msg_ij.sum() + 1e-300)

                    old = self.messages[(i, j)]
                    damped = (1.0 - self.damping) * old + self.damping * msg_ij
                    damped /= (damped.sum() + 1e-300)
                    new_messages[(i, j)] = damped
                    max_diff = max(max_diff, np.max(np.abs(damped - old)))

            self.messages = new_messages
            if max_diff < self.tol:
                break

        # Node beliefs b_i(σ_i) ∝ exp(h_i σ_i) Π_{k∈N(i)} m_{k->i}(σ_i)
        self.beliefs = {}
        for i in range(self.N):
            log_b = np.array([self.h[i] * (+1), self.h[i] * (-1)], dtype=np.float64)
            for k in self.neighbors[i]:
                log_b += np.log(self.messages[(k, i)] + 1e-300)
            b = np.exp(log_b - scipy_logsumexp(log_b))
            self.beliefs[i] = b

    def get_conditional(self, assigned_nodes, assigned_vals):
        """Approximate conditionals under partial assignment (clamping).

        For each unassigned node a:
        P(σ_a | assigned) ∝ exp(h_a σ_a) Π_{b∈N(a), unassigned} m_{b->a}(σ_a)
                             × Π_{b∈N(a), assigned} exp(J_ab σ_a σ_b)
        """
        assigned_map = dict(zip(assigned_nodes, assigned_vals))
        cond = {}
        for a in range(self.N):
            if a in assigned_map:
                continue

            log_p = np.array([self.h[a] * (+1), self.h[a] * (-1)], dtype=np.float64)
            for b in self.neighbors[a]:
                if b in assigned_map:
                    sb = assigned_map[b]
                    log_p[0] += self.J[a, b] * (+1) * sb
                    log_p[1] += self.J[a, b] * (-1) * sb
                else:
                    log_p += np.log(self.messages[(b, a)] + 1e-300)

            p = np.exp(log_p - scipy_logsumexp(log_p))
            cond[a] = {+1: float(p[0]), -1: float(p[1])}
        return cond

    def sample_completion_given_partial(self, rng, partial_vals, partial_nodes):
        """Sample full config consistent with partial assignment."""
        sigma = dict(zip(partial_nodes, partial_vals))
        unassigned = [i for i in range(self.N) if i not in sigma]
        for a in unassigned:
            cond = self.get_conditional(list(sigma.keys()), list(sigma.values()))
            probs = cond.get(a, {+1: 0.5, -1: 0.5})
            p_plus = probs.get(+1, 0.5)
            p_minus = probs.get(-1, 0.5)
            norm = p_plus + p_minus
            if norm < 1e-12:
                p_plus, p_minus = 0.5, 0.5
            else:
                p_plus, p_minus = p_plus / norm, p_minus / norm
            sigma[a] = rng.choice([+1, -1], p=[p_plus, p_minus])
        return [sigma[i] for i in range(self.N)]

