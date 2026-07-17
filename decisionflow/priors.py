"""
Proposal priors for Decision Flow.

A prior generates K paths sigma_1..sigma_K from its terminal marginal
pi_T(sigma) and returns the exact log importance weights

    log w_k = -E(sigma_k) - log pi_T(sigma_k),

so that for any partial assignment s_{t+1}

    (1/K) sum_{k : sigma_k consistent with s_{t+1}} w_k
        ->  sum_{sigma consistent with s_{t+1}} exp(-E(sigma))   as K -> inf,

which is exactly the DF / ancestral decision weight (paper Eq. 4; for the
PLP prior of Eqs. 20-22 the prior factors cancel analytically).

Two families are provided:

* ProductPrior        — PLP of Eqs. 20-22: pi_T(sigma) = prod_a q_a(sigma_a)
                        with q from Uniform / Mean-Field / BP marginals.
* LocalBoltzmannPrior — LBP: nodes assigned in a FIXED order, each spin
                        drawn from its local Gibbs conditional given the
                        already-assigned neighbors. The weight telescopes by
                        the energy-absorption identity

                            w(sigma) = prod_t 2 cosh(h^{loc}_{i_t}(S_t)),

                        because the exponent of the path probability sums to
                        -E(sigma) (each edge contributes exactly once, when
                        its later endpoint is assigned).

                        REQUIREMENT: the DF sampler must assign nodes in the
                        same fixed order, otherwise the prefix structure of
                        the paths does not match the sampler states and the
                        weights are invalid.
"""
from __future__ import annotations

import numpy as np

from .ising import IsingModel
from .solvers import mean_field_marginals, bp_marginals

_EPS = 1e-12


class Prior:
    """Interface: sample(K, rng) -> (paths (K,N) int8, log_w (K,) float64)."""

    #: if True, the DF sampler must use the same node order as the prior
    requires_fixed_order = False

    def __init__(self, model: IsingModel):
        self.model = model
        self.N = model.N

    def sample(self, K, rng):
        raise NotImplementedError

    def log_prob(self, paths):
        """log pi_T(paths) — used for diagnostics and tests."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Product-of-Local-Priors (PLP), paper Eqs. 20-22
# ---------------------------------------------------------------------------
class ProductPrior(Prior):
    """pi_T(sigma) = prod_a q_a(sigma_a), q: (N, 2) [P(-1), P(+1)]."""

    def __init__(self, model: IsingModel, q):
        super().__init__(model)
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.N, 2):
            raise ValueError(f"q must have shape ({self.N}, 2), got {q.shape}.")
        q = np.clip(q, _EPS, 1.0)
        q /= q.sum(axis=1, keepdims=True)
        self.q = q
        self.log_q = np.log(q)

    # ---------------------------------------------------------- constructors
    @classmethod
    def uniform(cls, model: IsingModel):
        return cls(model, np.full((model.N, 2), 0.5))

    @classmethod
    def from_mean_field(cls, model: IsingModel, **kw):
        return cls(model, mean_field_marginals(model, **kw))

    @classmethod
    def from_bp(cls, model: IsingModel, **kw):
        return cls(model, bp_marginals(model, **kw))

    # ------------------------------------------------------------- sampling
    def sample(self, K, rng):
        plus = rng.random((K, self.N)) < self.q[None, :, 1]
        paths = np.where(plus, np.int8(1), np.int8(-1))
        log_w = -self.model.energy(paths) - self.log_prob(paths)
        return paths, log_w

    def log_prob(self, paths):
        paths = np.asarray(paths)
        plus = paths == 1
        return plus @ self.log_q[:, 1] + (~plus) @ self.log_q[:, 0]


# ---------------------------------------------------------------------------
# Local-Boltzmann conditional Prior (LBP), fixed order
# ---------------------------------------------------------------------------
class LocalBoltzmannPrior(Prior):
    """Sequential prior: at step t, node i = order[t] is drawn from its local
    Gibbs conditional P(sigma_i) ∝ exp(sigma_i h^{loc}_i) with

        h^{loc}_i = h_i + sum_{j in N(i), j assigned earlier} J_ij sigma_j.

    Importance weight (exact, no path-probability bookkeeping needed):

        log w = sum_t log(2 cosh(h^{loc}_{order[t]})).
    """

    requires_fixed_order = True

    def __init__(self, model: IsingModel, order=None):
        super().__init__(model)
        self.order = list(range(model.N)) if order is None else list(order)
        if sorted(self.order) != list(range(model.N)):
            raise ValueError("order must be a permutation of range(N).")
        J = model.J
        nb = model.neighbors()
        pos = {node: t for t, node in enumerate(self.order)}
        # neighbors of each node that precede it in the order, with couplings
        self._prev_nb = []
        self._prev_J = []
        for i in self.order:
            prev = [j for j in nb[i] if pos[j] < pos[i]]
            self._prev_nb.append(np.asarray(prev, dtype=np.int64))
            self._prev_J.append(J[i, prev] if prev else np.zeros(0))

    def _local_fields(self, paths):
        """(K, N) array: h^{loc} seen at each step while generating `paths`."""
        K = paths.shape[0]
        lh = np.empty((K, len(self.order)))
        for t, i in enumerate(self.order):
            lh[:, t] = self.model.h[i]
            if len(self._prev_nb[t]):
                lh[:, t] += paths[:, self._prev_nb[t]] @ self._prev_J[t]
        return lh

    def sample(self, K, rng):
        paths = np.zeros((K, self.N), dtype=np.int8)
        log_w = np.zeros(K)
        for t, i in enumerate(self.order):
            lh = np.full(K, self.model.h[i], dtype=np.float64)
            if len(self._prev_nb[t]):
                lh += paths[:, self._prev_nb[t]] @ self._prev_J[t]
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * lh))
            p_plus = np.clip(p_plus, _EPS, 1.0 - _EPS)
            paths[:, i] = np.where(rng.random(K) < p_plus, np.int8(1), np.int8(-1))
            log_w += np.logaddexp(lh, -lh)        # log(2 cosh(lh))
        return paths, log_w

    def log_prob(self, paths):
        """log pi_T(paths) by replaying the chain rule along the order."""
        paths = np.asarray(paths, dtype=np.int8)
        lh = self._local_fields(paths)
        spins_t = paths[:, self.order]            # (K, N) spins in order
        return np.sum(spins_t * lh - np.logaddexp(lh, -lh), axis=1)
