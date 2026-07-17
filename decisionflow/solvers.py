"""
Variational singleton marginals: naive mean field (MF) and loopy belief
propagation (BP). Both return q of shape (N, 2) with columns
[P(sigma = -1), P(sigma = +1)].

These are used only to build PLP proposal priors (paper Appendix A, Eqs.
20-22): the DF correction makes the final sampler exact regardless of the
quality of q; better q only lowers the importance-sampling variance at
finite path budget K.
"""
from __future__ import annotations

import numpy as np

from .ising import IsingModel

_EPS = 1e-12


def _to_q(mag):
    """Magnetizations m_i in [-1, 1] -> q (N, 2) [P(-1), P(+1)]."""
    p_plus = np.clip(0.5 * (1.0 + mag), _EPS, 1.0 - _EPS)
    return np.stack([1.0 - p_plus, p_plus], axis=1)


def mean_field_marginals(model: IsingModel, max_iter=500, tol=1e-6):
    """Naive mean field: m = tanh(h + J m), synchronous updates."""
    J = model.J
    m = np.zeros(model.N)
    for _ in range(max_iter):
        m_new = np.tanh(model.h + J @ m)
        if np.max(np.abs(m_new - m)) < tol:
            m = m_new
            break
        m = m_new
    return _to_q(m)


def bp_marginals(model: IsingModel, max_iter=200, tol=1e-6, damping=0.5):
    """Loopy BP for the pairwise Ising model.

    Messages m_{i->j}(sigma_j) (log-domain, synchronous updates, damping in
    probability space):

        m_{i->j}(s_j) ∝ sum_{s_i} exp(J_ij s_i s_j + h_i s_i)
                                 * prod_{k in N(i)\\{j}} m_{k->i}(s_i)

    Beliefs: b_i(s_i) ∝ exp(h_i s_i) prod_{k in N(i)} m_{k->i}(s_i).
    """
    N = model.N
    ei, ej, ew = model.ei, model.ej, model.ew
    E = len(ew)
    # directed edges: 2 per undirected edge
    src = np.concatenate([ei, ej])          # message src[d] -> dst[d]
    dst = np.concatenate([ej, ei])
    Jd = np.concatenate([ew, ew])
    D = 2 * E

    # incoming directed edges per node: into[v] = [d : dst[d] == v]
    into = [np.where(dst == v)[0] for v in range(N)]
    # for each directed edge d = (i->j): edges into i except the one from j
    in_edges = [
        np.asarray([d2 for d2 in into[int(src[d])] if int(src[d2]) != int(dst[d])],
                   dtype=np.int64)
        for d in range(D)
    ]

    spins = np.array([-1.0, 1.0])
    log_msg = np.full((D, 2), -np.log(2.0))   # index: [s_dst = -1, +1]

    for _ in range(max_iter):
        new = np.empty_like(log_msg)
        for d in range(D):
            i, j = int(src[d]), int(dst[d])
            # log incoming product to i, as function of s_i
            log_in = log_msg[in_edges[d]].sum(axis=0) if len(in_edges[d]) else np.zeros(2)
            # msg(s_j) = logsumexp_{s_i} [ J s_i s_j + h_i s_i + log_in(s_i) ]
            # outer[s_i, s_j]
            outer = Jd[d] * np.outer(spins, spins) + model.h[i] * spins[:, None] + log_in[:, None]
            m = outer.max()
            msg = m + np.log(np.exp(outer - m).sum(axis=0))
            new[d] = msg - (msg.max() + np.log(np.exp(msg - msg.max()).sum()))
        if damping < 1.0:
            p_new = damping * np.exp(new) + (1.0 - damping) * np.exp(log_msg)
            p_new = np.clip(p_new, _EPS, 1.0)
            p_new /= p_new.sum(axis=1, keepdims=True)
            new = np.log(p_new)
        delta = np.max(np.abs(new - log_msg))
        log_msg = new
        if delta < tol:
            break

    # beliefs
    mag = np.zeros(N)
    for i in range(N):
        log_b = model.h[i] * spins + log_msg[into[i]].sum(axis=0)
        log_b -= log_b.max() + np.log(np.exp(log_b - log_b.max()).sum())
        b = np.exp(log_b)
        mag[i] = b[1] - b[0]
    return _to_q(mag)
