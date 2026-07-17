"""
Decision Flow samplers.

exact_sample : exact i.i.d. sampling from p(sigma) ∝ exp(-E(sigma)) by
               ancestral (sequential) conditioning on the 2^N enumeration.
               This is the K -> infinity limit of Algorithm 1 and serves as
               the reference implementation of Theorem 1 for small N.

alg1_sample  : Algorithm 1 of the paper. Given K prior paths with exact
               importance weights w_k = exp(-E(sigma_k)) / pi_T(sigma_k),
               each decision weight is the sequential-IS estimate

                   W(i, s) = sum_{k active, sigma_k[i] = s} w_k
                           ->  sum_{sigma consistent} exp(-E(sigma))  (K -> inf)

               independently of the proposal pi_T (full support suffices);
               the prior only controls the finite-K variance.

Implementation notes
--------------------
* Duplicate paths are aggregated into unique configurations with summed
  weights (exact identity, big speedup when K >> 2^N). The aggregated
  weighted point set is precisely the empirical pi_T of the paper's
  Algorithm 1.
* Sampling proceeds in chunks, vectorized over the chunk; `active` is kept
  as a float array so per-step weight sums are plain BLAS gemv calls.
"""
from __future__ import annotations

import numpy as np

from .ising import Enumeration

# column layout of the stacked mask matrix: col 2*i+0 -> spin -1, col 2*i+1 -> spin +1
_NEG, _POS = 0, 1


def _resolve_orders(order, rng, n_chunks, N):
    """Per-chunk node orders. order=None -> fresh random order each chunk."""
    if order is None:
        return [rng.permutation(N) for _ in range(n_chunks)]
    order = list(order)
    if sorted(order) != list(range(N)):
        raise ValueError("order must be a permutation of range(N).")
    return [order] * n_chunks


def aggregate_paths(paths, log_w):
    """Merge duplicate configurations, summing their importance weights.

    Returns (unique_paths (U, N) int8, log_w_agg (U,)) with weights defined
    up to an additive constant (the sampler renormalizes anyway).
    """
    uniq, inv = np.unique(paths, axis=0, return_inverse=True)
    lw = log_w - log_w.max()
    w = np.exp(lw)
    w_agg = np.zeros(len(uniq))
    np.add.at(w_agg, inv, w)
    return uniq, np.log(w_agg)


def exact_sample(enum: Enumeration, S, rng, order=None, chunk=1000):
    """Exact i.i.d. samples from p(sigma) ∝ exp(-E(sigma)).

    Ancestral sampling on the enumeration: given the active (consistent)
    configuration set, the next node/spin is chosen with probability
    proportional to the remaining Boltzmann mass

        P(i, s | state) ∝ sum_{sigma consistent, sigma_i = s} exp(-E(sigma)).

    The node order may be fixed or re-drawn per chunk (order=None); the
    output distribution is exactly p either way.
    """
    N = enum.N
    M = enum.mask_matrix.astype(np.float64)          # (2N, 2^N)
    W = M * enum.weights[None, :]                    # weighted masks
    n_chunks = int(np.ceil(S / chunk))
    orders = _resolve_orders(order, rng, n_chunks, N)

    samples = np.zeros((S, N), dtype=np.int8)
    done = 0
    w_tot = enum.weights                                # W_neg + W_pos = w_tot
    for ord_chunk in orders:
        B = min(chunk, S - done)
        active = np.ones((B, M.shape[1]), dtype=bool)
        states = np.zeros((B, N), dtype=np.int8)
        total = np.full(B, w_tot.sum())                 # active @ w_tot
        for i in ord_chunk:
            # one gemv per step: w_neg = total - w_pos; the branch weight
            # taken becomes the next step's total (no second gemv).
            w_pos = active @ W[2 * i + _POS]
            w_neg = np.maximum(total - w_pos, 0.0)
            p_pos = w_pos / np.where(total > 0, total, 1.0)
            plus = rng.random(B) < p_pos
            states[:, i] = np.where(plus, np.int8(1), np.int8(-1))
            active &= np.where(plus[:, None],
                               M[2 * i + _POS][None, :] > 0,
                               M[2 * i + _NEG][None, :] > 0)
            total = np.where(plus, w_pos, w_neg)
        samples[done:done + B] = states
        done += B
    return samples


def alg1_sample(paths, log_w, S, rng, order=None, chunk=1000, enum=None,
                on_starvation="exact", aggregate="auto", model=None):
    """Algorithm 1: DF sampling from K weighted prior paths.

    Parameters
    ----------
    paths : (K, N) int8           prior paths sigma_k
    log_w : (K,) float64          log importance weights (-E - log pi_T)
    S     : int                   number of posterior samples
    rng   : np.random.Generator
    order : permutation or None   node order; None -> fresh random per chunk.
                                  MUST be the prior's fixed order when the
                                  prior is a LocalBoltzmannPrior.
    enum  : Enumeration or None   enables the exact fallback for starved
                                  states (no active path left).
    on_starvation : "exact" | "raise" | "local_gibbs"
        "exact"       — exact child weights (requires enum; small N only);
        "raise"       — fail loudly;
        "local_gibbs" — approximate: draw from the local Gibbs conditional
                        given the current partial state (requires model;
                        available at any N; biased at finite K, vanishes as
                        K grows). The event count is reported in info.
    aggregate : "auto" | bool     merge duplicate paths first (exact identity;
                                  "auto" aggregates when 2^N <= K/4).
    model : IsingModel or None    required for on_starvation="local_gibbs".

    Returns
    -------
    samples : (S, N) int8
    info    : dict with 'n_starved' (fallback events) and 'n_unique'.
    """
    paths = np.asarray(paths, dtype=np.int8)
    log_w = np.asarray(log_w, dtype=np.float64)
    K, N = paths.shape

    if aggregate == "auto":
        aggregate = (2 ** N) <= (K // 4)
    if aggregate:
        paths, log_w = aggregate_paths(paths, log_w)
    K = len(paths)

    # normalized importance weights, projected onto each (node, spin) event
    lw = log_w - (log_w.max() + np.log(np.exp(log_w - log_w.max()).sum()))
    w = np.exp(lw)                                     # sums to 1
    W = np.stack([w * (paths[:, i] == -1) for i in range(N)] +
                 [w * (paths[:, i] == 1) for i in range(N)], axis=0)
    # rows 0..N-1: spin -1 for node i; rows N..2N-1: spin +1
    n_chunks = int(np.ceil(S / chunk))
    orders = _resolve_orders(order, rng, n_chunks, N)

    if on_starvation == "exact" and enum is None:
        raise ValueError("on_starvation='exact' requires an Enumeration.")
    if on_starvation == "local_gibbs" and model is None:
        raise ValueError("on_starvation='local_gibbs' requires model=.")
    if on_starvation not in ("exact", "raise", "local_gibbs"):
        raise ValueError(f"unknown on_starvation={on_starvation!r}")
    if on_starvation == "local_gibbs":
        _J = model.J
        _h = model.h

    samples = np.zeros((S, N), dtype=np.int8)
    n_starved = 0
    done = 0
    for ord_chunk in orders:
        B = min(chunk, S - done)
        active = np.ones((B, K))                     # float: cheap gemv below
        states = np.zeros((B, N), dtype=np.int8)
        total = np.full(B, w.sum())                  # active @ w  (W_neg + W_pos = w)
        for i in ord_chunk:
            # one gemv per step: w_neg = total - w_pos; the branch weight
            # taken becomes the next step's total (no second gemv).
            w_pos = active @ W[N + i]
            w_neg = np.maximum(total - w_pos, 0.0)

            starved = total <= 0.0
            if starved.any():
                if on_starvation == "raise":
                    raise RuntimeError(
                        f"IS starvation at node {i}: no active prior path "
                        f"for {int(starved.sum())}/{B} samples. Increase K "
                        f"or use a better-matched prior."
                    )
                n_starved += int(starved.sum())
                idx_b = np.where(starved)[0]
                if on_starvation == "local_gibbs":
                    # local Gibbs conditional given the partial state:
                    # h_loc = h_i + sum_{j assigned} J_ij sigma_j
                    # (unassigned spins are 0 in `states`, so they drop out)
                    h_loc = _h[i] + states[idx_b].astype(np.float64) @ _J[:, i]
                    w_neg[idx_b] = np.exp(-h_loc)
                    w_pos[idx_b] = np.exp(h_loc)
                    total[idx_b] = w_neg[idx_b] + w_pos[idx_b]
                else:
                    for b in idx_b:
                        w_neg[b], w_pos[b] = _exact_child_weights(
                            enum, states[b], i)
                        total[b] = w_neg[b] + w_pos[b]

            p_pos = w_pos / np.where(total > 0, total, 1.0)
            plus = rng.random(B) < p_pos
            states[:, i] = np.where(plus, np.int8(1), np.int8(-1))
            col = paths[:, i][None, :]                     # (1, K) in {-1, +1}
            active = active * np.where(plus[:, None], col == 1, col == -1)
            total = np.where(plus, w_pos, w_neg)
        samples[done:done + B] = states
        done += B

    return samples, {"n_starved": n_starved, "n_unique": K}


def _exact_child_weights(enum: Enumeration, state, i):
    """Exact DF child weights for node i given a partial state (0 = unset)."""
    base = np.ones(len(enum.configs), dtype=bool)
    for j in range(enum.N):
        if state[j] != 0:
            base &= enum.masks[(j, int(state[j]))]
    w_neg = enum.weights[base & enum.masks[(i, -1)]].sum()
    w_pos = enum.weights[base & enum.masks[(i, 1)]].sum()
    return w_neg, w_pos
