"""
Diagnostics:

  delta1, delta2 : relative singleton / pairwise moment mismatch (paper
                   Sec. 5 — relative normalization by |m_ref|, |c_ref|);
  kl_proxy       : K_dL(q||p) = E_q[log q + E]  (paper Eq. 23; differs from
                   KL(q||p) by the sampler-independent constant log Z);
  npll           : negative pseudo-log-likelihood (paper Eqs. 24-25).

All estimators are computed directly from the samples, so they work for
large N (no 2^N enumeration needed). References (mag, corr) can come from
an Enumeration (small N) or an ExactReference (N up to ~26) — anything with
`.mag` and `.corr` attributes.
"""
from __future__ import annotations

import numpy as np

from .ising import IsingModel, Enumeration

_EPS = 1e-12


def encode(samples):
    """Map configurations to int64 codes (coordinate a = bit N-1-a)."""
    s = np.asarray(samples)
    N = s.shape[1]
    bits = (s + 1) // 2
    return bits @ (1 << np.arange(N - 1, -1, -1, dtype=np.int64))


def empirical_probs(samples, enum: Enumeration):
    """Empirical distribution over the 2^N configurations (small N only)."""
    idx = enum.index(samples)
    counts = np.bincount(idx, minlength=len(enum.configs)).astype(np.float64)
    return counts / len(samples)


def sample_moments(samples):
    """Empirical magnetizations (N,) and correlations (N, N)."""
    s = np.asarray(samples, dtype=np.float64)
    return s.mean(axis=0), (s.T @ s) / len(s)


def delta1(mag, mag_ref):
    """Relative singleton mismatch (paper convention)."""
    return float(np.mean(np.abs(mag - mag_ref) / (np.abs(mag_ref) + _EPS)))


def delta2(corr, corr_ref):
    """Relative pairwise mismatch over i < j (paper convention)."""
    N = corr.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(np.mean(np.abs(corr[iu] - corr_ref[iu])
                         / (np.abs(corr_ref[iu]) + _EPS)))


def kl_proxy(samples, model: IsingModel):
    """K_dL(q||p) = (1/S) sum_k [log q_hat(sigma_k) + E(sigma_k)]  (Eq. 23).

    q_hat is the empirical distribution of the sample set; duplicate
    configurations are counted by sorting the int64 codes — no 2^N arrays.
    """
    samples = np.asarray(samples)
    S = len(samples)
    codes = encode(samples)
    _, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    log_q = np.log(counts[inv] / S)
    E = model.energy(samples)
    return float(np.mean(log_q + E))


def npll(samples, model: IsingModel):
    """Negative pseudo-log-likelihood (Eqs. 24-25), vectorized."""
    s = np.asarray(samples, dtype=np.float64)
    fields = model.local_fields(s)                    # (S, N)
    log_cond = s * fields - np.logaddexp(fields, -fields)
    return float(-np.mean(log_cond.sum(axis=1)))


def evaluate(samples, model: IsingModel, ref):
    """All four diagnostics. ref: any object with .mag and .corr
    (Enumeration or ExactReference)."""
    mag, corr = sample_moments(samples)
    return {
        "delta1": delta1(mag, ref.mag),
        "delta2": delta2(corr, ref.corr),
        "kl": kl_proxy(samples, model),
        "npll": npll(samples, model),
    }
