"""
Metropolis–Hastings baseline (single-spin-flip, random site order per sweep).
One "sweep" = N single-flip proposals, matching the legacy reference.
"""
from __future__ import annotations

import numpy as np

from .ising import IsingModel


def mh_sample(model: IsingModel, S, rng, burn_in=2000, thin=10, init=None):
    """Return (S, N) int8 samples.

    burn_in, thin are in units of sweeps (N single-spin-flip proposals each).
    """
    N = model.N
    J = model.J
    sigma = (rng.choice([-1, 1], size=N).astype(np.int8)
             if init is None else np.asarray(init, dtype=np.int8).copy())
    field = model.h + J @ sigma.astype(np.float64)   # f_i = h_i + sum_j J_ij s_j

    def step():
        nonlocal field
        i = int(rng.integers(0, N))
        dE = 2.0 * sigma[i] * field[i]               # E(flip) - E(current)
        if dE <= 0.0 or rng.random() < np.exp(-dE):
            old = sigma[i]
            sigma[i] = -old
            field = field + J[:, i] * (sigma[i] - old)   # incremental update

    def sweep():
        for _ in range(N):
            step()

    for _ in range(burn_in):
        sweep()
    out = np.empty((S, N), dtype=np.int8)
    for s in range(S):
        for _ in range(thin):
            sweep()
        out[s] = sigma
    return out
