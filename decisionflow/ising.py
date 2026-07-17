"""
Ising model specification and exact enumeration.

Target distribution (paper Eqs. 1 and 19):

    p(sigma) = (1/Z) exp(-E(sigma)),
    E(sigma) = - sum_{(a,b) in edges} J_ab sigma_a sigma_b - sum_a h_a sigma_a,

with sigma in {-1, +1}^N.

Conventions used everywhere in this package:
  * spins are int8 values in {-1, +1};
  * node marginals q are stored as (N, 2) arrays with columns
    [P(sigma = -1), P(sigma = +1)];
  * couplings are stored once per undirected edge (i < j).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class IsingModel:
    """Pairwise Ising model on N spins."""

    N: int
    h: np.ndarray       # (N,)  local fields
    ei: np.ndarray      # (E,)  edge endpoints, ei < ej
    ej: np.ndarray      # (E,)
    ew: np.ndarray      # (E,)  couplings J_ij

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_edges(cls, N, h, edges):
        """edges: iterable of (i, j, J_ij); each undirected edge given once."""
        h = np.asarray(h, dtype=np.float64)
        ei, ej, ew = zip(*edges) if edges else ([], [], [])
        return cls(
            N=int(N),
            h=h,
            ei=np.asarray(ei, dtype=np.int64),
            ej=np.asarray(ej, dtype=np.int64),
            ew=np.asarray(ew, dtype=np.float64),
        )

    @classmethod
    def grid(cls, n, m, J0=0.3, dJ=0.2, h_scale=0.1, seed=42):
        """Ferromagnetic grid, reproduces the legacy `create_ising_params`.

        h_i ~ U[-h_scale, h_scale];  J_ij = J0 + U[0, dJ] on nearest-neighbor
        edges of an n x m lattice (rng draws in legacy order: all h first,
        then edges row-major, horizontal before vertical at each cell).
        """
        rng = np.random.default_rng(seed)
        N = n * m

        def idx(r, c):
            return r * m + c

        h = rng.uniform(-h_scale, h_scale, size=N)
        edges = []
        for r in range(n):
            for c in range(m):
                i = idx(r, c)
                if c + 1 < m:
                    edges.append((i, idx(r, c + 1), J0 + rng.uniform(0, dJ)))
                if r + 1 < n:
                    edges.append((i, idx(r + 1, c), J0 + rng.uniform(0, dJ)))
        return cls.from_edges(N, h, edges)

    @classmethod
    def glassy_grid(cls, n, m, seed=42):
        """'Glassy' grid of paper Fig. 2: h_a, J_ab ~ U[-1, 1]."""
        rng = np.random.default_rng(seed)
        N = n * m
        h = rng.uniform(-1.0, 1.0, size=N)
        edges = []
        for r in range(n):
            for c in range(m):
                i = r * m + c
                if c + 1 < m:
                    edges.append((i, i + 1, rng.uniform(-1.0, 1.0)))
                if r + 1 < n:
                    edges.append((i, i + m, rng.uniform(-1.0, 1.0)))
        return cls.from_edges(N, h, edges)

    @classmethod
    def random_dense(cls, N, scale=0.3, seed=42):
        """Fully connected symmetric Gaussian J (legacy alg1_sample.py setup)."""
        rng = np.random.default_rng(seed)
        J = rng.normal(0.0, scale, size=(N, N))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        iu = np.triu_indices(N, k=1)
        edges = [(int(i), int(j), float(J[i, j])) for i, j in zip(*iu)]
        return cls.from_edges(N, np.zeros(N), edges)

    # ------------------------------------------------------------- derived
    @property
    def J(self):
        """Dense symmetric coupling matrix (N, N)."""
        J = np.zeros((self.N, self.N))
        J[self.ei, self.ej] = self.ew
        J[self.ej, self.ei] = self.ew
        return J

    def neighbors(self):
        """List of neighbor index arrays, one per node."""
        nb = [[] for _ in range(self.N)]
        for i, j in zip(self.ei, self.ej):
            nb[i].append(j)
            nb[j].append(i)
        return [np.asarray(x, dtype=np.int64) for x in nb]

    # ------------------------------------------------------------- physics
    def energy(self, sigmas):
        """E(sigma), vectorized. sigmas: (..., N) in {-1,+1} -> (...,)"""
        s = np.asarray(sigmas, dtype=np.float64)
        pair = (s[..., self.ei] * s[..., self.ej]) @ self.ew
        field = s @ self.h
        return -pair - field

    def local_fields(self, sigmas):
        """f_i = h_i + sum_j J_ij sigma_j, vectorized. sigmas: (..., N)."""
        s = np.asarray(sigmas, dtype=np.float64)
        return self.h + s @ self.J

    # ---------------------------------------------------- exact enumeration
    def enumerate(self):
        """Exact enumeration over all 2^N configurations (small N only)."""
        return Enumeration(self)


# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------
class Enumeration:
    """All 2^N configurations with exact probabilities and moments.

    Attributes
    ----------
    configs  : (2^N, N) int8   all configurations, row index = binary code
    energies : (2^N,)          E(sigma)
    probs    : (2^N,)          exact p(sigma) = exp(-E)/Z
    mag      : (N,)            exact singleton means  m_a = E[sigma_a]
    corr     : (N, N)          exact pairwise means   c_ab = E[sigma_a sigma_b]
    """

    def __init__(self, model: IsingModel):
        N = model.N
        if N > 24:
            raise ValueError(f"Enumeration infeasible for N={N} (>24).")
        self.model = model
        self.N = N
        # row index = binary code of the configuration (bit a = (sigma_a+1)/2)
        self.configs = np.array(list(product([-1, 1], repeat=N)), dtype=np.int8)
        self.energies = model.energy(self.configs)
        logp = -self.energies
        logp -= logp.max()
        p = np.exp(logp)
        self.probs = p / p.sum()
        self.logZ = float(np.log(p.sum()) + (-self.energies).max())
        self.mag = self.probs @ self.configs
        self.corr = (self.configs * self.probs[:, None]).T @ self.configs

        # decision-weight quantities used by the samplers
        # masks[(i, s)] : bool (2^N,) — configs with sigma_i = s
        self.masks = {
            (i, s): (self.configs[:, i] == s)
            for i in range(N)
            for s in (-1, 1)
        }
        # stacked view for vectorized ancestral sampling: column 2*i+(s==+1)
        self.mask_matrix = np.stack(
            [self.masks[(i, s)] for i in range(N) for s in (-1, 1)], axis=0
        )  # (2N, 2^N) bool
        w = np.exp(-self.energies - (-self.energies).max())
        self.weights = w  # unnormalized exp(-E), shifted

    def index(self, sigmas):
        """Map configurations to row indices in `configs`. sigmas: (..., N).

        `configs` is generated by itertools.product, so the LAST coordinate
        varies fastest, i.e. coordinate a is bit (N-1-a) of the row index.
        """
        s = np.asarray(sigmas)
        bits = (s + 1) // 2
        return bits @ (1 << np.arange(self.N - 1, -1, -1, dtype=np.int64))


# ---------------------------------------------------------------------------
# Memory-efficient exact reference (N up to ~26)
# ---------------------------------------------------------------------------
class ExactReference:
    """Exact moments and exact i.i.d. samples for N up to ~26, without
    materializing the (2^N, N) configuration table or the (2N, 2^N) mask
    matrices (which cost ~13 GB at N = 25).

    Configurations are integers 0 .. 2^N - 1; spin a of configuration c is
    2*bit_{N-1-a}(c) - 1. All sums are computed by bit-decoding passes over
    the index array with a few reusable buffers (peak memory ~ 5 x 2^N
    bytes for the weights plus scratch).

    Attributes
    ----------
    mag, corr : exact singleton / pairwise moments
    logZ      : exact log partition function
    """

    def __init__(self, model: IsingModel):
        N = model.N
        if N > 26:
            raise ValueError(f"ExactReference infeasible for N={N} (>26).")
        self.model = model
        self.N = N
        M = 1 << N
        idx = np.arange(M, dtype=np.uint32)
        t1 = np.empty(M, dtype=np.uint32)
        t2 = np.empty(M, dtype=np.uint32)
        F = np.empty(M, dtype=np.float64)
        # accumulate S = sum J si sj + sum h si  (= -E) directly into w
        w = np.zeros(M, dtype=np.float64)
        sh = (N - 1 - np.arange(N)).astype(np.uint32)

        def bit(a, out):
            np.right_shift(idx, sh[a], out=out)
            np.bitwise_and(out, 1, out=out)
            return out

        for i, j, Jij in zip(model.ei, model.ej, model.ew):
            np.bitwise_xor(bit(i, t1), bit(j, t2), out=t1)   # si*s_j = 1-2*xor
            np.multiply(t1, -2.0 * Jij, out=F)
            F += Jij
            w += F
        for i in range(N):
            np.multiply(bit(i, t1), 2.0 * model.h[i], out=F)  # si = 2*bit-1
            w += F
            w -= model.h[i]

        S_max = float(w.max())
        w -= S_max
        np.exp(w, out=w)                          # unnormalized exp(-E)
        Z = float(w.sum())
        self.logZ = float(np.log(Z) + S_max)

        # moments: m_i = (2*w.bit_i - Z)/Z ;  c_ij = (Z - 2*w.xor_ij)/Z
        mag = np.empty(N)
        for i in range(N):
            np.multiply(bit(i, t1), 1.0, out=F)
            mag[i] = (2.0 * float(F @ w) - Z) / Z
        corr = np.eye(N)
        for i in range(N):
            bi = bit(i, t1).copy()
            for j in range(i + 1, N):
                np.bitwise_xor(bi, bit(j, t2), out=t2)
                np.multiply(t2, 1.0, out=F)
                corr[i, j] = corr[j, i] = (Z - 2.0 * float(F @ w)) / Z
        self.mag = mag
        self.corr = corr

        # replace w by its cumsum -> exact i.i.d. sampling via inversion
        w.cumsum(out=w)
        self._cum = w
        self._Z = Z
        del idx, t1, t2, F

    def sample_exact(self, S, rng):
        """Exact i.i.d. samples from p(sigma) ∝ exp(-E(sigma))."""
        r = rng.random(S) * self._Z
        codes = np.searchsorted(self._cum, r).astype(np.int64)
        bits = (codes[:, None] >> (self.N - 1 - np.arange(self.N))) & 1
        return (2 * bits - 1).astype(np.int8)

