"""
Correctness tests for the Decision Flow package.

Run:  python tests/test_df.py        (or: pytest tests/test_df.py)

All tests are deterministic (fixed seeds) and validate against exact
enumeration on a 3x3 Ising grid.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisionflow import (
    IsingModel,
    ExactReference,
    ProductPrior,
    LocalBoltzmannPrior,
    exact_sample,
    alg1_sample,
    mh_sample,
    metrics,
)
from decisionflow.solvers import mean_field_marginals, bp_marginals


def _setup():
    model = IsingModel.grid(3, 3, seed=42)
    enum = model.enumerate()
    return model, enum


def test_energy_vectorization():
    model, enum = _setup()
    J = model.J
    for s in enum.configs[:64]:
        naive = -0.5 * s @ J @ s - model.h @ s
        assert abs(naive - model.energy(s)) < 1e-10


def test_enumeration_index():
    """Row index must equal the binary code of the configuration."""
    model, enum = _setup()
    for k in range(len(enum.configs)):
        assert enum.index(enum.configs[k]) == k
    assert np.allclose(enum.probs.sum(), 1.0)


def test_bp_beats_mf_prior():
    """On this instance BP singleton marginals are much closer to exact."""
    model, enum = _setup()
    q_mf = mean_field_marginals(model)
    q_bp = bp_marginals(model)
    m_mf = q_mf[:, 1] - q_mf[:, 0]
    m_bp = q_bp[:, 1] - q_bp[:, 0]
    d_mf = metrics.delta1(m_mf, enum.mag)
    d_bp = metrics.delta1(m_bp, enum.mag)
    print(f"  prior delta1: MF={d_mf:.4f}  BP={d_bp:.4f}")
    assert d_bp < d_mf


def test_lbp_weight_identity():
    """Energy absorption: w(sigma) * pi_T(sigma) == exp(-E(sigma))."""
    model, _ = _setup()
    rng = np.random.default_rng(0)
    prior = LocalBoltzmannPrior(model, order=list(range(model.N)))
    paths, log_w = prior.sample(2000, rng)
    lhs = log_w + prior.log_prob(paths)          # log w + log pi_T
    rhs = -model.energy(paths)                   # -E
    assert np.max(np.abs(lhs - rhs)) < 1e-10


def test_is_identity():
    """E_{pi_T}[ w * 1{sigma_i = s} ] = sum_{sigma_i = s} exp(-E(sigma)).

    With self-normalized weights (sum w = K), the estimate is directly the
    normalized Boltzmann mass Z(i, s) / Z_total.
    """
    model, enum = _setup()
    rng = np.random.default_rng(1)
    prior = ProductPrior.from_bp(model)
    K = 60000
    paths, log_w = prior.sample(K, rng)
    lw = log_w - (log_w.max() + np.log(np.exp(log_w - log_w.max()).sum()))
    w = np.exp(lw) * K
    Z = np.exp(-enum.energies - (-enum.energies).max())
    scale = Z.sum()
    for i in [0, 4, 8]:
        for s in (-1, 1):
            est = w[paths[:, i] == s].sum() / K          # -> Z(i,s)/Z_total
            ref = Z[enum.masks[(i, s)]].sum() / scale    # exact, normalized
            assert abs(est - ref) / ref < 0.05


def test_exact_sampler_moments():
    model, enum = _setup()
    rng = np.random.default_rng(2)
    samples = exact_sample(enum, 20000, rng)
    mag, corr = metrics.sample_moments(samples)
    assert np.max(np.abs(mag - enum.mag)) < 0.03
    assert np.max(np.abs(corr - enum.corr)) < 0.04


def test_alg1_converges_for_all_priors():
    """Algorithm 1 at moderate K reproduces exact moments for every prior.

    Tolerances are per-prior: the finite-K error is pure IS variance, whose
    scale is set by the proposal's effective sample size (ESS/K measured on
    this instance: lbp ~0.88, bp/uniform ~0.09, mf ~3e-4). The exact-sampler
    noise floor at S=20000 is max|m err| ~0.074.
    """
    model, enum = _setup()
    K = S = 20000
    priors = {
        "uniform": (ProductPrior.uniform(model), 0.16, 0.22),
        "mf": (ProductPrior.from_mean_field(model), 0.50, 0.60),
        "bp": (ProductPrior.from_bp(model), 0.16, 0.22),
        "lbp": (LocalBoltzmannPrior(model, order=list(range(model.N))), 0.12, 0.18),
    }
    for name, (prior, tol_m, tol_c) in priors.items():
        rng_p = np.random.default_rng(10)
        rng_s = np.random.default_rng(11)
        paths, log_w = prior.sample(K, rng_p)
        order = prior.order if prior.requires_fixed_order else None
        samples, info = alg1_sample(paths, log_w, S, rng_s, order=order, enum=enum)
        mag, corr = metrics.sample_moments(samples)
        err_m = np.max(np.abs(mag - enum.mag))
        err_c = np.max(np.abs(corr - enum.corr))
        print(f"  {name:>8}: max|m-m_ref|={err_m:.4f}  "
              f"max|c-c_ref|={err_c:.4f}  starved={info['n_starved']}")
        assert err_m < tol_m, name
        assert err_c < tol_c, name


def test_lbp_beats_uniform_at_small_K():
    """Finite-K ranking: a better proposal lowers the IS variance."""
    model, enum = _setup()
    K = S = 500
    n_seeds = 6
    err = {"uniform": [], "lbp": []}
    for seed in range(n_seeds):
        for name in err:
            prior = (ProductPrior.uniform(model) if name == "uniform"
                     else LocalBoltzmannPrior(model, order=list(range(model.N))))
            paths, log_w = prior.sample(K, np.random.default_rng(100 + seed))
            order = prior.order if prior.requires_fixed_order else None
            samples, _ = alg1_sample(paths, log_w, S,
                                     np.random.default_rng(200 + seed),
                                     order=order, enum=enum)
            mag, _ = metrics.sample_moments(samples)
            err[name].append(np.max(np.abs(mag - enum.mag)))
    print(f"  mean max|m err| @K=500: uniform={np.mean(err['uniform']):.4f}  "
          f"lbp={np.mean(err['lbp']):.4f}")
    assert np.mean(err["lbp"]) < np.mean(err["uniform"])


def test_exact_reference_matches_enumeration():
    """Bit-decoded ExactReference must equal the full enumeration (3x3)."""
    model, enum = _setup()
    ref = ExactReference(model)
    assert np.allclose(ref.mag, enum.mag, atol=1e-10)
    assert np.allclose(ref.corr, enum.corr, atol=1e-10)
    assert abs(ref.logZ - enum.logZ) < 1e-10
    s = ref.sample_exact(20000, np.random.default_rng(4))
    mag, corr = metrics.sample_moments(s)
    assert np.max(np.abs(mag - enum.mag)) < 0.03
    assert np.max(np.abs(corr - enum.corr)) < 0.04


def test_mcmc_baseline():
    model, enum = _setup()
    rng = np.random.default_rng(3)
    samples = mh_sample(model, 5000, rng, burn_in=2000, thin=10)
    mag, corr = metrics.sample_moments(samples)
    assert np.max(np.abs(mag - enum.mag)) < 0.06
    assert np.max(np.abs(corr - enum.corr)) < 0.08


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"{fn.__name__} ...")
        fn()
        print("  ok")
    print(f"\nAll {len(fns)} tests passed.")
