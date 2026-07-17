# ================================================================
# Algorithm 1 (finite K prior paths): S = K sweep.
# Corrected re-implementation of the legacy alg1_sk_sweep experiment.
#
#   - Methods: Uniform / MF / BP / LBP priors with Algorithm 1,
#     plus Exact (K = infinity) and MCMC references.
#   - K_values x n_seeds replicates, four diagnostics vs exact references.
#
# Reference strategy by system size:
#   N <= 16 : Enumeration       -> exact fallback on IS starvation
#   N >  16 : ExactReference    -> local-Gibbs fallback (approximate, counted)
#
# Usage:  python experiments/run_sk_sweep.py          (3x3 default)
#         python experiments/run_4x4.py               (4x4)
#         python experiments/run_5x5.py               (5x5)
# ================================================================
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisionflow import (
    IsingModel,
    Enumeration,
    ExactReference,
    ProductPrior,
    LocalBoltzmannPrior,
    exact_sample,
    alg1_sample,
    mh_sample,
    metrics,
)

CONFIG = dict(
    n=3, m=3, seed=42,                       # Ising instance
    K_values=[100, 500, 2000, 10000, 50000], # S = K at each point
    n_seeds=3,
    S_ref=5000,                              # sample size for references
    chunk=1000,
    out_png="alg1_sk_sweep_corrected.png",
    out_csv="alg1_sk_sweep_corrected.csv",
)

METHODS = ["Uniform-Alg1", "MF-Alg1", "BP-Alg1", "LBP-Alg1", "Exact-DF", "MCMC"]
METRICS = ["delta1", "delta2", "kl", "npll"]

ENUM_MAX_N = 16      # above this, use ExactReference + local-Gibbs fallback


def make_reference(model):
    """Enumeration (small N, enables exact fallback) or ExactReference."""
    if model.N <= ENUM_MAX_N:
        return model.enumerate()
    return ExactReference(model)


def build_priors(model, order):
    return {
        "Uniform-Alg1": ProductPrior.uniform(model),
        "MF-Alg1": ProductPrior.from_mean_field(model),
        "BP-Alg1": ProductPrior.from_bp(model),
        "LBP-Alg1": LocalBoltzmannPrior(model, order=order),
    }


def _exact_draw(ref, S, rng, chunk):
    if isinstance(ref, Enumeration):
        return exact_sample(ref, S, rng, chunk=chunk)
    return ref.sample_exact(S, rng)


def _alg1_draw(prior, model, ref, K, S, rng, chunk):
    paths, log_w = prior.sample(K, rng)
    ord_arg = prior.order if prior.requires_fixed_order else None
    if isinstance(ref, Enumeration):
        return alg1_sample(paths, log_w, S, rng, order=ord_arg, chunk=chunk,
                           enum=ref, on_starvation="exact", aggregate=True)
    return alg1_sample(paths, log_w, S, rng, order=ord_arg, chunk=chunk,
                       on_starvation="local_gibbs", model=model, aggregate=True)


def run_experiment(cfg=CONFIG, verbose=True):
    model = IsingModel.grid(cfg["n"], cfg["m"], seed=cfg["seed"])
    t_ref = time.time()
    ref = make_reference(model)
    N = model.N
    order = list(range(N))                     # fixed LBP order (prior & sampler)
    K_values, n_seeds = cfg["K_values"], cfg["n_seeds"]

    if verbose:
        print(f"Ising {cfg['n']}x{cfg['m']}  N={N}  configs={2**N}  "
              f"ref={type(ref).__name__} ({time.time()-t_ref:.0f}s)", flush=True)
        q_mf = ProductPrior.from_mean_field(model).q
        q_bp = ProductPrior.from_bp(model).q
        d1_mf = metrics.delta1(q_mf[:, 1] - q_mf[:, 0], ref.mag)
        d1_bp = metrics.delta1(q_bp[:, 1] - q_bp[:, 0], ref.mag)
        print(f"Prior delta1:  MF={d1_mf:.4f}  BP={d1_bp:.4f}", flush=True)

    results = {name: {met: np.zeros((n_seeds, len(K_values))) for met in METRICS}
               for name in METHODS}

    # ── references: Exact and MCMC at fixed S_ref ─────────────────────────
    if verbose:
        print(f"\nPre-computing Exact-DF and MCMC references "
              f"({n_seeds} seeds, S={cfg['S_ref']}) ...", flush=True)
    ref_stats = {name: {met: np.zeros(n_seeds) for met in METRICS}
                 for name in ["Exact-DF", "MCMC"]}
    for seed in range(n_seeds):
        s_ex = _exact_draw(ref, cfg["S_ref"], np.random.default_rng(seed * 10 + 8),
                           cfg["chunk"])
        s_mc = mh_sample(model, cfg["S_ref"], np.random.default_rng(seed * 10 + 9),
                         burn_in=2000, thin=10)
        for met in METRICS:
            ref_stats["Exact-DF"][met][seed] = metrics.evaluate(s_ex, model, ref)[met]
            ref_stats["MCMC"][met][seed] = metrics.evaluate(s_mc, model, ref)[met]

    # ── Algorithm 1 sweep ─────────────────────────────────────────────────
    if verbose:
        print(f"\nRunning Algorithm 1 (S=K sweep)  n_seeds={n_seeds}", flush=True)
        print(f"{'K=S':>7}  " + "  ".join(f"{m + ' D1':>12}" for m in
              ["Uni", "MF", "BP", "LBP"]) + f"  {'time':>6}", flush=True)

    t0 = time.time()
    alg1_names = ["Uniform-Alg1", "MF-Alg1", "BP-Alg1", "LBP-Alg1"]
    priors = build_priors(model, order)        # priors are K-independent
    for ki, K in enumerate(K_values):
        S = K
        starved = {name: [] for name in alg1_names}
        for seed in range(n_seeds):
            for pi, (name, prior) in enumerate(priors.items()):
                s, info = _alg1_draw(prior, model, ref, K, S,
                                     np.random.default_rng(seed * 100 + 2 * pi + 1),
                                     cfg["chunk"])
                starved[name].append(info["n_starved"])
                vals = metrics.evaluate(s, model, ref)
                for met in METRICS:
                    results[name][met][seed, ki] = vals[met]

            for met in METRICS:  # references are constant along K
                results["Exact-DF"][met][seed, ki] = ref_stats["Exact-DF"][met][seed]
                results["MCMC"][met][seed, ki] = ref_stats["MCMC"][met][seed]

        if verbose:
            means = [results[n]["delta1"][:, ki].mean() for n in alg1_names]
            st = [int(np.mean(starved[n])) for n in alg1_names]
            print(f"{K:>7}  " + "  ".join(f"{v:>12.4f}" for v in means) +
                  f"  {time.time()-t0:>5.0f}s   starved:{st}", flush=True)

    if verbose:
        print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)
    return model, ref, results


def plot_results(cfg, results):
    STYLE = {
        "Uniform-Alg1": dict(color="gray", ls="-.", marker="d", lw=2.0, ms=6),
        "MF-Alg1": dict(color="steelblue", ls="-", marker="o", lw=2.0, ms=6),
        "BP-Alg1": dict(color="firebrick", ls="--", marker="s", lw=2.0, ms=6),
        "LBP-Alg1": dict(color="darkorange", ls="-", marker="^", lw=2.0, ms=6),
        "Exact-DF": dict(color="purple", ls=":", marker="", lw=1.8, ms=0),
        "MCMC": dict(color="seagreen", ls="-.", marker="^", lw=1.5, ms=5),
    }
    INFO = {
        "kl": (r"KL proxy  $K_{dL}(q\|p)$", "KL proxy vs K=S"),
        "npll": ("NPLL", "NPLL vs K=S"),
        "delta1": (r"$\Delta_1$ (singleton mismatch)", r"$\Delta_1$ vs K=S"),
        "delta2": (r"$\Delta_2$ (pairwise mismatch)", r"$\Delta_2$ vs K=S"),
    }
    K_values = cfg["K_values"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Algorithm 1 (S = K), corrected importance weights\n"
        f"Ising {cfg['n']}x{cfg['m']} (N={cfg['n']*cfg['m']}), "
        f"{cfg['n_seeds']} seeds, shaded = ±1 std\n"
        f"Exact-DF / MCMC at S={cfg['S_ref']} (horizontal bands)",
        fontsize=10)
    for ax, met in zip(axes.flat, ["kl", "npll", "delta1", "delta2"]):
        ylabel, title = INFO[met]
        for name, sty in STYLE.items():
            data = results[name][met]
            mu, sd = data.mean(axis=0), data.std(axis=0)
            kw = dict(color=sty["color"], ls=sty["ls"], lw=sty["lw"])
            if sty["ms"] > 0:
                kw.update(marker=sty["marker"], markersize=sty["ms"])
            ax.plot(K_values, mu, label=name, **kw)
            ax.fill_between(K_values, mu - sd, mu + sd, color=sty["color"], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("K = S  (path budget = posterior samples)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3, lw=0.6)
    plt.tight_layout()
    plt.savefig(cfg["out_png"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved -> {cfg['out_png']}")


def save_csv(cfg, results):
    import csv
    with open(cfg["out_csv"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "K=S", "mean", "std"])
        for name in METHODS:
            for met in METRICS:
                for ki, K in enumerate(cfg["K_values"]):
                    col = results[name][met][:, ki]
                    w.writerow([name, met, K, f"{col.mean():.6f}", f"{col.std():.6f}"])
    print(f"Table saved  -> {cfg['out_csv']}")


if __name__ == "__main__":
    _, _, results = run_experiment(CONFIG)
    plot_results(CONFIG, results)
    save_csv(CONFIG, results)
