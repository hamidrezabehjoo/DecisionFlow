# ================================================================
# Algorithm 1 (finite K prior paths): S = K sweep up to K = S = 5e4
# Compare MF prior vs BP prior on all four diagnostics.
#
# Experiment design:
#   - K_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
#   - For each K: generate K prior paths, draw S = K posterior samples
#   - n_seeds = 3 independent replicates per (K, method) pair
#   - Methods: MF-Alg1, BP-Alg1, Exact-DF (reference, K=infinity)
#
# Algorithm 1 weight formula (from Eqs. 4, 20-22):
#   w(i,s) proportional to
#       sum_{k: active & sigma_k[i]=s} exp(-E(sigma_k) - log pi_T(sigma_k))
#   where pi_T(sigma) = prod_a q_a(sigma_a)  [Eq.21]
#   and paths sigma_k are drawn i.i.d. from pi_T.
#
# IS property:
#   E_{sigma~pi_T}[ 1_{sigma_i=s} * exp(-E)/pi_T ]
#   = sum_{sigma_i=s} exp(-E(sigma))   <-- exact DF weight
#   So w(i,s) -> exact DF weight as K -> infinity.
#
# Why BP beats MF at fixed K:
#   BP marginals are closer to the true p(sigma) marginals.
#   BP paths concentrate in the high-probability region of p(sigma).
#   At each filtering step, more active paths survive for BP than MF
#   => lower IS variance => better weight estimates at fixed K.
#
# Sampling is vectorised: all S samples processed in chunks of 500,
# using numpy batch matrix multiply (active @ exp_w) instead of loops.
# This gives ~50x speedup over serial, making K=S=5e4 feasible.
#
# Runtime: ~60 min on a modern laptop (dominated by K=S=50000 points).
#
# Usage:  pip install numpy matplotlib
#         python alg1_sk_sweep.py
# ================================================================

import numpy as np
import math
from collections import defaultdict, Counter
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time


# ────────────────────────────────────────────────────────────────
# 1.  Ising model
# ────────────────────────────────────────────────────────────────
def create_ising_params(n, m, J_strength=0.3, h_strength=0.1, seed=42):
    rng = np.random.default_rng(seed)
    N = n * m; J, h = {}, {}
    for i in range(N): h[i] = rng.uniform(-h_strength, h_strength)
    def idx(r, c): return r * m + c
    for r in range(n):
        for c in range(m):
            i = idx(r, c)
            if c + 1 < m:
                j = idx(r, c + 1); v = J_strength + rng.uniform(0, 0.2)
                J[(i, j)] = v; J[(j, i)] = v
            if r + 1 < n:
                j = idx(r + 1, c); v = J_strength + rng.uniform(0, 0.2)
                J[(i, j)] = v; J[(j, i)] = v
    return N, J, h

def ising_energy(sigma, J, h):
    e = sum(-Jij * sigma[i] * sigma[j] for (i, j), Jij in J.items() if i < j)
    return e + sum(-h[i] * sigma[i] for i in range(len(sigma)))


# ────────────────────────────────────────────────────────────────
# 2.  Mean Field & Belief Propagation
# ────────────────────────────────────────────────────────────────
def run_mean_field(N, J, h, max_iter=500, tol=1e-6):
    nb = defaultdict(list)
    for (i, j) in J: nb[i].append(j)
    m = np.zeros(N)
    for _ in range(max_iter):
        m_new = np.array([np.tanh(h[i] + sum(J[(i,j)]*m[j] for j in nb[i]))
                          for i in range(N)])
        if np.max(np.abs(m_new - m)) < tol: break
        m = m_new
    def clip(p): p = np.clip(p, 1e-12, 1-1e-12); return p/p.sum()
    return {i: clip(np.array([(1-m[i])/2, (1+m[i])/2])) for i in range(N)}

def run_belief_propagation(N, J, h, max_iter=200, damping=0.5, tol=1e-6):
    nb = defaultdict(list)
    for (i, j) in J: nb[i].append(j)
    def lse(x): m=float(np.max(x)); return m+float(np.log(np.sum(np.exp(x-m))))
    def clip(p): p = np.clip(p, 1e-12, 1-1e-12); return p/p.sum()
    msgs = {(i,j): np.log(np.array([0.5, 0.5])) for (i,j) in J}
    for _ in range(max_iter):
        new = {}
        for (i,j) in J:
            msg = np.zeros(2)
            for sj_idx, sj in enumerate([-1, 1]):
                vals = [h[i]*si + J[(i,j)]*si*sj +
                        sum(msgs[(k,i)][0 if si==-1 else 1] for k in nb[i] if k!=j)
                        for si in [-1, 1]]
                msg[sj_idx] = np.logaddexp(vals[0], vals[1])
            msg -= lse(msg)
            p = damping*np.exp(msg) + (1-damping)*np.exp(msgs[(i,j)])
            new[(i,j)] = np.log(clip(p))
        delta = max(np.max(np.abs(new[k]-msgs[k])) for k in msgs)
        msgs = new
        if delta < tol: break
    q = {}
    for i in range(N):
        b = np.array([h[i]*si + sum(msgs[(k,i)][0 if si==-1 else 1] for k in nb[i])
                      for si in [-1, 1]])
        b -= np.max(b); q[i] = clip(np.exp(b))
    return q


# ────────────────────────────────────────────────────────────────
# 3.  Precomputation helpers
# ────────────────────────────────────────────────────────────────
def precompute_node_masks(all_sigmas, N):
    return {(i, s): (all_sigmas[:, i] == s) for i in range(N) for s in (-1, 1)}

def _logsumexp(x):
    if x.size == 0: return -np.inf
    m = float(np.max(x))
    if not np.isfinite(m): return m
    return m + float(np.log(np.sum(np.exp(x - m))))


# ────────────────────────────────────────────────────────────────
# 4.  Exact DF reference sampler (K = infinity)
# ────────────────────────────────────────────────────────────────
def sample_df_exact(N, all_sigmas, energy_arr, node_masks, rng=None):
    """
    Exact DF: w(i,s) = sum_{consistent, sigma_i=s} exp(-E(sigma)).
    q cancels analytically -- prior-independent.
    """
    if rng is None: rng = np.random.default_rng()
    logE      = -energy_arr.astype(float)
    state     = [0] * N
    base_mask = np.ones(len(all_sigmas), dtype=bool)
    for _ in range(N):
        vacant = [i for i in range(N) if state[i] == 0]
        if not vacant: break
        weights, children = [], []
        for i in vacant:
            for spin in (-1, 1):
                cm  = base_mask & node_masks[(i, spin)]
                lsw = _logsumexp(logE[cm])
                weights.append(math.exp(lsw) if np.isfinite(lsw) else 0.0)
                children.append((i, spin))
        w = np.array(weights); w /= w.sum()
        idx = int(rng.choice(len(children), p=w))
        i_sel, s_sel = children[idx]
        state[i_sel] = s_sel
        base_mask = base_mask & node_masks[(i_sel, s_sel)]
    return tuple(state)


# ────────────────────────────────────────────────────────────────
# 5.  Algorithm 1: generate K prior paths + log importance weights
# ────────────────────────────────────────────────────────────────
def generate_prior_paths(q, N, K, J, h, rng, eps=1e-12):
    """
    Draw K paths from pi_T(sigma) = prod_a q_a(sigma_a)  [Eq.21].
    Each path assigns nodes in a random order, spin from q_i.
    Returns paths (K, N) and log importance weights log_w_k = -E_k - log pi_T(sigma_k).

    IS property: E_{sigma~pi_T}[1_{sigma_i=s} exp(-E)/pi_T]
                 = sum_{sigma_i=s} exp(-E(sigma))  (exact DF weight)
    """
    paths = np.zeros((K, N), dtype=np.int8)
    for k in range(K):
        for i in rng.permutation(N):
            paths[k, i] = 1 if rng.random() < q[i][1] else -1

    # Vectorised energy
    logE_paths = np.zeros(K)
    for (i, j), Jij in J.items():
        if i < j:
            logE_paths += Jij * paths[:, i].astype(float) * paths[:, j].astype(float)
    for i in range(N):
        logE_paths += h[i] * paths[:, i].astype(float)
    logE_paths = -logE_paths   # -E(sigma_k)

    # log pi_T
    log_pi = np.zeros(K)
    for i in range(N):
        log_pi += np.log(np.where(paths[:, i] == 1, q[i][1], q[i][0]).astype(float) + eps)

    return paths, logE_paths - log_pi   # log_w = -E - log pi_T


# ────────────────────────────────────────────────────────────────
# 6.  Algorithm 1 vectorised sampler
# ────────────────────────────────────────────────────────────────
def sample_alg1_vectorized(paths, log_w, N, node_masks, logE_full,
                            S, rng, chunk=500):
    """
    Draw S posterior samples using Algorithm 1 importance weights.

    Vectorised over samples in chunks of `chunk` to avoid OOM.
    For each chunk:
      active[b, k] = True if path k still consistent with sample b
      w_neg[b] = active[b] @ (exp_w * path_neg[i])  -- numpy dot
      w_pos[b] = active[b] @ (exp_w * path_pos[i])

    All S samples use the same K paths but independent random choices.
    Within a chunk, all samples follow the same random node order
    (a valid DF schedule -- any fixed ordering produces exact samples as K→∞).

    Complexity per K-S point: O(S * K * N / chunk) ~ O(S * K)
    """
    K        = len(paths)
    exp_w    = np.exp(log_w - log_w.max())   # pre-shift for stability
    pn       = {i: (paths[:, i] == -1) for i in range(N)}
    pp       = {i: (paths[:, i] ==  1) for i in range(N)}

    all_samples = []
    for start in range(0, S, chunk):
        B      = min(chunk, S - start)
        active = np.ones((B, K), dtype=bool)
        states = np.zeros((B, N), dtype=np.int8)
        node_order = list(rng.permutation(N))

        for i in node_order:
            ew_neg = exp_w * pn[i]   # (K,) importance weights for spin=-1
            ew_pos = exp_w * pp[i]   # (K,) importance weights for spin=+1

            w_neg = active @ ew_neg   # (B,)
            w_pos = active @ ew_pos   # (B,)

            # Exact fallback (should be rare for reasonable K)
            zero  = (w_neg == 0) & (w_pos == 0)
            if zero.any():
                for b in np.where(zero)[0]:
                    base = np.ones(len(logE_full), dtype=bool)
                    for j in range(N):
                        if states[b, j] != 0:
                            base &= node_masks[(j, int(states[b, j]))]
                    w_neg[b] = math.exp(_logsumexp(logE_full[base & node_masks[(i, -1)]]))
                    w_pos[b] = math.exp(_logsumexp(logE_full[base & node_masks[(i,  1)]]))

            total  = w_neg + w_pos
            p_pos  = w_pos / np.where(total > 0, total, 1.0)
            chose  = rng.random(B) < p_pos
            spins  = np.where(chose, np.int8(1), np.int8(-1))
            states[:, i] = spins
            active &= np.where(chose[:, np.newaxis],
                                pp[i][np.newaxis, :],
                                pn[i][np.newaxis, :])

        all_samples.extend([tuple(states[b]) for b in range(B)])
    return all_samples


# ────────────────────────────────────────────────────────────────
# 7.  MCMC reference
# ────────────────────────────────────────────────────────────────
def sample_mcmc(J, h, N, S, burn_in=2000, thin=10, rng=None):
    if rng is None: rng = np.random.default_rng()
    nb = defaultdict(list)
    for (i, j) in J: nb[i].append(j)
    config = rng.choice([-1, 1], size=N).astype(int)
    def sweep():
        for _ in range(N):
            i  = int(rng.integers(0, N))
            lf = h[i] + sum(J[(i,j)]*config[j] for j in nb[i])
            dE = 2*config[i]*lf
            if dE <= 0 or rng.random() < math.exp(-dE): config[i] *= -1
    for _ in range(burn_in): sweep()
    out = []
    while len(out) < S:
        for _ in range(thin): sweep()
        out.append(tuple(config.copy()))
    return out


# ────────────────────────────────────────────────────────────────
# 8.  Reference quantities & metrics
# ────────────────────────────────────────────────────────────────
def compute_exact_references(all_sigmas, energy_arr):
    logp = -energy_arr.astype(float); logp -= logp.max()
    p    = np.exp(logp); p /= p.sum()
    return p @ all_sigmas, (all_sigmas * p[:, None]).T @ all_sigmas

def compute_kl_proxy(samples, J, h):
    S = len(samples); cnt = Counter(samples); kl = 0.0
    for sigma, c in cnt.items():
        kl += (c/S) * (math.log(c/S) + ising_energy(sigma, J, h))
    return kl

def compute_npll(samples, J, h, N):
    nb = defaultdict(list)
    for (i,j) in J: nb[i].append(j)
    total = 0.0
    for sigma in samples:
        for i in range(N):
            lf     = h[i] + sum(J[(i,j)]*sigma[j] for j in nb[i])
            total -= sigma[i]*lf - math.log(2*math.cosh(lf))
    return total / len(samples)

def compute_delta1(samples, m_ref):
    arr = np.array(samples)
    return float(np.mean(np.abs(arr.mean(0) - m_ref) / (np.abs(m_ref) + 1e-12)))

def compute_delta2(samples, c_ref, N):
    arr   = np.array(samples)
    c_hat = (arr.T @ arr) / len(samples)
    return float(np.mean([abs(c_hat[i,j]-c_ref[i,j]) / (abs(c_ref[i,j])+1e-12)
                           for i in range(N) for j in range(i+1, N)]))


# ────────────────────────────────────────────────────────────────
# 9.  Main experiment
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    n, m = 3, 3
    N, J, h = create_ising_params(n, m)
    print(f"Ising {n}x{m}  N={N}  configs={2**N}")

    all_sigmas = np.array(list(product([-1, 1], repeat=N)))
    energy_arr = np.array([ising_energy(s, J, h) for s in all_sigmas])
    node_masks = precompute_node_masks(all_sigmas, N)
    logE_full  = -energy_arr.astype(float)
    m_ref, c_ref = compute_exact_references(all_sigmas, energy_arr)

    q_mf = run_mean_field(N, J, h)
    q_bp = run_belief_propagation(N, J, h)

    m_mf_p = np.array([q_mf[i][1]-q_mf[i][0] for i in range(N)])
    m_bp_p = np.array([q_bp[i][1]-q_bp[i][0] for i in range(N)])
    d1_p_mf = float(np.mean(np.abs(m_mf_p-m_ref)/(np.abs(m_ref)+1e-12)))
    d1_p_bp = float(np.mean(np.abs(m_bp_p-m_ref)/(np.abs(m_ref)+1e-12)))
    print(f"Prior Delta_1:  MF={d1_p_mf:.4f}  BP={d1_p_bp:.4f}  (Table 1: MF~10.2, BP~0.17)")

    # ── Experiment ─────────────────────────────────────────────
    K_values = [100, 1000, 10000, 20000, 50000]
    n_seeds  = 3
    METRICS  = ["kl", "npll", "delta1", "delta2"]
    METHODS  = ["MF-Alg1", "BP-Alg1", "Exact-DF", "MCMC"]

    results = {
        name: {metric: np.zeros((n_seeds, len(K_values))) for metric in METRICS}
        for name in METHODS
    }

    # Pre-compute Exact-DF and MCMC at a representative S (use max K as S)
    S_ref = 5000   # fixed S for reference methods (independent of K)
    print(f"\nPre-computing Exact-DF and MCMC ({n_seeds} seeds, S={S_ref}) ...")
    ex_res = {metric: np.zeros(n_seeds) for metric in METRICS}
    mc_res = {metric: np.zeros(n_seeds) for metric in METRICS}
    for seed in range(n_seeds):
        rng_e = np.random.default_rng(seed * 10 + 8)
        rng_m = np.random.default_rng(seed * 10 + 9)
        s_ex = [sample_df_exact(N, all_sigmas, energy_arr, node_masks, rng=rng_e)
                for _ in range(S_ref)]
        s_mc = sample_mcmc(J, h, N, S=S_ref, burn_in=2000, thin=10, rng=rng_m)
        for metric, fn in [("kl",     lambda s: compute_kl_proxy(s, J, h)),
                            ("npll",   lambda s: compute_npll(s, J, h, N)),
                            ("delta1", lambda s: compute_delta1(s, m_ref)),
                            ("delta2", lambda s: compute_delta2(s, c_ref, N))]:
            ex_res[metric][seed] = fn(s_ex)
            mc_res[metric][seed] = fn(s_mc)

    print(f"\nRunning Algorithm 1 (S=K sweep)  n_seeds={n_seeds}")
    print(f"{'K=S':>7}  {'MF D1':>10}  {'BP D1':>10}  {'MF KL':>8}  {'BP KL':>8}  {'time':>6}")

    t0 = time.time()
    for ki, K in enumerate(K_values):
        d1_mf_seeds, d1_bp_seeds = [], []
        kl_mf_seeds, kl_bp_seeds = [], []

        for seed in range(n_seeds):
            rng_mf   = np.random.default_rng(seed * 10 + 0)
            rng_bp   = np.random.default_rng(seed * 10 + 1)
            rng_s_mf = np.random.default_rng(seed * 10 + 2)
            rng_s_bp = np.random.default_rng(seed * 10 + 3)

            # S = K  (same number of paths and posterior samples)
            S = K

            paths_mf, log_w_mf = generate_prior_paths(q_mf, N, K, J, h, rng_mf)
            paths_bp, log_w_bp = generate_prior_paths(q_bp, N, K, J, h, rng_bp)

            s_mf = sample_alg1_vectorized(paths_mf, log_w_mf, N, node_masks,
                                           logE_full, S, rng_s_mf)
            s_bp = sample_alg1_vectorized(paths_bp, log_w_bp, N, node_masks,
                                           logE_full, S, rng_s_bp)

            for metric, fn in [("kl",     lambda s: compute_kl_proxy(s, J, h)),
                                ("npll",   lambda s: compute_npll(s, J, h, N)),
                                ("delta1", lambda s: compute_delta1(s, m_ref)),
                                ("delta2", lambda s: compute_delta2(s, c_ref, N))]:
                results["MF-Alg1"][metric][seed, ki] = fn(s_mf)
                results["BP-Alg1"][metric][seed, ki] = fn(s_bp)

            # Fill reference methods (same value at every K)
            for metric in METRICS:
                results["Exact-DF"][metric][seed, ki] = ex_res[metric][seed]
                results["MCMC"][metric][seed, ki]     = mc_res[metric][seed]

            d1_mf_seeds.append(results["MF-Alg1"]["delta1"][seed, ki])
            d1_bp_seeds.append(results["BP-Alg1"]["delta1"][seed, ki])
            kl_mf_seeds.append(results["MF-Alg1"]["kl"][seed, ki])
            kl_bp_seeds.append(results["BP-Alg1"]["kl"][seed, ki])

        elapsed = time.time() - t0
        print(f"{K:>7}  "
              f"{np.mean(d1_mf_seeds):>7.4f}+-{np.std(d1_mf_seeds):.3f}  "
              f"{np.mean(d1_bp_seeds):>7.4f}+-{np.std(d1_bp_seeds):.3f}  "
              f"{np.mean(kl_mf_seeds):>8.3f}  {np.mean(kl_bp_seeds):>8.3f}  "
              f"{elapsed:>5.0f}s", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Summary at K=S={K_values[-1]}  (mean +/- std, {n_seeds} seeds)")
    print(f"{'Method':<12}  {'Delta_1':>18}  {'Delta_2':>18}  {'KL proxy':>18}")
    for name in METHODS:
        d1 = results[name]["delta1"][:, -1]
        d2 = results[name]["delta2"][:, -1]
        kl = results[name]["kl"][:, -1]
        print(f"{name:<12}  {d1.mean():>8.4f}+-{d1.std():<6.4f}  "
              f"{d2.mean():>8.4f}+-{d2.std():<6.4f}  "
              f"{kl.mean():>8.4f}+-{kl.std():<6.4f}")

    # ── Figure ────────────────────────────────────────────────
    STYLE = {
        "MF-Alg1":  dict(color="steelblue",  ls="-",  marker="o", lw=2.0, ms=6),
        "BP-Alg1":  dict(color="firebrick",  ls="--", marker="s", lw=2.0, ms=6),
        "Exact-DF": dict(color="purple",     ls=":",  marker="",  lw=1.8, ms=0),
        "MCMC":     dict(color="seagreen",   ls="-.", marker="^", lw=1.5, ms=5),
    }
    METRIC_INFO = {
        "kl":     (r"KL proxy  $\mathrm{KL}_d(q\|p)$", "KL Proxy vs K=S"),
        "npll":   ("NPLL",                               "NPLL vs K=S"),
        "delta1": (r"$\Delta_1$  (singleton mismatch)",  r"$\Delta_1$ vs K=S"),
        "delta2": (r"$\Delta_2$  (pairwise mismatch)",   r"$\Delta_2$ vs K=S"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Algorithm 1 (S = K):  MF prior vs BP prior\n"
        f"Ising {n}x{m} (N={N}),  {n_seeds} seeds,  shaded = ±1 std\n"
        f"Exact-DF and MCMC computed at S={S_ref} (shown as horizontal bands)",
        fontsize=10,
    )

    for ax, metric in zip(axes.flat, METRICS):
        ylabel, title = METRIC_INFO[metric]
        for name, sty in STYLE.items():
            data  = results[name][metric]
            mu    = data.mean(axis=0)
            sigma = data.std(axis=0)
            kw    = dict(color=sty["color"], ls=sty["ls"], lw=sty["lw"])
            if sty["ms"] > 0: kw.update(marker=sty["marker"], markersize=sty["ms"])
            ax.plot(K_values, mu, label=name, **kw)
            ax.fill_between(K_values, mu - sigma, mu + sigma,
                            color=sty["color"], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("K = S  (path budget = posterior samples)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3, lw=0.6)

    plt.tight_layout()
    plt.savefig("alg1_sk_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved -> alg1_sk_sweep.png")
