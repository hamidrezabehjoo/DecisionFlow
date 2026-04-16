import numpy as np


def mh_sample_ising(
    J,
    h=None,
    N=None,
    n_samples=1000,
    burn_in=5000,
    thin=1,
    seed=42,
    init_sigma=None,
):
    """Metropolis–Hastings sampler for an Ising model (single-spin flips).

    Target: p(σ) ∝ exp(-E(σ)), with
      E(σ) = -0.5 σ^T J σ - h^T σ

    Args:
      J: (N,N) symmetric coupling matrix, diagonal assumed 0.
      h: (N,) local fields (optional).
      N: number of spins (optional; inferred from J if omitted).
      n_samples: number of returned samples.
      burn_in: number of MH steps before collecting.
      thin: collect one sample every `thin` MH steps.
      seed: RNG seed.
      init_sigma: optional initial configuration in {-1,+1}^N.

    Returns:
      List[tuple[int]] of length `n_samples`.
    """
    J = np.asarray(J, dtype=np.float64)
    if N is None:
        N = int(J.shape[0])
    h_vec = np.zeros(N, dtype=np.float64) if h is None else np.asarray(h, dtype=np.float64)

    rng = np.random.default_rng(seed)
    if init_sigma is None:
        sigma = rng.choice([-1, 1], size=N).astype(np.int8)
    else:
        sigma = np.asarray(init_sigma, dtype=np.int8).copy()
        if sigma.shape != (N,):
            raise ValueError(f"init_sigma must have shape {(N,)}, got {sigma.shape}")

    # Local field at each site: f_i = h_i + Σ_j J_ij σ_j
    field = h_vec + J @ sigma

    def mh_step():
        nonlocal sigma, field
        i = int(rng.integers(0, N))
        s_i = sigma[i]

        # For flipping σ_i -> -σ_i:
        # ΔE = E(new)-E(old) = 2 σ_i (h_i + Σ_{j≠i} J_ij σ_j)
        # With diagonal 0, field[i] = h_i + Σ_j J_ij σ_j = h_i + Σ_{j≠i} ...
        dE = 2.0 * float(s_i) * float(field[i])

        if dE <= 0.0 or rng.random() < np.exp(-dE):
            # accept: update sigma and all fields efficiently
            sigma[i] = -s_i
            # field_k = h_k + Σ_j J_kj σ_j, only σ_i changed:
            # new_field = old_field + J[:, i] * (σ_i_new - σ_i_old)
            delta = sigma[i] - s_i  # equals -2*s_i
            field = field + J[:, i] * float(delta)

    # burn-in
    for _ in range(int(burn_in)):
        mh_step()

    samples = []
    steps_per_sample = max(1, int(thin))
    while len(samples) < int(n_samples):
        for _ in range(steps_per_sample):
            mh_step()
        samples.append(tuple(int(x) for x in sigma))

    return samples

