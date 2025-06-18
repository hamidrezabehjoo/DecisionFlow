import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------------
# Helper functions (from det-prob.py)
# -------------------------------
def create_ising_params(n, m, det_J=0, ran_J=1, det_h=1, ran_h=0, seed=0):
    rng = np.random.default_rng(seed)
    N = n * m
    J = {}
    h = {}
    for i in range(N):
        h[i] = det_h + ran_h * rng.uniform(-1, 1)
    def idx(rr, cc):
        return rr * m + cc
    for rr in range(n):
        for cc in range(m):
            i = idx(rr, cc)
            if cc + 1 < m:
                j = idx(rr, cc + 1)
                val = det_J + ran_J * rng.uniform(-1, 1)
                J[(i, j)] = val
                J[(j, i)] = val
            if rr + 1 < n:
                j = idx(rr + 1, cc)
                val = det_J + ran_J * rng.uniform(-1, 1)
                J[(i, j)] = val
                J[(j, i)] = val
    return N, J, h

def ising_energy(config, J, h):
    # config is a dictionary: {site: spin} with spins ±1.
    e_pair = 0.0
    for (i, j), Jij in J.items():
        e_pair += -0.5 * Jij * config[i] * config[j]
    e_single = 0.0
    for i, hi in h.items():
        e_single += -hi * config[i]
    return e_pair + e_single

def state_to_dict(state):
    """Convert state tuple (of length N with entries in {0, ±1})
       to a dictionary {i: spin} for nonzero sites."""
    return {i: spin for i, spin in enumerate(state) if spin != 0}

def count_assigned(state):
    return sum(1 for x in state if x != 0)

def is_consistent(partial, full):
    # For every nonzero entry in partial, it must match full.
    return all(partial[i] == full[i] for i in range(len(partial)) if partial[i] != 0)

def next_states(state, J, h, N):
    """
    Given a partial state (tuple of length N with entries in {0, +1, -1}),
    return list of (child_state, probability) for each possible assignment at one vacant site.
    The rule: probability ∝ exp(candidate * (h[i] + sum_{j assigned} J_ij * spin_j)).
    """
    vacant_indices = [i for i in range(N) if state[i] == 0]
    if not vacant_indices:
        return []
    choices = []
    for i in vacant_indices:
        # Compute local field: h[i] + sum_{j assigned and (i,j) in J} J[(i,j)]*spin_j
        lf = h[i]
        config = state_to_dict(state)
        for j, spin in config.items():
            if (i, j) in J:
                lf += J[(i, j)] * spin
        for candidate in [1, -1]:
            score = math.exp(candidate * lf)
            new_state = list(state)
            new_state[i] = candidate
            choices.append((tuple(new_state), score))
    total = sum(score for (_, score) in choices)
    return [(s, score/total) for (s, score) in choices]

def enumerate_states(N, J, h):
    """
    Enumerate all full states (with all spins assigned) level-by-level.
    Returns: states_by_level: dict mapping level t to dict {state: probability}.
    """
    states_by_level = dict()
    init = (0,) * N
    states_by_level[0] = {init: 1.0}
    T = N  # number of spins = number of levels
    for t in range(T):
        states_by_level[t+1] = dict()
        for state, prob in states_by_level[t].items():
            for child, p_trans in next_states(state, J, h, N):
                states_by_level[t+1][child] = states_by_level[t+1].get(child, 0.0) + prob * p_trans
    return states_by_level

# -------------------------------
# Empirical DF functions (Appendix C)
# -------------------------------
def sample_prior_path(state, J, h, N):
    """Generate one full path (sequence of states) from empty to full using prior transitions."""
    path = [state]
    while count_assigned(state) < N:
        children = next_states(state, J, h, N)
        if not children:
            break
        probs = [p for (_, p) in children]
        chosen = np.random.choice(len(children), p=probs)
        child = children[chosen][0]
        path.append(child)
        state = child
    return path

def build_empirical_counts(prior_paths, T):
    """
    From a list of sample paths, build empirical counts for transitions and states at each level.
    Returns:
       trans_counts: dict keyed by (t, state, child) with counts.
       state_counts: dict keyed by (t, state) with counts.
    """
    trans_counts = defaultdict(int)
    state_counts = defaultdict(int)
    for path in prior_paths:
        for t, s in enumerate(path):
            state_counts[(t, s)] += 1
        for t in range(len(path)-1):
            s = path[t]
            child = path[t+1]
            trans_counts[(t, s, child)] += 1
    return trans_counts, state_counts

def get_empirical_transition_probs(trans_counts, state_counts, T):
    """
    Compute p^(prior-emp)_t(child | state) for each level t.
    Returns dict keyed by (t, state, child).
    """
    p_emp = {}
    for (t, s, child), cnt in trans_counts.items():
        total = state_counts[(t, s)]
        p_emp[(t, s, child)] = cnt / total
    return p_emp

def get_empirical_marginals(state_counts, K, T):
    """
    Compute π^(prior-emp)_t(state) = (# of hits at level t)/K.
    Returns dict keyed by (t, state).
    """
    pi_emp = {}
    for (t, s), cnt in state_counts.items():
        pi_emp[(t, s)] = cnt / K
    return pi_emp

def compute_empirical_green(T, p_emp, state_counts):
    """
    Compute the empirical Green functions G_emp(t, s, s_full) recursively.
    Only for observed states.
    Boundary: for each full state s_full at level T, set G_emp(T, s_full, s_full)=1.
    """
    G_emp = {}
    full_states = [s for (t, s) in state_counts if t == T]
    for s_full in full_states:
        G_emp[(T, s_full, s_full)] = 1.0
    for t in range(T-1, -1, -1):
        current_states = [s for (tt, s) in state_counts if tt == t]
        for s in current_states:
            for s_full in full_states:
                if not is_consistent(s, s_full):
                    continue
                total = 0.0
                for key in p_emp:
                    tt, s_key, child = key
                    if tt == t and s_key == s and is_consistent(child, s_full):
                        total += p_emp[key] * G_emp.get((t+1, child, s_full), 0.0)
                G_emp[(t, s, s_full)] = total
    return G_emp

def compute_multiplicative_factors(T, G_emp, pi_emp):
    """
    For each level t=1,...,T-1 and state s, compute:
       Υ_t(s) = sum_{s_full consistent with s} [ exp(-E(s_full)) * G_emp(t, s, s_full) / π_emp(T, s_full) ]
    Returns a dict keyed by (t, s).
    """
    Y = {}
    full_states = [s for (t_key, s) in pi_emp if t_key == T]
    for (t, s) in [(tt, s) for (tt, s) in pi_emp if 1 <= tt < T]:
        val = 0.0
        for (tt, s_full) in pi_emp:
            if tt == T and is_consistent(s, s_full):
                config = state_to_dict(s_full)
                weight = math.exp(-ising_energy(config, J, h))
                val += weight * G_emp.get((t, s, s_full), 0.0) / (pi_emp[(T, s_full)] + 1e-12)
        Y[(t, s)] = val
    return Y

def compute_optimal_empirical_transitions(T, p_emp, pi_emp, Y):
    """
    Compute optimal empirical transitions p^(*,emp)_t(s_next|s):
      For t = 0,...,T-2:
         ∝ p_emp(t, s, s_next) * Y[(t+1, s_next)]
      For t = T-1:
         ∝ p_emp(T-1, s, s_full) * (exp(-E(s_full)) / π_emp(T, s_full))
    Normalizes over children for each (t, s).
    Returns a dict keyed by (t, s, s_next).
    """
    p_opt = {}
    for key in p_emp:
        t, s, s_next = key
        if t < T-1:
            if (t+1, s_next) in Y:
                p_opt[key] = p_emp[key] * Y[(t+1, s_next)]
        elif t == T-1:
            config = state_to_dict(s_next)
            weight = math.exp(-ising_energy(config, J, h))
            p_opt[key] = p_emp[key] * (weight / (pi_emp.get((T, s_next), 1e-12)))
    p_opt_norm = {}
    groups = defaultdict(list)
    for (t, s, s_next), prob in p_opt.items():
        groups[(t, s)].append(((t, s, s_next), prob))
    for (t, s), items in groups.items():
        total = sum(prob for (_, prob) in items)
        for key, prob in items:
            p_opt_norm[key] = prob / (total + 1e-12)
    return p_opt_norm

def sample_posterior_path(state, T, p_opt_norm):
    """
    Starting from a given state at level 0, sample a full path using optimal empirical transitions.
    """
    path = [state]
    t = 0
    current = state
    while t < T:
        candidates = []
        probs = []
        for key, prob in p_opt_norm.items():
            tt, s, s_next = key
            if tt == t and s == current:
                candidates.append(s_next)
                probs.append(prob)
        if not candidates:
            # Fall back to prior transitions if none are found.
            children = next_states(current, J, h, T)
            if not children:
                break
            probs = [p for (_, p) in children]
            candidates = [child for (child, _) in children]
        chosen = np.random.choice(len(candidates), p=probs)
        next_state = candidates[chosen]
        path.append(next_state)
        current = next_state
        t += 1
    return path

def compute_correlations(samples, N):
    """
    Given a list of full state samples (tuples of length N with spins ±1),
    compute singleton magnetizations and pairwise correlations.
    Returns:
       m: dict mapping site index to average spin.
       c: dict mapping (i, j) with i<j to average product.
    """
    m = {i: 0.0 for i in range(N)}
    c = {}
    for i in range(N):
        for j in range(i+1, N):
            c[(i,j)] = 0.0
    S = len(samples)
    for state in samples:
        for i in range(N):
            m[i] += state[i]
        for i in range(N):
            for j in range(i+1, N):
                c[(i,j)] += state[i] * state[j]
    for i in m:
        m[i] /= S
    for key in c:
        c[key] /= S
    return m, c

def compute_exact_correlations(N, J, h):
    """
    Compute exact magnetizations and pairwise correlations via full enumeration.
    """
    states_by_level = enumerate_states(N, J, h)
    full_states = states_by_level[N]
    boltzmann = {}
    for state in full_states:
        config = {i: state[i] for i in range(N)}
        boltzmann[state] = math.exp(-ising_energy(config, J, h))
    Z = sum(boltzmann.values())
    P_target = {state: boltzmann[state]/Z for state in boltzmann}
    m = {i: 0.0 for i in range(N)}
    c = {}
    for i in range(N):
        for j in range(i+1, N):
            c[(i,j)] = 0.0
    for state, p in P_target.items():
        for i in range(N):
            m[i] += state[i] * p
        for i in range(N):
            for j in range(i+1, N):
                c[(i,j)] += state[i] * state[j] * p
    return m, c

# -------------------------------
# MCMC functions (Metropolis-Hastings)
# -------------------------------
N_MCMC_SAMPLES = 5000
MCMC_BURNIN = 2000
MCMC_THIN = 10
SEED = 123  # MCMC seed

def compute_dE(config, i, new_val, J, h):
    old_config = config.copy()
    old_energy = ising_energy(old_config, J, h)
    new_config = config.copy()
    new_config[i] = new_val
    new_energy = ising_energy(new_config, J, h)
    return new_energy - old_energy

def metropolis_hastings_ising(n, N, J, h,
                              n_samples=N_MCMC_SAMPLES,
                              burnin=MCMC_BURNIN,
                              thin=MCMC_THIN,
                              seed=SEED):
    rng = np.random.default_rng(seed)
    config = rng.choice([-1, 1], size=N)  # start from a random configuration
    samples = []
    total_steps = burnin + n_samples * thin
    for step in range(total_steps):
        i = rng.integers(0, N)
        old_val = config[i]
        new_val = -old_val  # flip the spin
        dE = compute_dE(config, i, new_val, J, h)
        if dE <= 0 or rng.random() < np.exp(-dE):
            config[i] = new_val  # accept move
        if step >= burnin and (step - burnin) % thin == 0:
            samples.append(config.copy())
    return samples

# -------------------------------
# Visualization and Analysis
# -------------------------------
def plot_energy_distribution(post, mcmc, J, h):
    """Plot energy distributions with detailed comparison metrics, formatted for IEEE two-column papers"""
    # IEEE column width is approximately 3.5 inches
    plt.figure(figsize=(3.5, 2.8))
    
    # Use IEEE-friendly font
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'legend.frameon': True,
        'legend.loc': 'upper right',
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'lightgray',
        'lines.linewidth': 1.5
    })
    
    # Convert to energies for visualization
    post_e = np.array([ising_energy(state_to_dict(s), J, h) for s in post])
    mcmc_e = np.array([ising_energy(state_to_dict(s), J, h) for s in mcmc])
    
    # Calculate histogram data with the same bins for both distributions
    bins = 10  # Reduced number of bins for clearer visualization
    hist_range = (min(min(post_e), min(mcmc_e)), max(max(post_e), max(mcmc_e)))
    
    # Create histograms
    post_hist, bin_edges = np.histogram(post_e, bins=bins, range=hist_range, density=True)
    mcmc_hist, _ = np.histogram(mcmc_e, bins=bins, range=hist_range, density=True)
    
    # Calculate KL divergence (approximation using histograms)
    def kl_divergence(p, q):
        # Add small epsilon to avoid division by zero or log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))
    
    kl_df_to_mcmc = kl_divergence(post_hist, mcmc_hist)
    kl_mcmc_to_df = kl_divergence(mcmc_hist, post_hist)
    
    # Calculate mean and standard deviation
    post_mean, post_std = np.mean(post_e), np.std(post_e)
    mcmc_mean, mcmc_std = np.mean(mcmc_e), np.std(mcmc_e)
    
    # Plot histograms with IEEE-friendly styling
    plt.hist(post_e, bins=bins, alpha=0.7, label=f'DF Posterior (μ={post_mean:.2f}, σ={post_std:.2f})', 
             density=True, histtype='step', linewidth=1.5, color='#1f77b4')
    plt.hist(mcmc_e, bins=bins, alpha=0.7, label=f'MCMC (μ={mcmc_mean:.2f}, σ={mcmc_std:.2f})', 
             density=True, histtype='step', linewidth=1.5, color='#ff7f0e')
    
    plt.xlabel("Energy")
    plt.ylabel("Probability Density")
    plt.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Add legend with a light background and border
    plt.legend(loc='best', fontsize=6.5, frameon=True, framealpha=0.9, edgecolor='lightgray')
    
    # Tighten layout and adjust margins for IEEE format
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    # Save with high DPI for print quality
    #plt.savefig("energy_distribution_comparison.png", dpi=600, bbox_inches='tight')
    plt.savefig("energy_distribution_comparison.pdf", bbox_inches='tight')
    plt.show()
    
    # Print additional statistics
    print("\nDistribution Comparison Statistics:")
    print(f"DF Posterior: Mean={post_mean:.4f}, Std={post_std:.4f}")
    print(f"MCMC: Mean={mcmc_mean:.4f}, Std={mcmc_std:.4f}")
    print(f"Mean Difference: {post_mean - mcmc_mean:.4f}")
    print(f"KL(DF||MCMC): {kl_df_to_mcmc:.4f}")
    print(f"KL(MCMC||DF): {kl_mcmc_to_df:.4f}")
    
    # Calculate and print energy percentiles
    percentiles = [5, 25, 50, 75, 95]
    post_percentiles = np.percentile(post_e, percentiles)
    mcmc_percentiles = np.percentile(mcmc_e, percentiles)
    
    print("\nEnergy Percentiles:")
    print(f"{'Percentile':>10} | {'DF Posterior':>12} | {'MCMC':>12} | {'Difference':>12}")
    print(f"{'-'*10} | {'-'*12} | {'-'*12} | {'-'*12}")
    for i, p in enumerate(percentiles):
        print(f"{p:>10}% | {post_percentiles[i]:>12.4f} | {mcmc_percentiles[i]:>12.4f} | {post_percentiles[i]-mcmc_percentiles[i]:>12.4f}")
    
    return post_mean, post_std, mcmc_mean, mcmc_std, kl_df_to_mcmc, kl_mcmc_to_df

def calculate_exact_boltzmann(N, J, h, T=1.0):
    """
    Calculate the exact Boltzmann distribution for an N×N Ising model.
    
    Parameters:
    -----------
    N : int
        Size of the Ising lattice (N×N)
    J : float
        Coupling constant
    h : float
        External field
    T : float
        Temperature (default=1.0)
    
    Returns:
    --------
    states : list
        List of all possible states
    energies : numpy.ndarray
        Energy of each state
    probabilities : numpy.ndarray
        Boltzmann probability of each state
    """
    if N > 3:
        print(f"Warning: Calculating exact distribution for {N}×{N} lattice may be computationally expensive.")
        
    # Generate all possible states for an N×N lattice
    n_sites = N
    n_states = 2**n_sites
    states = []
    energies = np.zeros(n_states)
    
    # Create all possible states
    for i in range(n_states):
        # Convert integer to binary representation
        binary = format(i, f'0{n_sites}b')
        # Convert binary to spin state (-1 or 1)
        state = np.array([1 if bit == '1' else -1 for bit in binary])
        # Create state dictionary
        state_dict = {}
        for idx, s in enumerate(state):
            state_dict[idx] = s
        
        # Calculate energy
        energy = ising_energy(state_dict, J, h)
        
        # Store state and energy
        states.append(state_dict)
        energies[i] = energy
    
    # Calculate Boltzmann probabilities
    boltzmann_factors = np.exp(-energies / T)
    partition_function = np.sum(boltzmann_factors)
    probabilities = boltzmann_factors / partition_function
    
    return states, energies, probabilities

def plot_energy_distribution_with_exact(post, N, J, h, mcmc=None):
    """
    Plot energy distribution of DF posterior compared to exact Boltzmann distribution and MCMC,
    formatted for IEEE two-column papers.
    
    Parameters:
    -----------
    post : list
        List of DF posterior samples
    N : int
        Size of the Ising lattice (N×N)
    J : float
        Coupling constant
    h : float
        External field
    mcmc : list, optional
        List of MCMC samples (if provided, will be included in the comparison)
    """
    # IEEE column width is approximately 3.5 inches
    plt.figure(figsize=(3.5, 2.8))
    
    # Use IEEE-friendly font
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'legend.frameon': True,
        'legend.loc': 'upper right',
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'lightgray',
        'lines.linewidth': 1.5
    })
    
    # Convert DF samples to energies
    post_e = np.array([ising_energy(state_to_dict(s), J, h) for s in post])
    
    # Convert MCMC samples to energies if provided
    if mcmc is not None:
        mcmc_e = np.array([ising_energy(state_to_dict(s), J, h) for s in mcmc])
    
    # Calculate exact Boltzmann distribution
    print(f"Calculating exact Boltzmann distribution for {N}×{N} Ising model...")
    exact_states, exact_energies, exact_probs = calculate_exact_boltzmann(N, J, h)
    
    # Calculate mean and standard deviation
    post_mean, post_std = np.mean(post_e), np.std(post_e)
    exact_mean = np.sum(exact_energies * exact_probs)
    exact_std = np.sqrt(np.sum(exact_probs * (exact_energies - exact_mean)**2))
    
    if mcmc is not None:
        mcmc_mean, mcmc_std = np.mean(mcmc_e), np.std(mcmc_e)
    
    # Calculate histogram data for DF posterior
    bins = 15  # Reduced for clearer visualization with 3 distributions
    
    # Determine range based on all available data
    if mcmc is not None:
        hist_range = (
            min(min(post_e), min(exact_energies), min(mcmc_e)), 
            max(max(post_e), max(exact_energies), max(mcmc_e))
        )
    else:
        hist_range = (min(min(post_e), min(exact_energies)), max(max(post_e), max(exact_energies)))
    
    # Create histograms
    post_hist, bin_edges = np.histogram(post_e, bins=bins, range=hist_range, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # For exact distribution, we need to bin the energies
    exact_hist, _ = np.histogram(exact_energies, bins=bins, range=hist_range, weights=exact_probs, density=False)
    # Normalize to make it a proper density
    exact_hist = exact_hist / np.sum(exact_hist * (bin_edges[1] - bin_edges[0]))
    
    if mcmc is not None:
        mcmc_hist, _ = np.histogram(mcmc_e, bins=bins, range=hist_range, density=True)
    
    # Calculate KL divergence
    def kl_divergence(p, q):
        # Add small epsilon to avoid division by zero or log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))
    
    kl_df_to_exact = kl_divergence(post_hist, exact_hist)
    if mcmc is not None:
        kl_mcmc_to_exact = kl_divergence(mcmc_hist, exact_hist)
        kl_df_to_mcmc = kl_divergence(post_hist, mcmc_hist)
    
    # Plot histograms with IEEE-friendly styling
    plt.hist(post_e, bins=bins, alpha=0.7, label=f'DF (μ={post_mean:.2f}, σ={post_std:.2f})', 
             density=True, histtype='step', linewidth=1.5, color='#1f77b4')
    
    # Plot exact distribution as a step function
    plt.step(bin_centers, exact_hist, where='mid', color='#d62728', linewidth=1.5, 
             label=f'Exact (μ={exact_mean:.2f}, σ={exact_std:.2f})', linestyle='-')
    
    # Plot MCMC if provided
    if mcmc is not None:
        plt.hist(mcmc_e, bins=bins, alpha=0.7, label=f'MCMC (μ={mcmc_mean:.2f}, σ={mcmc_std:.2f})', 
                 density=True, histtype='step', linewidth=1.5, color='#ff7f0e', linestyle='--')
    
    plt.xlabel("Energy")
    plt.ylabel("Probability Density")
    plt.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Add legend with a light background and border
    plt.legend(loc='best', fontsize=6, frameon=True, framealpha=0.9, edgecolor='lightgray')
    
    # Tighten layout and adjust margins for IEEE format
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    # Save with high DPI for print quality
    plt.savefig("energy_distribution_comparison.png", dpi=600, bbox_inches='tight')
    plt.savefig("energy_distribution_comparison.pdf", bbox_inches='tight')
    plt.show()
    
    # Print additional statistics
    print("\nDistribution Comparison Statistics:")
    print(f"DF Posterior: Mean={post_mean:.4f}, Std={post_std:.4f}")
    print(f"Exact Boltzmann: Mean={exact_mean:.4f}, Std={exact_std:.4f}")
    if mcmc is not None:
        print(f"MCMC: Mean={mcmc_mean:.4f}, Std={mcmc_std:.4f}")
    
    print(f"\nMean Differences:")
    print(f"DF vs Exact: {post_mean - exact_mean:.4f}")
    if mcmc is not None:
        print(f"MCMC vs Exact: {mcmc_mean - exact_mean:.4f}")
        print(f"DF vs MCMC: {post_mean - mcmc_mean:.4f}")
    
    print(f"\nKL Divergences:")
    print(f"KL(DF||Exact): {kl_df_to_exact:.4f}")
    if mcmc is not None:
        print(f"KL(MCMC||Exact): {kl_mcmc_to_exact:.4f}")
        print(f"KL(DF||MCMC): {kl_df_to_mcmc:.4f}")
    
    # Calculate and print energy percentiles
    percentiles = [5, 25, 50, 75, 95]
    post_percentiles = np.percentile(post_e, percentiles)
    
    # Calculate percentiles for exact distribution
    exact_percentiles = []
    cumulative_probs = np.cumsum(exact_probs)
    for p in percentiles:
        p_value = p / 100.0
        idx = np.searchsorted(cumulative_probs, p_value)
        if idx >= len(exact_energies):
            idx = len(exact_energies) - 1
        exact_percentiles.append(exact_energies[idx])
    
    if mcmc is not None:
        mcmc_percentiles = np.percentile(mcmc_e, percentiles)
        print("\nEnergy Percentiles:")
        print(f"{'Percentile':>10} | {'DF':>12} | {'Exact':>12} | {'MCMC':>12}")
        print(f"{'-'*10} | {'-'*12} | {'-'*12} | {'-'*12}")
        for i, p in enumerate(percentiles):
            print(f"{p:>10}% | {post_percentiles[i]:>12.4f} | {exact_percentiles[i]:>12.4f} | {mcmc_percentiles[i]:>12.4f}")
    else:
        print("\nEnergy Percentiles:")
        print(f"{'Percentile':>10} | {'DF Posterior':>12} | {'Exact':>12} | {'Difference':>12}")
        print(f"{'-'*10} | {'-'*12} | {'-'*12} | {'-'*12}")
        for i, p in enumerate(percentiles):
            print(f"{p:>10}% | {post_percentiles[i]:>12.4f} | {exact_percentiles[i]:>12.4f} | {post_percentiles[i]-exact_percentiles[i]:>12.4f}")
    
    if mcmc is not None:
        return post_mean, post_std, exact_mean, exact_std, mcmc_mean, mcmc_std, kl_df_to_exact, kl_mcmc_to_exact
    else:
        return post_mean, post_std, exact_mean, exact_std, kl_df_to_exact

# -------------------------------
# Main procedure
# -------------------------------
if __name__ == "__main__":
    # Grid parameters; note: exact enumeration is only performed if n*m < 10.
    n = 3
    m = 3
    N, J, h = create_ising_params(n, m, det_J=-0.3, ran_J=0.5, det_h=-0.1, ran_h=0.4, seed=42)
    T = N  # levels = number of spins

    print("Lattice size: {}x{} (N = {} spins)".format(n, m, N))
    print("Ising hyperparameters (h and J):")
    print(" h =", h)
    print(" J =", J)

    # -------------------------------
    # Empirical DF parameters and procedure
    # -------------------------------
    K = 1000  # number of prior sample paths
    S = 1000   # number of posterior sample paths
    init_state = (0,) * N

    # 1. Generate K prior sample paths using the prior transitions.
    prior_paths = []
    for k in range(K):
        path = sample_prior_path(init_state, J, h, N)
        if count_assigned(path[-1]) == N:
            prior_paths.append(path)
    print("Generated {} prior sample paths.".format(len(prior_paths)))

    # 2. Build empirical counts and transition probabilities.
    trans_counts, state_counts = build_empirical_counts(prior_paths, T)
    p_emp = get_empirical_transition_probs(trans_counts, state_counts, T)
    pi_emp = get_empirical_marginals(state_counts, K, T)

    # 3. Compute empirical Green functions.
    G_emp = compute_empirical_green(T, p_emp, state_counts)

    # 4. Compute multiplicative factors Υ_t(s) for levels 1,...,T-1.
    Y = compute_multiplicative_factors(T, G_emp, pi_emp)

    # 5. Compute optimal empirical transitions.
    p_opt = compute_optimal_empirical_transitions(T, p_emp, pi_emp, Y)

    # 6. Generate S posterior sample paths using the optimal empirical transitions.
    posterior_samples_emp = []
    for s_idx in range(S):
        path = sample_posterior_path(init_state, T, p_opt)
        if count_assigned(path[-1]) == N:
            posterior_samples_emp.append(path[-1])
    print("Generated {} posterior sample paths (Empirical DF).".format(len(posterior_samples_emp)))

    # Compute correlations from empirical DF samples.
    m_emp, c_emp = compute_correlations(posterior_samples_emp, N)

    # -------------------------------
    # MCMC procedure
    # -------------------------------
    mcmc_samples = metropolis_hastings_ising(n, N, J, h,
                                             n_samples=N_MCMC_SAMPLES,
                                             burnin=MCMC_BURNIN,
                                             thin=MCMC_THIN,
                                             seed=SEED)
    print("Generated {} MCMC samples.".format(len(mcmc_samples)))
    m_mcmc, c_mcmc = compute_correlations(mcmc_samples, N)

    # -------------------------------
    # Determine reference: use exact if system is small (n*m < 10), else use MCMC.
    # -------------------------------
    if N < 10:
        m_exact, c_exact = compute_exact_correlations(N, J, h)
        ref_label = "Exact"
        m_ref = m_exact
        c_ref = c_exact
    else:
        ref_label = "MCMC"
        m_ref = m_mcmc
        c_ref = c_mcmc

    # -------------------------------
    # Print out comparison of singleton magnetizations.
    # -------------------------------
    print("\nSingleton Magnetizations:")
    print("{:<10s} {:>12s} {:>12s} {:>12s}".format("Site", "Empirical", "MCMC", ref_label))
    for i in range(N):
        emp_val = m_emp[i]
        mcmc_val = m_mcmc[i]
        ref_val = m_ref[i]
        print("Site {:<3d}: {:>12.4f} {:>12.4f} {:>12.4f}".format(i, emp_val, mcmc_val, ref_val))
    
    # -------------------------------
    # Print out comparison of pairwise correlations.
    # -------------------------------
    print("\nPairwise Correlations:")
    print("{:<10s} {:>12s} {:>12s} {:>12s}".format("(i,j)", "Empirical", "MCMC", ref_label))
    for (i,j) in sorted(c_ref.keys()):
        emp_val = c_emp[(i,j)]
        mcmc_val = c_mcmc[(i,j)]
        ref_val = c_ref[(i,j)]
        print("({},{})   : {:>12.4f} {:>12.4f} {:>12.4f}".format(i, j, emp_val, mcmc_val, ref_val))
    
    # -------------------------------
    # Compute integrated mismatches Δ₁ and Δ₂.
    # Use small epsilon to avoid division by zero.
    # -------------------------------
    eps = 1e-12
    # Δ₁ = (1/N) * sum_i [ |m_method[i] - m_ref[i]| / (|m_ref[i]|) ]
    Delta1_emp = sum(abs(m_emp[i] - m_ref[i]) / (abs(m_ref[i]) + eps) for i in range(N)) / N
    Delta1_mcmc = sum(abs(m_mcmc[i] - m_ref[i]) / (abs(m_ref[i]) + eps) for i in range(N)) / N

    # Δ₂ = (1/(N*(N-1))) * sum_{i<j} [ 2 * |c_method[(i,j)] - c_ref[(i,j)]| / (|c_ref[(i,j)]|) ]
    pair_indices = [(i,j) for i in range(N) for j in range(i+1, N)]
    Delta2_emp = sum(2 * abs(c_emp[(i,j)] - c_ref[(i,j)]) / (abs(c_ref[(i,j)]) + eps) for (i,j) in pair_indices) / (N*(N-1))
    Delta2_mcmc = sum(2 * abs(c_mcmc[(i,j)] - c_ref[(i,j)]) / (abs(c_ref[(i,j)]) + eps) for (i,j) in pair_indices) / (N*(N-1))

    print("\nIntegrated Mismatches (relative errors):")
    print("Delta1 (singleton):")
    print("  Empirical DF vs {}: {:.4e}".format(ref_label, Delta1_emp))
    print("  MCMC vs {}:         {:.4e}".format(ref_label, Delta1_mcmc))
    print("Delta2 (pairwise):")
    print("  Empirical DF vs {}: {:.4e}".format(ref_label, Delta2_emp))
    print("  MCMC vs {}:         {:.4e}".format(ref_label, Delta2_mcmc))

    # -------------------------------
    # Plot energy distribution comparison
    # -------------------------------
    print("\nComparing DF posterior with exact Boltzmann distribution and MCMC...")
    #plot_energy_distribution_with_exact(posterior_samples_emp, N, J, h, mcmc=mcmc_samples)
    
    # Comment out the separate MCMC comparison since we now include it in the combined plot
    # print("\nComparing DF posterior with MCMC samples...")
    # plot_energy_distribution(posterior_samples_emp, mcmc_samples, J, h)
