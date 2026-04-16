import numpy as np


def prior_uniform_step(_s):
    return {+1: 0.5, -1: 0.5}


def prior_bp_step_simple(s, bp_solver, assigned_order):
    if len(s) >= len(assigned_order):
        return {+1: 0.5, -1: 0.5}
    next_node = assigned_order[len(s)]
    cond = bp_solver.get_conditional(assigned_order[: len(s)], s)
    return cond.get(next_node, {+1: 0.5, -1: 0.5})


def prior_mf_step_simple(s, mf_solver, assigned_order):
    """Sequential prior based on (fixed) MF marginals for each node."""
    if len(s) >= len(assigned_order):
        return {+1: 0.5, -1: 0.5}
    next_node = assigned_order[len(s)]
    return mf_solver.marginal(next_node)


def prior_mf_step_field_only(s, h):
    """Legacy: independent prior from local fields only (no couplings)."""
    t = len(s)
    if t >= len(h):
        return {+1: 0.5, -1: 0.5}
    p = 1.0 / (1.0 + np.exp(-2.0 * h[t]))
    return {+1: p, -1: 1.0 - p}

