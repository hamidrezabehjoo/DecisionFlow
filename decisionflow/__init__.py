"""Decision Flow: exact sequential sampling via Green-function correction of
proposal priors (Behjoo, Chertkov & Ahn).

Public API
----------
IsingModel                     model specification + exact enumeration
mean_field_marginals, bp_marginals
ProductPrior, LocalBoltzmannPrior
exact_sample, alg1_sample
mh_sample
metrics (delta1, delta2, kl_proxy, npll, evaluate)
"""
from .ising import IsingModel, Enumeration, ExactReference
from .solvers import mean_field_marginals, bp_marginals
from .priors import Prior, ProductPrior, LocalBoltzmannPrior
from .df import exact_sample, alg1_sample
from .mcmc import mh_sample
from . import metrics

__all__ = [
    "IsingModel",
    "Enumeration",
    "ExactReference",
    "mean_field_marginals",
    "bp_marginals",
    "Prior",
    "ProductPrior",
    "LocalBoltzmannPrior",
    "exact_sample",
    "alg1_sample",
    "mh_sample",
    "metrics",
]
