"""Sparse Betti matching: a topological loss for image segmentation computed on a
sparse cubical complex.

The compiled matcher (``betti_matching``, built from ``cpp/``) must be importable,
e.g. by adding its build directory to ``PYTHONPATH``.
"""

from .loss import (SparseBettiLoss, SparseBettiLossBatched, clamp_for_matching,
                   fine_pocket_labels, match_sparse, sparse_matching,
                   terms_on_arrays)
from .masks import CONE_BIRTH, lift_masked, make_sparse_pair, make_sparse_pair_torch
from .pocket_labels import PocketLabels, label_pockets
from .terms import BACKGROUND, PENALTIES, UNMATCHED_TARGETS, assemble_terms

__all__ = [
    "SparseBettiLoss", "SparseBettiLossBatched", "sparse_matching", "match_sparse",
    "clamp_for_matching", "fine_pocket_labels", "terms_on_arrays",
    "CONE_BIRTH", "make_sparse_pair", "make_sparse_pair_torch", "lift_masked",
    "PocketLabels", "label_pockets",
    "BACKGROUND", "PENALTIES", "UNMATCHED_TARGETS", "assemble_terms",
]
