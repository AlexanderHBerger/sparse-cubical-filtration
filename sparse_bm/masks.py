"""The sparse support of a (prediction, target) pair.

Convention: filtration values are ``g = 1 - p`` in [0, 1] (foreground low), and the
filtration is the sublevel filtration of ``g`` (V-construction).

A voxel is KEPT iff ``min(g0, g1) < tau_mask``, i.e. iff at least one of the two inputs
puts it above the foreground threshold. Every other voxel is replaced by ``+inf`` in
BOTH inputs; the matcher derives the kept complex K from those sentinels. Two facts the
construction relies on:

* every kept cell has value < tau_mask in the comparison image C = min(g0, g1) (a kept
  cell's corners are all kept, and the V-construction takes a max), so nothing is born
  at or above tau_mask on the C side. On the INPUT sides a kept cell can sit anywhere
  in [0, 1]: a voxel kept because the other input is below tau keeps its real value;
* every non-kept cell enters at ``CONE_BIRTH`` = 1, the background value, as one
  terminal slice ordered after every kept cell -- which is what the virtual pocket
  nodes of the matcher stand in for. For a binary target that slice IS the target's own
  background, so ``lift_masked(m1) == g1`` exactly.
"""

from __future__ import annotations

import numpy as np

#: Value carried by every VIRTUAL cell of the sparse complex -- cone edges, cone
#: 2-cells, dual pocket nodes. Mirrors ``CONE_BIRTH`` in ``cpp/src/config.h``. It is the
#: BACKGROUND value of the fg-low [0, 1] convention: the masked region is background in
#: both inputs, so it enters at the background value, ordered after every kept cell of
#: equal value. A class born at background dies at background -- zero persistence, never
#: a bar -- exactly as in the dense filtration. The matcher throws if a kept voxel
#: exceeds it.
CONE_BIRTH = 1.0


def make_sparse_pair(g0: np.ndarray, g1: np.ndarray, tau_mask: float):
    """Masked (+inf sentinel) float64 copies of a g-space input pair.

    ``keep = min(g0, g1) < tau_mask``; every non-kept voxel is set to ``+inf`` in both
    inputs. Returns ``(m0, m1, keep)``.
    """
    if g0.shape != g1.shape:
        raise ValueError(f"shape mismatch: {g0.shape} vs {g1.shape}")
    keep = np.minimum(g0, g1) < tau_mask
    m0 = g0.astype(np.float64, copy=True)
    m1 = g1.astype(np.float64, copy=True)
    m0[~keep] = np.inf
    m1[~keep] = np.inf
    return np.ascontiguousarray(m0), np.ascontiguousarray(m1), keep


def make_sparse_pair_torch(g0_t, g1_t, tau_mask: float):
    """Torch analog of ``make_sparse_pair`` (any device). Bitwise identical."""
    import torch

    if g0_t.shape != g1_t.shape:
        raise ValueError(f"shape mismatch: {g0_t.shape} vs {g1_t.shape}")
    if g0_t.dim() not in (2, 3):
        raise ValueError("2D or 3D inputs only")
    keep = torch.minimum(g0_t, g1_t) < tau_mask
    inf = torch.tensor(float("inf"), dtype=g0_t.dtype, device=g0_t.device)
    return torch.where(keep, g0_t, inf), torch.where(keep, g1_t, inf), keep


def lift_masked(m: np.ndarray, cone: float = CONE_BIRTH) -> np.ndarray:
    """The filtration the sparse matching models: ``where(keep, g, CONE_BIRTH)``.

    ``m`` is a masked array as ``make_sparse_pair`` returns it (+inf off the mask).
    Replacing the sentinel with ``CONE_BIRTH`` gives a plain finite array whose DENSE
    barcode is the reference ("oracle") for the sparse one: below CONE_BIRTH it IS the
    masked filtration of the kept complex, and at CONE_BIRTH the masked region enters as
    background and kills whatever the pockets fill. For a binary target this is the
    target itself; for the prediction it is the prediction with the agreed background
    pasted in.
    """
    out = np.asarray(m, dtype=np.float64).copy()
    out[~np.isfinite(out)] = float(cone)
    return np.ascontiguousarray(out)
