from __future__ import annotations

import numpy as np

# Background value of g = 1 - p. The masked region enters the sparse filtration at
# this value (CONE_BIRTH in cpp/src/config.h).
CONE_BIRTH = 1.0


def make_sparse_pair(g0: np.ndarray, g1: np.ndarray, tau_mask: float):
    """Mask a pair of filtrations (g = 1 - p) for the sparse matching.

    A voxel is kept iff ``min(g0, g1) < tau_mask``; all other voxels are set to +inf
    in both inputs. Returns ``(m0, m1, keep)``.
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
    """Torch version of ``make_sparse_pair``."""
    import torch

    if g0_t.shape != g1_t.shape:
        raise ValueError(f"shape mismatch: {g0_t.shape} vs {g1_t.shape}")
    if g0_t.dim() not in (2, 3):
        raise ValueError("2D or 3D inputs only")
    keep = torch.minimum(g0_t, g1_t) < tau_mask
    inf = torch.tensor(float("inf"), dtype=g0_t.dtype, device=g0_t.device)
    return torch.where(keep, g0_t, inf), torch.where(keep, g1_t, inf), keep


def lift_masked(m: np.ndarray, cone: float = CONE_BIRTH) -> np.ndarray:
    """Replace the +inf entries of a masked array by ``CONE_BIRTH``.

    The dense barcode of the result equals the barcode of the sparse complex.
    """
    out = np.asarray(m, dtype=np.float64).copy()
    out[~np.isfinite(out)] = float(cone)
    return np.ascontiguousarray(out)
