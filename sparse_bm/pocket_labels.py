"""Pockets of the sparse cubical complex.

A pocket is a connected component of the masked voxels. It is replaced by one virtual
node in the sparse complex. Pockets are 26-connected (8-connected in 2D): two cubes
merge through their shared 2-cell, and a non-kept 2-cell always has a masked corner,
so the components of the masked region are the 26-connected components of the masked
voxels, not the 6-connected components of the non-kept cubes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

__all__ = ["PocketLabels", "kept_cubes", "canonical_relabel", "label_pockets"]

try:
    import cc3d as _cc3d  # noqa: F401
    _HAVE_CC3D = True
except ImportError:
    _HAVE_CC3D = False


@dataclass(frozen=True)
class PocketLabels:
    """Pocket labels of one kept mask (int32, 0 on kept voxels and cubes)."""

    fine: np.ndarray
    n_fine: int
    cube_fine: np.ndarray
    fine_touches_boundary: np.ndarray

    @property
    def n_fine_interior(self) -> int:
        return int(self.n_fine - self.fine_touches_boundary.sum())


def kept_cubes(keep: np.ndarray) -> np.ndarray:
    """Top cells whose corners are all kept. Shape is ``keep.shape - 1``."""
    nd = keep.ndim
    if any(s < 2 for s in keep.shape):
        return keep[tuple(slice(0, max(s - 1, 0)) for s in keep.shape)]
    out = None
    for off in itertools.product((0, 1), repeat=nd):
        sl = tuple(slice(o, keep.shape[i] - 1 + o) for i, o in enumerate(off))
        out = keep[sl].copy() if out is None else (out & keep[sl])
    return out


def _xp(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy
    return np


def canonical_relabel(lab, n: int):
    """Renumber labels 1..n by raster order of first occurrence.

    The C++ matcher derives the order of the virtual cells from the pocket ids, so all
    backends have to produce the same labelling. Returns ``(labels, n)``.
    """
    xp = _xp(lab)
    lab = xp.ascontiguousarray(lab)
    if n == 0:
        return xp.zeros(lab.shape, dtype=xp.int32), 0
    flat = lab.reshape(-1)
    hi = int(flat.max())
    if hi == 0:
        return xp.zeros(lab.shape, dtype=xp.int32), 0

    if xp is np:
        # Duplicate fancy indices: numpy keeps the last write, so scattering in reverse
        # order leaves the first occurrence of every label.
        first = np.full(hi + 1, -1, dtype=np.int64)
        first[flat[::-1]] = np.arange(flat.size - 1, -1, -1, dtype=np.int64)
        present = np.flatnonzero(first >= 0)
    else:
        # cupy does not define the result for duplicate indices.
        import cupyx
        first = xp.full(hi + 1, flat.size, dtype=xp.int64)
        cupyx.scatter_min(first, flat, xp.arange(flat.size, dtype=xp.int64))
        present = xp.flatnonzero(first < flat.size)
    present = present[present != 0]
    order = present[xp.argsort(first[present], kind="stable")]
    lut = xp.zeros(hi + 1, dtype=xp.int32)
    lut[order] = xp.arange(1, order.size + 1, dtype=xp.int32)
    return lut[lab], int(order.size)


def _label_cpu(mask: np.ndarray, backend: str):
    # cc3d is not the default: in 2D it occasionally assigns a label to a background
    # pixel. label_pockets checks for this on every call.
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0
    if backend == "cc3d":
        import cc3d

        conn = {2: 8, 3: 26}[mask.ndim]
        lab = cc3d.connected_components(np.ascontiguousarray(mask), connectivity=conn)
        return lab.astype(np.int32, copy=False), int(lab.max())
    from scipy.ndimage import generate_binary_structure, label as ndi_label

    lab, n = ndi_label(mask, structure=generate_binary_structure(mask.ndim, mask.ndim))
    return lab.astype(np.int32, copy=False), int(n)


def _label_cupy(mask):
    import cupy as cp
    from cupyx.scipy.ndimage import label as cp_label

    if not bool(mask.any()):
        return cp.zeros(mask.shape, dtype=cp.int32), 0
    st = cp.ones((3,) * mask.ndim, dtype=cp.int32)
    lab, n = cp_label(mask, structure=st)
    return lab.astype(cp.int32, copy=False), int(n)


def label_pockets(keep: np.ndarray, backend: str = "auto") -> PocketLabels:
    """Pocket labels of a boolean kept mask (numpy or cupy).

    ``backend`` is "auto" (cupy for cupy arrays, scipy otherwise), "scipy", "cc3d" or
    "cupy". The result is always returned on the host.
    """
    if backend not in ("auto", "scipy", "cc3d", "cupy"):
        raise ValueError(f"unknown backend {backend!r}")
    is_device = type(keep).__module__.split(".")[0] == "cupy"
    if backend == "auto":
        backend = "cupy" if is_device else "scipy"
    if backend == "cupy" and not is_device:
        import cupy as cp

        keep = cp.asarray(keep)
        is_device = True
    if backend != "cupy" and is_device:
        keep = keep.get()
        is_device = False

    if keep.ndim not in (2, 3):
        raise ValueError("2D or 3D masks only")

    if is_device:
        import cupy as cp
        xp = cp
        keep_a = keep.astype(cp.bool_, copy=False)
        label_fn = _label_cupy
    else:
        xp = np
        keep_a = np.ascontiguousarray(keep.astype(bool, copy=False))
        label_fn = lambda m: _label_cpu(m, backend)  # noqa: E731

    fine_raw, n_fine = label_fn(~keep_a)
    kc = kept_cubes(keep_a)
    fine, n_fine = canonical_relabel(fine_raw, n_fine)

    # All masked corners of a cube lie in the same pocket, so the max over the corner
    # labels gives the pocket of a non-kept cube.
    nd = keep_a.ndim
    cube_fine = xp.zeros(kc.shape, dtype=xp.int32)
    if cube_fine.size:
        for off in itertools.product((0, 1), repeat=nd):
            sl = tuple(slice(o, keep_a.shape[i] - 1 + o) for i, o in enumerate(off))
            xp.maximum(cube_fine, fine[sl], out=cube_fine)
        cube_fine[kc] = 0

    touches = xp.zeros(n_fine + 1, dtype=xp.bool_)
    for axis in range(nd):
        for idx in (0, -1):
            face = xp.take(fine, idx, axis=axis)
            ids = xp.unique(face)
            touches[ids[ids != 0]] = True
    touches[0] = False

    if ((fine != 0) != (~keep_a)).any():
        raise RuntimeError(
            f"pocket labelling backend {backend!r} produced labels that are not "
            "exactly the masked voxels")

    if is_device:
        fine, cube_fine, touches = (cp.asnumpy(fine), cp.asnumpy(cube_fine),
                                    cp.asnumpy(touches))

    return PocketLabels(fine=np.ascontiguousarray(fine), n_fine=n_fine,
                        cube_fine=np.ascontiguousarray(cube_fine),
                        fine_touches_boundary=touches)
