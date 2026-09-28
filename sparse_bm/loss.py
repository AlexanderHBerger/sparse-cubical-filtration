"""sparseBM loss.

The loss runs in two steps: ``build_masks`` computes the masked inputs and the pocket
labels (on the GPU if available), and ``loss_from_masks`` runs the C++ matching on the
CPU, one thread per batch element, and assembles the loss terms. Filtration values are
g = 1 - p, so foreground is low.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .masks import make_sparse_pair, make_sparse_pair_torch
from .pocket_labels import label_pockets
from .terms import PENALTIES, assemble_terms, normalise_unmatched_target

_BM = None


def _get_bm():
    global _BM
    if _BM is None:
        import betti_matching as _bm
        _BM = _bm
    return _BM


def _value_dtype():
    return np.dtype(getattr(_get_bm(), "value_dtype", "float64"))


CLAMP_OFF = 1.0


def clamp_for_matching(m_p: np.ndarray, clamp_confident: float,
                       tau_mask: float) -> np.ndarray:
    """Set ``g_p <= 1 - clamp_confident`` to 0 in a copy of the matcher input.

    The ties make the matching cheaper. Only the matcher sees the clamped values; the
    loss terms read the original ones.
    """
    if clamp_confident >= CLAMP_OFF:
        return m_p
    g_hi = 1.0 - float(clamp_confident)
    if not (g_hi < tau_mask):
        raise ValueError(
            f"clamp_confident={clamp_confident} with tau_mask={tau_mask} would "
            f"move values across the mask threshold (1 - clamp = {g_hi} >= tau); "
            f"require clamp_confident > {1.0 - tau_mask}")
    out = np.array(m_p, copy=True)
    out[out <= g_hi] = 0
    return out


def fine_pocket_labels(keep) -> np.ndarray:
    """Pocket labels (int32, on the host) of a kept mask given as numpy array or torch
    tensor. CUDA tensors are labelled with cupy if it is installed."""
    if isinstance(keep, torch.Tensor):
        if keep.is_cuda:
            try:
                import cupy as cp

                dev = cp.from_dlpack(
                    keep.detach().to(torch.uint8).contiguous()).astype(cp.bool_)
                return np.ascontiguousarray(label_pockets(dev).fine)
            except ImportError:
                pass
        keep = keep.detach().cpu().numpy()
    return np.ascontiguousarray(label_pockets(np.asarray(keep)).fine)


def match_sparse(m_p: np.ndarray, m_t: np.ndarray, pocket_labels: np.ndarray,
                 tau_mask: float, clamp_confident: float = CLAMP_OFF):
    """Sparse matching of two masked filtrations (see ``make_sparse_pair``)."""
    return _get_bm().compute_matching(
        clamp_for_matching(m_p, clamp_confident, tau_mask), m_t,
        sparse=True, mask_threshold=tau_mask, pocket_labels=pocket_labels)


def sparse_matching(pred_np: np.ndarray, target_np: np.ndarray, tau_mask: float,
                    clamp_confident: float = CLAMP_OFF):
    """Mask, pocket labels and sparse matching for one pair of foreground probability
    maps. Returns ``(res, m_p, m_t, keep, labels)``."""
    vd = _value_dtype()
    g_p = np.ascontiguousarray((1.0 - pred_np).astype(np.float64))
    g_t = np.ascontiguousarray((1.0 - target_np).astype(np.float64))
    m_p, m_t, keep = make_sparse_pair(g_p, g_t, tau_mask)
    if vd != np.float64:
        m_p = np.ascontiguousarray(m_p, dtype=vd)
        m_t = np.ascontiguousarray(m_t, dtype=vd)
    labels = fine_pocket_labels(keep)
    res = match_sparse(m_p, m_t, labels, tau_mask,
                       clamp_confident=clamp_confident)
    return res, m_p, m_t, keep, labels


def terms_on_arrays(res, m_p, m_t, dims, unmatched_weight=1.0,
                    essential_weight=1.0, matched_weight=1.0,
                    unmatched_target="diagonal"):
    """Loss, gradient with respect to the prediction, and per-dimension loss for one
    matching result on numpy arrays (squared error only)."""
    g_p = torch.tensor(np.asarray(m_p, dtype=np.float64), requires_grad=True)
    loss, per_dim = assemble_terms(
        res, g_p, m_t, dims, unmatched_weight, essential_weight, "se",
        matched_weight, unmatched_target, "se")
    loss.backward()
    return float(loss.detach()), (-g_p.grad).numpy(), per_dim


def _batched_mask_build(preds, targets, tau, mask_backend):
    B = preds.shape[0]
    vd = _value_dtype()
    use_torch_mask = (mask_backend == "torch"
                      or (mask_backend == "auto" and preds.is_cuda))
    if not np.isfinite(tau):
        raise ValueError("the sparse matching requires a finite tau")
    m_p_list, m_t_list, keep_fracs, labels = [], [], [], []
    if use_torch_mask:
        with torch.no_grad():
            for b in range(B):
                g_p = (1.0 - preds[b].detach()).to(torch.float64)
                g_t = (1.0 - targets[b].detach()).to(torch.float64)
                mp_t, mt_t, keep_t = make_sparse_pair_torch(g_p, g_t, tau)
                labels.append(fine_pocket_labels(keep_t))
                keep_fracs.append(float(keep_t.float().mean().item()))
                m_p_list.append(np.ascontiguousarray(mp_t.cpu().numpy(), dtype=vd))
                m_t_list.append(np.ascontiguousarray(mt_t.cpu().numpy(), dtype=vd))
    else:
        for b in range(B):
            p_np = preds[b].detach().cpu().numpy().astype(np.float64)
            t_np = targets[b].detach().cpu().numpy().astype(np.float64)
            mp, mt, keep = make_sparse_pair(np.ascontiguousarray(1.0 - p_np),
                                            np.ascontiguousarray(1.0 - t_np), tau)
            if vd != np.float64:
                mp = np.ascontiguousarray(mp, dtype=vd)
                mt = np.ascontiguousarray(mt, dtype=vd)
            labels.append(fine_pocket_labels(keep))
            keep_fracs.append(float(np.asarray(keep).mean()))
            m_p_list.append(mp)
            m_t_list.append(mt)
    return m_p_list, m_t_list, keep_fracs, labels


def _match_batch(m_p_list, m_t_list, tau, clamp_confident, pocket_labels):
    match_p = ([clamp_for_matching(mp, clamp_confident, tau) for mp in m_p_list]
               if clamp_confident < CLAMP_OFF else m_p_list)
    return _get_bm().compute_matching(
        match_p, m_t_list, sparse=True, mask_threshold=tau,
        pocket_labels=list(pocket_labels))


def _loss_from_masks(preds, m_p_list, m_t_list, pocket_labels, dims,
                     unmatched_weight, essential_weight, tau, stats=None,
                     clamp_confident=CLAMP_OFF, penalty_matched="ce",
                     matched_weight=1.0, unmatched_target="diagonal",
                     penalty_unmatched="se", logits=None, fg_index=1):
    B = preds.shape[0]
    results = _match_batch(m_p_list, m_t_list, tau, clamp_confident, pocket_labels)
    total = torch.zeros((), dtype=torch.float64, device=preds.device)
    per_dim_sum = {int(d): 0.0 for d in dims}
    for b in range(B):
        lv, pd = assemble_terms(
            results[b], 1.0 - preds[b], m_t_list[b], dims, unmatched_weight,
            essential_weight, penalty_matched, matched_weight, unmatched_target,
            penalty_unmatched, logits=None if logits is None else logits[b],
            fg_index=fg_index)
        total = total + lv.to(torch.float64)
        for d in dims:
            per_dim_sum[int(d)] += pd[int(d)]
    if stats is not None:
        stats['per_dim'] = {int(d): per_dim_sum[int(d)] / B for d in dims}
    if total.grad_fn is None and preds.requires_grad:
        # Empty barcodes: keep the loss connected to the graph so backward() works.
        total = total + preds.reshape(-1)[0].to(torch.float64) * 0.0
    return (total / B).to(preds.dtype)


class SparseBettiLossBatched(torch.nn.Module):
    """sparseBM loss for a batch of foreground probabilities ``(B, *spatial)``, 2D or 3D.

    Returns the mean loss over the batch. ``forward`` runs ``build_masks`` and
    ``loss_from_masks``; calling them separately allows overlapping the CPU matching
    with GPU work.

    Parameters
    ----------
    unmatched_weight, matched_weight, essential_weight : float
        Weights of the unmatched, matched and essential (censored) terms.
    dims : tuple of int, optional
        Homology dimensions to include. Default: all dimensions of the input.
    theta, tau_mask : float
        Foreground threshold on p; tau_mask = 1 - theta unless given.
    mask_backend : str
        "auto" (torch on the GPU, scipy otherwise), "torch" or "scipy".
    capture_stats : bool
        Store per-dimension loss, kept fraction and pocket count in ``last_stats``.
    clamp_confident : float
        See ``clamp_for_matching``. 1.0 disables it.
    penalty_matched, penalty_unmatched : str
        "se" (squared error on g) or "ce" (cross-entropy on ``logits``).
    unmatched_target : str
        "diagonal" or "gt", see ``terms``.
    fg_index : int
        Foreground channel of ``logits``.
    """

    def __init__(self, unmatched_weight: float = 1.0,
                 essential_weight: float = 1.0,
                 dims: Optional[Tuple[int, ...]] = None, theta: float = 0.5,
                 tau_mask: Optional[float] = None, mask_backend: str = "auto",
                 capture_stats: bool = False,
                 clamp_confident: float = 0.99,
                 penalty_matched: str = "ce", matched_weight: float = 1.0,
                 unmatched_target: str = "diagonal",
                 penalty_unmatched: str = "se", fg_index: int = 1):
        super().__init__()
        if mask_backend not in ("auto", "scipy", "torch"):
            raise ValueError('mask_backend must be "auto", "scipy" or "torch"')
        if tau_mask is not None and not np.isfinite(float(tau_mask)):
            raise ValueError("tau_mask must be finite")
        self.unmatched_weight = float(unmatched_weight)
        self.essential_weight = float(essential_weight)
        self.dims = None if dims is None else tuple(int(d) for d in dims)
        self.theta = float(theta)
        self.tau_mask = None if tau_mask is None else float(tau_mask)
        self.mask_backend = str(mask_backend)
        self.capture_stats = bool(capture_stats)
        self.last_stats = None
        clamp_confident = float(clamp_confident)
        if not (0.0 < clamp_confident <= CLAMP_OFF):
            raise ValueError(
                f"clamp_confident must be in (0, 1]; got {clamp_confident}")
        self.clamp_confident = clamp_confident
        for name, pen in (("penalty_matched", penalty_matched),
                          ("penalty_unmatched", penalty_unmatched)):
            if pen not in PENALTIES:
                raise ValueError(f"{name} must be one of {PENALTIES}; got {pen!r}")
        self.penalty_matched = str(penalty_matched)
        self.penalty_unmatched = str(penalty_unmatched)
        self.matched_weight = float(matched_weight)
        self.unmatched_target = normalise_unmatched_target(str(unmatched_target))
        self.fg_index = int(fg_index)

    def _tau(self) -> float:
        return float(1.0 - self.theta) if self.tau_mask is None else float(self.tau_mask)

    def _dims(self, preds: torch.Tensor) -> Tuple[int, ...]:
        return self.dims if self.dims is not None else tuple(range(preds.ndim - 1))

    def build_masks(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        assert preds.ndim in (3, 4), "batched inputs: (B, *2D) or (B, *3D)"
        return _batched_mask_build(preds, targets, self._tau(), self.mask_backend)

    def loss_from_masks(self, preds: torch.Tensor, m_p_list, m_t_list,
                        keep_fracs, pocket_labels,
                        logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(pocket_labels) != len(m_p_list):
            raise ValueError(f"pocket_labels has {len(pocket_labels)} entries for "
                             f"{len(m_p_list)} samples")
        stats = {} if self.capture_stats else None
        out = _loss_from_masks(
            preds, m_p_list, m_t_list, pocket_labels, self._dims(preds),
            self.unmatched_weight, self.essential_weight, self._tau(), stats,
            self.clamp_confident, self.penalty_matched, self.matched_weight,
            self.unmatched_target, self.penalty_unmatched, logits=logits,
            fg_index=self.fg_index)
        if self.capture_stats:
            stats['fg_fraction'] = (float(np.mean(keep_fracs))
                                    if keep_fracs else 0.0)
            stats['n_pockets'] = (float(np.mean([int(l.max())
                                                 for l in pocket_labels]))
                                  if pocket_labels else 0.0)
            self.last_stats = stats
        return out

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.loss_from_masks(preds, *self.build_masks(preds, targets),
                                    logits=logits)


class SparseBettiLoss(SparseBettiLossBatched):
    """``SparseBettiLossBatched`` for a single unbatched pair."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(pred.unsqueeze(0), target.unsqueeze(0),
                               None if logits is None else logits.unsqueeze(0))
