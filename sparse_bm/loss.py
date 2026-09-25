"""Sparse Betti-matching loss.

The pipeline is

    masks + pocket labels (GPU if available, non-differentiable)   build_masks()
      -> ONE batched C++ matching (CPU, one task per sample)         loss_from_masks()
      -> autograd term assembly                                      terms.assemble_terms

The sparse complex keeps a voxel iff ``min(g_p, g_t) < tau`` (``g = 1 - p``) and adds
one virtual node per connected masked region (pocket). Its barcode is the persistent
homology of the kept complex at the inputs' own values, every class that survives in
it censored at ``CONE_BIRTH = 1``, the background value -- for a binary target
literally its dense barcode (see ``masks.lift_masked``). Censored bars arrive on the
essential channels of the matching result.

The loss terms (see ``terms``):
  * finite matched pairs -- birth and death pulled to the target's values;
  * matched essentials   -- birth pulled to the target's birth (no death term:
                            both deaths are censored);
  * unmatched prediction bars -- every endpoint pulled to the bar's other
                            endpoint, which for an essential is background, or
                            (``unmatched_target="gt"``) to the GT value there;
  * unmatched target bars -- no term, no gradient.

Two penalties, ``penalty_matched`` and ``penalty_unmatched``, each "ce" or "se",
defaulting to ce on matched and se on unmatched. "ce" is the ordinary
classification loss on the classifier's LOGITS, so a ce configuration passes
``logits=(B, C, *spatial)`` alongside the foreground probability.

Value convention: g = 1 - image (foreground low); tau_mask = 1 - theta unless
overridden. All matching decisions live in the C++ matcher; this module only turns
the returned coordinates into loss terms.
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
    """The compiled matcher (``betti_matching``, built from ``cpp/``)."""
    global _BM
    if _BM is None:
        import betti_matching as _bm  # resolved via PYTHONPATH
        _BM = _bm
    return _BM


def _value_dtype():
    """numpy dtype the loaded matcher build expects (float64, or float32 for the
    -DVALUE_T_FLOAT build)."""
    return np.dtype(getattr(_get_bm(), "value_dtype", "float64"))


CLAMP_OFF = 1.0


def clamp_for_matching(m_p: np.ndarray, clamp_confident: float,
                       tau_mask: float) -> np.ndarray:
    """Confident-foreground clamp, applied ONLY to the matching input.

    Collapses ``p >= clamp_confident`` (i.e. ``g_p <= 1 - clamp_confident``) onto a
    single value, manufacturing ties so more columns resolve as emergent pairs, which
    makes the matching cheaper.

    Returns a SEPARATE array. The caller keeps the unclamped ``m_p`` for the term
    assembly, so the clamp changes only WHICH cells pair with which; every
    birth/death value the loss and gradient use is still exact.

    Only the confident FOREGROUND end is clamped: a confident *background* voxel
    survives the mask only where the ground truth says foreground, so those voxels
    are exactly the model's false negatives, and clamping them would zero the
    gradient on its own mistakes.

    The kept set is unchanged: masked cells are +inf, and every clamped value moves
    from ``(0, 1 - clamp_confident]`` to 0 while staying below ``tau_mask``, so
    ``keep = min(g_p, g_t) < tau`` cannot flip. That requires
    ``1 - clamp_confident < tau_mask``, which is asserted.
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
    """Pocket labelling of a kept mask -> host int32 array.

    ``keep`` may be a numpy array or a torch tensor. A CUDA tensor is labelled
    on-device through cupy when cupy is importable (``label_pockets`` accepts a cupy
    array and returns host tables, byte-identical to the CPU backend), and otherwise
    falls back to a host transfer -- correctness never depends on the GPU being
    available.
    """
    if isinstance(keep, torch.Tensor):
        if keep.is_cuda:
            try:
                import cupy as cp

                # uint8 rather than bool: torch's dlpack bool support is version
                # dependent, uint8 is not, and the astype below is free.
                dev = cp.from_dlpack(
                    keep.detach().to(torch.uint8).contiguous()).astype(cp.bool_)
                return np.ascontiguousarray(label_pockets(dev).fine)
            except ImportError:
                pass
        keep = keep.detach().cpu().numpy()
    return np.ascontiguousarray(label_pockets(np.asarray(keep)).fine)


def match_sparse(m_p: np.ndarray, m_t: np.ndarray, pocket_labels: np.ndarray,
                 tau_mask: float, clamp_confident: float = CLAMP_OFF):
    """Run the sparse matcher on already-masked arrays (see ``masks``).

    The arrays go in unmodified apart from the optional confident-foreground clamp.
    The matcher re-derives the kept set from the +inf sentinels and validates
    ``pocket_labels`` against it on every call.
    """
    return _get_bm().compute_matching(
        clamp_for_matching(m_p, clamp_confident, tau_mask), m_t,
        sparse=True, mask_threshold=tau_mask, pocket_labels=pocket_labels)


def sparse_matching(pred_np: np.ndarray, target_np: np.ndarray, tau_mask: float,
                    clamp_confident: float = CLAMP_OFF):
    """Masks + pocket labels + matching for one (prediction, target) pair.

    ``pred_np`` / ``target_np`` are foreground probabilities (image values, fg high).
    Returns ``(res, m_p, m_t, keep, labels)``; ``m_p`` / ``m_t`` are the masked
    filtrations (``g = 1 - image``, +inf off the mask) the loss terms read, cast to
    the build's value dtype.
    """
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
    """(loss, d loss / d pred, per_dim) for one matching result, se on both halves.

    The same assembly the training path runs, on numpy arrays: ``m_p`` is the masked
    prediction filtration (+inf off the mask), the returned gradient is with respect
    to the PREDICTION (``d g_p / d pred = -1`` folded in) and is zero off the critical
    cells. A "ce" penalty needs the classifier's logits, so it has no array form; use
    ``terms.assemble_terms`` directly for it.
    """
    g_p = torch.tensor(np.asarray(m_p, dtype=np.float64), requires_grad=True)
    loss, per_dim = assemble_terms(
        res, g_p, m_t, dims, unmatched_weight, essential_weight, "se",
        matched_weight, unmatched_target, "se")
    loss.backward()
    return float(loss.detach()), (-g_p.grad).numpy(), per_dim


# -----------------------------------------------------------------------------
# Batched mask build (GPU side) and the live loss path
# -----------------------------------------------------------------------------

def _batched_mask_build(preds, targets, tau, mask_backend):
    """Masks + PER-SAMPLE pocket labels for a batch (CPU numpy, build dtype).

    On GPU (``make_sparse_pair_torch``, cupy labelling if available) when preds is
    CUDA; scipy otherwise. Both are non-differentiable functions of the thresholded
    inputs, so doing them here -- separately from the differentiable loss -- loses
    no gradient and lets a training loop issue the next GPU forward before the
    (CPU-heavy) matching runs.

    Returns ``(m_p_list, m_t_list, keep_fracs, pocket_labels)``.
    """
    B = preds.shape[0]
    vd = _value_dtype()
    use_torch_mask = (mask_backend == "torch"
                      or (mask_backend == "auto" and preds.is_cuda))
    if not np.isfinite(tau):
        raise ValueError("the sparse matching requires a finite tau: the whole "
                         "construction is defined relative to it")
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
    """The one batched C++ call (one std::async task per sample). The confident
    clamp is MATCHER-ONLY: the terms read the exact masks."""
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
    """C++ matching + the autograd term assembly, from pre-built masks -> mean loss.

    ``preds`` is the foreground probability (B, *spatial) and carries the gradient
    for every "se" term; ``logits`` is the classifier output (B, C, *spatial) and
    carries it for every "ce" term. Returns the batch-MEAN loss as a live tensor.
    """
    B = preds.shape[0]
    results = _match_batch(m_p_list, m_t_list, tau, clamp_confident, pocket_labels)
    # float64 for the batch mean too, so a float32 prediction's loss is the float64
    # one to the last bit; only the finished scalar is cast back.
    total = torch.zeros((), dtype=torch.float64, device=preds.device)
    per_dim_sum = {int(d): 0.0 for d in dims}
    for b in range(B):
        # g_p is finite everywhere -- the +inf sentinels live in m_p_list. No term
        # ever gathers a masked voxel (critical cells are always kept), so the graph
        # stays free of infinities.
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
        # An empty barcode across the whole batch -- a correctly-empty patch, or a
        # prediction with nothing above the mask threshold -- leaves `total` a
        # constant, and .backward() on it would raise. One scalar op restores a grad
        # path without a reduction over the volume.
        total = total + preds.reshape(-1)[0].to(torch.float64) * 0.0
    return (total / B).to(preds.dtype)


class SparseBettiLossBatched(torch.nn.Module):
    """Batched sparse Betti-matching loss.

    ``forward(preds, targets, logits=None)`` takes stacked ``(B, *spatial)`` tensors
    (foreground probabilities; 2D or 3D) and runs the B per-sample matchings in ONE
    ``compute_matching([...], [...])`` call, which the C++ matcher parallelises
    across the batch. Returns the batch MEAN loss. When capture_stats=True,
    ``last_stats`` holds the batch-mean per-dimension loss, the mean kept fraction
    and the mean pocket count.

    Split into ``build_masks()`` (GPU, non-differentiable) and ``loss_from_masks()``
    (CPU C++ matching, differentiable) so a training loop can issue the next forward
    between them and overlap the CPU matching with the GPU work. ``forward()`` runs
    both back-to-back.

    Parameters
    ----------
    unmatched_weight : float
        Weight of the unmatched-prediction terms (finite bars to their other
        endpoint, essentials to background / GT).
    essential_weight : float
        Weight of the essential (censored) terms (matched birth term + unmatched
        essentials). 0 disables them.
    dims : tuple of int or None
        Homology dimensions to include. None (default) uses every dimension of the
        input: (0, 1) for 2D, (0, 1, 2) for 3D.
    theta, tau_mask : float
        Foreground threshold on IMAGE values; tau_mask = 1 - theta unless given.
        Must be finite.
    mask_backend : str
        "auto" (torch when the prediction is on the GPU, scipy otherwise), "torch"
        or "scipy". Both produce identical masks.
    capture_stats : bool
        Populate ``last_stats`` (monitoring only; never affects the loss).
    clamp_confident : float
        Collapse ``p >= clamp_confident`` onto one value in the array handed to the
        matcher (ties -> cheaper matching). The loss terms still read EXACT values,
        so this changes only which cells pair with which. 1.0 disables it.
    penalty_matched, penalty_unmatched : str
        "ce" or "se", independent; default ce on matched, se on unmatched. A "ce"
        half needs ``logits`` at call time.
    matched_weight : float
        Weight on the matched terms (finite pairs + matched essentials).
    unmatched_target : str
        "diagonal" (every endpoint -> the bar's other endpoint; background for an
        essential) or "gt" (the GT value at the voxel).
    fg_index : int
        Which softmax channel is the foreground, for the ce terms' logit gather.
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
            raise ValueError("tau_mask must be finite: the mask and the pockets are "
                             "defined relative to it")
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
        """GPU-side prepare step -> ``(m_p_list, m_t_list, keep_fracs, pocket_labels)``.

        Feed it straight through: ``loss_from_masks(preds, *build_masks(...))``.
        """
        assert preds.shape == targets.shape
        assert preds.ndim in (3, 4), "batched inputs: (B, *2D) or (B, *3D)"
        return _batched_mask_build(preds, targets, self._tau(), self.mask_backend)

    def loss_from_masks(self, preds: torch.Tensor, m_p_list, m_t_list,
                        keep_fracs, pocket_labels,
                        logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CPU-heavy C++ matching + assembly on pre-built masks -> mean loss.

        ``preds`` is the foreground probability ``(B, *spatial)`` -- the field the
        matching ran on, and the one every "se" term reads. ``logits`` is the raw
        classifier output ``(B, C, *spatial)``, required whenever either penalty is
        "ce": cross-entropy is computed there, never on the probability.
        """
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
    """Per-sample convenience: ``forward(pred, target, logits=None)`` on ONE
    unbatched ``(*spatial)`` pair (``logits`` as ``(C, *spatial)``). Same knobs,
    same code -- it is the batched class on a batch of one."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(pred.unsqueeze(0), target.unsqueeze(0),
                               None if logits is None else logits.unsqueeze(0))
