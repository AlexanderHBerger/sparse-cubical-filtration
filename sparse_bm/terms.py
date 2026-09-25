"""Differentiable torch assembly of the four loss-relevant Betti-matching channels.

The loss is ordinary torch expressions over the filtration values gathered at the
critical cells the matcher reported; autograd supplies the chain rule, the +g/-g pair
on a symmetric relative bar, and the accumulation when one voxel is the critical cell
of several bars (backward through advanced indexing accumulates duplicates, like
``np.add.at``). The four channels, and why each looks the way it does:

  matched finite       birth AND death terms; both targets read from the TARGET
                       complex at the TARGET's own coordinates. Both deaths are real
                       cells: a class censored on one side is censored on the other
                       (H_n(K) is the same space on both sides), so a censored class
                       never reaches this channel.
  matched essential    birth term only. Both deaths are censored at the background
                       value, so the death term would be A (1 - 1)^2 = 0 identically
                       and there is no death coordinate to gather.
  unmatched finite     to "diagonal" (the bar's other endpoint) or "gt" (each
                       endpoint's own GT value).
  unmatched essential  to "diagonal", which for a censored bar is its death, the
                       BACKGROUND value -- or to "gt", the GT value at the birth.

PENALTIES. ``penalty_matched`` and ``penalty_unmatched`` are independent, each "ce" or
"se", in every combination. The defaults are ce on matched and se on unmatched.

  se  ``A (g_pred - g_target)^2`` on the filtration values (g = 1 - p_fg).
  ce  the ORDINARY classification loss on the classifier's LOGITS: gather the full logit
      vector at the critical voxels and hand F.cross_entropy the target distribution
      over (background, foreground). A hard GT label is the one-hot case of that, so
      hard and soft targets go down ONE path with no branch. Gradient
      ``softmax(z) - target``: bounded, no reciprocal, no epsilon, and alive at a
      confident false negative -- which is the point. A matched term against a
      foreground target under se is ``2A(g - 0)`` in g, and the softmax chain rule
      multiplies it by ``p(1 - p)``, so at p = 1e-3 the term has 1e-3 of its nominal
      gradient left; under ce it has ``p - 1 = -0.999``.

WHY CE IS NOT COMPUTED ON g. Reconstructing a logit-space gradient from g needs a
division by ``p (1 - p)`` and hence an epsilon clip, which dominates the gradient on
confident voxels. Here the logits are in hand, the cancellation is exact and no epsilon
is needed.

SEPARATION OF CONCERNS. The matcher needs a scalar filtration; the loss needs the
classifier's full output. Only the first is a reduction, so only the first is reduced --
and at the point of use, not upstream. This module therefore takes BOTH:

  ``g_p``    (*spatial)     the scalar field the matching ran on, and the field every
                            se term reads. May carry +inf sentinels outside the mask.
  ``logits`` (C, *spatial)  the classifier output, which every ce term reads. Channel
                            selection happens here, in the gather.

They must agree on the kept voxels: ``g_p == 1 - softmax(logits, 0)[fg_index]``,
asserted on entry, because desynchronising them means the loss and the matching read
different fields.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .masks import CONE_BIRTH

#: The two penalties, each applicable to either half.
PENALTIES = ("se", "ce")

#: Where an unmatched prediction bar is sent. "diagonal" is RELATIVE: every endpoint
#: goes to the bar's other endpoint, which for an essential is BACKGROUND. "gt" is the
#: GT label at the voxel.
UNMATCHED_TARGETS = ("diagonal", "gt")

#: The background value of the fg-low convention (g = 1, p = 0). It is CONE_BIRTH: the
#: value every censored bar dies at, hence the "other endpoint" of an unmatched
#: essential.
BACKGROUND = CONE_BIRTH


def normalise_unmatched_target(unmatched_target: str) -> str:
    if unmatched_target not in UNMATCHED_TARGETS:
        raise ValueError(f"unmatched_target must be one of {UNMATCHED_TARGETS}; "
                         f"got {unmatched_target!r}")
    return unmatched_target


def g_from_logits(logits, fg_index=1):
    """(C, *spatial) logits -> the scalar filtration g = 1 - p_fg the matcher ran on."""
    return 1.0 - torch.softmax(logits, dim=0)[fg_index]


def assemble_terms(res, g_p, m_t_np, dims, unmatched_weight=1.0,
                   essential_weight=1.0, penalty_matched="ce",
                   matched_weight=1.0, unmatched_target="diagonal",
                   penalty_unmatched="se", logits=None, fg_index=1,
                   bg_index=0):
    """One BettiMatchingResult -> (loss tensor, per_dim dict).

    ``g_p`` is a differentiable tensor holding 1 - p_fg -- the field the matching ran on
    and the field every se term reads. ``logits`` is the classifier output
    ``(C, *spatial)``, required whenever either penalty is "ce". ``m_t_np`` is the numpy
    target filtration, used for the matched targets and the "gt" targets -- it carries no
    gradient. The returned loss is a 0-dim tensor on ``g_p``'s device.
    """
    dev = g_p.device
    # ACCUMULATE IN FLOAT64, then cast the scalar back. The terms are a few thousand
    # gathered critical-cell values -- not the volume -- so the promotion is free, and it
    # keeps a float32 prediction's loss identical to the float64 one to the last bit.
    # Returning float64 would instead promote the whole training loss.
    acc = torch.float64
    loss = torch.zeros((), dtype=acc, device=dev)
    per_dim = {int(d): 0.0 for d in dims}
    has_ess = hasattr(res, "num_matched_essentials")

    unmatched_target = normalise_unmatched_target(unmatched_target)
    for name, pen in (("penalty_matched", penalty_matched),
                      ("penalty_unmatched", penalty_unmatched)):
        if pen not in PENALTIES:
            raise ValueError(f"{name} must be one of {PENALTIES}; got {pen!r}")
    if "ce" in (penalty_matched, penalty_unmatched):
        if logits is None:
            raise ValueError(
                'a "ce" penalty needs the classifier output `logits`, shape '
                "(C, *spatial): ce is computed on the logits, never on g (see the "
                "module docstring).")
        if logits.dim() != g_p.dim() + 1 or logits.shape[1:] != g_p.shape:
            raise ValueError(f"logits{tuple(logits.shape)} must be (C, *g_p.shape) "
                             f"with g_p.shape={tuple(g_p.shape)}")
        # The two representations must be the same field. Cheap next to the C++
        # matching. Only the KEPT voxels are compared: g_p may carry +inf sentinels
        # outside the mask, and no term ever reads one (critical cells are always kept).
        fin = torch.isfinite(g_p)
        if not torch.allclose(g_p[fin], g_from_logits(logits, fg_index)[fin],
                              rtol=1e-4, atol=1e-6):
            raise ValueError(
                "g_p disagrees with 1 - softmax(logits)[fg_index] on the kept voxels. "
                "Build g_p from the SAME logits via g_from_logits, or the loss and the "
                "matching read different fields.")
    m_t_np = np.asarray(m_t_np, dtype=np.float64)

    def _ix(a):
        return tuple(np.asarray(a).T)

    def gp(a):
        """Differentiable gather of g_p at an (N, ndim) coordinate array."""
        c = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=dev)
        return g_p[tuple(c[:, i] for i in range(c.shape[1]))].to(acc)

    def gt(a):
        """Constant gather of the target filtration."""
        v = m_t_np[_ix(a)]
        if not np.isfinite(v).all():
            raise ValueError("a target critical cell is masked: the matcher reported "
                             "a coordinate outside the kept complex")
        return torch.as_tensor(v, dtype=acc, device=dev)

    def ce(coords, g_target, A):
        """The ordinary classification loss at a set of critical voxels.

        ``g_target`` is the target in g-space (0 = foreground), as a tensor. It is turned
        into a distribution over (bg_index, fg_index) and handed to F.cross_entropy with
        the gathered logit vectors. A hard GT label is the one-hot case, so a hard target
        and a soft/relative one take the same path and agree exactly with the
        class-index form where both apply.
        """
        c = torch.as_tensor(np.ascontiguousarray(coords), dtype=torch.long, device=dev)
        ix = tuple(c[:, i] for i in range(c.shape[1]))
        z = logits[(slice(None),) + ix].T.to(acc)                # (N, C)
        q = torch.zeros_like(z)
        q[:, bg_index] = g_target
        q[:, fg_index] = 1.0 - g_target
        return A * F.cross_entropy(z, q, reduction="sum")

    def term(coords, g_target, A, penalty):
        """One group of terms against a common target, under either penalty."""
        if penalty == "ce":
            return ce(coords, g_target, A)
        return (A * (gp(coords) - g_target) ** 2).sum()

    for d in dims:
        before = loss

        # ---- 1. finite matched pairs ------------------------------------------------
        b1 = np.asarray(res.input1_matched_birth_coordinates[d])
        if b1.shape[0]:
            d1 = np.asarray(res.input1_matched_death_coordinates[d])
            b2 = np.asarray(res.input2_matched_birth_coordinates[d])
            d2 = np.asarray(res.input2_matched_death_coordinates[d])
            loss = loss + term(b1, gt(b2), matched_weight, penalty_matched)
            loss = loss + term(d1, gt(d2), matched_weight, penalty_matched)

        # ---- 2. finite unmatched prediction bars ------------------------------------
        ub = np.asarray(res.input1_unmatched_birth_coordinates[d])
        if ub.shape[0]:
            ud = np.asarray(res.input1_unmatched_death_coordinates[d])
            A = unmatched_weight * 0.5
            if unmatched_target == "gt":
                loss = loss + term(ub, gt(ub), A, penalty_unmatched)
                loss = loss + term(ud, gt(ud), A, penalty_unmatched)
            elif penalty_unmatched == "se":
                # A RELATIVE target. ONE symmetric term; autograd delivers +g to the
                # birth and -g to the death.
                loss = loss + (A * (gp(ub) - gp(ud)) ** 2).sum()
            else:
                # ce is not symmetric, so it is two terms, each pulling one endpoint
                # toward the other HELD FIXED -- the one place a detach is required.
                loss = loss + ce(ub, gp(ud).detach(), A)
                loss = loss + ce(ud, gp(ub).detach(), A)

        # ---- 3 + 4. essentials ------------------------------------------------------
        if essential_weight != 0.0 and has_ess:
            eb1 = np.asarray(res.input1_matched_essential_birth_coordinates[d])
            if eb1.shape[0]:
                eb2 = np.asarray(res.input2_matched_essential_birth_coordinates[d])
                loss = loss + term(eb1, gt(eb2), essential_weight * matched_weight,
                                   penalty_matched)
            ueb = np.asarray(res.input1_unmatched_essential_birth_coordinates[d])
            if ueb.shape[0]:
                A = essential_weight * unmatched_weight * 0.5
                if unmatched_target == "gt":
                    loss = loss + term(ueb, gt(ueb), A, penalty_unmatched)
                else:
                    bg = torch.full((ueb.shape[0],), float(BACKGROUND), dtype=acc,
                                    device=dev)
                    loss = loss + term(ueb, bg, A, penalty_unmatched)

        per_dim[int(d)] = float((loss - before).detach())
    return loss.to(g_p.dtype), per_dim
