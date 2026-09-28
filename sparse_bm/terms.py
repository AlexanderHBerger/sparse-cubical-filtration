"""Loss terms of sparseBM, assembled with autograd from the critical cells of a matching.

  matched pairs            birth and death pulled to the matched target values
  matched essentials       birth pulled to the target birth (both deaths are censored)
  unmatched pairs          to the other endpoint of the bar ("diagonal") or to the
                           label value ("gt")
  unmatched essentials     to the background value ("diagonal") or to the label ("gt")

Each group uses squared error on g = 1 - p ("se") or cross-entropy on the logits ("ce").
Unmatched target bars get no term.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .masks import CONE_BIRTH

PENALTIES = ("se", "ce")
UNMATCHED_TARGETS = ("diagonal", "gt")
BACKGROUND = CONE_BIRTH


def normalise_unmatched_target(unmatched_target: str) -> str:
    if unmatched_target not in UNMATCHED_TARGETS:
        raise ValueError(f"unmatched_target must be one of {UNMATCHED_TARGETS}; "
                         f"got {unmatched_target!r}")
    return unmatched_target


def g_from_logits(logits, fg_index=1):
    return 1.0 - torch.softmax(logits, dim=0)[fg_index]


def assemble_terms(res, g_p, m_t_np, dims, unmatched_weight=1.0,
                   essential_weight=1.0, penalty_matched="ce",
                   matched_weight=1.0, unmatched_target="diagonal",
                   penalty_unmatched="se", logits=None, fg_index=1,
                   bg_index=0):
    """Loss of one matching result. Returns ``(loss, per_dim)``.

    ``g_p`` is the prediction filtration (1 - p_fg) as a differentiable tensor, ``m_t_np``
    the masked target filtration, and ``logits`` the network output ``(C, *spatial)``,
    which is required for "ce".
    """
    dev = g_p.device
    # Accumulated in float64 and cast back at the end.
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
                'a "ce" penalty needs the network output `logits`, shape (C, *spatial)')
        if logits.dim() != g_p.dim() + 1 or logits.shape[1:] != g_p.shape:
            raise ValueError(f"logits{tuple(logits.shape)} must be (C, *g_p.shape) "
                             f"with g_p.shape={tuple(g_p.shape)}")
        fin = torch.isfinite(g_p)
        if not torch.allclose(g_p[fin], g_from_logits(logits, fg_index)[fin],
                              rtol=1e-4, atol=1e-6):
            raise ValueError(
                "g_p does not match 1 - softmax(logits)[fg_index] on the kept voxels")
    m_t_np = np.asarray(m_t_np, dtype=np.float64)

    def _ix(a):
        return tuple(np.asarray(a).T)

    def gp(a):
        c = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=dev)
        return g_p[tuple(c[:, i] for i in range(c.shape[1]))].to(acc)

    def gt(a):
        v = m_t_np[_ix(a)]
        if not np.isfinite(v).all():
            raise ValueError("the matcher reported a masked target cell")
        return torch.as_tensor(v, dtype=acc, device=dev)

    def ce(coords, g_target, A):
        # Cross-entropy against the target distribution (g_target, 1 - g_target) over
        # (background, foreground).
        c = torch.as_tensor(np.ascontiguousarray(coords), dtype=torch.long, device=dev)
        ix = tuple(c[:, i] for i in range(c.shape[1]))
        z = logits[(slice(None),) + ix].T.to(acc)                # (N, C)
        q = torch.zeros_like(z)
        q[:, bg_index] = g_target
        q[:, fg_index] = 1.0 - g_target
        return A * F.cross_entropy(z, q, reduction="sum")

    def term(coords, g_target, A, penalty):
        if penalty == "ce":
            return ce(coords, g_target, A)
        return (A * (gp(coords) - g_target) ** 2).sum()

    for d in dims:
        before = loss

        b1 = np.asarray(res.input1_matched_birth_coordinates[d])
        if b1.shape[0]:
            d1 = np.asarray(res.input1_matched_death_coordinates[d])
            b2 = np.asarray(res.input2_matched_birth_coordinates[d])
            d2 = np.asarray(res.input2_matched_death_coordinates[d])
            loss = loss + term(b1, gt(b2), matched_weight, penalty_matched)
            loss = loss + term(d1, gt(d2), matched_weight, penalty_matched)

        ub = np.asarray(res.input1_unmatched_birth_coordinates[d])
        if ub.shape[0]:
            ud = np.asarray(res.input1_unmatched_death_coordinates[d])
            A = unmatched_weight * 0.5
            if unmatched_target == "gt":
                loss = loss + term(ub, gt(ub), A, penalty_unmatched)
                loss = loss + term(ud, gt(ud), A, penalty_unmatched)
            elif penalty_unmatched == "se":
                loss = loss + (A * (gp(ub) - gp(ud)) ** 2).sum()
            else:
                # ce is not symmetric: one term per endpoint, with the other one fixed.
                loss = loss + ce(ub, gp(ud).detach(), A)
                loss = loss + ce(ud, gp(ub).detach(), A)

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
