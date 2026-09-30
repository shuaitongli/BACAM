"""Compressed KL and behavior-aware state loss."""

from __future__ import annotations

import torch


EPS = 1e-12


def compressed_topk_kl(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    teacher_other: torch.Tensor,
    direction: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return one KL value per response token.

    The teacher distribution is stored as its top-32 tokens plus one bucket for
    every other token.  ``forward`` is KL(teacher || student); ``reverse`` is
    KL(student || teacher).  The tail is a compressed approximation by design.
    """
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"direction must be forward or reverse, got {direction!r}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    student_log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
    student_top_log = torch.gather(
        student_log_probs, dim=-1, index=topk_ids.long())
    student_top = student_top_log.exp()
    student_other = (1.0 - student_top.sum(dim=-1)).clamp_min(EPS)

    teacher_top = teacher_topk_probs.float().clamp_min(EPS)
    teacher_tail = teacher_other.float().clamp_min(EPS)
    teacher_log_top = teacher_top.log()
    teacher_log_tail = teacher_tail.log()

    if direction == "forward":
        top_term = teacher_top * (teacher_log_top - student_top_log)
        tail_term = teacher_tail * (teacher_log_tail - student_other.log())
    else:
        student_log_tail = student_other.log()
        top_term = student_top * (student_top_log - teacher_log_top)
        tail_term = student_other * (student_log_tail - teacher_log_tail)
    return ((top_term.sum(dim=-1) + tail_term) * (temperature ** 2)).clamp_min(0.0)


def behavior_aware_state_loss(token_losses: torch.Tensor,
                              critical_mask: torch.Tensor) -> torch.Tensor:
    """Give behavior and non-behavior token groups equal mass when both exist."""
    values = token_losses.reshape(-1)
    mask = critical_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    if values.numel() == 0 or values.numel() != mask.numel():
        raise ValueError("token losses and critical mask must have the same nonzero length")
    if bool(mask.any()) and bool((~mask).any()):
        return 0.5 * values[mask].mean() + 0.5 * values[~mask].mean()
    return values.mean()


def raw_kl_drift(value: torch.Tensor | float,
                 at_old: float) -> torch.Tensor | float:
    """Return old-task KL increase above the old-model endpoint."""
    return value - at_old
