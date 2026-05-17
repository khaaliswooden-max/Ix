"""
ix_strategy_synthesis.planner.constraints
──────────────────────────────────────────
Hard constraints applied to candidate actions before the planner scores
them. The constraint surface in v0.1 is small but non-negotiable:

  1. RISK_OFF proposals from any hypothesis dominate normal selection.
     If ANY hypothesis emits RISK_OFF, the planner emits RISK_OFF.
     (LiquidationAvoider in particular.)

  2. Size fractions are clipped to a configured maximum per call. The
     planner cannot deploy more than this fraction even if a hypothesis
     proposes it. (Phase 4 will refine sizing via Kelly with Brier
     damping; v0.1 just clips.)

  3. P(ruin) ≤ ε is enforced as a *bid-ask spread* on confidence: a
     hypothesis with confidence < epsilon_floor is rejected. The full
     formal constraint π* = argmax E[R] s.t. P(ruin) ≤ ε requires Phase
     4's risk infrastructure; v0.1 implements the upper-bound proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..contracts.events import ActionKind, ActionProposal


@dataclass(frozen=True)
class ConstraintConfig:
    """
    max_size_fraction:
        Absolute clip on |size_fraction|. v0.1 default 0.25 = 25% of
        capital max per single action. Phase 4 will replace this with
        Kelly-derived sizing.

    epsilon_floor:
        Hypotheses with confidence below this floor are rejected
        (returned as HOLD). Proxy for P(ruin) constraint until Phase 4.

    risk_off_dominates:
        If any proposal is RISK_OFF, the planner emits RISK_OFF
        regardless of other proposals. The chosen proposal is the
        highest-confidence RISK_OFF.
    """
    max_size_fraction: Decimal = Decimal("0.25")
    epsilon_floor: float = 0.05
    risk_off_dominates: bool = True


def filter_and_clip(
    proposals: list[ActionProposal],
    config: ConstraintConfig | None = None,
) -> list[ActionProposal]:
    """
    Apply hard constraints to a list of proposals.

    Returns a new list (does not mutate). May replace proposals with
    HOLD-equivalent variants when constraints reject them.

    Order of operations:
      1. If risk_off_dominates and any RISK_OFF exists, return only the
         highest-confidence RISK_OFF.
      2. Filter out proposals with confidence < epsilon_floor (replaced
         with HOLD).
      3. Clip |size_fraction| to max_size_fraction (preserving sign).
    """
    cfg = config or ConstraintConfig()

    # Stage 1: RISK_OFF domination
    if cfg.risk_off_dominates:
        risk_offs = [p for p in proposals if p.action == ActionKind.RISK_OFF]
        if risk_offs:
            top = max(risk_offs, key=lambda p: p.confidence)
            return [top]

    out: list[ActionProposal] = []
    for p in proposals:
        # Stage 2: epsilon floor — HOLDs are exempt (their confidence is 0 by design)
        if p.action != ActionKind.HOLD and p.confidence < cfg.epsilon_floor:
            out.append(_to_hold(p, f"rejected: confidence {p.confidence:.3f} < floor"))
            continue

        # Stage 3: size clipping (preserve sign and action)
        size = p.size_fraction
        if abs(size) > cfg.max_size_fraction:
            clipped = cfg.max_size_fraction if size > 0 else -cfg.max_size_fraction
            out.append(ActionProposal(
                hypothesis=p.hypothesis,
                symbol=p.symbol,
                window_end=p.window_end,
                action=p.action,
                size_fraction=clipped,
                confidence=p.confidence,
                description_length_bits=p.description_length_bits,
                rationale=f"{p.rationale} (size clipped from {size} to {clipped})",
            ))
            continue

        out.append(p)
    return out


def _to_hold(p: ActionProposal, rationale: str) -> ActionProposal:
    return ActionProposal(
        hypothesis=p.hypothesis,
        symbol=p.symbol,
        window_end=p.window_end,
        action=ActionKind.HOLD,
        size_fraction=Decimal(0),
        confidence=0.0,
        description_length_bits=p.description_length_bits,
        rationale=rationale,
    )
