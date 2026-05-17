"""
ix_strategy_synthesis.planner.mcts
───────────────────────────────────
Monte Carlo Tree Search planner over hypothesis-generated action
proposals. v0.1 is single-step: given N proposals at one snapshot, score
each via simulated rollouts and pick the highest-scoring one (subject to
the complexity penalty and hard constraints).

Multi-step lookahead (proper MCTS with selection/expansion/simulation/
backpropagation across multiple time-steps) belongs in v0.2, after Phase
3 emits real fill outcomes that the planner can roll forward against. In
v0.1 we use the hypothesis-reported confidence × size as the value
estimate, which is the AIXI-correct quantity in the single-step limit.

Why single-step is the right scope for v0.1:

  1. Multi-step MCTS requires a forward model of how the market evolves
     given an action. That forward model is Phase 1's regime classifier
     run forward in time, which we have as a one-step prediction but not
     as a multi-step trajectory generator. Building the trajectory
     generator is Phase 6 territory (self-improvement learns the
     trajectory model from realized data).

  2. The complexity-penalty mechanism — the actual non-obvious gap — is
     orthogonal to depth. Single-step demonstrates the penalty cleanly;
     multi-step adds rollout depth but does not change the gap claim.

  3. Live latency budget at the operating cadence (minute-scale) does
     not require deep search. Phase 0's substrate emits ticks at ~1 Hz;
     a 1-minute decision cadence has ample headroom for deeper search
     in v0.2 if backtests justify it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.events import ActionKind, ActionProposal, ExecutionRequest
from .complexity_penalty import ComplexityPenaltyConfig, apply_penalty
from .constraints import ConstraintConfig, filter_and_clip

# Phase 1 types — soft import for isolated tests
try:
    from ix_world_model.contracts.events import MarketSnapshot
except ImportError:  # pragma: no cover
    MarketSnapshot = object  # type: ignore


@dataclass(frozen=True)
class PlannerConfig:
    """
    rollouts_per_proposal:
        Number of Monte Carlo rollouts per candidate. v0.1 simulates a
        single-step outcome under each proposal, so this is really just
        the noise-averaging count. Set to 1 for deterministic value.

    rollout_horizon_steps:
        For v0.2 multi-step MCTS. Ignored in v0.1 single-step.

    hold_baseline_value:
        Value assigned to HOLD action. Default 0.0 — no expected return,
        no expected loss. Other actions must exceed this (after
        complexity penalty) to be chosen.
    """
    rollouts_per_proposal: int = 1
    rollout_horizon_steps: int = 1
    hold_baseline_value: float = 0.0


class MCTSPlanner:
    """
    Single-step planner over hypothesis proposals.

    Usage:
        planner = MCTSPlanner()
        proposals = [h.propose(snapshot) for h in roster]
        request = planner.choose(snapshot, proposals)
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
        complexity_config: ComplexityPenaltyConfig | None = None,
        constraint_config: ConstraintConfig | None = None,
    ):
        self.cfg = config or PlannerConfig()
        self.complexity_cfg = complexity_config or ComplexityPenaltyConfig()
        self.constraint_cfg = constraint_config or ConstraintConfig()

    def choose(
        self,
        snapshot: MarketSnapshot,
        proposals: list[ActionProposal],
    ) -> ExecutionRequest:
        if not proposals:
            raise ValueError("proposals cannot be empty")
        for p in proposals:
            if p.symbol != snapshot.symbol:
                raise ValueError(
                    f"proposal symbol {p.symbol!r} != snapshot {snapshot.symbol!r}"
                )
            if p.window_end != snapshot.window_end:
                raise ValueError(
                    "proposal window_end does not match snapshot"
                )

        # Apply hard constraints first. May reduce the candidate set to
        # a single RISK_OFF.
        filtered = filter_and_clip(proposals, self.constraint_cfg)

        # Score each surviving proposal under the complexity-penalty
        # objective. HOLD's raw value (hold_baseline_value) is the
        # implicit floor: any non-HOLD action must beat HOLD on
        # *penalized* value, or HOLD wins by normal scoring.
        scored: list[tuple[ActionProposal, float, float]] = []  # (prop, penalized_value, penalty)
        for p in filtered:
            raw = self._raw_value(p)
            penalized, penalty = apply_penalty(
                raw, p.description_length_bits, self.complexity_cfg
            )
            scored.append((p, penalized, penalty))

        # Choose the highest penalized-value proposal. Tie-breaks favor
        # lower complexity (smaller bits), then lexicographic hypothesis name.
        scored.sort(key=lambda t: (-t[1], t[0].description_length_bits, t[0].hypothesis))
        chosen, chosen_value, chosen_penalty = scored[0]

        # Safety case: if the chosen action is non-HOLD and its
        # penalized value is below the minimum floor, AND no HOLD is
        # already in the candidate set, synthesize one as a fallback.
        # If a HOLD already exists in the candidates, it would have
        # been chosen by normal scoring — no synthesis needed.
        candidate_list = list(filtered)
        if (
            chosen.action != ActionKind.HOLD
            and chosen_value < self.complexity_cfg.min_value_to_overcome_penalty
            and not any(p.action == ActionKind.HOLD for p in filtered)
        ):
            fallback = _synth_hold(snapshot, "all non-HOLD proposals failed penalty floor")
            chosen = fallback
            chosen_value = self.cfg.hold_baseline_value
            chosen_penalty = self.complexity_cfg.coefficient * fallback.description_length_bits
            candidate_list = candidate_list + [fallback]

        return ExecutionRequest(
            symbol=snapshot.symbol,
            window_end=snapshot.window_end,
            chosen_proposal=chosen,
            candidate_proposals=tuple(candidate_list),
            planner_value=chosen_value,
            complexity_penalty_applied=chosen_penalty,
            regime_at_decision=snapshot.regime.mode.value,
            regime_mode_probability=snapshot.regime.mode_probability,
        )

    # ── value model (v0.1 single-step) ─────────────────────────

    def _raw_value(self, proposal: ActionProposal) -> float:
        """
        Single-step value estimate:
            V = confidence × |size_fraction| × scale - HOLD baseline.

        HOLD is anchored at hold_baseline_value (default 0). RISK_OFF is
        treated as a moderate positive value when triggered (avoiding
        ruin is worth something), but the constraint layer will already
        have promoted it to the chosen action if appropriate.

        This is intentionally simple. v0.2 with multi-step MCTS will
        replace this with a proper rollout value.
        """
        if proposal.action == ActionKind.HOLD:
            return self.cfg.hold_baseline_value
        if proposal.action == ActionKind.RISK_OFF:
            # Worth a moderate positive value — survival is part of return.
            return 5.0 * proposal.confidence
        if proposal.action == ActionKind.CLOSE:
            return 1.0  # always slightly preferred over HOLD when proposed
        # LONG / SHORT
        return float(abs(proposal.size_fraction)) * proposal.confidence * 10.0


def _synth_hold(snapshot, rationale: str) -> ActionProposal:
    from decimal import Decimal
    return ActionProposal(
        hypothesis="planner_fallback",
        symbol=snapshot.symbol,
        window_end=snapshot.window_end,
        action=ActionKind.HOLD,
        size_fraction=Decimal(0),
        confidence=0.0,
        description_length_bits=1.0,
        rationale=rationale,
    )
