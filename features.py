"""
ix_world_model.hypotheses.ensemble
───────────────────────────────────
Bayesian credit assignment over a finite ensemble of named strategy
hypotheses. This is the second AIXI-style mixture in IX (the first being
the regime classifier).

KEY ARCHITECTURAL DECISION
──────────────────────────
The ensemble does NOT execute strategies. It maintains *weights* over
hypotheses based on their *realized out-of-sample PnL* on a rolling
window. Phase 2 reads these weights and decides which hypothesis (or
which weighted combination) to execute. Phase 4 reads them and bounds
position size by the *agreement* in the ensemble.

This separation is the difference between AIXI-correct and AIXI-flavored:
the world model's job is induction (what does the data say about which
theories are right?), not action (what should I do?).

Update rule
───────────
For each hypothesis h, we maintain:

    cumulative_pnl(h)    -- realized OOS PnL on rolling window
    score(h)             -- bounded transform of cumulative_pnl

We normalize scores by softmax to get weights. The softmax temperature
controls exploration vs. exploitation: high temperature → flatter
weights (the ensemble disagrees more, exploration encouraged);
low temperature → sharp weights (concentrated on the historical leader).

We separately track the *empirical disagreement* of the ensemble. High
disagreement = the hypotheses don't agree about what to do, which is
itself a signal Phase 2/4 can use.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from ..contracts.events import HypothesisWeights


# ── Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EnsembleConfig:
    """
    pnl_window:
        Number of recent observations of realized PnL to retain per
        hypothesis. Older PnL falls off — implements an effective
        rolling-window evaluation.

    softmax_temperature:
        Controls how sharply weights concentrate on the leader.
        1.0 = neutral; <1.0 = sharpen (exploit); >1.0 = flatten (explore).

    min_weight:
        Floor on each hypothesis's weight after softmax. Prevents any
        hypothesis from being permanently locked out of the ensemble.

    score_scale:
        Divisor applied to cumulative PnL before softmax. Tunes how much
        PnL difference is needed to meaningfully shift weights. Should
        be on the order of typical hypothesis-level PnL variance.
    """
    pnl_window: int = 256
    softmax_temperature: float = 1.0
    min_weight: float = 0.01
    score_scale: float = 100.0  # in PnL units (e.g. USD)


# ── Per-hypothesis state ─────────────────────────────────────────────────

@dataclass
class _HypothesisState:
    name: str
    pnl_window: int
    samples: deque[Decimal] = field(default_factory=deque)
    cumulative: Decimal = Decimal(0)

    def push(self, pnl: Decimal) -> None:
        self.samples.append(pnl)
        self.cumulative += pnl
        if len(self.samples) > self.pnl_window:
            old = self.samples.popleft()
            self.cumulative -= old


# ── Ensemble ─────────────────────────────────────────────────────────────

class HypothesisEnsemble:
    """
    Maintains weights over a fixed roster of named hypotheses. One
    ensemble per symbol.

    Usage:
        ens = HypothesisEnsemble("BTC/USDT", ["trend", "meanrev", "carry"])
        # Phase 2 / Phase 3 report realized PnL to us:
        ens.report_pnl({"trend": Decimal("12.50"),
                        "meanrev": Decimal("-3.20"),
                        "carry": Decimal("4.10")})
        # We emit current weights for downstream consumption:
        w = ens.weights(window_end=now)
    """

    def __init__(
        self,
        symbol: str,
        hypothesis_names: list[str],
        config: EnsembleConfig | None = None,
    ):
        if not hypothesis_names:
            raise ValueError("hypothesis_names cannot be empty")
        if len(set(hypothesis_names)) != len(hypothesis_names):
            raise ValueError("hypothesis_names contains duplicates")

        self.symbol = symbol
        self.cfg = config or EnsembleConfig()
        self._states: dict[str, _HypothesisState] = {
            n: _HypothesisState(name=n, pnl_window=self.cfg.pnl_window)
            for n in hypothesis_names
        }
        self._observations = 0

    def report_pnl(self, pnl_by_hypothesis: Mapping[str, Decimal]) -> None:
        """
        Each hypothesis must report its realized PnL for this step. A
        hypothesis that did not trade reports Decimal(0); a hypothesis
        that does not exist in the ensemble is rejected.
        """
        for name, pnl in pnl_by_hypothesis.items():
            if name not in self._states:
                raise ValueError(f"unknown hypothesis: {name!r}")
            self._states[name].push(pnl)
        # Any hypothesis NOT reported in this call is treated as 0 PnL.
        for name, state in self._states.items():
            if name not in pnl_by_hypothesis:
                state.push(Decimal(0))
        self._observations += 1

    def weights(self, window_end: datetime) -> HypothesisWeights:
        """Compute current weights via softmax over cumulative PnL scores."""
        names = list(self._states.keys())
        scores = [
            float(self._states[n].cumulative) / self.cfg.score_scale / self.cfg.softmax_temperature
            for n in names
        ]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        Z = sum(exps)
        raw_weights = [e / Z for e in exps]

        # Apply min-weight floor by water-filling: any hypothesis below
        # the floor is clipped to the floor; remaining mass is renormalized
        # over the unclipped hypotheses. The floor is then guaranteed to
        # hold post-normalization (not just pre-).
        floor = self.cfg.min_weight
        n_hyp = len(raw_weights)
        if floor * n_hyp > 1.0:
            raise ValueError(
                f"min_weight={floor} × {n_hyp} hypotheses exceeds 1.0 — "
                f"reduce min_weight to at most {1.0 / n_hyp:.4f}"
            )

        normalized = list(raw_weights)
        for _ in range(n_hyp):  # at most N passes to reach fixed point
            below = [i for i, w in enumerate(normalized) if w < floor]
            if not below:
                break
            mass_clipped = floor * len(below)
            mass_remaining = 1.0 - mass_clipped
            above_indices = [i for i in range(n_hyp) if i not in below]
            above_total = sum(normalized[i] for i in above_indices)
            if above_total <= 0 or not above_indices:
                # Edge case: every hypothesis is at floor; fall back to uniform
                normalized = [1.0 / n_hyp] * n_hyp
                break
            for i in below:
                normalized[i] = floor
            for i in above_indices:
                normalized[i] = normalized[i] / above_total * mass_remaining

        weight_map = {n: w for n, w in zip(names, normalized)}
        pnl_map = {n: self._states[n].cumulative for n in names}

        return HypothesisWeights(
            symbol=self.symbol,
            window_end=window_end,
            weights=weight_map,
            realized_pnl_window=pnl_map,
            observations_used=max(self._observations, 1),
        )
