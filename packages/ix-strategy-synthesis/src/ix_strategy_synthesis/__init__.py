"""
ix_strategy_synthesis — Phase 2 of the IX agent.

Public surface:

    from ix_strategy_synthesis.contracts.events import (
        ActionKind, ActionProposal, ExecutionRequest, RealizationReport,
    )
    from ix_strategy_synthesis.synthesis.orchestrator import (
        StrategySynthesizer, SynthesizerConfig, FillSimulator,
    )
    from ix_strategy_synthesis.hypotheses.specialists import default_roster
    from ix_strategy_synthesis.planner.mcts import MCTSPlanner
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
