"""
ix_substrate — Phase 0 of the IX agent.

Public surface:

    from ix_substrate.contracts.events import TickEvent, DisagreementEvent
    from ix_substrate.reconciliation.disagreement import (
        PriceDisagreementDetector, DisagreementConfig,
    )
    from ix_substrate.reconciliation.alignment import align_two, AlignConfig
    from ix_substrate.storage.duckdb_store import DuckDBStore
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
