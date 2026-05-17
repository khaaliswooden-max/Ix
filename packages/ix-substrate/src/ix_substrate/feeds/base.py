"""
ix_substrate.feeds.base
────────────────────────
Protocol all feed sources implement. Async-first so the substrate can run
many concurrent venues without thread-pool overhead.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..contracts.events import TickEvent


@runtime_checkable
class Feed(Protocol):
    """A streaming source of TickEvent."""

    venue: str
    symbols: tuple[str, ...]

    async def __aenter__(self) -> "Feed": ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    def stream(self) -> AsyncIterator[TickEvent]:
        """Yield TickEvents until the feed is closed."""
        ...
