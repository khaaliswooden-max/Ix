"""
ix_substrate.config
────────────────────
YAML config parsing for `ix-substrate run`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeedSpec:
    kind: str       # "cex"
    venue: str
    symbol: str
    poll_ms: int = 1000


@dataclass
class SubstrateConfig:
    feeds: list[FeedSpec] = field(default_factory=list)
    db_path: str = "ix_substrate.duckdb"
    bucket_ms: int = 250
    threshold_bps: float = 3.0
    expected_band_multiplier: float = 1.5

    @classmethod
    def load(cls, path: str | Path) -> "SubstrateConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        feeds = [FeedSpec(**f) for f in raw.pop("feeds", [])]
        return cls(feeds=feeds, **raw)
