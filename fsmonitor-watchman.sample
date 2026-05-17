"""
ix_world_model.regimes.classifier
──────────────────────────────────
Bayesian classifier over discrete regime states.

Architecture
─────────────
For each Regime r, we maintain:

  - a *profile*: the expected (mean) feature vector under that regime,
    plus per-feature standard deviations
  - a *Dirichlet prior count* representing how often r has been
    "credited" with explaining observed features

On each new FeatureVector:

  1. Compute likelihood P(features | regime) under a diagonal Gaussian
     centered on the profile for each regime.
  2. Update Dirichlet counts via soft credit assignment (proportional to
     likelihood).
  3. Emit a RegimePosterior by normalizing Dirichlet counts.

Why this design
───────────────
This is *AIXI's mixture-of-experts pattern, finitized*. Each regime is an
"expert hypothesis" about how the market behaves; the Dirichlet
posterior is the AIXI-style credit assignment, restricted to a finite
tractable model class. Solomonoff's universal prior is replaced by a
uniform Dirichlet prior over regimes; the mixture weighting comes from
realized likelihoods rather than program length.

Profiles are configurable. v0.1 ships with literature-default profiles
that work as cold-start priors. Phase 6 (self-improvement) will update
profile parameters from realized PnL — but that's deferred until Phase
1-2-3 are running.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from ..contracts.events import Regime, RegimePosterior
from .features import FeatureVector


# ── Regime profiles (the model class) ───────────────────────────────────

@dataclass(frozen=True)
class RegimeProfile:
    """
    Expected feature values under a regime, with per-feature spread used
    as the Gaussian likelihood scale.

    All scales must be > 0.
    """
    realized_vol_mean:           float
    realized_vol_scale:          float
    autocorr_lag1_mean:          float
    autocorr_lag1_scale:         float
    spread_bps_mean:             float
    spread_bps_scale:            float
    disagreement_intensity_mean: float
    disagreement_intensity_scale: float

    def log_likelihood(self, fv: FeatureVector) -> float:
        """log P(features | this regime) under independent Gaussians."""
        return (
            _log_gauss(fv.realized_vol_annualized,    self.realized_vol_mean,    self.realized_vol_scale)
            + _log_gauss(fv.autocorr_lag1,            self.autocorr_lag1_mean,   self.autocorr_lag1_scale)
            + _log_gauss(fv.spread_bps_mean,          self.spread_bps_mean,      self.spread_bps_scale)
            + _log_gauss(fv.disagreement_intensity_bps, self.disagreement_intensity_mean, self.disagreement_intensity_scale)
        )


def _log_gauss(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma})")
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)


# ── Cold-start default profiles ──────────────────────────────────────────
#
# IMPORTANT: These defaults are calibrated for **minute-cadence** crypto
# data (one observation per minute, annualization factor 365*24*60 ≈ 525K).
#
# If you feed second-cadence data into a classifier using these profiles,
# realized vol will be much higher than the profiles expect and the
# classifier will lock onto VOLATILE or BROKEN. Either:
#
#   (a) downsample your tick stream to minute-cadence before classification, or
#   (b) supply custom profiles whose feature scales match your cadence, or
#   (c) preprocess realized_vol_annualized to match the calibration cadence
#       (multiply by sqrt(target_cadence / actual_cadence)).
#
# Phase 6 (self-improvement) will fit these profiles to realized data
# automatically, eliminating the manual calibration step. Until then,
# document the cadence assumption alongside any new strategy that uses
# the classifier.

DEFAULT_PROFILES: dict[Regime, RegimeProfile] = {
    Regime.TREND_UP: RegimeProfile(
        realized_vol_mean=0.40,           realized_vol_scale=0.20,
        autocorr_lag1_mean=0.15,          autocorr_lag1_scale=0.10,
        spread_bps_mean=3.0,              spread_bps_scale=2.0,
        disagreement_intensity_mean=2.0,  disagreement_intensity_scale=2.0,
    ),
    Regime.TREND_DOWN: RegimeProfile(
        realized_vol_mean=0.60,           realized_vol_scale=0.30,
        autocorr_lag1_mean=0.15,          autocorr_lag1_scale=0.10,
        spread_bps_mean=4.0,              spread_bps_scale=3.0,
        disagreement_intensity_mean=3.0,  disagreement_intensity_scale=2.5,
    ),
    Regime.MEAN_REVERT: RegimeProfile(
        realized_vol_mean=0.30,           realized_vol_scale=0.15,
        autocorr_lag1_mean=-0.10,         autocorr_lag1_scale=0.08,
        spread_bps_mean=2.5,              spread_bps_scale=1.5,
        disagreement_intensity_mean=1.5,  disagreement_intensity_scale=1.5,
    ),
    Regime.VOLATILE: RegimeProfile(
        realized_vol_mean=1.00,           realized_vol_scale=0.40,
        autocorr_lag1_mean=0.0,           autocorr_lag1_scale=0.15,
        spread_bps_mean=8.0,              spread_bps_scale=4.0,
        disagreement_intensity_mean=6.0,  disagreement_intensity_scale=4.0,
    ),
    Regime.BROKEN: RegimeProfile(
        realized_vol_mean=2.50,           realized_vol_scale=1.0,
        autocorr_lag1_mean=0.0,           autocorr_lag1_scale=0.30,
        spread_bps_mean=40.0,             spread_bps_scale=30.0,
        disagreement_intensity_mean=40.0, disagreement_intensity_scale=25.0,
    ),
}


# ── Classifier ───────────────────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    """
    prior_strength:
        Total pseudo-count in the symmetric Dirichlet prior. Higher =
        slower to update, more conservative. 5.0 is a reasonable start
        (one effective observation per regime).

    decay:
        Per-step multiplicative decay applied to counts before each
        update. Implements an effective rolling-window over observations
        so the posterior tracks regime changes instead of accumulating
        forever. 0.99 = roughly 100-observation half-life.

    min_count:
        Floor on each Dirichlet count after decay, to keep probabilities
        bounded away from 0/1 (which would prevent regime switching).
    """
    prior_strength: float = 5.0
    decay: float = 0.99
    min_count: float = 0.05


class RegimeClassifier:
    """
    One classifier per symbol. Pump FeatureVectors in; get
    RegimePosteriors out.
    """

    def __init__(
        self,
        symbol: str,
        profiles: dict[Regime, RegimeProfile] | None = None,
        config: ClassifierConfig | None = None,
    ):
        self.symbol = symbol
        self.profiles = profiles or DEFAULT_PROFILES
        self.cfg = config or ClassifierConfig()
        n_regimes = len(self.profiles)
        initial = self.cfg.prior_strength / n_regimes
        self._counts: dict[Regime, float] = {r: initial for r in self.profiles}
        self._observations = 0
        self._last_mode: Regime | None = None
        self._mode_run_length: int = 0

    def observe(self, fv: FeatureVector) -> RegimePosterior:
        """Update with one feature vector; emit current posterior."""
        # 1. Decay counts (rolling-window via exponential weighting)
        for r in self._counts:
            self._counts[r] = max(self.cfg.min_count, self._counts[r] * self.cfg.decay)

        # 2. Log-likelihoods per regime
        log_liks = {r: prof.log_likelihood(fv) for r, prof in self.profiles.items()}

        # 3. Softmax to get normalized likelihoods (numerically stable)
        max_ll = max(log_liks.values())
        exps = {r: math.exp(ll - max_ll) for r, ll in log_liks.items()}
        Z = sum(exps.values())
        liks = {r: e / Z for r, e in exps.items()}

        # 4. Soft credit assignment: add likelihood to each count
        for r, l in liks.items():
            self._counts[r] += l

        self._observations += 1

        # 5. Normalize to posterior probabilities
        total = sum(self._counts.values())
        probs = {r: c / total for r, c in self._counts.items()}

        # 6. Track mode persistence
        current_mode = max(probs.items(), key=lambda kv: kv[1])[0]
        if current_mode == self._last_mode:
            self._mode_run_length += 1
        else:
            self._last_mode = current_mode
            self._mode_run_length = 1

        return RegimePosterior(
            symbol=self.symbol,
            window_end=fv.timestamp,
            probabilities=probs,
            observations_used=self._observations,
            half_life_observations=float(self._mode_run_length),
        )

    @property
    def counts(self) -> dict[Regime, float]:
        """Read-only view of current Dirichlet counts (testing / debugging)."""
        return dict(self._counts)
