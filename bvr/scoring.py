"""Shared competition score used for locked eval, live monitoring, and reports."""

from __future__ import annotations

SCORE_MISSION_WEIGHT = 0.6
SCORE_KILL_WEIGHT = 0.25
SCORE_EFF_WEIGHT = 0.15
SCORE_FORMULA_LABEL = "0.6×mission + 0.25×kill + 0.15×missile eff."


def competition_score(mission_rate: float, kill_rate: float, missile_efficiency: float) -> float:
    """Composite score in [0, 1] (approximately)."""
    return (
        SCORE_MISSION_WEIGHT * float(mission_rate)
        + SCORE_KILL_WEIGHT * float(kill_rate)
        + SCORE_EFF_WEIGHT * float(missile_efficiency)
    )
