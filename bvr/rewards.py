"""Reward / penalty configuration.

This is the ONLY learning signal students are allowed to tune. Each term has a
weight; setting a weight to 0.0 cleanly disables that feedback. The environment
produces a dictionary of raw "term" values every step (events are 0/1, shaping
terms are continuous magnitudes) and `compute()` turns them into a scalar reward
plus a per-term contribution breakdown used by the dashboard and analysis tools.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

# Large, sparse, end-of-episode style signals.
EVENT_TERMS = [
    "mission_completed",  # reached the goal area alive
    "hit_enemy",          # neutralized an enemy aircraft
    "was_hit",            # got neutralized (should be negative)
    "fire_missile",       # fired a missile (small cost discourages spamming)
    "miss_missile",       # a fired missile expired without a hit
]

# Small, dense, per-step signals that guide exploration.
SHAPING_TERMS = [
    "mission_shaping",  # progress (km) toward the goal this step
    "maintain_track",   # enemy held on radar this step
    "lost_track",       # lost radar lock this step (should be negative)
    "closing_bonus",    # closing range toward the enemy this step
    "wez_advantage",    # enemy inside my Weapon Engagement Zone but I am not in theirs
]

REWARD_TERMS = EVENT_TERMS + SHAPING_TERMS

# Sensible defaults inspired by the B-ACE reward table.
DEFAULT_REWARDS: Dict[str, float] = {
    "global_scale": 1.0,
    "mission_completed": 10.0,
    "hit_enemy": 3.0,
    "was_hit": -5.0,
    "fire_missile": -0.1,
    "miss_missile": -0.5,
    "mission_shaping": 0.02,
    "maintain_track": 0.001,
    "lost_track": -0.1,
    "closing_bonus": 0.0,
    "wez_advantage": 0.0,
}


def load_rewards(path: str) -> Dict[str, float]:
    """Load a rewards config, filling any missing keys with defaults."""
    cfg = dict(DEFAULT_REWARDS)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_rewards(path: str, cfg: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    clean = {"global_scale": float(cfg.get("global_scale", 1.0))}
    for k in REWARD_TERMS:
        clean[k] = float(cfg.get(k, DEFAULT_REWARDS[k]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def compute(terms: Dict[str, float], cfg: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Combine raw term values with the student weights.

    Returns (total_reward, contributions) where contributions[k] is the signed
    reward contributed by term k this step (already scaled by global_scale).
    """
    scale = float(cfg.get("global_scale", 1.0))
    contributions: Dict[str, float] = {}
    total = 0.0
    for k in REWARD_TERMS:
        weight = float(cfg.get(k, 0.0))
        contribution = weight * float(terms.get(k, 0.0)) * scale
        contributions[k] = contribution
        total += contribution
    return total, contributions
