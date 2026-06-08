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
from typing import Dict, Optional, Tuple

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

# Short help shown as tooltips in the student UI (English to match the dashboard).
TERM_HELP: Dict[str, str] = {
    "global_scale": "Multiplies every reward/penalty term. Leave at 1.0 unless rescaling the whole signal.",
    "mission_completed": "Large bonus when you reach the goal area alive. Primary mission objective.",
    "hit_enemy": "Bonus for shooting down the enemy. Helps survival but should not dominate mission.",
    "was_hit": "Weight when shot down. Negative = penalty; positive would reward getting hit (unusual).",
    "fire_missile": "Weight per missile launch. Negative = cost; positive = bonus for firing.",
    "miss_missile": "Weight when a missile misses. Negative = penalty; positive = bonus.",
    "mission_shaping": "Small per-step reward for moving toward the goal. Helps early learning.",
    "maintain_track": "Tiny per-step reward while the enemy stays on your radar.",
    "lost_track": "Weight when radar lock is lost. Negative = penalty; positive = bonus.",
    "closing_bonus": "Per-step reward for closing distance to the enemy (optional aggression).",
    "wez_advantage": "Per-step reward when you are in missile range but the enemy is not.",
}

# Default min/max for student inputs (admin can override via platform config).
# All weight terms allow positive (reward) or negative (penalty); sign is up to the student.
DEFAULT_RANGES: Dict[str, Dict[str, float]] = {
    "global_scale": {"min": 0.0, "max": 10.0, "step": 0.1},
    "mission_completed": {"min": -100.0, "max": 100.0, "step": 0.5},
    "hit_enemy": {"min": -100.0, "max": 100.0, "step": 0.5},
    "was_hit": {"min": -100.0, "max": 100.0, "step": 0.5},
    "fire_missile": {"min": -10.0, "max": 10.0, "step": 0.05},
    "miss_missile": {"min": -10.0, "max": 10.0, "step": 0.1},
    "mission_shaping": {"min": -10.0, "max": 10.0, "step": 0.01},
    "maintain_track": {"min": -1.0, "max": 1.0, "step": 0.001},
    "lost_track": {"min": -10.0, "max": 10.0, "step": 0.05},
    "closing_bonus": {"min": -10.0, "max": 10.0, "step": 0.01},
    "wez_advantage": {"min": -10.0, "max": 10.0, "step": 0.01},
}

ALL_REWARD_KEYS = ["global_scale"] + REWARD_TERMS


def parse_json_map(raw: str, fallback: Dict) -> Dict:
    if not raw or not str(raw).strip():
        return dict(fallback)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else dict(fallback)
    except json.JSONDecodeError:
        return dict(fallback)


def reward_defaults_from_config(platform_cfg: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Student form defaults: start at zero or instructor-configured values."""
    base = configured_reward_defaults(platform_cfg)
    cfg = platform_cfg or {}
    if cfg.get("rewards_start_zero", "0") == "1":
        for k in ALL_REWARD_KEYS:
            base[k] = 0.0
    return base


def configured_reward_defaults(platform_cfg: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Instructor defaults from platform config (ignores start-at-zero)."""
    cfg = platform_cfg or {}
    base = dict(DEFAULT_REWARDS)
    overrides = parse_json_map(cfg.get("reward_defaults_json", ""), {})
    for k in ALL_REWARD_KEYS:
        if k in overrides:
            base[k] = float(overrides[k])
    return base


def reward_ranges_from_config(platform_cfg: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
    cfg = platform_cfg or {}
    base = {k: dict(v) for k, v in DEFAULT_RANGES.items()}
    overrides = parse_json_map(cfg.get("reward_ranges_json", ""), {})
    for k, spec in overrides.items():
        if k not in base or not isinstance(spec, dict):
            continue
        for field in ("min", "max", "step"):
            if field in spec:
                base[k][field] = float(spec[field])
    return base


def clamp_rewards(rewards: Dict[str, float], ranges: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Clamp submitted weights to allowed ranges."""
    ranges = ranges or DEFAULT_RANGES
    out = {"global_scale": float(rewards.get("global_scale", 0.0))}
    spec_gs = ranges.get("global_scale", DEFAULT_RANGES["global_scale"])
    out["global_scale"] = max(spec_gs["min"], min(spec_gs["max"], out["global_scale"]))
    for k in REWARD_TERMS:
        val = float(rewards.get(k, 0.0))
        spec = ranges.get(k, DEFAULT_RANGES.get(k, {"min": -50, "max": 50}))
        out[k] = max(spec["min"], min(spec["max"], val))
    return out


def reward_editor_payload(platform_cfg: Optional[Dict[str, str]] = None) -> Dict:
    cfg = platform_cfg or {}
    return {
        "start_zero": cfg.get("rewards_start_zero", "0") == "1",
        "defaults": reward_defaults_from_config(cfg),
        "configured_defaults": configured_reward_defaults(cfg),
        "ranges": reward_ranges_from_config(cfg),
        "help": TERM_HELP,
        "terms": {"event": EVENT_TERMS, "shaping": SHAPING_TERMS},
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
