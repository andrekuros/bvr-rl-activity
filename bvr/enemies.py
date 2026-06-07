"""Red-team (enemy) behaviors.

Five hand-coded Finite State Machine (FSM) agents plus a random agent. Each is
parameterized by when it decides to *shoot*, *crank* (turn ~45 deg to keep the
target on the radar edge while extending), and *break* (hard defensive turn to
defeat an incoming missile) - the same idea as the B-ACE dShot / lCrank / lBreak
parameters.

Optimized reference opponents B1..B10 live in ``config/reference_enemies.json``
(produced by ``python -m bvr.fsm_optimize``).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

ENEMY_TYPES = ["duck", "defensive", "balanced", "aggressive", "sniper", "random"]
SELECTABLE_TYPES = ["duck", "defensive", "balanced", "aggressive", "sniper"]
REFERENCE_PREFIX = "B"
REFERENCE_COUNT = 10

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_REFERENCE_PATH = os.path.join(_CONFIG_DIR, "reference_enemies.json")
_REFERENCE_CACHE: Optional[Dict[str, Dict]] = None


def _angle_to(frm, to):
    return np.arctan2(to[1] - frm[1], to[0] - frm[0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _steer(cur_heading, desired_heading):
    """Return a normalized turn command in [-1, 1] toward desired heading."""
    err = _wrap(desired_heading - cur_heading)
    return float(np.clip(err / np.deg2rad(25.0), -1.0, 1.0))


# Default FSM parameters for the five archetypes (used as optimization seeds).
BASE_FSM_PARAMS: Dict[str, Dict] = {
    "duck": {"shot_frac": 0.0, "crank_frac": 0.0, "break_dist": 0.0,
             "aggressive": False, "can_fire": False},
    "defensive": {"shot_frac": 0.8, "crank_frac": 0.95, "break_dist": 34.0,
                  "aggressive": False, "can_fire": True},
    "balanced": {"shot_frac": 0.9, "crank_frac": 0.9, "break_dist": 26.0,
                 "aggressive": True, "can_fire": True},
    "aggressive": {"shot_frac": 1.0, "crank_frac": 0.75, "break_dist": 18.0,
                   "aggressive": True, "can_fire": True},
    "sniper": {"shot_frac": 1.0, "crank_frac": 1.0, "break_dist": 36.0,
               "aggressive": False, "can_fire": True},
}


class FSMEnemy:
    """Configurable BVR FSM agent.

    Parameters are fractions of the missile max range (wez_rmax):
      shot_frac  : fire when target distance <= shot_frac * rmax
      crank_frac : start cranking when distance <= crank_frac * rmax
      break_dist : break hard if an incoming missile is closer than this (km)
      aggressive : if True, flies toward the opponent; else flies to its goal
    """

    def __init__(self, name, shot_frac, crank_frac, break_dist,
                 aggressive=True, can_fire=True):
        self.name = name
        self.shot_frac = float(shot_frac)
        self.crank_frac = float(crank_frac)
        self.break_dist = float(break_dist)
        self.aggressive = bool(aggressive)
        self.can_fire = bool(can_fire)

    @classmethod
    def from_params(cls, name: str, params: Dict) -> "FSMEnemy":
        p = dict(params)
        return cls(
            name,
            shot_frac=p.get("shot_frac", 0.9),
            crank_frac=p.get("crank_frac", 0.9),
            break_dist=p.get("break_dist", 26.0),
            aggressive=bool(p.get("aggressive", True)),
            can_fire=bool(p.get("can_fire", True)),
        )

    def to_params(self) -> Dict:
        return {
            "shot_frac": self.shot_frac,
            "crank_frac": self.crank_frac,
            "break_dist": self.break_dist,
            "aggressive": self.aggressive,
            "can_fire": self.can_fire,
        }

    def reset(self):
        pass

    def act(self, state):
        pos = state["pos"]
        heading = state["heading"]
        opp = state["opp_pos"]
        dist = state["opp_dist"]
        rmax = state["wez_rmax"]
        goal = state["goal"]

        fire = 0.0
        if state["incoming_missile"] and state["incoming_dist"] < self.break_dist:
            los = _angle_to(pos, opp)
            desired = los + np.deg2rad(135.0)
            return np.array([_steer(heading, desired), 0.0, 0.0], dtype=np.float32)

        los_to_opp = _angle_to(pos, opp)

        if state["opp_tracked"] and self.can_fire and state["missiles"] > 0:
            if dist <= self.shot_frac * rmax and state["fire_ready"]:
                fire = 1.0
            if dist <= self.crank_frac * rmax:
                desired = los_to_opp + np.deg2rad(40.0)
                return np.array([_steer(heading, desired), 0.0, fire], dtype=np.float32)

        target = opp if (self.aggressive and state["opp_tracked"]) else goal
        desired = _angle_to(pos, target)
        return np.array([_steer(heading, desired), 0.0, fire], dtype=np.float32)


class RandomEnemy:
    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def reset(self):
        pass

    def act(self, state):
        turn = self.rng.uniform(-1.0, 1.0)
        fire = 1.0 if self.rng.uniform() < 0.02 else 0.0
        return np.array([turn, 0.0, fire], dtype=np.float32)


def load_reference_enemies(reload: bool = False) -> Dict[str, Dict]:
    """Load B1..B10 parameter sets from ``config/reference_enemies.json``."""
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is not None and not reload:
        return _REFERENCE_CACHE
    refs: Dict[str, Dict] = {}
    if os.path.exists(_REFERENCE_PATH):
        with open(_REFERENCE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, val in raw.items():
            if key.upper().startswith(REFERENCE_PREFIX) and isinstance(val, dict):
                refs[key.upper()] = val
    _REFERENCE_CACHE = refs
    return refs


def reference_types() -> list:
    """Return sorted B1..B10 names that exist in the reference file."""
    refs = load_reference_enemies()
    names = sorted(refs.keys(), key=lambda n: int(n[1:]) if n[1:].isdigit() else 999)
    return names


def enemy_catalog() -> Dict:
    """Metadata for the UI: names, ranks, and optimization scores."""
    refs = load_reference_enemies()
    if refs:
        names = reference_types()
        return {
            "mode": "reference",
            "names": names,
            "info": {
                n: {
                    "rank": refs[n].get("rank", int(n[1:]) if n[1:].isdigit() else 0),
                    "score": refs[n].get("score"),
                    "label": f"{n}  (#{refs[n].get('rank', n[1:])}, score {float(refs[n].get('score', 0)):.2f})",
                }
                for n in names
            },
        }
    return {
        "mode": "archetype",
        "names": list(SELECTABLE_TYPES),
        "info": {n: {"label": n} for n in SELECTABLE_TYPES},
    }


def training_enemy_pool() -> list:
    """Enemies used for student training: reference B1..B10 if present, else archetypes."""
    return enemy_catalog()["names"]


def make_enemy(name, rng=None):
    """Factory mapping a scenario name to an enemy instance."""
    name = str(name)
    key = name.upper()
    refs = load_reference_enemies()
    if key in refs:
        return FSMEnemy.from_params(key, refs[key])

    name = name.lower()
    if name in BASE_FSM_PARAMS:
        return FSMEnemy.from_params(name, BASE_FSM_PARAMS[name])
    if name == "random":
        return RandomEnemy(rng)
    raise ValueError(f"Unknown enemy type: {name!r}. Options: {ENEMY_TYPES + reference_types()}")
