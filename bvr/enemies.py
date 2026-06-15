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

ENEMY_TYPES = ["duck", "defensive", "balanced", "aggressive", "sniper",
               "super_aggressive", "super_defensive", "random"]
SELECTABLE_TYPES = ["duck", "defensive", "balanced", "aggressive", "sniper",
                    "super_aggressive", "super_defensive"]
REFERENCE_PREFIX = "B"
REFERENCE_COUNT = 10
# References shown in the student training picker (the rest stay for locked eval).
TRAINING_REFERENCE_COUNT = 5
# Fixed hand-coded opponents always offered for training alongside the references.
TRAINING_ARCHETYPES = ["super_aggressive", "super_defensive", "duck", "balanced"]
FSM_STOCH_NOISE = 0.03  # ±3% on shoot / crank / break thresholds (per decision)

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
    # Charges straight at the opponent and fires as soon as it can; almost never
    # breaks defensively.
    "super_aggressive": {"shot_frac": 1.0, "crank_frac": 0.0, "break_dist": 6.0,
                         "aggressive": True, "can_fire": True},
    # Always running / breaking; keeps maximum distance and only fires point-blank.
    "super_defensive": {"shot_frac": 0.45, "crank_frac": 1.0, "break_dist": 55.0,
                        "aggressive": False, "can_fire": True},
}

# Difficulty scores (0..10) used to color the picker for archetypes.
ARCHETYPE_SCORES = {
    "duck": 2.0, "defensive": 8.5, "balanced": 6.5, "aggressive": 7.5,
    "sniper": 9.0, "super_aggressive": 7.0, "super_defensive": 8.0,
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
                 aggressive=True, can_fire=True, rng=None):
        self.name = name
        self.shot_frac = float(shot_frac)
        self.crank_frac = float(crank_frac)
        self.break_dist = float(break_dist)
        self.aggressive = bool(aggressive)
        self.can_fire = bool(can_fire)
        self.rng = rng or np.random.default_rng()

    @classmethod
    def from_params(cls, name: str, params: Dict, rng=None) -> "FSMEnemy":
        p = dict(params)
        return cls(
            name,
            shot_frac=p.get("shot_frac", 0.9),
            crank_frac=p.get("crank_frac", 0.9),
            break_dist=p.get("break_dist", 26.0),
            aggressive=bool(p.get("aggressive", True)),
            can_fire=bool(p.get("can_fire", True)),
            rng=rng,
        )

    def _stoch_mult(self) -> float:
        return float(self.rng.uniform(1.0 - FSM_STOCH_NOISE, 1.0 + FSM_STOCH_NOISE))

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
        if state["incoming_missile"] and state["incoming_dist"] < self.break_dist * self._stoch_mult():
            los = _angle_to(pos, opp)
            desired = los + np.deg2rad(135.0)
            return np.array([_steer(heading, desired), 0.0, 0.0], dtype=np.float32)

        los_to_opp = _angle_to(pos, opp)

        if state["opp_tracked"] and self.can_fire and state["missiles"] > 0:
            if dist <= self.shot_frac * rmax * self._stoch_mult() and state["fire_ready"]:
                fire = 1.0
            if dist <= self.crank_frac * rmax * self._stoch_mult():
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


def _ref_feature(name: str) -> np.ndarray:
    """Behavioral feature vector for a reference, used to pick a diverse subset."""
    p = load_reference_enemies().get(name, {})
    return np.array([
        float(p.get("shot_frac", 0.9)),
        float(p.get("crank_frac", 0.9)),
        float(p.get("break_dist", 30.0)) / 40.0,
        1.0 if p.get("aggressive", True) else 0.0,
    ], dtype=float)


def diverse_reference_subset(k: int = TRAINING_REFERENCE_COUNT) -> list:
    """Pick ``k`` references that span the behavior space: start from the
    top-ranked one, then greedily add the opponent farthest from those already
    chosen (farthest-point sampling). Returned in rank order for display."""
    names = reference_types()
    if len(names) <= k:
        return names
    chosen = [names[0]]
    while len(chosen) < k:
        best_n, best_d = None, -1.0
        for n in names:
            if n in chosen:
                continue
            d = min(float(np.linalg.norm(_ref_feature(n) - _ref_feature(c))) for c in chosen)
            if d > best_d:
                best_d, best_n = d, n
        if best_n is None:
            break
        chosen.append(best_n)
    return [n for n in names if n in chosen]


def _catalog_entry(name: str, rank: int) -> Dict:
    refs = load_reference_enemies()
    if name in refs:
        r = refs[name]
        return {
            "rank": r.get("rank", rank),
            "score": float(r.get("score", 0)),
            "kind": "reference",
            "label": f"{name}  (#{r.get('rank', rank)}, score {float(r.get('score', 0)):.2f})",
            "params": {
                "shot_frac": float(r.get("shot_frac", 0.9)),
                "crank_frac": float(r.get("crank_frac", 0.9)),
                "break_dist": float(r.get("break_dist", 30.0)),
                "aggressive": bool(r.get("aggressive", True)),
                "can_fire": bool(r.get("can_fire", True)),
            },
        }
    params = BASE_FSM_PARAMS.get(name, BASE_FSM_PARAMS["balanced"])
    return {
        "rank": rank,
        "score": ARCHETYPE_SCORES.get(name, 5.0),
        "kind": "archetype",
        "label": name.replace("_", " "),
        "params": dict(params),
    }


def enemy_catalog() -> Dict:
    """Metadata for the student training picker: diverse references + fixed
    archetypes, each with FSM params and a difficulty score."""
    names = training_enemy_pool()
    mode = "reference" if load_reference_enemies() else "archetype"
    return {
        "mode": mode,
        "names": names,
        "info": {n: _catalog_entry(n, i + 1) for i, n in enumerate(names)},
    }


def training_enemy_pool() -> list:
    """Opponents a student may pick for training: a diverse subset of the
    references (if any) plus the fixed hand-coded archetypes."""
    refs = diverse_reference_subset()
    if refs:
        archetypes = [a for a in TRAINING_ARCHETYPES if a in BASE_FSM_PARAMS]
        return refs + [a for a in archetypes if a not in refs]
    return list(SELECTABLE_TYPES)


def eval_enemy_catalog() -> Dict:
    """Catalog metadata for every locked eval opponent (profile map)."""
    names = eval_enemy_pool()
    refs = load_reference_enemies()
    info = {}
    for i, n in enumerate(names):
        if n in refs:
            r = refs[n]
            info[n] = {
                "rank": int(r.get("rank", i + 1)),
                "score": float(r.get("score", 0)),
                "kind": "reference",
                "label": n,
                "params": {
                    "shot_frac": float(r.get("shot_frac", 0.9)),
                    "crank_frac": float(r.get("crank_frac", 0.9)),
                    "break_dist": float(r.get("break_dist", 30.0)),
                    "aggressive": bool(r.get("aggressive", True)),
                    "can_fire": bool(r.get("can_fire", True)),
                },
            }
        else:
            info[n] = _catalog_entry(n, i + 1)
    return {"mode": "eval", "names": names, "info": info}


def eval_enemy_pool() -> list:
    """Fixed opponent set for locked scoring / final competition: every
    reference plus every static FSM archetype (students cannot change this)."""
    refs = reference_types()
    statics = [t for t in ENEMY_TYPES if t != "random"]
    return refs + statics if refs else statics


def make_enemy(name, rng=None):
    """Factory mapping a scenario name to an enemy instance."""
    rng = rng or np.random.default_rng()
    name = str(name)
    key = name.upper()
    refs = load_reference_enemies()
    if key in refs:
        return FSMEnemy.from_params(key, refs[key], rng=rng)

    name = name.lower()
    if name in BASE_FSM_PARAMS:
        return FSMEnemy.from_params(name, BASE_FSM_PARAMS[name], rng=rng)
    if name == "random":
        return RandomEnemy(rng)
    raise ValueError(f"Unknown enemy type: {name!r}. Options: {ENEMY_TYPES + reference_types()}")
