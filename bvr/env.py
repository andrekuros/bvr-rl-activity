"""Simplified 2D Beyond-Visual-Range (BVR) air-combat environment.

A single trained blue agent flies a mission against one red enemy. The world is
a top-down 2D arena (km); altitude is kept as a single scalar dimension so the
flight-level idea from B-ACE survives without the cost of full 3D. Each side
carries radar-guided missiles with a simplified Weapon Engagement Zone (WEZ):
shots are "good" inside RMax and nearly inescapable inside the NEZ, but a target
that turns away (beams/cranks) can make an inbound missile run out of energy.

The environment follows the Gymnasium single-agent API. Reward weights come from
`config/rewards.json` (see bvr/rewards.py); the env only produces the raw term
values, so students change *behavior* purely by changing *weights*.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import rewards as rewards_mod
from .enemies import make_enemy

# ----------------------------------------------------------------------------
# World constants (abstract but internally consistent; distances in km).
# ----------------------------------------------------------------------------
ARENA = 120.0          # square arena side length (km)
AC_SPEED = 1.0         # aircraft ground speed (km per step)
MAX_TURN = math.radians(25.0)   # max heading change per step (rad)
RADAR_RANGE = 60.0     # detection range (km)
START_SEPARATION = 88.0   # km between the two aircraft at spawn (> radar range)
N_MISSILES = 4
FIRE_COOLDOWN = 5      # steps between shots
MISSILE_SPEED = 2.5    # km per step
MISSILE_FUEL = 26      # max steps a missile can fly (~65 km if it flew straight)
MISSILE_TURN = math.radians(18.0)  # missile seeker turn limit -> beaming works
HIT_RADIUS = 2.0       # km, missile detonation radius
WEZ_RMAX = 40.0        # km, edge of the engagement zone
WEZ_NEZ = 18.0         # km, no-escape zone
GOAL_RADIUS = 8.0      # km, mission goal capture radius

OBS_DIM = 20
ACT_DIM = 3


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Missile:
    __slots__ = ("owner", "pos", "heading", "fuel", "alive")

    def __init__(self, owner: str, pos: np.ndarray, heading: float):
        self.owner = owner          # "blue" or "red"
        self.pos = pos.astype(np.float64)
        self.heading = heading
        self.fuel = MISSILE_FUEL
        self.alive = True


class Aircraft:
    __slots__ = ("pos", "heading", "alt", "missiles", "cooldown", "alive", "goal")

    def __init__(self, pos, heading, goal):
        self.pos = np.array(pos, dtype=np.float64)
        self.heading = float(heading)
        self.alt = 0.5
        self.missiles = N_MISSILES
        self.cooldown = 0
        self.alive = True
        self.goal = np.array(goal, dtype=np.float64)


class BVREnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_config: Optional[Dict] = None,
                 scenario: Optional[Dict] = None, seed: Optional[int] = None):
        super().__init__()
        self.reward_config = reward_config or dict(rewards_mod.DEFAULT_REWARDS)
        scenario = scenario or {}
        self.enemy_pool: List[str] = list(scenario.get("enemies", ["balanced"]))
        self.random_enemy_prob: float = float(scenario.get("random_enemy_prob", 0.0))
        self.enemy_sampling: str = str(scenario.get("enemy_sampling", "round_robin"))
        self.max_cycles: int = int(scenario.get("max_cycles", 260))
        self._forced_enemy: Optional[str] = None
        self._rr_idx: int = 0

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.blue: Optional[Aircraft] = None
        self.red: Optional[Aircraft] = None
        self.enemy = None
        self.enemy_name = "balanced"
        self.external_red_action = None  # set by a duel runner to control red
        self.missiles: List[Missile] = []
        self.step_count = 0
        self._prev_goal_dist = 0.0
        self._prev_enemy_dist = 0.0
        self._tracked_prev = False
        self.last_terms: Dict[str, float] = {}
        self.last_contributions: Dict[str, float] = {}
        self.episode_result = "pending"

    # -- configuration helpers ------------------------------------------------
    def set_enemy(self, name: Optional[str]):
        """Force the next reset(s) to use a specific enemy (used for evaluation)."""
        self._forced_enemy = name

    def set_reward_config(self, cfg: Dict):
        self.reward_config = dict(cfg)

    # -- gym API --------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.missiles = []
        self.external_red_action = None
        self.episode_result = "pending"

        # Blue starts west, mission goal is far east. Red mirrors it. The
        # aircraft spawn START_SEPARATION km apart (beyond radar range), so the
        # agent must navigate and close the distance before any engagement is
        # possible - it cannot simply open fire from the start.
        cy = ARENA / 2.0
        margin = (ARENA - START_SEPARATION) / 2.0
        jitter = lambda: float(self.rng.uniform(-18.0, 18.0))
        bx, rx = margin, ARENA - margin
        self.blue = Aircraft(pos=(bx, cy + jitter()), heading=0.0, goal=(rx, cy))
        self.red = Aircraft(pos=(rx, cy + jitter()), heading=math.pi, goal=(bx, cy))

        if self._forced_enemy is not None:
            self.enemy_name = self._forced_enemy
        elif self.random_enemy_prob > 0 and self.rng.uniform() < self.random_enemy_prob:
            self.enemy_name = "random"
        elif self.enemy_sampling == "round_robin":
            pool = [e for e in self.enemy_pool if e != "random"] or list(self.enemy_pool)
            self.enemy_name = str(pool[self._rr_idx % len(pool)])
            self._rr_idx = (self._rr_idx + 1) % max(len(pool), 1)
        else:
            self.enemy_name = str(self.rng.choice(self.enemy_pool))
        self.enemy = make_enemy(self.enemy_name, rng=self.rng)

        self._prev_goal_dist = float(np.linalg.norm(self.blue.pos - self.blue.goal))
        self._prev_enemy_dist = float(np.linalg.norm(self.blue.pos - self.red.pos))
        self._tracked_prev = self._prev_enemy_dist <= RADAR_RANGE

        return self._obs(), {"enemy": self.enemy_name}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        terms = {k: 0.0 for k in rewards_mod.REWARD_TERMS}

        # 1) Apply blue action.
        fired_blue = self._apply_action(self.blue, action)
        if fired_blue:
            terms["fire_missile"] = 1.0

        # 2) Enemy decides + acts. In a student-vs-student duel an external
        #    policy supplies the red action; otherwise the FSM enemy decides.
        if self.red is not None and self.red.alive:
            if self.external_red_action is not None:
                self._apply_action(self.red, self.external_red_action)
            else:
                self._apply_action(self.red, self.enemy.act(self._enemy_state()))

        # 3) Advance missiles, resolve hits/misses.
        hit_enemy, blue_was_hit, blue_missile_missed = self._advance_missiles()
        if hit_enemy:
            terms["hit_enemy"] = 1.0
        if blue_was_hit:
            terms["was_hit"] = 1.0
        if blue_missile_missed:
            terms["miss_missile"] = 1.0

        self.step_count += 1

        # 4) Geometry / tracking after movement.
        enemy_alive = self.red is not None and self.red.alive
        enemy_dist = float(np.linalg.norm(self.blue.pos - self.red.pos)) if enemy_alive else RADAR_RANGE + 1
        tracked = enemy_alive and enemy_dist <= RADAR_RANGE
        goal_dist = float(np.linalg.norm(self.blue.pos - self.blue.goal))

        # 5) Shaping terms.
        terms["mission_shaping"] = self._prev_goal_dist - goal_dist
        terms["maintain_track"] = 1.0 if tracked else 0.0
        terms["lost_track"] = 1.0 if (self._tracked_prev and not tracked and enemy_alive) else 0.0
        terms["closing_bonus"] = (self._prev_enemy_dist - enemy_dist) if tracked else 0.0
        in_my_wez, in_enemy_wez = self._wez_pair(self.blue, self.red)
        terms["wez_advantage"] = 1.0 if (in_my_wez and not in_enemy_wez) else 0.0

        # 6) Termination. The mission is the objective: reach the goal alive.
        #    Killing the enemy is a means to survive, not the win condition.
        terminated = False
        truncated = False
        if not self.blue.alive:
            self.episode_result = "shot_down"
            terminated = True
        elif goal_dist <= GOAL_RADIUS:
            terms["mission_completed"] = 1.0
            self.episode_result = "mission"
            terminated = True
        elif self.step_count >= self.max_cycles:
            self.episode_result = "timeout"
            truncated = True

        reward, contributions = rewards_mod.compute(terms, self.reward_config)
        self.last_terms = terms
        self.last_contributions = contributions

        self._prev_goal_dist = goal_dist
        self._prev_enemy_dist = enemy_dist if enemy_alive else self._prev_enemy_dist
        self._tracked_prev = tracked

        info = {
            "terms": terms,
            "contributions": contributions,
            "enemy": self.enemy_name,
            "frame": self._frame(tracked),
        }
        if terminated or truncated:
            info["result"] = self.episode_result
            info["enemy_alive"] = enemy_alive
        return self._obs(), float(reward), terminated, truncated, info

    # -- internals ------------------------------------------------------------
    def _apply_action(self, ac: Aircraft, action) -> bool:
        """Move an aircraft; return True if it fired a missile this step."""
        if ac is None or not ac.alive:
            return False
        turn = float(np.clip(action[0], -1.0, 1.0))
        alt_cmd = float(np.clip(action[1], -1.0, 1.0)) if len(action) > 1 else 0.0
        fire = float(action[2]) if len(action) > 2 else 0.0

        ac.heading = _wrap(ac.heading + turn * MAX_TURN)
        ac.pos[0] = float(np.clip(ac.pos[0] + AC_SPEED * math.cos(ac.heading), 0.0, ARENA))
        ac.pos[1] = float(np.clip(ac.pos[1] + AC_SPEED * math.sin(ac.heading), 0.0, ARENA))
        ac.alt = float(np.clip(ac.alt + alt_cmd * 0.05, 0.0, 1.0))
        if ac.cooldown > 0:
            ac.cooldown -= 1

        fired = False
        if fire > 0.0 and ac.missiles > 0 and ac.cooldown == 0:
            owner = "blue" if ac is self.blue else "red"
            self.missiles.append(Missile(owner, ac.pos.copy(), ac.heading))
            ac.missiles -= 1
            ac.cooldown = FIRE_COOLDOWN
            fired = True
        return fired

    def _advance_missiles(self):
        hit_enemy = False
        blue_was_hit = False
        blue_missile_missed = False
        survivors: List[Missile] = []
        for m in self.missiles:
            target = self.red if m.owner == "blue" else self.blue
            if target is None or not target.alive:
                continue  # nothing to chase -> drop quietly
            desired = math.atan2(target.pos[1] - m.pos[1], target.pos[0] - m.pos[0])
            err = _wrap(desired - m.heading)
            m.heading = _wrap(m.heading + float(np.clip(err, -MISSILE_TURN, MISSILE_TURN)))
            m.pos[0] += MISSILE_SPEED * math.cos(m.heading)
            m.pos[1] += MISSILE_SPEED * math.sin(m.heading)
            m.fuel -= 1

            dist = float(np.linalg.norm(m.pos - target.pos))
            if dist <= HIT_RADIUS:
                target.alive = False
                if m.owner == "blue":
                    hit_enemy = True
                else:
                    blue_was_hit = True
                continue  # missile consumed
            if m.fuel <= 0:
                if m.owner == "blue":
                    blue_missile_missed = True
                continue  # ran out of energy -> miss
            survivors.append(m)
        self.missiles = survivors
        return hit_enemy, blue_was_hit, blue_missile_missed

    def _wez_pair(self, me: Aircraft, opp: Aircraft):
        """A coarse WEZ from `me`'s perspective: nose-on aspect inside RMax with
        a missile available counts as having the shot. Returns (me_has_shot,
        opp_has_shot) and works symmetrically for either aircraft."""
        if me is None or opp is None or not me.alive or not opp.alive:
            return False, False
        dist = float(np.linalg.norm(me.pos - opp.pos))
        los = math.atan2(opp.pos[1] - me.pos[1], opp.pos[0] - me.pos[0])
        me_aspect = math.cos(_wrap(los - me.heading))
        opp_aspect = math.cos(_wrap((los + math.pi) - opp.heading))
        me_shot = dist <= WEZ_RMAX and me_aspect > 0.3 and me.missiles > 0
        opp_shot = dist <= WEZ_RMAX and opp_aspect > 0.3 and opp.missiles > 0
        return me_shot, opp_shot

    def _incoming_missile(self, ac: Aircraft):
        side = "red" if ac is self.blue else "blue"
        best = None
        best_dist = 1e9
        for m in self.missiles:
            if m.owner == side:
                d = float(np.linalg.norm(m.pos - ac.pos))
                if d < best_dist:
                    best, best_dist = m, d
        return best, best_dist

    def _enemy_state(self) -> Dict:
        """God-view state handed to the FSM enemy (red aircraft)."""
        return self._aircraft_state(self.red, self.blue)

    def _aircraft_state(self, me: Aircraft, opp: Aircraft) -> Dict:
        """God-view FSM state from ``me``'s perspective (used for red or blue FSM)."""
        m, mdist = self._incoming_missile(me)
        opp_dist = float(np.linalg.norm(me.pos - opp.pos)) if opp is not None else RADAR_RANGE + 1
        opp_alive = opp is not None and opp.alive
        return {
            "pos": me.pos,
            "heading": me.heading,
            "missiles": me.missiles,
            "fire_ready": me.cooldown == 0,
            "goal": me.goal,
            "opp_pos": opp.pos if opp is not None else me.pos,
            "opp_heading": opp.heading if opp is not None else me.heading,
            "opp_missiles": opp.missiles if opp is not None else 0,
            "opp_tracked": opp_alive and opp_dist <= RADAR_RANGE,
            "opp_dist": opp_dist,
            "incoming_missile": m is not None,
            "incoming_dist": mdist,
            "wez_rmax": WEZ_RMAX,
            "wez_nez": WEZ_NEZ,
            "max_turn": MAX_TURN,
        }

    def _build_obs(self, me: Aircraft, opp: Aircraft, goal: np.ndarray) -> np.ndarray:
        """Build the 20-dim observation from `me`'s point of view. Used for the
        blue agent and, in duels, for a red agent (with roles swapped)."""
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0] = me.pos[0] / ARENA * 2 - 1
        o[1] = me.pos[1] / ARENA * 2 - 1
        o[2] = me.alt * 2 - 1
        o[3] = math.sin(me.heading)
        o[4] = math.cos(me.heading)
        o[5] = me.missiles / N_MISSILES * 2 - 1

        goal_vec = goal - me.pos
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_brg = _wrap(math.atan2(goal_vec[1], goal_vec[0]) - me.heading)
        o[6] = min(goal_dist / ARENA, 1.0) * 2 - 1
        o[7] = math.sin(goal_brg)
        o[8] = math.cos(goal_brg)

        opp_alive = opp is not None and opp.alive
        opp_dist = float(np.linalg.norm(me.pos - opp.pos)) if opp_alive else RADAR_RANGE + 1
        tracked = opp_alive and opp_dist <= RADAR_RANGE
        if tracked:
            evec = opp.pos - me.pos
            los = math.atan2(evec[1], evec[0])
            brg = _wrap(los - me.heading)
            o[9] = 1.0
            o[10] = min(opp_dist / RADAR_RANGE, 1.0) * 2 - 1
            o[11] = math.sin(brg)
            o[12] = math.cos(brg)
            aspect = _wrap((los + math.pi) - opp.heading)
            o[13] = math.sin(aspect)
            o[14] = math.cos(aspect)
            me_shot, opp_shot = self._wez_pair(me, opp)
            o[15] = 1.0 if me_shot else 0.0
            o[16] = 1.0 if opp_shot else 0.0
            o[17] = opp.missiles / N_MISSILES * 2 - 1

        m, mdist = self._incoming_missile(me)
        if m is not None:
            o[18] = 1.0
            o[19] = min(mdist / RADAR_RANGE, 1.0) * 2 - 1
        else:
            o[19] = 1.0
        return np.clip(o, -1.0, 1.0)

    def _obs(self) -> np.ndarray:
        return self._build_obs(self.blue, self.red, self.blue.goal)

    def red_obs(self) -> np.ndarray:
        """Red's observation (mirror of the blue agent) for student-vs-student duels."""
        return self._build_obs(self.red, self.blue, self.red.goal)

    def _frame(self, tracked: bool) -> Dict:
        """Lightweight JSON-friendly snapshot for the dashboard canvas."""
        return {
            "t": self.step_count,
            "arena": ARENA,
            "enemy": self.enemy_name,
            "blue": {
                "x": round(float(self.blue.pos[0]), 2),
                "y": round(float(self.blue.pos[1]), 2),
                "hdg": round(float(self.blue.heading), 3),
                "alive": bool(self.blue.alive),
                "missiles": int(self.blue.missiles),
                "alt": round(float(self.blue.alt), 2),
                "goal": [round(float(self.blue.goal[0]), 1), round(float(self.blue.goal[1]), 1)],
            },
            "red": None if self.red is None else {
                "x": round(float(self.red.pos[0]), 2),
                "y": round(float(self.red.pos[1]), 2),
                "hdg": round(float(self.red.heading), 3),
                "alive": bool(self.red.alive),
                "missiles": int(self.red.missiles),
                "tracked": bool(tracked),
            },
            "missiles": [
                {"x": round(float(m.pos[0]), 2), "y": round(float(m.pos[1]), 2), "owner": m.owner}
                for m in self.missiles
            ],
            "result": self.episode_result,
        }
