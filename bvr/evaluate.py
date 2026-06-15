"""Evaluation and replay utilities.

Used by the dashboard (watch a trained model fly in real time) and by the
competition server (score a submission against the locked enemy set).
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from . import rewards as rewards_mod
from .env import BVREnv, N_MISSILES
from .enemies import SELECTABLE_TYPES, eval_enemy_pool, training_enemy_pool
from .scoring import SCORE_FORMULA_LABEL, competition_score
from .train import DEFAULT_MODEL_PATH, load_configs

RESULT_RANK = {"shot_down": 0, "timeout": 1, "mission": 2}


def pick_replay_episode_index(episodes: List[Dict]) -> int:
    """When eval outcomes differ, pick the most negative episode for replay."""
    if not episodes:
        return 0
    outcomes = {e.get("result") for e in episodes}
    if len(outcomes) <= 1:
        return 0

    def rank(i: int):
        ep = episodes[i]
        r = ep.get("result", "timeout")
        return (RESULT_RANK.get(r, 1), float(ep.get("reward", 0)))

    return min(range(len(episodes)), key=rank)


def load_model(model_path: str) -> PPO:
    return PPO.load(model_path, device="cpu")


def play_episode(model: PPO, reward_config: Dict, scenario: Dict, enemy: str,
                 seed: int = 0, on_frame: Optional[Callable] = None) -> Dict:
    """Run a single deterministic episode. If on_frame is given it is called with
    every frame dict (used for real-time streaming replay)."""
    env = BVREnv(reward_config=reward_config, scenario=scenario, seed=seed)
    env.set_enemy(enemy)
    obs, _ = env.reset()
    if on_frame:
        on_frame(env._frame(False))
    done = False
    ep_reward = 0.0
    term_totals = {k: 0.0 for k in rewards_mod.REWARD_TERMS}
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        for k, v in info["contributions"].items():
            term_totals[k] += v
        if on_frame:
            on_frame(info["frame"])
        done = terminated or truncated
    result = info.get("result", "timeout")
    mission = 1 if result == "mission" else 0
    return {
        "enemy": enemy,
        "result": result,
        "mission": mission,
        "win": mission,  # win == mission accomplished (reached the goal alive)
        "survived": 1 if env.blue.alive else 0,
        "killed_enemy": 1 if (env.red is not None and not env.red.alive) else 0,
        "reward": round(float(ep_reward), 3),
        "steps": steps,
        "missiles_used": N_MISSILES - env.blue.missiles,
        "contributions": {k: round(v, 4) for k, v in term_totals.items()},
    }


def evaluate_model(model_path: str, reward_config: Optional[Dict] = None,
                   scenario: Optional[Dict] = None, enemies: Optional[List[str]] = None,
                   episodes_per_enemy: int = 10, seed: int = 1000) -> Dict:
    """Aggregate performance across enemies. Returns per-enemy and overall stats.

    The competition `score` (0..1) is 0.6*mission_rate + 0.25*kill_rate +
    0.15*missile_efficiency, so combat effectiveness and not wasting missiles
    both count on top of the primary mission objective.
    """
    if reward_config is None or scenario is None:
        r, s = load_configs()
        reward_config = reward_config or r
        scenario = scenario or s
    enemies = enemies or scenario.get("enemies", training_enemy_pool()) or training_enemy_pool()
    model = load_model(model_path)

    per_enemy = {}
    all_eps: List[Dict] = []
    for enemy in enemies:
        eps = [play_episode(model, reward_config, scenario, enemy, seed=seed + j)
               for j in range(episodes_per_enemy)]
        all_eps.extend(eps)
        per_enemy[enemy] = {
            "mission_rate": round(float(np.mean([e["mission"] for e in eps])), 3),
            "winrate": round(float(np.mean([e["mission"] for e in eps])), 3),
            "survival": round(float(np.mean([e["survived"] for e in eps])), 3),
            "kills": round(float(np.mean([e["killed_enemy"] for e in eps])), 3),
            "mean_reward": round(float(np.mean([e["reward"] for e in eps])), 3),
            "missiles_used": round(float(np.mean([e["missiles_used"] for e in eps])), 2),
            "episodes": eps,
        }

    mission_rate = float(np.mean([e["mission"] for e in all_eps]))
    survival_rate = float(np.mean([e["survived"] for e in all_eps]))
    kill_rate = float(np.mean([e["killed_enemy"] for e in all_eps]))
    mean_reward = float(np.mean([e["reward"] for e in all_eps]))
    fired = float(np.sum([e["missiles_used"] for e in all_eps]))
    kills = float(np.sum([e["killed_enemy"] for e in all_eps]))
    efficiency = round(kills / fired, 3) if fired > 0 else 0.0
    score = competition_score(mission_rate, kill_rate, efficiency)
    return {
        "score": round(score, 4),
        "mission_rate": round(mission_rate, 3),
        "winrate": round(mission_rate, 3),
        "survival_rate": round(survival_rate, 3),
        "kill_rate": round(kill_rate, 3),
        "mean_reward": round(mean_reward, 3),
        "missile_efficiency": efficiency,
        "episodes_per_enemy": episodes_per_enemy,
        "n_episodes": len(all_eps),
        "per_enemy": per_enemy,
    }


def main():
    import json as _json
    parser = argparse.ArgumentParser(description="Evaluate a trained BVR model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print stats as a single JSON blob.")
    parser.add_argument("--locked", action="store_true",
                        help="Evaluate against the locked competition enemy set / scenario.")
    args = parser.parse_args()
    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train one first with `python -m bvr.train`.")

    reward_config = scenario = enemies = None
    if args.locked:
        reward_config = dict(rewards_mod.DEFAULT_REWARDS)
        scenario = {"enemies": list(eval_enemy_pool()), "random_enemy_prob": 0.0, "max_cycles": 260}
        enemies = list(eval_enemy_pool())
    stats = evaluate_model(args.model, reward_config=reward_config, scenario=scenario,
                           enemies=enemies, episodes_per_enemy=args.episodes)
    if args.json:
        print(_json.dumps(stats))
        return
    print(f"Score: {stats['score']:.1%}  ({SCORE_FORMULA_LABEL})")
    print(f"Mission rate: {stats['mission_rate']:.0%}  |  kill rate: {stats['kill_rate']:.0%}  "
          f"|  survival: {stats['survival_rate']:.0%}")
    print(f"Mean reward: {stats['mean_reward']:.2f}  |  missile efficiency: {stats['missile_efficiency']}")
    for enemy, s in stats["per_enemy"].items():
        print(f"  {enemy:<11} mission={s['mission_rate']:.0%}  survive={s['survival']:.0%}  "
              f"kills={s['kills']:.0%}  reward={s['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
