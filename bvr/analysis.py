"""Statistical analysis of a trained model.

Produces a set of PNG charts and a stats dictionary:
  - win rate / survival / kill rate per enemy type
  - reward-term contribution breakdown (where did the reward come from?)
  - missile efficiency (kills per missile fired)
  - blue trajectory heatmap (where does the learned policy like to fly?)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import rewards as rewards_mod
from .env import ARENA, BVREnv, N_MISSILES
from .enemies import training_enemy_pool
from .evaluate import evaluate_model, load_model
from .train import DEFAULT_MODEL_PATH, load_configs, PROJECT_ROOT

DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_output")


def _collect_trajectories(model, reward_config, scenario, enemies, episodes_per_enemy, seed):
    xs: List[float] = []
    ys: List[float] = []
    for enemy in enemies:
        for j in range(episodes_per_enemy):
            env = BVREnv(reward_config=reward_config, scenario=scenario, seed=seed + j)
            env.set_enemy(enemy)
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                xs.append(float(env.blue.pos[0]))
                ys.append(float(env.blue.pos[1]))
                done = terminated or truncated
    return np.array(xs), np.array(ys)


def analyze_model(model_path: str, reward_config: Optional[Dict] = None,
                  scenario: Optional[Dict] = None, out_dir: str = DEFAULT_OUT_DIR,
                  episodes_per_enemy: int = 12, seed: int = 2000) -> Dict:
    if reward_config is None or scenario is None:
        r, s = load_configs()
        reward_config = reward_config or r
        scenario = scenario or s
    enemies = scenario.get("enemies", training_enemy_pool()) or training_enemy_pool()
    os.makedirs(out_dir, exist_ok=True)

    stats = evaluate_model(model_path, reward_config, scenario, enemies,
                           episodes_per_enemy=episodes_per_enemy, seed=seed)
    plots: Dict[str, str] = {}

    # 1) Outcomes per enemy.
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(enemies))
    w = 0.27
    ax.bar(x - w, [stats["per_enemy"][e]["mission_rate"] for e in enemies], w, label="mission")
    ax.bar(x, [stats["per_enemy"][e]["survival"] for e in enemies], w, label="survive")
    ax.bar(x + w, [stats["per_enemy"][e]["kills"] for e in enemies], w, label="kill")
    ax.set_xticks(x); ax.set_xticklabels(enemies, rotation=20)
    ax.set_ylim(0, 1); ax.set_ylabel("rate"); ax.set_title("Outcomes per enemy")
    ax.legend()
    plots["outcomes"] = _save(fig, out_dir, "outcomes_per_enemy.png")

    # 2) Reward-term contribution breakdown (summed over all eval episodes).
    totals = {k: 0.0 for k in rewards_mod.REWARD_TERMS}
    for e in enemies:
        for ep in stats["per_enemy"][e]["episodes"]:
            for k, v in ep["contributions"].items():
                totals[k] += v
    fig, ax = plt.subplots(figsize=(7, 4))
    terms = rewards_mod.REWARD_TERMS
    vals = [totals[k] for k in terms]
    colors = ["#4c9be8" if v >= 0 else "#e8624c" for v in vals]
    ax.barh(terms, vals, color=colors)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_title("Reward contribution by term (sum over eval)")
    ax.invert_yaxis()
    plots["contributions"] = _save(fig, out_dir, "reward_contributions.png")

    # 3) Missile efficiency per enemy.
    fig, ax = plt.subplots(figsize=(7, 4))
    used = [stats["per_enemy"][e]["missiles_used"] for e in enemies]
    kills = [stats["per_enemy"][e]["kills"] for e in enemies]
    ax.bar(x - 0.2, used, 0.4, label="missiles used (avg)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, kills, 0.4, color="#5ec27a", label="kill rate")
    ax.set_xticks(x); ax.set_xticklabels(enemies, rotation=20)
    ax.set_ylabel("missiles used"); ax2.set_ylabel("kill rate"); ax2.set_ylim(0, 1)
    ax.set_title("Missile usage vs. kill rate")
    plots["efficiency"] = _save(fig, out_dir, "missile_efficiency.png")

    # 4) Trajectory heatmap.
    xs, ys = _collect_trajectories(model_path_to_model(model_path), reward_config,
                                   scenario, enemies, max(4, episodes_per_enemy // 3), seed)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if len(xs) > 0:
        ax.hist2d(xs, ys, bins=40, range=[[0, ARENA], [0, ARENA]], cmap="magma")
    ax.set_title("Blue trajectory heatmap"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    plots["heatmap"] = _save(fig, out_dir, "trajectory_heatmap.png")

    return {"stats": stats, "plots": plots, "out_dir": out_dir}


_MODEL_CACHE = {}


def model_path_to_model(model_path: str):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = load_model(model_path)
    return _MODEL_CACHE[model_path]


def _save(fig, out_dir, name) -> str:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of a trained BVR model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train one first.")
    result = analyze_model(args.model, out_dir=args.out, episodes_per_enemy=args.episodes)
    s = result["stats"]
    print(f"Score: {s['score']:.1%}  |  mission {s['mission_rate']:.0%}  "
          f"kill {s['kill_rate']:.0%}  survive {s['survival_rate']:.0%}")
    print("Charts written to:")
    for name, path in result["plots"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
