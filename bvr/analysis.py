"""Statistical analysis of a trained model.

Produces charts and a stats dictionary:
  - win rate / survival / kill rate per enemy type
  - reward-term contribution breakdown
  - missile efficiency
  - agent profile map (offense vs defense, like the training picker)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import rewards as rewards_mod
from .enemies import eval_enemy_catalog, eval_enemy_pool, training_enemy_pool
from .evaluate import evaluate_model, load_model, pick_replay_episode_index, play_episode
from .train import DEFAULT_MODEL_PATH, _subsample, load_configs, PROJECT_ROOT

DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_output")
ANALYSIS_REPLAY_FRAME_CAP = 120
REPLAYS_FILENAME = "replays.json"


def _offense(params: Dict) -> float:
    ag = 1.12 if params.get("aggressive") else 0.88
    cf = 0.2 if params.get("can_fire") is False else 1.0
    return float(params.get("shot_frac", 0.9)) * ag * cf


def _defense(params: Dict) -> float:
    return float(params.get("crank_frac", 0.9)) * 0.55 + float(params.get("break_dist", 30.0)) / 90.0


def build_agent_profile(stats: Dict, catalog: Optional[Dict] = None) -> Dict:
    """Opponent positions + estimated learned-agent position on offense/defense map."""
    catalog = catalog or eval_enemy_catalog()
    per = stats.get("per_enemy") or {}
    opponents = []
    for name in catalog.get("names", []):
        info = (catalog.get("info") or {}).get(name, {})
        params = info.get("params") or {}
        pe = per.get(name) or {}
        opponents.append({
            "name": name,
            "offense": round(_offense(params), 3),
            "defense": round(_defense(params), 3),
            "score": float(info.get("score", 5)),
            "mission_rate": float(pe.get("mission_rate", 0)),
        })
    s = stats
    student = {
        "label": "Your agent",
        "offense": round(min(1.2, float(s.get("kill_rate", 0)) * 0.9
                             + float(s.get("missile_efficiency", 0)) * 0.35), 3),
        "defense": round(min(1.2, float(s.get("mission_rate", 0)) * 0.55
                             + float(s.get("survival_rate", 0)) * 0.45), 3),
        "score": float(s.get("score", 0)),
        "mission_rate": float(s.get("mission_rate", 0)),
        "kill_rate": float(s.get("kill_rate", 0)),
    }
    return {"opponents": opponents, "student": student}


def plot_agent_profile(profile: Dict, out_dir: str) -> str:
    """Scatter map matching the dashboard enemy picker axes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    opps = profile.get("opponents") or []
    if not opps:
        ax.text(0.5, 0.5, "No opponents", ha="center", va="center", transform=ax.transAxes)
    xs = [p["offense"] for p in opps]
    ys = [p["defense"] for p in opps]
    scores = [p["score"] for p in opps]
    min_s, max_s = (min(scores), max(scores)) if scores else (0, 10)
    colors = []
    for sc in scores:
        t = (sc - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        colors.append((0.3 + t * 0.6, 0.8 - t * 0.5, 0.28))
    ax.scatter(xs, ys, c=colors, s=70, edgecolors="#28324a", linewidths=0.8, zorder=2)
    for p in opps:
        ax.annotate(p["name"], (p["offense"], p["defense"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color="#555")
    st = profile.get("student") or {}
    ax.scatter([st.get("offense", 0)], [st.get("defense", 0)], marker="*",
               s=280, c="#4c9be8", edgecolors="white", linewidths=1.2, zorder=5,
               label=f"Your agent (score {st.get('score', 0):.0%})")
    ax.set_xlabel("Offense →")
    ax.set_ylabel("Defense ↑")
    ax.set_title("Agent profile map (opponents + your learned policy)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    return _save(fig, out_dir, "agent_profile.png")


def collect_eval_replays(model, reward_config: Dict, scenario: Dict,
                         enemies: List[str], stats: Dict,
                         seed: int = 2000) -> Dict[str, Dict]:
    """One replay per opponent; uses worst episode when eval outcomes differ."""
    replays: Dict[str, Dict] = {}
    for enemy in enemies:
        frames: List[Dict] = []
        pe = (stats.get("per_enemy") or {}).get(enemy) or {}
        eps_list = pe.get("episodes") or []
        j = pick_replay_episode_index(eps_list)

        def on_frame(frame, _frames=frames):
            _frames.append(frame)

        result = play_episode(model, reward_config, scenario, enemy,
                              seed=seed + j, on_frame=on_frame)
        replays[enemy] = {
            "enemy": enemy,
            "result": result["result"],
            "mission": int(result.get("mission", 0)),
            "steps": result["steps"],
            "reward": result["reward"],
            "mission_rate": float(pe.get("mission_rate", 0)),
            "kill_rate": float(pe.get("kills", 0)),
            "replay_episode": j + 1,
            "picked_worst": len({e.get("result") for e in eps_list}) > 1,
            "frames": _subsample(frames, cap=ANALYSIS_REPLAY_FRAME_CAP),
        }
    return replays


def save_replays(replays: Dict[str, Dict], enemies: List[str], out_dir: str) -> str:
    path = os.path.join(out_dir, REPLAYS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"enemies": enemies, "replays": replays}, f)
    return path


def load_replays(out_dir: str) -> Optional[Dict]:
    path = os.path.join(out_dir, REPLAYS_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "replays" in data:
        return data
    if isinstance(data, dict):
        enemies = list(data.keys())
        return {"enemies": enemies, "replays": data}
    return None


def analyze_model(model_path: str, reward_config: Optional[Dict] = None,
                  scenario: Optional[Dict] = None, out_dir: str = DEFAULT_OUT_DIR,
                  episodes_per_enemy: int = 12, seed: int = 2000,
                  with_replays: bool = True) -> Dict:
    if reward_config is None or scenario is None:
        r, s = load_configs()
        reward_config = reward_config or r
        scenario = scenario or s
    enemies = list(eval_enemy_pool()) or list(training_enemy_pool())
    scenario = {**scenario, "enemies": enemies}
    os.makedirs(out_dir, exist_ok=True)

    stats = evaluate_model(model_path, reward_config, scenario, enemies,
                           episodes_per_enemy=episodes_per_enemy, seed=seed)
    plots: Dict[str, str] = {}
    replays: Dict[str, Dict] = {}
    agent_profile = build_agent_profile(stats)
    plots["agent_profile"] = plot_agent_profile(agent_profile, out_dir)

    if with_replays:
        model = model_path_to_model(model_path)
        replays = collect_eval_replays(model, reward_config, scenario, enemies, stats, seed=seed)
        save_replays(replays, enemies, out_dir)

    x = np.arange(len(enemies))
    w = 0.27

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, [stats["per_enemy"][e]["mission_rate"] for e in enemies], w, label="mission")
    ax.bar(x, [stats["per_enemy"][e]["survival"] for e in enemies], w, label="survive")
    ax.bar(x + w, [stats["per_enemy"][e]["kills"] for e in enemies], w, label="kill")
    ax.set_xticks(x)
    ax.set_xticklabels(enemies, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Outcomes per enemy")
    ax.legend()
    plots["outcomes"] = _save(fig, out_dir, "outcomes_per_enemy.png")

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

    fig, ax = plt.subplots(figsize=(7, 4))
    used = [stats["per_enemy"][e]["missiles_used"] for e in enemies]
    kills = [stats["per_enemy"][e]["kills"] for e in enemies]
    ax.bar(x - 0.2, used, 0.4, label="missiles used (avg)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, kills, 0.4, color="#5ec27a", label="kill rate")
    ax.set_xticks(x)
    ax.set_xticklabels(enemies, rotation=20)
    ax.set_ylabel("missiles used")
    ax2.set_ylabel("kill rate")
    ax2.set_ylim(0, 1)
    ax.set_title("Missile usage vs. kill rate")
    plots["efficiency"] = _save(fig, out_dir, "missile_efficiency.png")

    return {
        "stats": stats,
        "plots": plots,
        "out_dir": out_dir,
        "replays": replays,
        "replay_enemies": enemies,
        "agent_profile": agent_profile,
    }


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
