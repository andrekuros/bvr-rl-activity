"""End-of-class final competition and analysis.

Three pieces, used by the instructor's "Run final competition" admin command:

1. STATIC results - how each student's agent does against the fixed FSM enemies
   (this is the normal competition score).
2. POOL (student-vs-student) - a round-robin where one student's policy flies
   blue and another's flies red, swapping sides for fairness. Ranked by points
   (win=3, draw=1).
3. CAUSALITY - because every student trains the SAME fixed network and only the
   reward weights differ, we can correlate each reward/penalty value with the
   final outcomes to see which choices actually drove performance.

`final_report()` ties them together and writes charts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import rewards as rewards_mod
from .env import BVREnv, GOAL_RADIUS
from .evaluate import evaluate_model, load_model

LOCKED_MAX_CYCLES = 260


# ---------------------------------------------------------------------------
# Student-vs-student duel.
# ---------------------------------------------------------------------------
def duel(model_blue, model_red, seed: int = 0, max_cycles: int = LOCKED_MAX_CYCLES) -> str:
    """Run one head-to-head episode. Returns 'blue', 'red' or 'draw'.

    Win conditions: neutralize the opponent, or reach your own goal first.
    """
    env = BVREnv(reward_config=dict(rewards_mod.DEFAULT_REWARDS),
                 scenario={"enemies": ["duck"], "max_cycles": max_cycles}, seed=seed)
    env.set_enemy("duck")  # placeholder; red is driven externally below
    obs_b, _ = env.reset()
    for _ in range(max_cycles + 1):
        obs_r = env.red_obs()
        a_b, _ = model_blue.predict(obs_b, deterministic=True)
        a_r, _ = model_red.predict(obs_r, deterministic=True)
        env.external_red_action = np.asarray(a_r, dtype=np.float64).reshape(-1)
        obs_b, _, terminated, truncated, info = env.step(a_b)

        blue_alive, red_alive = env.blue.alive, env.red.alive
        if not blue_alive or not red_alive:
            if blue_alive and not red_alive:
                return "blue"
            if red_alive and not blue_alive:
                return "red"
            return "draw"  # mutual kill
        if info.get("result") == "mission":  # blue reached its goal
            return "blue"
        if float(np.linalg.norm(env.red.pos - env.red.goal)) <= GOAL_RADIUS:
            return "red"
        if terminated or truncated:
            return "draw"
    return "draw"


def round_robin(models: Dict[str, object], seeds: int = 2,
                progress: Optional[Callable] = None):
    """Every agent plays every other agent on both sides. Returns standings,
    a head-to-head win matrix, and the ranking."""
    names = list(models)
    standings = {n: {"points": 0, "wins": 0, "losses": 0, "draws": 0, "games": 0} for n in names}
    matrix = {a: {b: None for b in names} for a in names}
    total = max(len(names) * (len(names) - 1), 1)
    done = 0
    for a in names:
        for b in names:
            if a == b:
                continue
            results = [duel(models[a], models[b], seed=s) for s in range(seeds)]
            a_wins = results.count("blue")
            b_wins = results.count("red")
            draws = results.count("draw")
            matrix[a][b] = round(a_wins / seeds, 3)
            standings[a]["wins"] += a_wins
            standings[a]["losses"] += b_wins
            standings[a]["draws"] += draws
            standings[a]["points"] += a_wins * 3 + draws
            standings[a]["games"] += seeds
            done += 1
            if progress:
                progress(done, total)
    for n in standings:
        s = standings[n]
        s["winrate"] = round(s["wins"] / max(s["games"], 1), 3)
    ranking = sorted(names, key=lambda n: (standings[n]["points"], standings[n]["winrate"]), reverse=True)
    return standings, matrix, ranking


# ---------------------------------------------------------------------------
# Causality (reward weights -> outcomes).
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def causality(rows: List[Dict], terms: List[str]) -> Dict[str, Optional[Dict]]:
    """Correlate each reward term value (across students) with the outcomes."""
    out: Dict[str, Optional[Dict]] = {}
    score = np.array([r["score"] for r in rows], dtype=float)
    mission = np.array([r["mission_rate"] for r in rows], dtype=float)
    kill = np.array([r["kill_rate"] for r in rows], dtype=float)
    pool = np.array([r.get("pool_points", 0) for r in rows], dtype=float)
    for term in terms:
        x = np.array([float(r["rewards"].get(term, 0.0)) for r in rows], dtype=float)
        if np.std(x) < 1e-9:
            out[term] = None  # everyone used the same value -> no signal
            continue
        out[term] = {
            "corr_score": _pearson(x, score),
            "corr_mission": _pearson(x, mission),
            "corr_kill": _pearson(x, kill),
            "corr_pool": _pearson(x, pool),
        }
    return out


# ---------------------------------------------------------------------------
# Full report.
# ---------------------------------------------------------------------------
def final_report(entries: List[Dict], out_dir: str, seeds: int = 2,
                 progress: Optional[Callable] = None) -> Dict:
    """entries: list of {name, model_path, rewards(dict), score, mission_rate,
    kill_rate}. Static metrics may be precomputed; if missing they are evaluated.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ensure static metrics exist.
    for e in entries:
        if e.get("score") is None:
            stats = evaluate_model(e["model_path"], episodes_per_enemy=12)
            e.update(score=stats["score"], mission_rate=stats["mission_rate"],
                     kill_rate=stats["kill_rate"])

    models = {e["name"]: load_model(e["model_path"]) for e in entries}
    names = list(models)

    pool = {"standings": {}, "matrix": {}, "ranking": []}
    if len(names) >= 2:
        standings, matrix, ranking = round_robin(models, seeds=seeds, progress=progress)
        pool = {"standings": standings, "matrix": matrix, "ranking": ranking}
        for e in entries:
            e["pool_points"] = standings[e["name"]]["points"]
            e["pool_winrate"] = standings[e["name"]]["winrate"]
    else:
        for e in entries:
            e["pool_points"] = 0
            e["pool_winrate"] = 0.0

    static_ranking = sorted(entries, key=lambda e: e["score"], reverse=True)
    caus = causality(entries, rewards_mod.REWARD_TERMS) if len(entries) >= 3 else {}

    plots = _make_plots(entries, pool, caus, out_dir)
    return {
        "n_students": len(entries),
        "static_ranking": [{"name": e["name"], "score": e["score"],
                            "mission_rate": e["mission_rate"], "kill_rate": e["kill_rate"]}
                           for e in static_ranking],
        "pool": {"ranking": pool["ranking"], "standings": pool["standings"], "matrix": pool["matrix"]},
        "causality": caus,
        "entries": [{"name": e["name"], "score": e["score"], "mission_rate": e["mission_rate"],
                     "kill_rate": e["kill_rate"], "pool_points": e.get("pool_points", 0),
                     "pool_winrate": e.get("pool_winrate", 0.0), "rewards": e["rewards"]}
                    for e in entries],
        "plots": plots,
    }


def _make_plots(entries, pool, caus, out_dir) -> Dict[str, str]:
    plots = {}

    # 1) Static score vs pool points (do the static winners also win head-to-head?)
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [e["score"] for e in entries]
    ys = [e.get("pool_points", 0) for e in entries]
    ax.scatter(xs, ys, color="#4c9be8")
    for e in entries:
        ax.annotate(e["name"], (e["score"], e.get("pool_points", 0)), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("static score (vs FSM enemies)")
    ax.set_ylabel("pool points (vs other students)")
    ax.set_title("Static skill vs head-to-head skill")
    plots["static_vs_pool"] = _save(fig, out_dir, "static_vs_pool.png")

    # 2) Causality: correlation of each reward term with score and pool points.
    if caus:
        terms = [t for t, v in caus.items() if v is not None]
        cs = [caus[t]["corr_score"] or 0 for t in terms]
        cp = [caus[t]["corr_pool"] or 0 for t in terms]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        y = np.arange(len(terms))
        ax.barh(y - 0.2, cs, 0.4, label="vs static score", color="#4c9be8")
        ax.barh(y + 0.2, cp, 0.4, label="vs pool points", color="#5ec27a")
        ax.set_yticks(y); ax.set_yticklabels(terms); ax.invert_yaxis()
        ax.axvline(0, color="#888", lw=0.8); ax.set_xlim(-1, 1)
        ax.set_xlabel("Pearson correlation"); ax.set_title("Which reward choices drove results?")
        ax.legend()
        plots["causality"] = _save(fig, out_dir, "causality.png")

    # 3) Head-to-head heatmap.
    if pool["ranking"]:
        order = pool["ranking"]
        M = np.array([[pool["matrix"][a].get(b) if pool["matrix"][a].get(b) is not None else np.nan
                       for b in order] for a in order], dtype=float)
        fig, ax = plt.subplots(figsize=(1.1 + 0.6 * len(order), 1.1 + 0.6 * len(order)))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        ax.set_title("Head-to-head: row (blue) win rate vs column")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plots["matrix"] = _save(fig, out_dir, "head_to_head.png")

    return plots


def _save(fig, out_dir, name) -> str:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    """Local use: point at a folder of <name>.zip models (each may have an
    optional <name>.rewards.json sidecar with the reward weights)."""
    parser = argparse.ArgumentParser(description="Run the final competition over a folder of models.")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--out", default="final_output")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    entries = []
    for path in sorted(glob.glob(os.path.join(args.models_dir, "*.zip"))):
        name = os.path.splitext(os.path.basename(path))[0]
        rewards = dict(rewards_mod.DEFAULT_REWARDS)
        sidecar = os.path.join(args.models_dir, name + ".rewards.json")
        if os.path.exists(sidecar):
            rewards.update(json.load(open(sidecar)))
        entries.append({"name": name, "model_path": path, "rewards": rewards, "score": None})
    if not entries:
        raise SystemExit("No .zip models found in " + args.models_dir)

    report = final_report(entries, args.out, seeds=args.seeds,
                          progress=lambda d, t: print(f"  duels {d}/{t}", end="\r"))
    print("\nStatic ranking:")
    for i, e in enumerate(report["static_ranking"]):
        print(f"  {i+1}. {e['name']:<16} score={e['score']:.1%}")
    print("Pool ranking (head-to-head):")
    for i, n in enumerate(report["pool"]["ranking"]):
        s = report["pool"]["standings"][n]
        print(f"  {i+1}. {n:<16} pts={s['points']} W{s['wins']}-D{s['draws']}-L{s['losses']}")
    print("Charts in:", args.out)


if __name__ == "__main__":
    main()
