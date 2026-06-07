"""Evolutionary search for stronger FSM reference opponents (B1..B10).

Each candidate is scored by fighting the current elite pool (round-robin as red
vs each elite blue FSM). Score from the red perspective:

  score = win_rate * 8 + ((1 + max_missiles - launched_missiles) / max_missiles) * 2

The search runs for 50 generations; the top 10 agents become B1 (best) .. B10.

Usage:
  python -m bvr.fsm_optimize
  python -m bvr.fsm_optimize --iterations 50 --seeds 3 --out config/reference_enemies.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .enemies import BASE_FSM_PARAMS, FSMEnemy, REFERENCE_COUNT, SELECTABLE_TYPES
from .env import BVREnv, GOAL_RADIUS, N_MISSILES

LOCKED_MAX_CYCLES = 260
ELITE_SIZE = 10


def clamp_params(p: Dict) -> Dict:
    q = dict(p)
    q["shot_frac"] = float(np.clip(q.get("shot_frac", 0.9), 0.0, 1.2))
    q["crank_frac"] = float(np.clip(q.get("crank_frac", 0.9), 0.0, 1.2))
    q["break_dist"] = float(np.clip(q.get("break_dist", 20.0), 0.0, 45.0))
    q["aggressive"] = bool(q.get("aggressive", True))
    q["can_fire"] = bool(q.get("can_fire", True))
    return q


def mutate_params(p: Dict, strength: float = 1.0, rng: random.Random = None) -> Dict:
    rng = rng or random
    q = clamp_params(copy.deepcopy(p))
    q["shot_frac"] += rng.uniform(-0.1, 0.1) * strength
    q["crank_frac"] += rng.uniform(-0.1, 0.1) * strength
    q["break_dist"] += rng.uniform(-5.0, 5.0) * strength
    if rng.random() < 0.12:
        q["aggressive"] = not q["aggressive"]
    if rng.random() < 0.08:
        q["can_fire"] = not q["can_fire"]
    return clamp_params(q)


def crossover(a: Dict, b: Dict, rng: random.Random = None) -> Dict:
    rng = rng or random
    child = {}
    for k in ("shot_frac", "crank_frac", "break_dist"):
        child[k] = (a[k] + b[k]) / 2.0 + rng.uniform(-0.03, 0.03)
    child["aggressive"] = a["aggressive"] if rng.random() < 0.5 else b["aggressive"]
    child["can_fire"] = a["can_fire"] if rng.random() < 0.5 else b["can_fire"]
    return clamp_params(child)


def run_fsm_match(red: FSMEnemy, blue: FSMEnemy, seed: int = 0,
                  max_cycles: int = LOCKED_MAX_CYCLES) -> Tuple[str, int]:
    """Run one FSM-vs-FSM episode. Returns (winner, red_missiles_launched)."""
    env = BVREnv(scenario={"enemies": ["duck"], "max_cycles": max_cycles}, seed=seed)
    env.reset(seed=seed)
    env.enemy = red
    env.enemy_name = red.name

    for _ in range(max_cycles + 1):
        if not env.blue.alive or not env.red.alive:
            break
        blue_goal = float(np.linalg.norm(env.blue.pos - env.blue.goal))
        red_goal = float(np.linalg.norm(env.red.pos - env.red.goal))
        if blue_goal <= GOAL_RADIUS:
            launched = N_MISSILES - env.red.missiles
            return "blue", launched
        if red_goal <= GOAL_RADIUS:
            launched = N_MISSILES - env.red.missiles
            return "red", launched

        blue_act = blue.act(env._aircraft_state(env.blue, env.red))
        red_act = red.act(env._enemy_state())
        env.external_red_action = np.asarray(red_act, dtype=np.float64).reshape(-1)
        _, _, terminated, truncated, _ = env.step(blue_act)
        if terminated or truncated:
            break

    launched = N_MISSILES - env.red.missiles
    if not env.blue.alive and env.red.alive:
        return "red", launched
    if env.blue.alive and not env.red.alive:
        return "blue", launched
    blue_goal = float(np.linalg.norm(env.blue.pos - env.blue.goal))
    if blue_goal <= GOAL_RADIUS:
        return "blue", launched
    red_goal = float(np.linalg.norm(env.red.pos - env.red.goal))
    if red_goal <= GOAL_RADIUS:
        return "red", launched
    return "draw", launched


def matchup_score(red_params: Dict, blue_params: Dict, seeds: int,
                  rng: random.Random) -> float:
    """Score for ``red_params`` fighting ``blue_params`` (red perspective)."""
    red = FSMEnemy.from_params("red", red_params)
    blue = FSMEnemy.from_params("blue", blue_params)
    wins = 0
    launched_total = 0
    for i in range(seeds):
        seed = rng.randint(0, 1_000_000)
        winner, launched = run_fsm_match(red, blue, seed=seed)
        wins += 1 if winner == "red" else 0
        launched_total += launched
    win_rate = wins / max(seeds, 1)
    avg_launched = launched_total / max(seeds, 1)
    missile_term = (1.0 + N_MISSILES - avg_launched) / N_MISSILES
    return win_rate * 8.0 + missile_term * 2.0


def evaluate_vs_elite(candidate: Dict, elite: List[Dict], seeds: int,
                      rng: random.Random) -> float:
    """Average score vs every member of the elite pool."""
    if not elite:
        return 0.0
    return float(np.mean([matchup_score(candidate, opp, seeds, rng) for opp in elite]))


def initial_population(rng: random.Random) -> List[Dict]:
    """Seed from each archetype vs each archetype (5x5 cross grid)."""
    pop: List[Dict] = []
    for red_name in SELECTABLE_TYPES:
        for blue_name in SELECTABLE_TYPES:
            base = dict(BASE_FSM_PARAMS[red_name])
            # Nudge toward beating this specific opponent archetype.
            opp = BASE_FSM_PARAMS[blue_name]
            if opp["aggressive"]:
                base["break_dist"] = min(45.0, base["break_dist"] + 2.0)
                base["crank_frac"] = min(1.2, base["crank_frac"] + 0.03)
            if opp["can_fire"] and opp["shot_frac"] > 0.8:
                base["shot_frac"] = min(1.2, base["shot_frac"] + 0.05)
            pop.append(clamp_params(mutate_params(base, strength=0.3, rng=rng)))
    # Fill with mutations of the strongest archetypes.
    while len(pop) < 40:
        pop.append(mutate_params(BASE_FSM_PARAMS[rng.choice(SELECTABLE_TYPES)], rng=rng))
    return pop


def optimize(iterations: int = 50, seeds: int = 3, population_size: int = 40,
             progress: Optional[Callable[[int, int, List[Dict], List[float]], None]] = None,
             rng: Optional[random.Random] = None) -> Tuple[List[Dict], List[float]]:
    """Run the evolutionary search. Returns (elite_top10_params, elite_scores)."""
    rng = rng or random.Random(42)
    population = initial_population(rng)

    # Rank seeds by performance vs the five hand-coded archetypes.
    base_elite = [dict(BASE_FSM_PARAMS[n]) for n in SELECTABLE_TYPES]
    scores = [evaluate_vs_elite(cand, base_elite, seeds, rng) for cand in population]

    elite_idx = np.argsort(scores)[::-1][:ELITE_SIZE]
    elite = [copy.deepcopy(population[i]) for i in elite_idx]
    elite_scores = [scores[i] for i in elite_idx]

    for gen in range(iterations):
        offspring: List[Dict] = []
        while len(offspring) < population_size - ELITE_SIZE:
            if rng.random() < 0.35:
                parent = copy.deepcopy(rng.choice(elite))
                offspring.append(mutate_params(parent, strength=0.8, rng=rng))
            else:
                a, b = rng.sample(elite, 2)
                offspring.append(crossover(a, b, rng=rng))

        new_scores = [evaluate_vs_elite(cand, elite, seeds, rng) for cand in offspring]

        combined = [(copy.deepcopy(e), elite_scores[i]) for i, e in enumerate(elite)]
        combined += [(offspring[i], new_scores[i]) for i in range(len(offspring))]
        combined.sort(key=lambda x: x[1], reverse=True)

        elite = [copy.deepcopy(combined[i][0]) for i in range(ELITE_SIZE)]
        elite_scores = [combined[i][1] for i in range(ELITE_SIZE)]

        if progress:
            progress(gen + 1, iterations, elite, elite_scores)

    return elite, elite_scores


def save_reference_enemies(elite: List[Dict], scores: List[float], out_path: str) -> Dict:
    """Write B1..B10 to JSON and return the payload."""
    payload = {}
    for rank, (params, score) in enumerate(zip(elite[:REFERENCE_COUNT], scores[:REFERENCE_COUNT]), start=1):
        name = f"B{rank}"
        entry = clamp_params(copy.deepcopy(params))
        entry["score"] = round(float(score), 4)
        entry["rank"] = rank
        payload[name] = entry

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    from . import enemies as enemies_mod
    enemies_mod.load_reference_enemies(reload=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Optimize FSM reference enemies B1..B10.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3, help="Episodes per elite matchup.")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--out", default=os.path.join("config", "reference_enemies.json"))
    args = parser.parse_args()

    def prog(gen, total, elite, scores):
        top = elite[0]
        print(f"[gen {gen:>2}/{total}] best={scores[0]:.3f}  "
              f"shot={top['shot_frac']:.2f} crank={top['crank_frac']:.2f} "
              f"break={top['break_dist']:.1f}", flush=True)

    print(f"Optimizing FSM references ({args.iterations} generations, "
          f"elite={ELITE_SIZE}, seeds={args.seeds})...")
    elite, scores = optimize(
        iterations=args.iterations,
        seeds=args.seeds,
        population_size=args.population,
        progress=prog,
    )

    payload = save_reference_enemies(elite, scores, args.out)
    print(f"\nSaved {len(payload)} reference enemies to {args.out}:")
    for name in sorted(payload.keys(), key=lambda n: int(n[1:])):
        e = payload[name]
        print(f"  {name}  score={e['score']:.3f}  shot={e['shot_frac']:.2f}  "
              f"crank={e['crank_frac']:.2f}  break={e['break_dist']:.1f}  "
              f"agg={e['aggressive']}  fire={e['can_fire']}")


if __name__ == "__main__":
    main()
