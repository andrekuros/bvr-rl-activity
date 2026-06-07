"""PPO training driver.

Builds the LOCKED model (bvr/policy.py) around the BVR environment, trains with
the student's reward weights, and streams live metrics + evaluation rollouts to
an `on_event` callback so the dashboard can render training in real time.

CLI:
    python -m bvr.train            # full run (timesteps from scenario.json)
    python -m bvr.train --quick    # short smoke run
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from . import rewards as rewards_mod
from .env import BVREnv
from .enemies import SELECTABLE_TYPES, reference_types, training_enemy_pool
from .policy import make_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "student_model.zip")


# ---------------------------------------------------------------------------
# Config helpers (shared by evaluate.py / analysis.py / server).
# ---------------------------------------------------------------------------
def load_scenario(config_dir: str = DEFAULT_CONFIG_DIR) -> Dict:
    path = os.path.join(config_dir, "scenario.json")
    default = {"enemies": list(training_enemy_pool()), "random_enemy_prob": 0.0,
               "max_cycles": 260, "train_timesteps": 200000, "seed": 0,
               "enemy_sampling": "round_robin", "eval_episodes_per_enemy": 1,
               "eval_every_rollouts": 2}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            default.update(json.load(f))
    # When optimized references exist, they replace the archetype list for training.
    refs = reference_types()
    if refs:
        default["enemies"] = refs
    return default


def load_configs(config_dir: str = DEFAULT_CONFIG_DIR):
    rewards = rewards_mod.load_rewards(os.path.join(config_dir, "rewards.json"))
    scenario = load_scenario(config_dir)
    return rewards, scenario


def make_env_fn(reward_config: Dict, scenario: Dict, seed: int = 0):
    def _f():
        return Monitor(BVREnv(reward_config=reward_config, scenario=scenario, seed=seed))
    return _f


# ---------------------------------------------------------------------------
# Evaluation used both live (during training) and standalone.
# ---------------------------------------------------------------------------
def _subsample(frames: List[Dict], cap: int = 140) -> List[Dict]:
    if len(frames) <= cap:
        return frames
    stride = int(np.ceil(len(frames) / cap))
    return frames[::stride]


# Same composite as competition scoring (see bvr/evaluate.py).
EVAL_MISSION_WEIGHT = 0.7
EVAL_KILL_WEIGHT = 0.3


def _competition_score(mission: int, kill: int) -> float:
    return EVAL_MISSION_WEIGHT * mission + EVAL_KILL_WEIGHT * kill


def _run_one_eval_episode(model, reward_config: Dict, scenario: Dict, enemy: str,
                          seed: int, with_frames: bool) -> Dict:
    """Single deterministic eval episode against one enemy."""
    env = BVREnv(reward_config=reward_config, scenario=scenario, seed=seed)
    env.set_enemy(enemy)
    obs, _ = env.reset()
    frames = [env._frame(False)] if with_frames else []
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
        if with_frames:
            frames.append(info["frame"])
        done = terminated or truncated
    result = info.get("result", "timeout")
    mission = 1 if result == "mission" else 0
    kill = 1 if (env.red is not None and not env.red.alive) else 0
    score = _competition_score(mission, kill)
    return {
        "enemy": enemy,
        "result": result,
        "win": mission,
        "mission": mission,
        "kill": kill,
        "score": round(score, 3),
        "reward": round(float(ep_reward), 3),
        "steps": steps,
        "blue_missiles_used": BVREnv_missiles_used(env, "blue"),
        "contributions": {k: round(v, 4) for k, v in term_totals.items()},
        "frames": _subsample(frames) if with_frames else [],
    }


def run_eval_episodes(model, reward_config: Dict, scenario: Dict,
                      enemies: List[str], seed: int = 1234,
                      episodes_per_enemy: int = 1, with_frames: bool = True,
                      on_progress: Optional[Callable] = None) -> List[Dict]:
    """Run one (or N) deterministic episode(s) per enemy for live monitoring."""
    episodes_per_enemy = max(1, int(episodes_per_enemy))
    total = len(enemies) * episodes_per_enemy
    finished = 0
    summaries = []

    for i, enemy in enumerate(enemies):
        runs: List[Dict] = []
        sample_frames: List[Dict] = []
        for j in range(episodes_per_enemy):
            ep = _run_one_eval_episode(
                model, reward_config, scenario, enemy,
                seed=seed + i * 1000 + j,
                with_frames=with_frames and j == 0,
            )
            runs.append(ep)
            if ep.get("frames"):
                sample_frames = ep["frames"]
            finished += 1

        missions = int(sum(r["win"] for r in runs))
        kills = int(sum(r.get("kill", 0) for r in runs))
        mission_rate = missions / episodes_per_enemy
        kill_rate = kills / episodes_per_enemy
        mean_score = float(np.mean([r["score"] for r in runs]))
        mean_reward = float(np.mean([r["reward"] for r in runs]))
        mean_steps = float(np.mean([r["steps"] for r in runs]))
        last = runs[-1]
        summary = {
            "enemy": enemy,
            "result": last["result"],
            "win": float(mission_rate),
            "mission_rate": round(mission_rate, 3),
            "kill_rate": round(kill_rate, 3),
            "missions": missions,
            "kills": kills,
            "score": round(mean_score, 3),
            "n_episodes": episodes_per_enemy,
            "reward": round(mean_reward, 3),
            "mean_reward": round(mean_reward, 3),
            "steps": round(mean_steps, 1),
            "mean_steps": round(mean_steps, 1),
            "blue_missiles_used": round(float(np.mean([r["blue_missiles_used"] for r in runs])), 2),
            "frames": sample_frames,
        }
        summaries.append(summary)
        if on_progress:
            on_progress({
                "finished": finished,
                "total": total,
                "enemy_index": i + 1,
                "enemy_total": len(enemies),
                "running_enemy": enemy,
                "last_result": last["result"],
                "enemy_done": True,
                "enemy_summary": summary,
            })
    return summaries


def BVREnv_missiles_used(env: BVREnv, side: str) -> int:
    from .env import N_MISSILES
    ac = env.blue if side == "blue" else env.red
    return N_MISSILES - (ac.missiles if ac is not None else N_MISSILES)


class LiveCallback(BaseCallback):
    """Emits training metrics every rollout; full multi-enemy eval only periodically."""

    def __init__(self, on_event: Callable, stop_flag: Optional[Callable],
                 reward_config: Dict, scenario: Dict, total_timesteps: int,
                 eval_enemies: List[str]):
        super().__init__()
        self.on_event = on_event
        self.stop_flag = stop_flag
        self.reward_config = reward_config
        self.scenario = scenario
        self.total_timesteps = total_timesteps
        self.eval_enemies = eval_enemies
        self.eval_every_rollouts = max(1, int(scenario.get("eval_every_rollouts", 2)))
        self._rollout_count = 0
        self._last_episodes: List[Dict] = []
        self._last_eval_score = 0.0

    def _on_step(self) -> bool:
        if self.stop_flag is not None and self.stop_flag():
            self.on_event({"type": "status", "state": "stopping"})
            return False
        return True

    def _emit_update(self, ep_rew_mean: float, ep_len_mean: float, episodes: List[Dict],
                     eval_score: float, eval_ran: bool, eval_summary: Optional[Dict] = None) -> None:
        self.on_event({
            "type": "update",
            "timesteps": int(self.num_timesteps),
            "progress": min(self.num_timesteps / max(self.total_timesteps, 1), 1.0),
            "ep_rew_mean": round(ep_rew_mean, 3),
            "ep_len_mean": round(ep_len_mean, 1),
            "eval_score": round(eval_score, 3),
            "winrate": round(eval_score, 3),  # legacy field used by learning curve
            "episodes": episodes,
            "eval_ran": eval_ran,
            "eval_summary": eval_summary,
        })

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        ep_rew_mean = float(np.mean([e["r"] for e in buf])) if len(buf) else 0.0
        ep_len_mean = float(np.mean([e["l"] for e in buf])) if len(buf) else 0.0
        self._rollout_count += 1

        # Full eval is for monitoring only — skip most rollouts so training stays fast.
        if self._rollout_count % self.eval_every_rollouts != 0:
            self._emit_update(ep_rew_mean, ep_len_mean, self._last_episodes,
                              self._last_eval_score, eval_ran=False)
            return

        eps_per = max(1, int(self.scenario.get("eval_episodes_per_enemy", 1)))
        total_sims = len(self.eval_enemies) * eps_per

        def prog(p: Dict) -> None:
            self.on_event({
                "type": "eval_progress",
                "timesteps": int(self.num_timesteps),
                "progress": min(self.num_timesteps / max(self.total_timesteps, 1), 1.0),
                "ep_rew_mean": round(ep_rew_mean, 3),
                "ep_len_mean": round(ep_len_mean, 1),
                "episodes_per_enemy": eps_per,
                **p,
            })

        self.on_event({
            "type": "eval_progress",
            "timesteps": int(self.num_timesteps),
            "finished": 0,
            "total": total_sims,
            "enemy_index": 0,
            "enemy_total": len(self.eval_enemies),
            "running_enemy": self.eval_enemies[0] if self.eval_enemies else "",
            "episodes_per_enemy": eps_per,
            "state": "starting",
        })
        episodes = run_eval_episodes(
            self.model, self.reward_config, self.scenario, self.eval_enemies,
            episodes_per_enemy=eps_per, with_frames=True, on_progress=prog,
        )
        avg_score = float(np.mean([e["score"] for e in episodes])) if episodes else 0.0
        mean_eval_reward = float(np.mean([e["mean_reward"] for e in episodes])) if episodes else 0.0
        self._last_episodes = episodes
        self._last_eval_score = avg_score
        self._emit_update(ep_rew_mean, ep_len_mean, episodes, avg_score, eval_ran=True, eval_summary={
            "episodes_per_enemy": eps_per,
            "total_simulations": total_sims,
            "finished": total_sims,
            "score": round(avg_score, 3),
            "mean_eval_reward": round(mean_eval_reward, 3),
            "monitoring_only": True,
        })


def _default_on_event(event: Dict) -> None:
    if event.get("type") == "update":
        score = event.get("eval_score", event.get("winrate", 0))
        print(f"[t={event['timesteps']:>7}] rew={event['ep_rew_mean']:.2f} "
              f"eval_score={score:.0%} ({int(event['progress']*100)}%)")
    elif event.get("type") == "status":
        print(f"[status] {event.get('state')}")


def run_training(reward_config: Optional[Dict] = None, scenario: Optional[Dict] = None,
                 on_event: Optional[Callable] = None, stop_flag: Optional[Callable] = None,
                 model_out: str = DEFAULT_MODEL_PATH, total_timesteps: Optional[int] = None,
                 seed: int = 0) -> str:
    """Train and save a model. Returns the saved model path."""
    if reward_config is None or scenario is None:
        r, s = load_configs()
        reward_config = reward_config or r
        scenario = scenario or s
    on_event = on_event or _default_on_event
    total_timesteps = int(total_timesteps or scenario.get("train_timesteps", 200000))
    eval_enemies = scenario.get("enemies", SELECTABLE_TYPES) or SELECTABLE_TYPES

    env = DummyVecEnv([make_env_fn(reward_config, scenario, seed)])
    model = make_model(env, seed=seed, verbose=0)

    on_event({"type": "status", "state": "training", "total_timesteps": total_timesteps})
    cb = LiveCallback(on_event, stop_flag, reward_config, scenario, total_timesteps, eval_enemies)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    model.save(model_out)
    env.close()
    on_event({"type": "status", "state": "done", "model_path": model_out})
    return model_out


def main():
    parser = argparse.ArgumentParser(description="Train the BVR PPO agent.")
    parser.add_argument("--quick", action="store_true", help="Short smoke run (~8k steps).")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--emit-json", action="store_true",
                        help="Emit each event as a JSON line on stdout (used by the online platform).")
    args = parser.parse_args()

    reward_config, scenario = load_configs(args.config_dir)
    timesteps = args.timesteps
    if args.quick:
        timesteps = 8000

    on_event = None
    if args.emit_json:
        def on_event(event):
            print(json.dumps(event), flush=True)

    run_training(reward_config, scenario, on_event=on_event, total_timesteps=timesteps,
                 model_out=args.out, seed=int(scenario.get("seed", 0)))


if __name__ == "__main__":
    main()
