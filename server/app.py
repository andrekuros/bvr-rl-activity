"""Browser dashboard backend (FastAPI + WebSocket).

Responsibilities:
  - serve the static dashboard UI
  - read/write config/rewards.json and config/scenario.json
  - start/stop a background training thread and stream live updates over a
    WebSocket (metrics + evaluation rollouts for the multi-view canvas)
  - stream a real-time replay of a saved model

Run with:
    uvicorn server.app:app --reload          (from the project root)
or simply:
    python server/app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from typing import Dict, Optional, Set

# Allow running both as `python server/app.py` and `uvicorn server.app:app`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from bvr import rewards as rewards_mod  # noqa: E402
from bvr.enemies import SELECTABLE_TYPES, enemy_catalog, reference_types, training_enemy_pool  # noqa: E402
from bvr.train import (DEFAULT_CONFIG_DIR, DEFAULT_MODEL_PATH, load_configs,  # noqa: E402
                       load_scenario, run_training)
from bvr.evaluate import play_episode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="BVR RL Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Hub:
    """Bridges the worker thread (training/replay) to async WebSocket clients."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: Set[WebSocket] = set()
        self.last_update: Optional[Dict] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def push(self, event: Dict) -> None:
        """Thread-safe: schedule a broadcast on the asyncio loop."""
        if event.get("type") == "update":
            self.last_update = event
        if self.loop is not None:
            self.loop.call_soon_threadsafe(asyncio.create_task, self._broadcast(event))

    async def _broadcast(self, event: Dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


hub = Hub()


def _allowed_enemies():
    return set(training_enemy_pool()) | set(SELECTABLE_TYPES) | set(reference_types())


@app.on_event("startup")
async def _startup():
    hub.loop = asyncio.get_running_loop()
    cat = enemy_catalog()
    print(f"[dashboard] opponents: {cat['names']} (mode={cat['mode']})")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/enemies")
async def get_enemies():
    """Public list of train/competition opponents (B1..B10 when optimized)."""
    catalog = enemy_catalog()
    return catalog


@app.get("/api/config")
async def get_config():
    rewards, scenario = load_configs()
    catalog = enemy_catalog()
    editor = rewards_mod.reward_editor_payload()
    editor["defaults"] = dict(rewards)
    return {
        "rewards": rewards,
        "scenario": scenario,
        "reward_editor": editor,
        "enemy_types": catalog["names"],
        "enemy_catalog": catalog,
        "busy": hub.is_busy(),
        "model_exists": os.path.exists(DEFAULT_MODEL_PATH),
    }


@app.post("/api/config")
async def save_config(payload: Dict):
    rewards = payload.get("rewards", {})
    editor = rewards_mod.reward_editor_payload()
    rewards = rewards_mod.clamp_rewards(rewards, editor["ranges"])
    scenario = payload.get("scenario", {})
    rewards_mod.save_rewards(os.path.join(DEFAULT_CONFIG_DIR, "rewards.json"), rewards)
    current = load_scenario()
    current.update({
        "enemies": [e for e in scenario.get("enemies", current["enemies"]) if e in _allowed_enemies()]
                   or list(training_enemy_pool()),
        "random_enemy_prob": float(scenario.get("random_enemy_prob", current.get("random_enemy_prob", 0.0))),
        "max_cycles": int(scenario.get("max_cycles", current.get("max_cycles", 260))),
        "train_timesteps": int(scenario.get("train_timesteps", current.get("train_timesteps", 200000))),
        "eval_episodes_per_enemy": int(scenario.get("eval_episodes_per_enemy",
                               current.get("eval_episodes_per_enemy", 1))),
        "eval_every_rollouts": int(scenario.get("eval_every_rollouts",
                               current.get("eval_every_rollouts", 2))),
        "enemy_sampling": scenario.get("enemy_sampling", current.get("enemy_sampling", "round_robin")),
    })
    with open(os.path.join(DEFAULT_CONFIG_DIR, "scenario.json"), "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return {"ok": True}


@app.post("/api/train/start")
async def train_start(payload: Optional[Dict] = None):
    if hub.is_busy():
        return JSONResponse({"ok": False, "error": "A job is already running."}, status_code=409)
    payload = payload or {}
    rewards, scenario = load_configs()
    timesteps = scenario.get("train_timesteps", 200000)
    if payload.get("quick"):
        timesteps = 8000
    if payload.get("timesteps"):
        timesteps = int(payload["timesteps"])

    hub.stop_event.clear()

    def worker():
        try:
            run_training(rewards, scenario, on_event=hub.push,
                         stop_flag=hub.stop_event.is_set, total_timesteps=timesteps,
                         seed=int(scenario.get("seed", 0)))
        except Exception as exc:  # surface errors to the UI
            hub.push({"type": "status", "state": "error", "message": str(exc)})

    hub.worker = threading.Thread(target=worker, daemon=True)
    hub.worker.start()
    return {"ok": True, "timesteps": timesteps}


@app.post("/api/train/stop")
async def train_stop():
    hub.stop_event.set()
    return {"ok": True}


@app.post("/api/replay/start")
async def replay_start(payload: Optional[Dict] = None):
    if hub.is_busy():
        return JSONResponse({"ok": False, "error": "A job is already running."}, status_code=409)
    if not os.path.exists(DEFAULT_MODEL_PATH):
        return JSONResponse({"ok": False, "error": "No trained model found. Train first."}, status_code=400)
    payload = payload or {}
    enemy = payload.get("enemy", "balanced")
    rewards, scenario = load_configs()
    from bvr.evaluate import load_model
    model = load_model(DEFAULT_MODEL_PATH)
    hub.stop_event.clear()

    def worker():
        hub.push({"type": "replay_start", "enemy": enemy})

        def on_frame(frame):
            if hub.stop_event.is_set():
                raise KeyboardInterrupt
            hub.push({"type": "replay_frame", "frame": frame})
            time.sleep(0.05)
        try:
            result = play_episode(model, rewards, scenario, enemy, seed=int(time.time()) % 10000,
                                  on_frame=on_frame)
            hub.push({"type": "replay_end", "result": result})
        except KeyboardInterrupt:
            hub.push({"type": "replay_end", "result": {"result": "stopped"}})
        except Exception as exc:
            hub.push({"type": "status", "state": "error", "message": str(exc)})

    hub.worker = threading.Thread(target=worker, daemon=True)
    hub.worker.start()
    return {"ok": True}


@app.post("/api/analysis")
async def run_analysis():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        return JSONResponse({"ok": False, "error": "No trained model found. Train first."}, status_code=400)
    from bvr.analysis import analyze_model
    result = analyze_model(DEFAULT_MODEL_PATH, episodes_per_enemy=10)
    plots = {name: "/analysis/" + os.path.basename(path) for name, path in result["plots"].items()}
    return {"ok": True, "stats": result["stats"], "plots": plots}


@app.get("/analysis/{name}")
async def analysis_file(name: str):
    from bvr.analysis import DEFAULT_OUT_DIR
    path = os.path.join(DEFAULT_OUT_DIR, os.path.basename(name))
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    if hub.last_update is not None:
        await ws.send_json(hub.last_update)
    try:
        while True:
            await ws.receive_text()  # keepalive; clients don't need to send
    except WebSocketDisconnect:
        hub.clients.discard(ws)
    except Exception:
        hub.clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
