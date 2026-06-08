"""Online training platform.

Everything the students need, hosted in one place:
  - self-registration with a class access code (admin-configurable)
  - per-student training QUOTA (N runs of M steps per rolling time window)
  - a job QUEUE with a worker pool that launches each training as a separate
    PROCESS (so a multi-core server trains several students in parallel)
  - live multi-view streaming of each run over a WebSocket
  - submit your best run to the competition leaderboard
  - an admin page to configure the access code, quotas and concurrency

Run with:
    python online_server/main.py            (listens on :8002)
Admin bootstrap (first launch) via env vars ADMIN_NAME / ADMIN_PASSWORD
(defaults to admin / admin - change it!).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from queue import Queue, Empty
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Body, Cookie, FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from bvr import rewards as rewards_mod  # noqa: E402
from bvr.enemies import (SELECTABLE_TYPES, enemy_catalog, eval_enemy_pool,  # noqa: E402
                         reference_types, training_enemy_pool)
from online_server import db  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
STATIC_DIR = os.path.join(HERE, "static")
MODELS_DIR = os.path.join(db.DATA_DIR, "models")
JOBS_DIR = os.path.join(db.DATA_DIR, "jobs")
FINAL_DIR = os.path.join(db.DATA_DIR, "final")
ANALYSIS_DIR = os.path.join(db.DATA_DIR, "analysis")


def _allowed_enemies():
    return (set(training_enemy_pool()) | set(SELECTABLE_TYPES)
            | set(reference_types()) | set(eval_enemy_pool()))


def _cleanup_run_files(run_id: int, run: Dict) -> None:
    """Remove on-disk artifacts for a deleted run."""
    for folder in (os.path.join(JOBS_DIR, f"run_{run_id}"),
                   os.path.join(ANALYSIS_DIR, f"run_{run_id}")):
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
    model = run.get("model_path")
    if model and os.path.isfile(model):
        try:
            os.remove(model)
        except OSError:
            pass


def _user_run(user: Dict, run_id: int) -> Optional[Dict]:
    run = db.get_run(run_id)
    if run is None:
        return None
    if run["user_id"] != user["id"] and not user.get("is_admin"):
        return None
    return run


def _load_run_config(run_id: int, run: Dict):
    job_dir = os.path.join(JOBS_DIR, f"run_{run_id}")
    rewards_path = os.path.join(job_dir, "rewards.json")
    scenario_path = os.path.join(job_dir, "scenario.json")
    if os.path.exists(rewards_path):
        rewards = rewards_mod.load_rewards(rewards_path)
    else:
        rewards = json.loads(run["rewards_json"]) if run.get("rewards_json") else dict(rewards_mod.DEFAULT_REWARDS)
    if os.path.exists(scenario_path):
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)
    else:
        enemies = json.loads(run["enemies"]) if run.get("enemies") else list(training_enemy_pool())
        scenario = {"enemies": enemies, "random_enemy_prob": 0.0, "max_cycles": 260}
    return rewards, scenario

app = FastAPI(title="BVR Online Training Platform")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# WebSocket hub (per user).
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: Dict[int, Set[WebSocket]] = {}

    def register(self, user_id: int, ws: WebSocket):
        self.clients.setdefault(user_id, set()).add(ws)

    def unregister(self, user_id: int, ws: WebSocket):
        self.clients.get(user_id, set()).discard(ws)

    def push(self, user_id: int, event: Dict):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(asyncio.create_task, self._send(user_id, event))

    async def _send(self, user_id: int, event: Dict):
        dead = []
        for ws in list(self.clients.get(user_id, set())):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(user_id, ws)


hub = Hub()


class ReplayManager:
    """One replay at a time per user (streams frames over the WebSocket)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active: Dict[int, int] = {}  # user_id -> run_id
        self.stop = threading.Event()

    def busy(self, user_id: int) -> bool:
        with self.lock:
            return user_id in self.active

    def start(self, user_id: int, run_id: int) -> bool:
        with self.lock:
            if user_id in self.active:
                return False
            self.active[user_id] = run_id
        self.stop.clear()
        return True

    def finish(self, user_id: int) -> None:
        with self.lock:
            self.active.pop(user_id, None)

    def cancel(self, user_id: int) -> None:
        with self.lock:
            if user_id in self.active:
                self.stop.set()


replay_mgr = ReplayManager()


# ---------------------------------------------------------------------------
# Job queue + worker pool (each job is a separate training PROCESS).
# ---------------------------------------------------------------------------
class JobManager:
    def __init__(self):
        self.queue: "Queue[int]" = Queue()
        self.active: Dict[int, subprocess.Popen] = {}
        self.cancelled: Set[int] = set()
        self.lock = threading.Lock()

    def max_concurrent(self) -> int:
        try:
            return max(1, int(db.get_config().get("max_concurrent", "2")))
        except ValueError:
            return 2

    def enqueue(self, run_id: int):
        self.queue.put(run_id)

    def cancel(self, run_id: int):
        with self.lock:
            self.cancelled.add(run_id)
            proc = self.active.get(run_id)
        if proc and proc.poll() is None:
            proc.terminate()

    def start(self):
        threading.Thread(target=self._dispatch_loop, daemon=True).start()
        # Re-enqueue runs that were left queued/running from a previous launch.
        for r in db.list_queued_runs():
            self.enqueue(r["id"])

    def _dispatch_loop(self):
        while True:
            if len(self.active) >= self.max_concurrent():
                time.sleep(0.5)
                continue
            try:
                run_id = self.queue.get(timeout=0.5)
            except Empty:
                continue
            threading.Thread(target=self._run_job, args=(run_id,), daemon=True).start()

    def _run_job(self, run_id: int):
        run = db.get_run(run_id)
        if run is None:
            return
        user_id = run["user_id"]
        position_note = {"type": "status", "run_id": run_id, "state": "running"}
        try:
            job_dir = os.path.join(JOBS_DIR, f"run_{run_id}")
            os.makedirs(job_dir, exist_ok=True)
            os.makedirs(MODELS_DIR, exist_ok=True)
            # Write the per-student config the subprocess will read.
            rewards = json.loads(run["rewards_json"])
            rewards_mod.save_rewards(os.path.join(job_dir, "rewards.json"), rewards)
            enemies = json.loads(run["enemies"]) or list(training_enemy_pool())
            enemy_weights = json.loads(run["enemy_weights_json"]) if run.get("enemy_weights_json") else {}
            scenario = {"enemies": enemies, "random_enemy_prob": 0.0,
                        "enemy_sampling": "weighted" if enemy_weights else "round_robin",
                        "enemy_weights": enemy_weights,
                        "max_cycles": 260, "train_timesteps": run["steps"], "seed": 0,
                        "eval_episodes_per_enemy": int(db.get_config().get("eval_episodes_per_enemy", 1)),
                        "eval_every_rollouts": int(db.get_config().get("eval_every_rollouts", 8)),
                        "live_eval_max_enemies": int(db.get_config().get("live_eval_max_enemies", 4))}
            with open(os.path.join(job_dir, "scenario.json"), "w", encoding="utf-8") as f:
                json.dump(scenario, f)

            model_path = os.path.join(MODELS_DIR, f"run_{run_id}.zip")
            db.update_run(run_id, status="running", started_at=time.time(), model_path=model_path)
            hub.push(user_id, position_note)

            # --- training subprocess (separate core) ---------------------
            cmd = [sys.executable, "-m", "bvr.train", "--emit-json",
                   "--config-dir", job_dir, "--timesteps", str(run["steps"]),
                   "--out", model_path]
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            with self.lock:
                self.active[run_id] = proc

            tail: List[str] = []
            curve = {"t": [], "reward": [], "score": []}
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    tail.append(line)
                    tail[:] = tail[-8:]
                    continue
                event["run_id"] = run_id
                if event.get("type") == "update" and "timesteps" in event:
                    curve["t"].append(int(event["timesteps"]))
                    curve["reward"].append(round(float(event.get("ep_rew_mean", 0.0)), 3))
                    curve["score"].append(round(float(event.get("eval_score", event.get("winrate", 0.0))), 3))
                hub.push(user_id, event)
            proc.wait()

            with self.lock:
                self.active.pop(run_id, None)
                cancelled = run_id in self.cancelled
                self.cancelled.discard(run_id)

            if cancelled:
                db.update_run(run_id, status="stopped", finished_at=time.time())
                hub.push(user_id, {"type": "status", "run_id": run_id, "state": "stopped"})
                return
            if proc.returncode != 0:
                raise RuntimeError("training failed: " + " | ".join(tail[-4:]))

            # --- evaluation subprocess (locked competition scoring) -------
            hub.push(user_id, {"type": "status", "run_id": run_id, "state": "evaluating"})
            locked_eps = int(db.get_config().get("locked_eval_episodes_per_enemy", 30))
            ev_cmd = [sys.executable, "-m", "bvr.evaluate", "--model", model_path,
                      "--json", "--locked", "--episodes", str(locked_eps)]
            out = subprocess.run(ev_cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
            stats = json.loads(out.stdout.strip().splitlines()[-1])
            db.update_run(run_id, status="done", finished_at=time.time(),
                          score=stats["score"], mission_rate=stats["mission_rate"],
                          kill_rate=stats["kill_rate"], survival_rate=stats["survival_rate"],
                          mean_reward=stats["mean_reward"],
                          missile_efficiency=stats["missile_efficiency"],
                          curve_json=json.dumps(curve), analysis_status="pending")
            hub.push(user_id, {"type": "done", "run_id": run_id, "stats": stats})

            # --- automatic post-run analysis report (in-process) ----------
            hub.push(user_id, {"type": "report", "run_id": run_id, "state": "pending"})
            try:
                run_row = db.get_run(run_id)
                rewards, scenario = _load_run_config(run_id, run_row)
                # Analyse against the full locked opponent set (same as scoring),
                # not just the opponents the student trained on.
                scenario = {**scenario, "enemies": list(eval_enemy_pool())}
                out_dir = os.path.join(ANALYSIS_DIR, f"run_{run_id}")
                from bvr.analysis import analyze_model
                analysis_eps = int(db.get_config().get("analysis_episodes_per_enemy", 10))
                result = analyze_model(model_path, reward_config=rewards, scenario=scenario,
                                       out_dir=out_dir, episodes_per_enemy=analysis_eps)
                report = {
                    "stats": result["stats"],
                    "plots": {name: os.path.basename(path)
                              for name, path in result["plots"].items()},
                }
                db.update_run(run_id, analysis_json=json.dumps(report), analysis_status="ready")
                hub.push(user_id, {"type": "report", "run_id": run_id, "state": "ready"})
            except Exception as exc:
                db.update_run(run_id, analysis_status="error",
                              analysis_json=json.dumps({"error": str(exc)[:400]}))
                hub.push(user_id, {"type": "report", "run_id": run_id, "state": "error",
                                   "message": str(exc)[:300]})
        except Exception as exc:
            with self.lock:
                self.active.pop(run_id, None)
            db.update_run(run_id, status="error", finished_at=time.time(), error=str(exc)[:500])
            hub.push(user_id, {"type": "status", "run_id": run_id, "state": "error", "message": str(exc)[:300]})


jobs = JobManager()


# ---------------------------------------------------------------------------
# Auth helpers.
# ---------------------------------------------------------------------------
def current_user(sid: Optional[str]) -> Optional[Dict]:
    return db.session_user(sid)


def quota_info(user_id: int) -> Dict:
    cfg = db.get_config()
    window_hours = float(cfg["window_hours"])
    per_window = int(cfg["runs_per_window"])
    used = db.count_runs_in_window(user_id, window_hours)
    return {
        "used": used,
        "per_window": per_window,
        "remaining": max(0, per_window - used),
        "window_hours": window_hours,
        "steps_per_run": int(cfg["steps_per_run"]),
        "steps_editable": cfg.get("steps_editable", "0") == "1",
        "locked_eval_episodes_per_enemy": int(cfg.get("locked_eval_episodes_per_enemy", 30)),
        "analysis_episodes_per_enemy": int(cfg.get("analysis_episodes_per_enemy", 10)),
    }


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    db.init_db()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    # Bootstrap an admin account on first launch.
    if db.user_count() == 0:
        name = os.environ.get("ADMIN_NAME", "admin")
        pw = os.environ.get("ADMIN_PASSWORD", "admin")
        db.create_user(name, pw, is_admin=True)
        print(f"[platform] created admin user '{name}' / password '{pw}' "
              f"(override with ADMIN_NAME / ADMIN_PASSWORD env vars).")
    hub.loop = asyncio.get_running_loop()
    jobs.start()
    cat = enemy_catalog()
    print(f"[platform] opponents: {cat['names']} (mode={cat['mode']})")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/enemies")
async def get_enemies():
    """Public opponent list (B1..B10 when reference_enemies.json exists)."""
    return enemy_catalog()


# --- auth routes ---------------------------------------------------------
@app.post("/api/register")
async def register(payload: Dict):
    cfg = db.get_config()
    if cfg.get("registration_open", "1") != "1":
        return JSONResponse({"ok": False, "error": "Registration is closed."}, status_code=403)
    if payload.get("access_code", "") != cfg["class_access_code"]:
        return JSONResponse({"ok": False, "error": "Invalid class access code."}, status_code=403)
    user = db.create_user(payload.get("name", ""), payload.get("password", ""))
    if user is None:
        return JSONResponse({"ok": False, "error": "Name taken or invalid."}, status_code=400)
    token = db.create_session(user["id"])
    resp = JSONResponse({"ok": True, "name": user["name"]})
    resp.set_cookie("sid", token, httponly=True, samesite="lax", max_age=7 * 86400)
    return resp


@app.post("/api/login")
async def login(payload: Dict):
    user = db.verify_user(payload.get("name", ""), payload.get("password", ""))
    if user is None:
        return JSONResponse({"ok": False, "error": "Wrong name or password."}, status_code=401)
    token = db.create_session(user["id"])
    resp = JSONResponse({"ok": True, "name": user["name"], "is_admin": bool(user["is_admin"])})
    resp.set_cookie("sid", token, httponly=True, samesite="lax", max_age=7 * 86400)
    return resp


@app.post("/api/logout")
async def logout(sid: Optional[str] = Cookie(None)):
    db.delete_session(sid)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("sid")
    return resp


@app.get("/api/me")
async def me(sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "id": user["id"],
        "name": user["name"],
        "is_admin": bool(user["is_admin"]),
        "quota": quota_info(user["id"]),
        "reward_editor": rewards_mod.reward_editor_payload(db.get_config()),
        "enemy_types": enemy_catalog()["names"],
        "enemy_catalog": enemy_catalog(),
    }


# --- training / runs -----------------------------------------------------
@app.get("/api/runs")
async def my_runs(sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    runs = db.all_runs(100) if user["is_admin"] else db.list_user_runs(user["id"])
    return {"runs": runs, "quota": quota_info(user["id"]), "is_admin": bool(user["is_admin"])}


@app.post("/api/train")
async def start_train(payload: Dict, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    q = quota_info(user["id"])
    if q["remaining"] <= 0:
        return JSONResponse({"ok": False, "error": "Quota exhausted for this time window."}, status_code=429)

    rewards_in = payload.get("rewards", {})
    platform_cfg = db.get_config()
    editor = rewards_mod.reward_editor_payload(platform_cfg)
    defaults = editor["defaults"]
    ranges = editor["ranges"]
    rewards = {"global_scale": float(rewards_in.get("global_scale", defaults["global_scale"]))}
    for k in rewards_mod.REWARD_TERMS:
        rewards[k] = float(rewards_in.get(k, defaults.get(k, 0.0)))
    rewards = rewards_mod.clamp_rewards(rewards, ranges)
    allowed = _allowed_enemies()
    enemies = [e for e in payload.get("enemies", []) if e in allowed] or list(training_enemy_pool())
    # Per-enemy priority (0..1) controlling how often each opponent appears.
    raw_weights = payload.get("enemy_weights") or {}
    enemy_weights = {}
    for e in enemies:
        try:
            enemy_weights[e] = max(0.0, min(1.0, float(raw_weights.get(e, 1.0))))
        except (TypeError, ValueError):
            enemy_weights[e] = 1.0
    # Drop opponents whose priority is 0 (treated as deselected).
    enemies = [e for e in enemies if enemy_weights.get(e, 1.0) > 0] or list(training_enemy_pool())
    cfg = db.get_config()
    max_steps = int(cfg.get("steps_per_run", q["steps_per_run"]))
    if cfg.get("steps_editable", "0") == "1":
        steps = min(int(payload.get("steps", max_steps)), max_steps)
    else:
        steps = max_steps
    steps = max(2000, steps)

    run_id = db.create_run(user["id"], steps, json.dumps(rewards), json.dumps(enemies),
                           json.dumps(enemy_weights))
    jobs.enqueue(run_id)
    return {
        "ok": True, "run_id": run_id, "run_uid": f"R{run_id:06d}",
        "steps": steps, "quota": quota_info(user["id"]),
    }


@app.get("/api/run/{run_id}")
async def get_run(run_id: int, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    run = _user_run(user, run_id)
    if run is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    model_ok = bool(run.get("model_path")) and os.path.exists(run["model_path"])
    return {
        "run": {
            "id": run["id"],
            "status": run["status"],
            "steps": run["steps"],
            "score": run.get("score"),
            "model_ready": model_ok,
        }
    }


@app.post("/api/run/{run_id}/stop")
async def stop_run(run_id: int, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    run = db.get_run(run_id)
    if user is None or run is None or run["user_id"] != user["id"]:
        return JSONResponse({"error": "not found"}, status_code=404)
    jobs.cancel(run_id)
    return {"ok": True}


@app.post("/api/run/{run_id}/submit")
async def submit_run(run_id: int, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    run = db.get_run(run_id)
    if user is None or run is None or run["user_id"] != user["id"]:
        return JSONResponse({"error": "not found"}, status_code=404)
    if run["status"] != "done" or run["score"] is None:
        return JSONResponse({"ok": False, "error": "Run is not scored yet."}, status_code=400)
    db.set_submitted_exclusive(user["id"], run_id)
    return {"ok": True}


def _report_context(user_id: int) -> Dict:
    """Auto-derived context for the student deliverable: their submitted run
    summary plus best/worst enemy matchups from its analysis."""
    run = db.user_submitted_run(user_id)
    if run is None:
        return {"has_submission": False}
    ctx = {
        "has_submission": True,
        "run_uid": run.get("run_uid") or f"R{run['id']:06d}",
        "score": run.get("score"),
        "mission_rate": run.get("mission_rate"),
        "kill_rate": run.get("kill_rate"),
        "survival_rate": run.get("survival_rate"),
        "missile_efficiency": run.get("missile_efficiency"),
        "mean_reward": run.get("mean_reward"),
        "steps": run.get("steps"),
        "enemies": json.loads(run["enemies"]) if run.get("enemies") else [],
        "best": [], "worst": [],
    }
    try:
        report = json.loads(run["analysis_json"]) if run.get("analysis_json") else None
        per_enemy = (report or {}).get("stats", {}).get("per_enemy", {})
        ranked = sorted(
            per_enemy.items(),
            key=lambda kv: (kv[1].get("mission_rate", 0), kv[1].get("kills", 0)),
        )
        fmt = lambda kv: {"enemy": kv[0], "mission_rate": kv[1].get("mission_rate"),
                          "kills": kv[1].get("kills")}
        ctx["worst"] = [fmt(kv) for kv in ranked[:3]]
        ctx["best"] = [fmt(kv) for kv in ranked[-3:][::-1]]
    except Exception:
        pass
    return ctx


@app.get("/api/report")
async def get_my_report(sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    rep = db.get_user_report(user["id"])
    return {
        "ok": True,
        "data": json.loads(rep["data"]) if rep else {},
        "updated_at": rep["updated_at"] if rep else None,
        "context": _report_context(user["id"]),
    }


@app.post("/api/report")
async def save_my_report(payload: Dict, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Invalid report data."}, status_code=400)
    # Keep it bounded so a single field can't bloat the DB.
    clean = {str(k): str(v)[:8000] for k, v in data.items()}
    db.save_user_report(user["id"], json.dumps(clean))
    return {"ok": True}


@app.get("/api/run/{run_id}/report")
async def run_report(run_id: int, sid: Optional[str] = Cookie(None)):
    """View-only run report: selected parameters, learning curve and the
    analysis that was generated automatically when the run finished."""
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    run = _user_run(user, run_id)
    if run is None:
        return JSONResponse({"ok": False, "error": "Run not found or access denied."}, status_code=404)

    rewards, scenario = _load_run_config(run_id, run)
    enemies = scenario.get("enemies") or (json.loads(run["enemies"]) if run.get("enemies") else [])
    curve = json.loads(run["curve_json"]) if run.get("curve_json") else {"t": [], "reward": [], "score": []}
    analysis_status = run.get("analysis_status")
    report = json.loads(run["analysis_json"]) if run.get("analysis_json") else None
    stats = plots = None
    if report and "stats" in report:
        stats = report["stats"]
        plots = {name: f"/api/run/{run_id}/analysis/{os.path.basename(fn)}"
                 for name, fn in (report.get("plots") or {}).items()}

    return {
        "ok": True,
        "run": {
            "id": run["id"], "run_uid": run.get("run_uid") or f"R{run['id']:06d}",
            "status": run["status"], "steps": run["steps"],
            "score": run["score"], "mission_rate": run["mission_rate"],
            "kill_rate": run["kill_rate"], "survival_rate": run["survival_rate"],
            "mean_reward": run["mean_reward"], "missile_efficiency": run["missile_efficiency"],
            "submitted": bool(run["submitted"]),
        },
        "params": {"steps": run["steps"], "enemies": enemies, "rewards": rewards,
                   "enemy_weights": json.loads(run["enemy_weights_json"]) if run.get("enemy_weights_json") else {}},
        "curve": curve,
        "analysis_status": analysis_status,
        "stats": stats,
        "plots": plots,
        "error": report.get("error") if report else None,
    }


@app.get("/api/run/{run_id}/analysis/{name}")
async def analysis_file(run_id: int, name: str, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=403)
    if _user_run(user, run_id) is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = os.path.join(ANALYSIS_DIR, f"run_{run_id}", os.path.basename(name))
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.post("/api/run/{run_id}/replay/start")
async def replay_run(run_id: int, payload: Dict = Body(default={}), sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    run = _user_run(user, run_id)
    if run is None:
        return JSONResponse({"ok": False, "error": "Run not found or access denied."}, status_code=404)
    if run["status"] != "done" or not run.get("model_path"):
        return JSONResponse({"ok": False, "error": "Run is not finished."}, status_code=400)
    if not os.path.exists(run["model_path"]):
        return JSONResponse({"ok": False, "error": "Model file missing."}, status_code=400)
    if replay_mgr.busy(user["id"]):
        return JSONResponse({"ok": False, "error": "A replay is already running."}, status_code=409)
    enemy = payload.get("enemy", "B1")
    if enemy not in _allowed_enemies():
        enemy = list(training_enemy_pool())[0]
    rewards, scenario = _load_run_config(run_id, run)
    if not replay_mgr.start(user["id"], run_id):
        return JSONResponse({"ok": False, "error": "Could not start replay."}, status_code=409)
    user_id = user["id"]

    def worker():
        from bvr.evaluate import load_model, play_episode
        try:
            model = load_model(run["model_path"])
            hub.push(user_id, {"type": "replay_start", "run_id": run_id, "enemy": enemy})

            def on_frame(frame):
                if replay_mgr.stop.is_set():
                    raise KeyboardInterrupt
                hub.push(user_id, {"type": "replay_frame", "run_id": run_id, "frame": frame})
                time.sleep(0.05)

            result = play_episode(model, rewards, scenario, enemy,
                                  seed=int(time.time()) % 10000, on_frame=on_frame)
            hub.push(user_id, {"type": "replay_end", "run_id": run_id, "result": result})
        except KeyboardInterrupt:
            hub.push(user_id, {"type": "replay_end", "run_id": run_id, "result": {"result": "stopped"}})
        except Exception as exc:
            hub.push(user_id, {"type": "status", "run_id": run_id, "state": "error", "message": str(exc)[:300]})
        finally:
            replay_mgr.finish(user_id)

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "enemy": enemy}


@app.post("/api/run/{run_id}/replay/stop")
async def replay_stop(run_id: int, sid: Optional[str] = Cookie(None)):
    user = current_user(sid)
    if user is None:
        return JSONResponse({"error": "auth"}, status_code=401)
    replay_mgr.cancel(user["id"])
    return {"ok": True}


@app.get("/api/leaderboard")
async def leaderboard():
    return {"leaderboard": db.leaderboard(), "enemies": list(eval_enemy_pool())}


# --- admin ---------------------------------------------------------------
def _require_admin(sid):
    user = current_user(sid)
    if user is None or not user["is_admin"]:
        return None
    return user


@app.get("/api/admin/config")
async def admin_get(sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    return {"config": db.get_config(), "runs": db.all_runs(100),
            "reward_editor": rewards_mod.reward_editor_payload(db.get_config()),
            "code_defaults": rewards_mod.DEFAULT_REWARDS,
            "code_ranges": rewards_mod.DEFAULT_RANGES}


@app.post("/api/admin/config")
async def admin_set(payload: Dict, sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    allowed = {"class_access_code", "runs_per_window", "steps_per_run",
               "window_hours", "max_concurrent", "registration_open",
               "rewards_start_zero", "reward_defaults_json", "reward_ranges_json",
               "eval_every_rollouts", "eval_episodes_per_enemy", "live_eval_max_enemies",
               "locked_eval_episodes_per_enemy", "analysis_episodes_per_enemy",
               "steps_editable"}
    db.set_config({k: v for k, v in payload.items() if k in allowed})
    return {"ok": True, "config": db.get_config()}


@app.delete("/api/admin/run/{run_id}")
async def admin_delete_run(run_id: int, sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    run = db.get_run(run_id)
    if run is None:
        return JSONResponse({"ok": False, "error": "Run not found."}, status_code=404)
    jobs.cancel(run_id)
    _cleanup_run_files(run_id, run)
    db.delete_run(run_id)
    return {"ok": True}


# --- admin: users panel --------------------------------------------------
@app.get("/api/admin/users")
async def admin_users(sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    cfg = db.get_config()
    window_hours = float(cfg["window_hours"])
    per_window = int(cfg["runs_per_window"])
    users = []
    for u in db.list_users():
        used = db.count_runs_in_window(u["id"], window_hours)
        users.append({
            **u,
            "quota_used": used,
            "quota_per_window": per_window,
            "quota_remaining": max(0, per_window - used),
            "has_report": bool(u.get("has_report")),
        })
    return {"ok": True, "users": users, "window_hours": window_hours}


@app.post("/api/admin/user/{user_id}/reset-quota")
async def admin_reset_quota(user_id: int, sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    if db.get_user(user_id) is None:
        return JSONResponse({"ok": False, "error": "User not found."}, status_code=404)
    db.reset_user_quota(user_id)
    return {"ok": True}


@app.get("/api/admin/user/{user_id}/report")
async def admin_user_report(user_id: int, sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    user = db.get_user(user_id)
    if user is None:
        return JSONResponse({"ok": False, "error": "User not found."}, status_code=404)
    rep = db.get_user_report(user_id)
    return {
        "ok": True,
        "user": {"id": user["id"], "name": user["name"]},
        "data": json.loads(rep["data"]) if rep else {},
        "updated_at": rep["updated_at"] if rep else None,
        "context": _report_context(user_id),
        "runs": db.list_user_runs(user_id),
    }


@app.get("/api/admin/reports/export")
async def admin_export_reports(sid: Optional[str] = Cookie(None)):
    """Download every student's written report + their submitted-run context as
    one JSON file, for offline grading/review."""
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    fields = ["strategy", "reward_link", "curve", "results", "matchups", "next_step"]
    out = []
    for u in db.list_users():
        if u.get("is_admin"):
            continue
        rep = db.get_user_report(u["id"])
        data = json.loads(rep["data"]) if rep else {}
        out.append({
            "user": u["name"],
            "updated_at": rep["updated_at"] if rep else None,
            "report": {k: data.get(k, "") for k in fields},
            "submission": _report_context(u["id"]),
        })
    payload = {"exported_at": time.time(), "students": out}
    headers = {"Content-Disposition": "attachment; filename=student_reports.json"}
    return JSONResponse(payload, headers=headers)


# --- final competition (admin command) -----------------------------------
final_state: Dict = {"running": False, "progress": ""}


def _load_final_report() -> Optional[Dict]:
    path = os.path.join(FINAL_DIR, "report.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@app.post("/api/admin/final/start")
async def final_start(sid: Optional[str] = Cookie(None)):
    admin = _require_admin(sid)
    if admin is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    if final_state["running"]:
        return JSONResponse({"ok": False, "error": "Final competition already running."}, status_code=409)
    subs = db.submitted_entries()
    if len(subs) < 2:
        return JSONResponse({"ok": False, "error": "Need at least 2 submissions."}, status_code=400)

    final_state.update(running=True, progress="starting")
    admin_id = admin["id"]

    def worker():
        try:
            from bvr.tournament import final_report
            entries = [{
                "name": s["name"], "model_path": s["model_path"],
                "rewards": json.loads(s["rewards_json"]) if s["rewards_json"] else {},
                "score": s["score"], "mission_rate": s["mission_rate"], "kill_rate": s["kill_rate"],
            } for s in subs]

            def prog(done, total):
                final_state["progress"] = f"duels {done}/{total}"
                hub.push(admin_id, {"type": "final", "state": "running", "progress": f"{done}/{total}"})

            os.makedirs(FINAL_DIR, exist_ok=True)
            report = final_report(entries, FINAL_DIR, seeds=2, progress=prog)
            with open(os.path.join(FINAL_DIR, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f)
            hub.push(admin_id, {"type": "final", "state": "done"})
        except Exception as exc:
            hub.push(admin_id, {"type": "final", "state": "error", "message": str(exc)[:300]})
        finally:
            final_state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "n_students": len(subs)}


@app.get("/api/admin/final")
async def final_get(sid: Optional[str] = Cookie(None)):
    if _require_admin(sid) is None:
        return JSONResponse({"error": "admin only"}, status_code=403)
    report = _load_final_report()
    plots = {}
    if report:
        plots = {k: "/final/" + os.path.basename(v) for k, v in report.get("plots", {}).items()}
    return {"running": final_state["running"], "progress": final_state["progress"],
            "report": report, "plots": plots}


@app.get("/final/{name}")
async def final_file(name: str, sid: Optional[str] = Cookie(None)):
    if current_user(sid) is None:
        return JSONResponse({"error": "auth"}, status_code=403)
    path = os.path.join(FINAL_DIR, os.path.basename(name))
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


# --- websocket -----------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sid = ws.cookies.get("sid")
    user = current_user(sid)
    if user is None:
        await ws.close(code=4401)
        return
    user_id = user["id"]
    hub.register(user_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.unregister(user_id, ws)
    except Exception:
        hub.unregister(user_id, ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
