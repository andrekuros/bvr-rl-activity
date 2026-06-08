"""Minimal online competition server.

Run this on a public host. Students POST their trained weights to /submit; the
server re-evaluates the weights against the LOCKED enemy set under fixed
conditions, records the score (win rate), and serves a leaderboard.

Run with:
    python competition_server/main.py            (listens on :8001)
or:
    uvicorn competition_server.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import zipfile

# Make the bvr package importable when launched from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from bvr import rewards as rewards_mod  # noqa: E402
from bvr.enemies import training_enemy_pool  # noqa: E402
from bvr.evaluate import evaluate_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_PATH = os.path.join(HERE, "leaderboard.json")
SUBMISSIONS_DIR = os.path.join(HERE, "submissions")
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Locked evaluation contract: every submission faces the same opponents.
LOCKED_SCENARIO = {
    "enemies": list(training_enemy_pool()),
    "random_enemy_prob": 0.0,
    "max_cycles": 260,
}
EPISODES_PER_ENEMY = 20

app = FastAPI(title="BVR Competition Server")


def _load_board():
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_board(board):
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(board, f, indent=2)


@app.post("/submit")
async def submit(name: str = Form(...), submission: UploadFile = None):
    if submission is None:
        return JSONResponse({"ok": False, "error": "missing submission file"}, status_code=400)

    raw = await submission.read()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "bundle.zip")
        with open(bundle, "wb") as f:
            f.write(raw)
        try:
            with zipfile.ZipFile(bundle) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            return JSONResponse({"ok": False, "error": "invalid zip"}, status_code=400)

        model_path = os.path.join(tmp, "model.zip")
        if not os.path.exists(model_path):
            return JSONResponse({"ok": False, "error": "model.zip not found in submission"}, status_code=400)

        try:
            stats = evaluate_model(model_path, reward_config=dict(rewards_mod.DEFAULT_REWARDS),
                                   scenario=LOCKED_SCENARIO, enemies=LOCKED_SCENARIO["enemies"],
                                   episodes_per_enemy=EPISODES_PER_ENEMY, seed=4242)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"evaluation failed: {exc}"}, status_code=400)

    entry = {
        "name": name[:48],
        "score": stats["score"],
        "mission_rate": stats["mission_rate"],
        "kill_rate": stats["kill_rate"],
        "survival_rate": stats["survival_rate"],
        "mean_reward": stats["mean_reward"],
        "missile_efficiency": stats["missile_efficiency"],
        "per_enemy": {e: stats["per_enemy"][e]["mission_rate"] for e in stats["per_enemy"]},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    board = _load_board()
    # Keep only the best submission per name.
    board = [b for b in board if b["name"] != entry["name"]]
    board.append(entry)
    board.sort(key=lambda b: (b["score"], b.get("mission_rate", 0), b["mean_reward"]), reverse=True)
    _save_board(board)

    rank = next((i + 1 for i, b in enumerate(board) if b["name"] == entry["name"]), len(board))
    return {"ok": True, "score": entry["score"], "winrate": entry["winrate"],
            "rank": rank, "total": len(board)}


@app.get("/api/leaderboard")
async def api_leaderboard():
    return {"leaderboard": _load_board(), "enemies": LOCKED_SCENARIO["enemies"],
            "episodes_per_enemy": EPISODES_PER_ENEMY}


@app.get("/", response_class=HTMLResponse)
async def leaderboard_page():
    board = _load_board()
    enemies = LOCKED_SCENARIO["enemies"]
    head = "".join(f"<th>{e}</th>" for e in enemies)
    rows = ""
    for i, b in enumerate(board):
        per = "".join(f"<td>{int(b['per_enemy'].get(e, 0)*100)}%</td>" for e in enemies)
        rows += (f"<tr><td>{i+1}</td><td class='name'>{b['name']}</td>"
                 f"<td class='score'>{b['score']*100:.1f}%</td>"
                 f"<td>{b.get('mission_rate',0)*100:.0f}%</td><td>{b.get('kill_rate',0)*100:.0f}%</td>"
                 f"<td>{b['mean_reward']}</td><td>{b['missile_efficiency']}</td>{per}"
                 f"<td class='ts'>{b['timestamp']}</td></tr>")
    if not rows:
        rows = f"<tr><td colspan='{8+len(enemies)}'>No submissions yet.</td></tr>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>BVR Competition Leaderboard</title>
<meta http-equiv="refresh" content="20"/>
<style>
 body{{font-family:Segoe UI,system-ui,sans-serif;background:#0e1320;color:#e6ecf5;margin:0;padding:30px}}
 h1{{font-size:22px}} .sub{{color:#8b97ad;font-size:13px;margin-bottom:18px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border:1px solid #28324a;padding:7px 10px;text-align:center}}
 th{{background:#161d2e;color:#8b97ad;text-transform:uppercase;font-size:11px}}
 td.name{{text-align:left;font-weight:600}} td.score{{color:#5ec27a;font-weight:700}}
 td.ts{{color:#8b97ad;font-size:11px}} tr:nth-child(even){{background:#131a29}}
</style></head><body>
<h1>BVR Reinforcement Learning - Competition Leaderboard</h1>
<div class="sub">Each submission is evaluated over {EPISODES_PER_ENEMY} episodes vs every locked enemy. Score = 0.6 x mission rate + 0.25 x kill rate + 0.15 x missile efficiency. Auto-refreshes every 20s.</div>
<table><tr><th>#</th><th>name</th><th>score</th><th>mission</th><th>kill</th><th>reward</th><th>missile eff.</th>{head}<th>submitted</th></tr>
{rows}</table></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
