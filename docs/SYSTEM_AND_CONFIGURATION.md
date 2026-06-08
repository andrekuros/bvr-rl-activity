# System & Configuration Guide

Technical reference for instructors and administrators: architecture, configuration files, deployment, and scoring.

For the student-facing walkthrough, see [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

---

## 1. What this system is

**BVR RL Activity** is a teaching simulator for *Reinforcement Learning applied to simulation*. Students do **not** implement PPO or change the neural network. They design the **reward function** (weights on mission, combat, and shaping terms). A fixed PPO policy learns in a simplified 2D Beyond Visual Range (BVR) air-combat environment.

| Layer | Role |
|-------|------|
| `bvr/env.py` | Gymnasium environment (physics, radar, missiles, mission goal) |
| `bvr/policy.py` | **Locked** MLP policy architecture |
| `bvr/train.py` | **Locked** PPO training; students only pass reward/scenario config |
| `bvr/evaluate.py` | Official competition scoring (multi-episode, deterministic) |
| `bvr/analysis.py` | Post-training stats, charts, heatmaps |
| `server/` | Local dashboard (`:8000`) — single user, full tooling |
| `online_server/` | Class platform (`:8002`) — logins, quotas, queue, leaderboard, admin |
| `competition_server/` | Optional legacy upload-only leaderboard (`:8001`) |

---

## 2. Architecture overview

```
                    ┌─────────────────────────────────────┐
                    │  Browser UI (static HTML/JS)        │
                    └──────────────┬──────────────────────┘
                                   │ HTTP / WebSocket
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   server/app.py           online_server/main.py     competition_server/
   (:8000 local)           (:8002 class platform)      (:8001 optional)
          │                        │
          │ thread                 │ job queue + subprocess pool
          ▼                        ▼
   bvr/train.py              bvr/train.py --emit-json
   bvr/evaluate.py           bvr/evaluate.py --locked
   bvr/analysis.py           bvr/analysis.py (on demand)
```

### Training flow (online platform)

1. Student queues a run → row inserted in SQLite (`runs` table).
2. Worker picks run → writes `online_server/data/jobs/run_{id}/rewards.json` + `scenario.json`.
3. Subprocess: `python -m bvr.train --emit-json --config-dir … --out models/run_{id}.zip`.
4. JSON lines on stdout → WebSocket events (progress, live eval, learning curve).
5. On success → subprocess: `python -m bvr.evaluate --model … --json --locked --episodes 12`.
6. Scores stored in DB; student can **submit** one run to the leaderboard.

### What is persisted

| Location | Contents |
|----------|----------|
| `config/rewards.json` | Default reward weights (local dashboard) |
| `config/scenario.json` | Default scenario (local dashboard) |
| `config/reference_enemies.json` | Optimized FSM opponents B1–B10 |
| `models/student_model.zip` | Last local training run |
| `online_server/data/platform.db` | Users, sessions, runs, admin settings |
| `online_server/data/models/run_{id}.zip` | One model per online run |
| `online_server/data/jobs/run_{id}/` | Per-run config snapshot |
| `online_server/data/analysis/run_{id}/` | Analysis plots per run |
| `analysis_output/` | Local analysis output |

**Not persisted:** live WebSocket metrics during training (only in browser until refresh).

---

## 3. Configuration files

### 3.1 `config/rewards.json`

The **only learning signal** students tune. Each key is a weight; `0.0` disables that term.

| Term | Type | Meaning |
|------|------|---------|
| `global_scale` | multiplier | Scales all terms |
| `mission_completed` | event | Large bonus for reaching goal alive |
| `hit_enemy` | event | Bonus for killing/reducing enemy |
| `was_hit` | event | Penalty when shot down (keep negative) |
| `fire_missile` | event | Small cost per launch (anti-spam) |
| `miss_missile` | event | Penalty when a missile expires without hit |
| `mission_shaping` | shaping/step | Progress toward goal this step |
| `maintain_track` | shaping/step | Enemy on radar |
| `lost_track` | shaping/step | Lost radar lock (usually negative) |
| `closing_bonus` | shaping/step | Closing distance to enemy |
| `wez_advantage` | shaping/step | Enemy in your WEZ but not vice versa |

### 3.2 `config/scenario.json`

Controls training and live monitoring (not all keys are exposed in every UI).

| Key | Default | Meaning |
|-----|---------|---------|
| `enemies` | B1–B10 | Opponent list for training/eval |
| `random_enemy_prob` | `0.0` | Probability of random opponent each episode |
| `enemy_sampling` | `round_robin` | `round_robin` cycles B1→…→B10 each episode; alternatives in code |
| `max_cycles` | `260` | Max steps per episode |
| `train_timesteps` | `200000` | PPO environment steps per training run |
| `seed` | `0` | RNG seed for reproducibility |
| `eval_episodes_per_enemy` | `1` | Live eval runs per opponent (monitoring only) |
| `eval_every_rollouts` | `2` | Run live eval every N PPO rollouts (~2048 steps each) |

**Note:** Online students pick enemies and steps in the UI; the worker writes a per-run `scenario.json` under `jobs/run_{id}/`. Admin caps steps via `steps_per_run`.

### 3.3 `config/reference_enemies.json`

Generated by evolutionary search (`python -m bvr.fsm_optimize`). When present, **training and locked competition** use **B1 (strongest) … B10** instead of hand-coded archetypes (`duck`, `defensive`, etc.).

Regenerate (instructor, offline):

```bash
python -m bvr.fsm_optimize --iterations 50 --seeds 3
```

---

## 4. Scoring & evaluation

### Mission vs score

- **Mission success:** blue reaches the goal area **alive** (primary objective).
- **Episode outcomes:** `mission`, `shot_down`, `timeout`.

### Competition score (official)

Evaluated over **12 deterministic episodes per enemy** (locked eval):

```
score = 0.6 × mission_rate + 0.25 × kill_rate + 0.15 × missile_efficiency
```

Reported per run on the online platform after training completes.

### Live eval (during training)

- Runs **1 episode per enemy** every few PPO rollouts.
- Same score formula, but **monitoring only** — does not change gradients.
- UI shows **average score across all eval runs** (not per-enemy leaderboard during training).

---

## 5. Deployment

### 5.1 Local lab (`:8000`)

For instructor testing or solo use:

```bash
cd bvr-rl-activity
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python server/app.py
```

Open http://localhost:8000

Features: full reward editor, training, live eval replays, manual replay, statistical analysis.

### 5.2 Online class platform (`:8002`)

**Students use this in class.** Instructor runs:

```bash
set ADMIN_NAME=admin
set ADMIN_PASSWORD=your-secure-password
python online_server/main.py
```

Open http://localhost:8002 (or server IP).

**First launch:** creates admin user and SQLite DB in `online_server/data/`.

**Reset class:** stop server, delete `online_server/data/`, restart.

### 5.3 Admin panel settings

Log in as admin → **Admin** tab:

| Setting | Default | Purpose |
|---------|---------|---------|
| `class_access_code` | `BVR2026` | Required to register |
| `runs_per_window` | `10` | Max queued runs per student per window |
| `steps_per_run` | `200000` | Max timesteps cap |
| `window_hours` | `1` | Rolling quota window |
| `max_concurrent` | `2` | Parallel training subprocesses |
| `registration_open` | `1` | `0` closes new sign-ups |

### 5.4 Final competition

Admin → **Run final competition** (needs ≥2 submissions):

1. **Static ranking** — submitted models vs B1–B10 (same as leaderboard score).
2. **Pool ranking** — student vs student round-robin (policies swap sides).
3. **Causality** — Pearson correlation between each reward weight and outcomes.

Offline equivalent:

```bash
python -m bvr.tournament --models-dir path/to/models --out final_output --seeds 2
```

---

## 6. CLI reference (instructor / debugging)

```bash
python -m bvr.train --quick                    # ~8k steps smoke test
python -m bvr.train                            # full run from scenario.json
python -m bvr.evaluate --locked                # score models/student_model.zip
python -m bvr.analysis                         # charts → analysis_output/
python -m bvr.submit --name "Team" --server http://host:8001
```

---

## 7. Locked vs editable (enforce in grading)

| Editable by students | Locked |
|----------------------|--------|
| All weights in `rewards.json` | `bvr/policy.py` (network) |
| Enemy checklist (within allowed set) | PPO hyperparameters in `bvr/train.py` |
| Training timesteps (within quota) | Environment physics in `bvr/env.py` |
| | Reference enemies B1–B10 (instructor-generated) |

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Port 8002 already in use | Old server still running | Kill process on 8002, restart |
| `/api/enemies` 404 | Stale server without latest code | Restart `online_server/main.py` |
| Homepage 404 on `:8002` | Old build missing `/` route | Pull latest, restart |
| Admin login fails | Wrong password or empty DB | Reset password via `db.set_password` or delete `data/` |
| Students see no B1–B10 | Missing `reference_enemies.json` | Run `fsm_optimize` |
| Live curve empty | Training just started | Wait for 2+ eval points |

---

## 9. Project layout

```
bvr-rl-activity/
  config/                 rewards.json, scenario.json, reference_enemies.json
  bvr/                    core sim + train + eval + analysis + tournament
  server/                 local dashboard (:8000)
  online_server/          class platform (:8002)
  competition_server/     optional upload leaderboard (:8001)
  docs/                   this guide + STUDENT_GUIDE.md
  models/                 local saved model (gitignored)
  analysis_output/        local analysis plots (gitignored)
```
