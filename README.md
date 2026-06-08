# BVR Reinforcement Learning - Classroom Activity

A simplified **Beyond Visual Range (BVR) air-combat** environment for teaching
Reinforcement Learning, inspired by [B-ACE](https://github.com/andrekuros/B-ACE)
but written in pure Python (no game engine) so it runs anywhere in minutes.

You train a fighter agent (PPO) to complete a mission against AI enemies. **The
neural network and the learning algorithm are fixed.** Your job is to design the
*reward function*: decide what the agent should be rewarded and penalized for.
Then you submit your trained model to the class competition.

## Documentation

| Guide | Audience |
|-------|----------|
| **[docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md)** | Students — what each UI element means, how to train, submit, and interpret results |
| **[docs/SYSTEM_AND_CONFIGURATION.md](docs/SYSTEM_AND_CONFIGURATION.md)** | Instructors — architecture, config files, deployment, scoring, admin |

---

## What you may change vs. what is locked

| You CAN edit                          | LOCKED (do not edit)                 |
| ------------------------------------- | ------------------------------------ |
| `config/rewards.json` (all weights)   | `bvr/policy.py` (network)            |
| `config/scenario.json` (enemy mix)    | PPO hyperparameters in `bvr/train.py`|

Setting any reward weight to `0.0` simply turns that signal off, including the
shaping rewards and the global rewards.

---

## 1. Install

```bash
cd bvr-rl-activity
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Open the dashboard (recommended way to work)

```bash
python server/app.py
```

Open http://localhost:8000. From the dashboard you can:

- edit every reward / penalty (and click `0` to disable a signal),
- choose which of the 5 enemies (+ optional random agent) to train against,
- **Start Training** and watch live: one tactical view per enemy, learning
  curves, and win rate per enemy, all updating during training,
- **Watch trained model** to replay a match in real time after training,
- **Run analysis** to get win/survival/kill rates, reward-term breakdown,
  missile efficiency, and a trajectory heatmap.

## 3. Or use the command line

```bash
python -m bvr.train --quick      # short smoke run
python -m bvr.train              # full run (timesteps from scenario.json)
python -m bvr.evaluate           # score the saved model vs every enemy
python -m bvr.analysis           # write charts to analysis_output/
```

The trained model is saved to `models/student_model.zip`.

## 4. Submit to the competition

```bash
python -m bvr.submit --name "Your Team" --server http://<SERVER>:8001
```

The server re-runs your weights against the locked enemy set and ranks you by
score. View the leaderboard at `http://<SERVER>:8001/`.

---

## Online training arena (hosted, with logins + quotas)

For a fully online experience, the instructor runs the **online platform** and
students do everything in the browser - no local install required:

```bash
# Instructor, on a multi-core server:
set ADMIN_PASSWORD=change-me      # Windows (use export on macOS/Linux)
python online_server/main.py      # listens on :8002
```

Students open `http://<SERVER>:8002/` and:

1. **Register** with the class access code (the instructor sets it).
2. Tune their rewards/enemies and **queue a training run**. Runs execute
   server-side as separate processes (true multi-core parallelism) and are
   limited by a per-student **quota** (e.g. 10 runs of 200k steps per hour).
3. Watch the run live (multi-view + learning curve), then **submit their best
   run** to the competition leaderboard (only one submission counts per student).

### Admin

Log in as the admin account (bootstrapped on first launch from `ADMIN_NAME` /
`ADMIN_PASSWORD`, default `admin` / `admin`). The **Admin** tab configures, live:

| Setting              | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| `class_access_code`  | Code required to register.                          |
| `runs_per_window`    | Training runs allowed per student per window.       |
| `steps_per_run`      | Max timesteps per run (caps what students request). |
| `window_hours`       | Length of the rolling quota window.                 |
| `max_concurrent`     | How many training processes run at once.            |
| `registration_open`  | `1`/`0` to open/close sign-ups.                     |

State is stored in `online_server/data/` (SQLite + saved models); delete that
folder to reset the class.

### Final competition (end-of-class command)

When everyone has submitted, the admin presses **Run final competition** in the
Admin tab. It produces a combined report with three views:

- **Static ranking** - each submitted agent vs the fixed FSM enemies (the normal
  competition score).
- **Pool ranking** - a student-vs-student round-robin: one student's policy flies
  blue and another's flies red, sides swapped for fairness, ranked by points
  (win=3, draw=1). This shows who actually beats the *other students*, not just
  the scripted bots.
- **Causality** - since every student trains the *same fixed network* and only
  the reward/penalty weights differ, the report correlates each reward term with
  the outcomes (score, mission rate, kill rate, pool points). The Pearson `r`
  table and chart highlight which reward choices actually drove performance.

You can also run it offline over a folder of models:

```bash
python -m bvr.tournament --models-dir path/to/models --out final_output --seeds 2
```

Each `<name>.zip` model may have an optional `<name>.rewards.json` sidecar with
the reward weights used (so the causality analysis has values to correlate).

---

## The environment in one paragraph

A blue fighter must reach a goal area on the far side of a 120x120 km arena
while a red enemy tries to shoot it down. The two aircraft spawn ~88 km apart -
beyond the 60 km radar range - so the agent must navigate and close the distance
before any engagement is possible (no instant shooting). Both carry radar-guided
missiles with a simplified Weapon Engagement Zone; a missile that is
out-maneuvered runs out of energy and misses. **Observation:** own state,
relative geometry to the enemy and goal, WEZ flags, and incoming-missile warning.
**Action:** `[heading_change, flight_level_change, fire]`. Heading and fire drive the 2D fight; flight level is a normalized altitude state the agent can change (visible in observations and replay) but **does not yet affect** radar, missiles, or WEZ in this simplified sim.

The **objective is the mission**: reach the goal area alive. Killing the enemy is
a means to survive, not the win condition. The competition **score** reflects
this: `score = 0.6 x mission_rate + 0.25 x kill_rate + 0.15 x missile_efficiency`, evaluated over many
episodes against every locked enemy.

### The five archetypes (`bvr/enemies.py`)

| Enemy        | Behavior                                              |
| ------------ | ----------------------------------------------------- |
| `duck`       | Flies straight to its goal, never fires (easiest).    |
| `defensive`  | Shoots in range, then cranks/breaks very early - hard to kill. |
| `balanced`   | Pursues, shoots medium range, defends solidly.        |
| `aggressive` | Pushes in hard, shoots long, defends late.            |
| `sniper`     | Fires at max range then breaks early to defend.       |
| `random`     | Random actions (optional, via `random_enemy_prob`).   |

### Reference opponents B1..B10 (optimized FSM)

Hand-tuned archetypes are seeds only. Run the evolutionary search to produce
stronger reference opponents saved in `config/reference_enemies.json`:

```bash
python -m bvr.fsm_optimize --iterations 50 --seeds 3
```

The search:

1. Seeds a population from every archetype-vs-archetype pair (5×5 grid).
2. Runs **50 generations**; each new candidate is scored by fighting the current
   **top-10 elite** pool (round-robin as red vs each elite blue FSM).
3. Fitness (red perspective):  
   `score = win_rate × 8 + ((1 + max_missiles − launched_missiles) / max_missiles) × 2`
4. Saves the best 10 as **B1** (strongest) through **B10**.

Once `reference_enemies.json` exists, **training and locked competition scoring**
automatically use B1..B10 instead of the five archetypes.

### Reward terms (`config/rewards.json`)

Events: `mission_completed`, `hit_enemy`, `was_hit`, `fire_missile`,
`miss_missile`. Shaping (per step): `mission_shaping`, `maintain_track`,
`lost_track`, `closing_bonus`, `wez_advantage`. Plus a `global_scale` multiplier.

## Project layout

```
bvr-rl-activity/
  docs/          STUDENT_GUIDE.md, SYSTEM_AND_CONFIGURATION.md
  config/        rewards.json, scenario.json, reference_enemies.json (B1..B10)
  bvr/           env, enemies, rewards, policy(LOCKED), train, evaluate, analysis, fsm_optimize, tournament, submit
  server/        local single-user dashboard (backend + static UI)
  competition_server/  simple leaderboard server (upload weights)
  online_server/ full online platform: logins, quotas, job queue, leaderboard, admin
```
