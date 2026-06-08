# Student Guide — BVR Reinforcement Learning Lab

Welcome. In this activity you **do not code the AI algorithm**. You tune **rewards and penalties** so a fixed learning agent (PPO) learns to fly a BVR mission: **reach the goal area alive** while facing smart enemies.

Your class uses the **online platform** at `http://<server>:8002/`. Everything below refers to that website unless noted.

---

## 1. What you are trying to achieve

You are the **mission designer**:

- The simulator and neural network are fixed.
- You choose what behavior is **rewarded** or **punished**.
- A good design helps the agent learn quickly **and** score well on the final test.

**Winning the competition** means a high **score**:

```
score = 70% mission success + 30% kill rate
```

- **Mission** = reach the green goal zone **alive** (this is the real objective).
- **Kill** = shooting down the enemy helps survival, but killing alone is not enough.

---

## 2. Getting started

1. Open the class URL (e.g. `http://localhost:8002`).
2. **Register** with your name, password, and the **class access code** from your instructor (default is often `BVR2026`).
3. **Log in** and go to the **Train** tab.

You have a limited number of training runs per hour (shown in the top bar as **quota**). Use them wisely.

---

## 3. The mission (in plain language)

- You control a **blue fighter** in a 120×120 km arena.
- You must fly to a **goal** on the far side.
- A **red enemy** tries to shoot you with radar-guided missiles.
- You start **far apart** (outside radar range) — you must navigate and close before fighting.
- Actions: change heading, change altitude, fire missile.

---

## 4. What you can change (Train tab — left panel)

### Rewards & penalties

These are the **only** knobs you tune. Set any value to **0** to turn that signal off.

| Name | What it does | Tip |
|------|--------------|-----|
| **global_scale** | Multiplies all rewards | Leave at 1.0 unless instructor says otherwise |
| **mission_completed** | Big reward for finishing alive | Should stay strong and positive |
| **hit_enemy** | Reward for killing the enemy | Helps survival; don’t make it bigger than mission |
| **was_hit** | Penalty when you are shot down | Keep **negative** |
| **fire_missile** | Small cost each shot | Stops spamming missiles |
| **miss_missile** | Penalty when a shot misses | Encourages careful shots |
| **mission_shaping** | Small reward each step toward goal | Helps early learning |
| **maintain_track** | Small reward for keeping enemy on radar | Navigation aid |
| **lost_track** | Penalty when radar lock is lost | Usually negative |
| **closing_bonus** | Reward for closing distance | Optional aggression hint |
| **wez_advantage** | Reward when you are in range but enemy is not | Optional tactical hint |

**Common mistakes:**

- Rewarding kills more than the mission → agent fights but never completes the mission.
- No shaping terms → learning is very slow at the start.
- No missile cost → agent fires constantly and wastes shots.

### Reference opponents (B1–B10)

Checkboxes for **B1 … B10** (B1 is strongest). Leave all selected unless your instructor says otherwise. Training cycles through them (**round robin**) so you learn against everyone.

### Training steps

How long each run trains ( capped by instructor ). More steps usually helps, but costs one quota slot.

Click **Queue training run** to start.

---

## 5. What you see while training (Train tab — right panel)

### Live run metrics

| Metric | Meaning |
|--------|---------|
| **progress** | How much of your training budget is done |
| **avg score** | Average eval score so far (70% mission + 30% kill) |
| **train reward** | Mean reward the agent receives while learning (can be negative early) |
| **state** | `queued`, `running`, `evaluating`, `done`, etc. |
| **eval runs** | How many monitoring eval flights finished (e.g. 7/10) |

### Live eval

While training, the system runs **short test flights** against each enemy (monitoring only — does not change learning). You see:

- **Average score** headline (focus on this number).
- **Mini replays** — small animations as each test finishes.
- Outcome tags: `mission`, `shot_down`, or `timeout`.

### Learning curve

Graph with two lines:

- **Blue — train reward** (left axis): what the agent optimizes step-by-step.
- **Green — avg score** (right axis, 0–100%): how well it performs on test flights over time.

If green goes up over time, your reward design is working.

### Run complete panel

When training finishes, a new section appears:

- **Run analysis** — detailed breakdown + charts (same idea as the instructor’s analysis tool).
- **Watch replay** — pick an enemy and watch one full deterministic flight.

---

## 6. After training — My runs tab

Every queued run is listed here.

| Column | Meaning |
|--------|---------|
| **score** | Official score (12 test episodes per enemy) — **this is what counts** |
| **mission** | % of episodes where you reached the goal alive |
| **kill** | % of episodes with a kill |
| **reward** | Mean environment reward during eval |

**Actions:**

- **submit** — send this run to the **Leaderboard** (only **one** submission counts; pick your best).
- **review** — open analysis + replay for that run.
- **stop** — cancel a run still queued or running.

---

## 7. Leaderboard tab

Shows class rankings by **official score** vs the fixed opponents B1–B10. Submit your best run before the deadline.

---

## 8. How to work effectively

1. **Start from defaults** — run once without changes to see baseline behavior.
2. **Change one thing at a time** — e.g. increase `mission_shaping`, then run again.
3. **Watch the learning curve** — flat green line = reward design may not be helping.
4. **Use analysis after a run** — check which enemies you fail against.
5. **Use replay** — see *why* you lost ( flew wrong direction? fired too early? ).
6. **Submit early, improve later** — you can submit once; update strategy on remaining quota runs.

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **BVR** | Beyond Visual Range — fight with radar/missiles before visual contact |
| **PPO** | The fixed learning algorithm (you don’t configure it) |
| **Shaping** | Small per-step rewards that guide learning |
| **Event reward** | Large reward/penalty at mission milestones |
| **Live eval** | Quick tests during training (1 flight per enemy) |
| **Locked eval** | Official scoring after training (12 flights per enemy) |
| **WEZ** | Weapon Engagement Zone — missile effective range |
| **Quota** | Limit on how many runs you can start per time window |

---

## 10. What you must NOT do

- Do not edit Python source files unless your instructor explicitly allows it.
- Do not share accounts (quota is per student).
- Do not expect **train reward** and **score** to match — they measure different things.

---

## 11. Local install (optional)

If your instructor also provides the local dashboard (`:8000`), it is the same activity with extra tools for solo experimentation. Class work should still go through `:8002` so scores and submissions are recorded.

---

## 12. Need help?

| Problem | What to try |
|---------|-------------|
| Can’t register | Check class access code; ask if registration is closed |
| Quota exhausted | Wait for the time window to reset |
| Score is 0% | Review replay; increase mission shaping; check `mission_completed` weight |
| Training stuck on queued | Server busy; wait or ask instructor to raise `max_concurrent` |
| Page looks broken | Hard refresh: **Ctrl+F5** |

For system setup and admin settings, your instructor should refer to [SYSTEM_AND_CONFIGURATION.md](SYSTEM_AND_CONFIGURATION.md).
