"use strict";
const $ = (id) => document.getElementById(id);
let ME = null;
let watchedRun = null;
const history = { t: [], reward: [], mission: [] };
let evalLiveEpisodes = [];
let matchEpisodes = [];
let matchIdx = 0;
let lastDoneRunId = null;
let detailRunId = null;
let replayRunId = null;
let replayInfoEl = null;
let replayCanvasId = null;
const replayBuffer = [];
let replayIdx = 0;

// ---------------------------------------------------------------- auth
function showAuth(which) {
  $("login-form").classList.toggle("active", which === "login");
  $("register-form").classList.toggle("active", which === "register");
  $("tab-login").classList.toggle("primary", which === "login");
  $("tab-register").classList.toggle("primary", which === "register");
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}) });
  return { ok: r.ok, status: r.status, data: await r.json().catch(() => ({})) };
}
async function doLogin(e) {
  e.preventDefault();
  const res = await postJSON("/api/login", { name: $("login-name").value, password: $("login-pw").value });
  if (!res.data.ok) { $("login-err").textContent = res.data.error || "Login failed"; return false; }
  await enterApp();
  return false;
}
async function doRegister(e) {
  e.preventDefault();
  const res = await postJSON("/api/register", { name: $("reg-name").value, password: $("reg-pw").value,
    access_code: $("reg-code").value });
  if (!res.data.ok) { $("reg-err").textContent = res.data.error || "Registration failed"; return false; }
  await enterApp();
  return false;
}
async function doLogout() { await postJSON("/api/logout"); location.reload(); }

async function checkAuth() {
  ME = await (await fetch("/api/me")).json();
  if (ME.authenticated) await enterApp(); else { $("auth-view").classList.remove("hidden"); $("app-view").classList.add("hidden"); }
}
async function enterApp() {
  const [meRes, catalog] = await Promise.all([fetch("/api/me"), fetchEnemyCatalog()]);
  ME = await meRes.json();
  if (!ME.authenticated) return;
  ME.enemy_types = catalog.names;
  ME.enemy_catalog = catalog;
  $("auth-view").classList.add("hidden");
  $("app-view").classList.remove("hidden");
  $("whoami").textContent = ME.name;
  if (ME.is_admin) $("admin-tab").classList.remove("hidden");
  renderRewardEditor();
  renderEnemies(catalog);
  $("steps").value = ME.quota.steps_per_run;
  $("steps-hint").textContent = `max ${ME.quota.steps_per_run} steps/run`;
  updateQuota(ME.quota);
  connectWS();
  loadRuns();
}

function updateQuota(q) {
  const pill = $("quota");
  pill.textContent = `${q.remaining}/${q.per_window} runs left (per ${q.window_hours}h)`;
  pill.classList.toggle("low", q.remaining <= 0);
  $("train-btn").disabled = q.remaining <= 0;
}

// ---------------------------------------------------------------- views
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("view-" + name).classList.add("active");
  document.querySelectorAll(".navtabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "runs") loadRuns();
  if (name === "board") loadBoard();
  if (name === "admin") { loadAdmin(); loadFinal(); }
}

// ---------------------------------------------------------------- reward editor
function renderRewardEditor() {
  $("global_scale").value = ME.defaults.global_scale;
  renderTerms("event-terms", ME.reward_terms.event);
  renderTerms("shaping-terms", ME.reward_terms.shaping);
}
function renderTerms(containerId, terms) {
  const c = $(containerId); c.innerHTML = "";
  terms.forEach((term) => {
    const val = ME.defaults[term];
    const row = document.createElement("div");
    row.className = "term-row" + (Number(val) === 0 ? " disabled" : "");
    row.innerHTML = `<label>${term}</label><input type="number" step="0.1" data-term="${term}" value="${val}" /><button class="zero">0</button>`;
    const input = row.querySelector("input");
    input.addEventListener("input", () => row.classList.toggle("disabled", Number(input.value) === 0));
    row.querySelector(".zero").addEventListener("click", () => { input.value = 0; row.classList.add("disabled"); });
    c.appendChild(row);
  });
}
function enemyLabel(name, catalog) {
  const cat = catalog || (ME && ME.enemy_catalog) || {};
  const info = (cat.info && cat.info[name]) || {};
  return info.label || name;
}
async function fetchEnemyCatalog() {
  const res = await fetch("/api/enemies");
  if (!res.ok) throw new Error("Could not load enemy list");
  return res.json();
}
function renderEnemies(catalog) {
  const cat = catalog || ME.enemy_catalog || {};
  ME.enemy_types = cat.names || ME.enemy_types || [];
  ME.enemy_catalog = cat;
  const list = $("enemy-list"); list.innerHTML = "";
  const hint = $("enemy-hint");
  if (hint) {
    hint.textContent = (cat.mode === "reference")
      ? "Optimized FSM references B1 (strongest) through B10. All selected by default."
      : "Pick which hand-coded opponents to train against.";
  }
  if (!ME.enemy_types.length) {
    list.innerHTML = "<p class='hint'>No opponents loaded. Restart the server after running fsm_optimize.</p>";
    return;
  }
  ME.enemy_types.forEach((e) => {
    const w = document.createElement("label");
    w.className = "enemy-item";
    w.innerHTML = `<input type="checkbox" value="${e}" checked/> ${enemyLabel(e, cat)}`;
    list.appendChild(w);
  });
}
function gatherRewards() {
  const rewards = { global_scale: Number($("global_scale").value) };
  document.querySelectorAll("[data-term]").forEach((i) => (rewards[i.dataset.term] = Number(i.value)));
  return rewards;
}

// ---------------------------------------------------------------- training
async function queueRun() {
  const enemies = Array.from(document.querySelectorAll("#enemy-list input:checked")).map((c) => c.value);
  const res = await postJSON("/api/train", { rewards: gatherRewards(), enemies, steps: Number($("steps").value) });
  if (!res.data.ok) { alert(res.data.error || "Could not start"); return; }
  watchedRun = res.data.run_id;
  history.t.length = 0; history.reward.length = 0; history.mission.length = 0;
  $("post-run-panel")?.classList.add("hidden");
  $("live-run").textContent = `#${res.data.run_id} (queued)`;
  $("m-state").textContent = "queued";
  updateQuota(res.data.quota);
  loadRuns();
}

// ---------------------------------------------------------------- websocket
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connectWS, 1500);
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
}
function avgEvalScore(eps) {
  const ok = (eps || []).filter(Boolean);
  if (!ok.length) return null;
  return ok.reduce((a, e) => a + (e.score ?? 0), 0) / ok.length;
}
function fmtScore(v) {
  return v == null ? "-" : (v * 100).toFixed(0) + "%";
}
function showEvalProgress(ev) {
  const box = $("eval-progress");
  if (!box) return;
  box.classList.remove("hidden");
  if (ev.state === "starting") {
    evalLiveEpisodes = [];
    matchEpisodes = [];
    matchIdx = 0;
    if ($("match-grid")) $("match-grid").innerHTML = "";
    if ($("eval-overall")) $("eval-overall").textContent = "Running eval pass…";
  }
  const finished = ev.finished ?? 0;
  const total = ev.total ?? 0;
  const pct = total ? Math.round((finished / total) * 100) : 0;
  $("eval-count").textContent = `${finished}/${total} runs`;
  $("eval-progress-bar").style.width = pct + "%";
  $("m-eval-sims").textContent = `${finished}/${total}`;
  $("eval-status").textContent = ev.state === "starting"
    ? "Starting eval pass…"
    : `Evaluating ${finished}/${total} runs`;
  if (ev.last_result) {
    $("eval-detail").textContent = `Last result: ${ev.last_result}`;
  } else if (total) {
    $("eval-detail").textContent = `Run ${Math.min(finished + 1, total)}/${total} in progress…`;
  }
  if (ev.enemy_done && ev.enemy_summary) {
    const idx = (ev.enemy_index || 1) - 1;
    evalLiveEpisodes[idx] = ev.enemy_summary;
    matchEpisodes[idx] = ev.enemy_summary;
    renderEvalCard(ev.enemy_summary, idx);
    const avg = avgEvalScore(evalLiveEpisodes);
    const done = evalLiveEpisodes.filter(Boolean).length;
    if (avg != null) {
      $("m-mission").textContent = fmtScore(avg);
      if ($("eval-overall")) {
        $("eval-overall").textContent =
          `Average score: ${fmtScore(avg)} (${done}/${ev.enemy_total || ev.total} runs complete)`;
      }
    }
  }
}
function renderEvalCard(ep, i) {
  const grid = $("match-grid");
  if (!grid) return;
  let wrap = document.getElementById("match-" + i);
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "match";
    wrap.id = "match-" + i;
    wrap.innerHTML = `<div class="cap"><b>${ep.enemy}</b><span class="tag"></span></div>
      <div class="meta hint"></div><canvas width="170" height="170" id="mc${i}"></canvas>`;
    grid.appendChild(wrap);
  }
  const tag = wrap.querySelector(".tag");
  const meta = wrap.querySelector(".meta");
  const tc = ep.result === "mission" ? "win" : (ep.result === "shot_down" ? "lost" : "timeout");
  tag.className = "tag " + tc;
  tag.textContent = ep.result;
  meta.textContent = `${Math.round(ep.steps || 0)} steps`;
}
function buildMatchGrid(episodes) {
  const grid = $("match-grid");
  if (!grid) return;
  grid.innerHTML = "";
  episodes.forEach((ep, i) => renderEvalCard(ep, i));
}
function hideEvalProgress() {
  const box = $("eval-progress");
  if (box) box.classList.add("hidden");
}
function applyEvalSummary(ev) {
  const s = ev.eval_summary;
  if (!s) return;
  const score = s.score != null ? s.score : 0;
  $("m-mission").textContent = fmtScore(score);
  $("m-eval-sims").textContent = `${s.finished}/${s.total_simulations}`;
  if ($("eval-overall")) {
    $("eval-overall").textContent =
      `Average score: ${fmtScore(score)} (0.7×mission + 0.3×kill over ${s.total_simulations} runs) · monitoring only`;
  }
  hideEvalProgress();
}
function handleEvent(ev) {
  if (ev.type === "replay_start" || ev.type === "replay_frame" || ev.type === "replay_end") {
    if (ev.run_id !== replayRunId) return;
    if (ev.type === "replay_start") {
      replayBuffer.length = 0;
      replayIdx = 0;
      if (replayInfoEl) replayInfoEl.textContent = "vs " + ev.enemy;
    } else if (ev.type === "replay_frame") {
      replayBuffer.push(ev.frame);
    } else if (ev.type === "replay_end") {
      const r = ev.result || {};
      if (replayInfoEl) replayInfoEl.textContent = `result: ${r.result || "?"} (reward ${r.reward ?? "-"})`;
      replayRunId = null;
      document.querySelectorAll("#post-replay-stop, #detail-replay-stop").forEach((b) => b.classList.add("hidden"));
    }
    return;
  }
  if (watchedRun === null && ev.run_id) watchedRun = ev.run_id;
  if (ev.run_id && ev.run_id !== watchedRun) return;
  if (ev.type === "status") {
    $("m-state").textContent = ev.state;
    $("live-run").textContent = `#${ev.run_id} (${ev.state})`;
    if (ev.state === "error") alert("Run failed: " + (ev.message || ""));
    if (["done", "error", "stopped", "evaluating"].includes(ev.state)) loadRuns();
  } else if (ev.type === "eval_progress") {
    $("m-state").textContent = "evaluating";
    showEvalProgress(ev);
    if (ev.timesteps) { /* keep run id visible */ }
    if (ev.progress != null) {
      $("m-progress").textContent = Math.round(ev.progress * 100) + "%";
      $("progress-bar").style.width = ev.progress * 100 + "%";
    }
    if (ev.ep_rew_mean != null) $("m-reward").textContent = ev.ep_rew_mean;
  } else if (ev.type === "update") {
    $("m-progress").textContent = Math.round(ev.progress * 100) + "%";
    $("progress-bar").style.width = ev.progress * 100 + "%";
    const score = ev.eval_score != null ? ev.eval_score : ev.winrate;
    $("m-mission").textContent = fmtScore(score);
    $("m-reward").textContent = ev.ep_rew_mean;
    if (ev.eval_ran === false) hideEvalProgress();
    else applyEvalSummary(ev);
    history.t.push(ev.timesteps); history.reward.push(ev.ep_rew_mean);
    history.mission.push(ev.eval_score != null ? ev.eval_score : ev.winrate);
    if (ev.episodes && ev.episodes.length) {
      matchEpisodes = ev.episodes;
      matchIdx = 0;
      buildMatchGrid(matchEpisodes);
    }
    drawCurve();
  } else if (ev.type === "done") {
    $("m-state").textContent = "done";
    showPostRunPanel(ev.run_id);
    loadRuns();
  } else if (ev.type === "final") {
    if (ev.state === "running") $("final-progress").textContent = "running... duels " + (ev.progress || "");
    else if (ev.state === "done") { $("final-progress").textContent = "done"; loadFinal(); }
    else if (ev.state === "error") $("final-progress").textContent = "error: " + (ev.message || "");
  }
}

// ---------------------------------------------------------------- my runs
async function loadRuns() {
  const data = await (await fetch("/api/runs")).json();
  if (data.quota) updateQuota(data.quota);
  const rows = (data.runs || []).map((r) => {
    const sc = r.score != null ? (r.score * 100).toFixed(1) + "%" : "-";
    const miss = r.mission_rate != null ? (r.mission_rate * 100).toFixed(0) + "%" : "-";
    const kill = r.kill_rate != null ? (r.kill_rate * 100).toFixed(0) + "%" : "-";
    const canSubmit = r.status === "done" && r.score != null;
    const submitBtn = canSubmit
      ? `<button class="${r.submitted ? "primary" : ""}" onclick="submitRun(${r.id})">${r.submitted ? "submitted" : "submit"}</button>`
      : "-";
    const stopBtn = (r.status === "running" || r.status === "queued")
      ? `<button class="danger btn-sm" onclick="stopRun(${r.id})">stop</button>` : "";
    const reviewBtn = r.status === "done"
      ? `<button class="btn-sm" onclick="openRunDetail(${r.id})">review</button>` : "";
    return `<tr><td>#${r.id}</td><td><span class="status-badge s-${r.status}">${r.status}</span></td>
      <td>${r.steps}</td><td class="score">${sc}</td><td>${miss}</td><td>${kill}</td>
      <td>${r.mean_reward ?? "-"}</td><td>${submitBtn} ${reviewBtn} ${stopBtn}</td></tr>`;
  }).join("");
  $("runs-table").innerHTML = `<table><tr><th>run</th><th>status</th><th>steps</th><th>score</th>
    <th>mission</th><th>kill</th><th>reward</th><th>actions</th></tr>${rows || "<tr><td colspan='8'>No runs yet.</td></tr>"}</table>`;
}
async function submitRun(id) { await postJSON(`/api/run/${id}/submit`); loadRuns(); }
async function stopRun(id) { await postJSON(`/api/run/${id}/stop`); loadRuns(); }

function populateReplaySelect(selectId) {
  const sel = $(selectId);
  if (!sel) return;
  sel.innerHTML = "";
  (ME.enemy_types || []).forEach((e) => {
    const opt = document.createElement("option");
    opt.value = e;
    opt.textContent = enemyLabel(e);
    sel.appendChild(opt);
  });
}

function showPostRunPanel(runId) {
  lastDoneRunId = runId;
  const panel = $("post-run-panel");
  if (!panel) return;
  panel.classList.remove("hidden");
  $("post-run-label").textContent = `#${runId}`;
  $("post-analysis-out").innerHTML = "";
  populateReplaySelect("post-replay-enemy");
}

function openRunDetail(runId) {
  detailRunId = runId;
  showView("runs");
  $("run-detail").classList.remove("hidden");
  $("run-detail-label").textContent = `#${runId}`;
  $("detail-analysis-out").innerHTML = "";
  populateReplaySelect("detail-replay-enemy");
}

function renderAnalysisOut(stats, plots, outId) {
  const out = $(outId);
  if (!out) return;
  let html = `<p>Score <b>${(stats.score * 100).toFixed(0)}%</b> · mission <b>${(stats.mission_rate * 100).toFixed(0)}%</b> · kill <b>${(stats.kill_rate * 100).toFixed(0)}%</b> · survive <b>${(stats.survival_rate * 100).toFixed(0)}%</b> · mean reward <b>${stats.mean_reward}</b> · missile eff. <b>${stats.missile_efficiency}</b></p>`;
  html += "<table class='analysis-table'><tr><th>enemy</th><th>mission</th><th>survive</th><th>kill</th><th>reward</th><th>missiles</th></tr>";
  for (const [enemy, e] of Object.entries(stats.per_enemy || {})) {
    html += `<tr><td>${enemy}</td><td>${(e.mission_rate * 100).toFixed(0)}%</td><td>${(e.survival * 100).toFixed(0)}%</td><td>${(e.kills * 100).toFixed(0)}%</td><td>${e.mean_reward}</td><td>${e.missiles_used}</td></tr>`;
  }
  html += "</table><div>";
  for (const path of Object.values(plots || {})) html += `<img src="${path}?t=${Date.now()}"/>`;
  html += "</div>";
  out.innerHTML = html;
}

async function runAnalysis(runId, outId) {
  const out = $(outId);
  if (out) out.innerHTML = "<p class='hint'>Running analysis (10 episodes per enemy)…</p>";
  const res = await postJSON(`/api/run/${runId}/analysis`);
  if (!res.data.ok) {
    if (out) out.innerHTML = `<p class="hint">${res.data.error || "Analysis failed."}</p>`;
    return;
  }
  renderAnalysisOut(res.data.stats, res.data.plots, outId);
}

async function startRunReplay(runId, enemySelectId, infoId, canvasId, stopBtnId) {
  const enemy = $(enemySelectId)?.value || "B1";
  replayRunId = runId;
  replayInfoEl = $(infoId);
  replayCanvasId = canvasId;
  replayBuffer.length = 0;
  replayIdx = 0;
  if (replayInfoEl) replayInfoEl.textContent = "loading…";
  const stopBtn = $(stopBtnId);
  if (stopBtn) stopBtn.classList.remove("hidden");
  const res = await postJSON(`/api/run/${runId}/replay/start`, { enemy });
  if (!res.data.ok) {
    replayRunId = null;
    if (replayInfoEl) replayInfoEl.textContent = res.data.error || "Could not start replay.";
    if (stopBtn) stopBtn.classList.add("hidden");
  }
}

async function stopRunReplay(stopBtnId) {
  if (detailRunId) await postJSON(`/api/run/${detailRunId}/replay/stop`);
  else if (lastDoneRunId) await postJSON(`/api/run/${lastDoneRunId}/replay/stop`);
  replayRunId = null;
  replayBuffer.length = 0;
  const stopBtn = $(stopBtnId);
  if (stopBtn) stopBtn.classList.add("hidden");
}

// ---------------------------------------------------------------- leaderboard
async function loadBoard() {
  const data = await (await fetch("/api/leaderboard")).json();
  const rows = (data.leaderboard || []).map((b, i) =>
    `<tr><td>${i + 1}</td><td class="name">${b.name}</td><td class="score">${(b.score * 100).toFixed(1)}%</td>
     <td>${(b.mission_rate * 100).toFixed(0)}%</td><td>${(b.kill_rate * 100).toFixed(0)}%</td>
     <td>${(b.survival_rate * 100).toFixed(0)}%</td><td>${b.mean_reward}</td></tr>`).join("");
  const refs = (data.enemies || []).join(", ");
  $("board-table").innerHTML = `<p class="hint">Locked opponents: ${refs || "B1..B10"}</p>
    <table><tr><th>#</th><th>name</th><th>score</th><th>mission</th><th>kill</th>
    <th>survive</th><th>reward</th></tr>${rows || "<tr><td colspan='7'>No submissions yet.</td></tr>"}</table>`;
}

// ---------------------------------------------------------------- admin
async function loadAdmin() {
  const data = await (await fetch("/api/admin/config")).json();
  if (!data.config) return;
  for (const [k, v] of Object.entries(data.config)) { const el = $("cfg-" + k); if (el) el.value = v; }
  const rows = (data.runs || []).map((r) =>
    `<tr><td>#${r.id}</td><td class="name">${r.user_name}</td><td><span class="status-badge s-${r.status}">${r.status}</span></td>
     <td>${r.steps}</td><td class="score">${r.score != null ? (r.score * 100).toFixed(1) + "%" : "-"}</td>
     <td>${r.submitted ? "yes" : ""}</td></tr>`).join("");
  $("admin-runs").innerHTML = `<table><tr><th>run</th><th>user</th><th>status</th><th>steps</th><th>score</th><th>submitted</th></tr>${rows}</table>`;
}
async function saveAdmin() {
  const keys = ["class_access_code", "runs_per_window", "steps_per_run", "window_hours", "max_concurrent", "registration_open"];
  const body = {}; keys.forEach((k) => (body[k] = $("cfg-" + k).value));
  await postJSON("/api/admin/config", body);
  alert("Saved.");
}
async function runFinal() {
  const res = await postJSON("/api/admin/final/start");
  if (!res.data.ok) { alert(res.data.error || "Could not start"); return; }
  $("final-progress").textContent = `running over ${res.data.n_students} students...`;
}
async function loadFinal() {
  const d = await (await fetch("/api/admin/final")).json();
  if (d.running) $("final-progress").textContent = "running... " + (d.progress || "");
  if (!d.report) return;
  const r = d.report;
  let h = `<h3>Static ranking (vs FSM enemies)</h3><table><tr><th>#</th><th>name</th><th>score</th><th>mission</th><th>kill</th></tr>`;
  r.static_ranking.forEach((e, i) => h += `<tr><td>${i+1}</td><td class="name">${e.name}</td><td class="score">${(e.score*100).toFixed(1)}%</td><td>${(e.mission_rate*100).toFixed(0)}%</td><td>${(e.kill_rate*100).toFixed(0)}%</td></tr>`);
  h += "</table>";
  if (r.pool && r.pool.ranking && r.pool.ranking.length) {
    h += `<h3>Pool ranking (student vs student)</h3><table><tr><th>#</th><th>name</th><th>points</th><th>W-D-L</th><th>winrate</th></tr>`;
    r.pool.ranking.forEach((n, i) => { const s = r.pool.standings[n]; h += `<tr><td>${i+1}</td><td class="name">${n}</td><td class="score">${s.points}</td><td>${s.wins}-${s.draws}-${s.losses}</td><td>${(s.winrate*100).toFixed(0)}%</td></tr>`; });
    h += "</table>";
  }
  if (r.causality && Object.keys(r.causality).length) {
    h += `<h3>Causality: reward choice vs results (Pearson r)</h3><table><tr><th>term</th><th>vs score</th><th>vs mission</th><th>vs kill</th><th>vs pool</th></tr>`;
    for (const [term, v] of Object.entries(r.causality)) {
      if (!v) { h += `<tr><td>${term}</td><td colspan="4" class="hint">(all students used the same value)</td></tr>`; continue; }
      const f = (x) => x == null ? "-" : x.toFixed(2);
      h += `<tr><td>${term}</td><td>${f(v.corr_score)}</td><td>${f(v.corr_mission)}</td><td>${f(v.corr_kill)}</td><td>${f(v.corr_pool)}</td></tr>`;
    }
    h += "</table><p class='hint'>Only reward weights matter (the network is fixed), so these associations point to which reward choices actually drove behavior. |r| near 1 = strong.</p>";
  }
  h += "<div>";
  for (const [k, p] of Object.entries(d.plots || {})) h += `<img src="${p}?t=${Date.now()}" style="max-width:420px;border:1px solid var(--line);border-radius:8px;margin:8px 8px 0 0"/>`;
  h += "</div>";
  $("final-out").innerHTML = h;
}

// ---------------------------------------------------------------- canvas drawing
function w2c(x, y, arena, w, h) { return [x / arena * w, h - y / arena * h]; }
function drawFrame(ctx, frame, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1322"; ctx.fillRect(0, 0, w, h);
  const arena = frame.arena || 120;
  ctx.strokeStyle = "#1b2438"; ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    ctx.beginPath(); ctx.moveTo(i / 5 * w, 0); ctx.lineTo(i / 5 * w, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i / 5 * h); ctx.lineTo(w, i / 5 * h); ctx.stroke();
  }
  const b = frame.blue, rd = frame.red;
  if (b && b.goal) {
    const [gx, gy] = w2c(b.goal[0], b.goal[1], arena, w, h);
    ctx.strokeStyle = "#5ec27a"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(gx, gy, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx - 9, gy); ctx.lineTo(gx + 9, gy); ctx.moveTo(gx, gy - 9); ctx.lineTo(gx, gy + 9); ctx.stroke();
  }
  if (b && rd && rd.tracked && rd.alive && b.alive) {
    const [bx, by] = w2c(b.x, b.y, arena, w, h); const [rx, ry] = w2c(rd.x, rd.y, arena, w, h);
    ctx.strokeStyle = "rgba(232,98,76,0.35)"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(rx, ry); ctx.stroke(); ctx.setLineDash([]);
  }
  (frame.missiles || []).forEach((m) => {
    const [mx, my] = w2c(m.x, m.y, arena, w, h);
    ctx.fillStyle = m.owner === "blue" ? "#9fd0ff" : "#ffb3a8";
    ctx.beginPath(); ctx.arc(mx, my, 2.2, 0, 2 * Math.PI); ctx.fill();
  });
  if (b) drawAircraft(ctx, b, "#4c9be8", arena, w, h);
  if (rd) drawAircraft(ctx, rd, "#e8624c", arena, w, h);
}
function drawAircraft(ctx, ac, color, arena, w, h) {
  const [cx, cy] = w2c(ac.x, ac.y, arena, w, h);
  if (!ac.alive) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx - 5, cy - 5); ctx.lineTo(cx + 5, cy + 5); ctx.moveTo(cx + 5, cy - 5); ctx.lineTo(cx - 5, cy + 5); ctx.stroke();
    return;
  }
  const a = -ac.hdg, s = 7;
  const nose = [cx + Math.cos(a) * s, cy + Math.sin(a) * s];
  const left = [cx + Math.cos(a + 2.5) * s * 0.8, cy + Math.sin(a + 2.5) * s * 0.8];
  const right = [cx + Math.cos(a - 2.5) * s * 0.8, cy + Math.sin(a - 2.5) * s * 0.8];
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.moveTo(...nose); ctx.lineTo(...left); ctx.lineTo(...right); ctx.closePath(); ctx.fill();
}
function tickVisuals() {
  if (matchEpisodes.length) {
    const withFrames = matchEpisodes.filter((e) => e && e.frames && e.frames.length);
    if (withFrames.length) {
      const maxLen = Math.max(...withFrames.map((e) => e.frames.length));
      matchEpisodes.forEach((ep, i) => {
        const cv = $("mc" + i);
        if (!cv || !ep || !ep.frames || !ep.frames.length) return;
        const f = ep.frames[Math.min(matchIdx, ep.frames.length - 1)];
        drawFrame(cv.getContext("2d"), f, cv.width, cv.height);
      });
      matchIdx = (matchIdx + 1) % Math.max(maxLen, 1);
    }
  }
  if (replayBuffer.length && replayCanvasId) {
    const cv = $(replayCanvasId);
    if (cv && replayIdx < replayBuffer.length) {
      drawFrame(cv.getContext("2d"), replayBuffer[replayIdx], cv.width, cv.height);
      replayIdx++;
    }
  }
  setTimeout(tickVisuals, 95);
}
function drawCurve() {
  drawLearningCurve("curve", history);
}

// ----------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  checkAuth();
  tickVisuals();
  $("train-btn").onclick = queueRun;
  $("admin-save").onclick = saveAdmin;
  $("final-btn").onclick = runFinal;
  $("post-analysis-btn").onclick = () => lastDoneRunId && runAnalysis(lastDoneRunId, "post-analysis-out");
  $("post-replay-btn").onclick = () => lastDoneRunId && startRunReplay(lastDoneRunId, "post-replay-enemy", "post-replay-info", "post-replay", "post-replay-stop");
  $("post-replay-stop").onclick = () => stopRunReplay("post-replay-stop");
  $("detail-analysis-btn").onclick = () => detailRunId && runAnalysis(detailRunId, "detail-analysis-out");
  $("detail-replay-btn").onclick = () => detailRunId && startRunReplay(detailRunId, "detail-replay-enemy", "detail-replay-info", "detail-replay", "detail-replay-stop");
  $("detail-replay-stop").onclick = () => stopRunReplay("detail-replay-stop");
  setInterval(() => { if (ME && ME.authenticated && $("view-board").classList.contains("active")) loadBoard(); }, 15000);
});
