"use strict";
const $ = (id) => document.getElementById(id);
let ME = null;
let watchedRun = null;
const history = { t: [], reward: [], mission: [], score: [] };
const SCORE_FORMULA = "0.6×mission + 0.25×kill + 0.15×missile eff.";
let evalLiveEpisodes = [];
let evalEpochs = [];
let selectedEpochIdx = -1;
let liveEpochId = null;
let matchEpisodes = [];
const analysisReplayCache = {};  // prefix -> { enemies, replays }
let liveReplayPlaying = null;    // { cardIdx, idx, hold, canvasId, frames }
let analysisFrameReplay = null;  // { prefix, idx, hold, canvasId, frames, enemy }
const MATCH_HOLD_FRAMES = 55;
let lastDoneRunId = null;
let detailRunId = null;
let adminRewardEditor = null;

// ---------------------------------------------------------------- auth
function showAuth(which) {
  $("login-form").classList.toggle("active", which === "login");
  $("register-form").classList.toggle("active", which === "register");
  $("tab-login").classList.toggle("primary", which === "login");
  $("tab-register").classList.toggle("primary", which === "register");
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify(body ?? {}),
  });
  return { ok: r.ok, status: r.status, data: await r.json().catch(() => ({})) };
}
function apiError(res, data) {
  if (res.status === 404 && !data.error) {
    return "Endpoint not found — restart the server: python online_server/main.py";
  }
  return data.error || `Request failed (HTTP ${res.status})`;
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
  applyStepsConfig();
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
  if (name === "report") loadMyReport();
  if (name === "board") loadBoard();
  if (name === "admin") { loadAdmin(); loadFinal(); loadUsers(); }
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------- reward editor
function renderRewardEditor() {
  const ed = RewardEditor.renderAll(ME.reward_editor, ME.reward_editor && ME.reward_editor.defaults);
  if (!ed) return;
  const hint = $("reward-editor-hint");
  if (hint) {
    hint.innerHTML = ed.start_zero
      ? "All weights start at <b>0</b>. Use sliders below — <b>+</b> reward, <b>−</b> penalty."
      : "Slider or type a value. <b>+</b> = reward, <b>−</b> = penalty, <b>0</b> = off. Hover a name for help.";
  }
}
function renderTerms(containerId, terms) {
  /* legacy hook — use renderRewardEditor */
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
  const hint = $("enemy-hint");
  if (hint) {
    hint.textContent = (cat.mode === "reference")
      ? "Click bubbles to toggle. Pool = 5 varied optimized agents (B#) + fixed archetypes. Locked scoring uses ALL agents."
      : "Click bubbles to toggle opponents. Position = offense vs defense style.";
  }
  if (!ME.enemy_types.length) {
    const box = $("enemy-picker");
    if (box) box.innerHTML = "<p class='hint'>No opponents loaded. Run fsm_optimize and restart the server.</p>";
    return;
  }
  EnemyPicker.render("enemy-picker", cat, { selected: cat.names });
}
function gatherSelectedEnemies() {
  return EnemyPicker.getSelected();
}
function gatherRewards() {
  return RewardEditor.gather();
}
function applyStepsConfig(quota) {
  const q = quota || (ME && ME.quota);
  const stepsEl = $("steps");
  const hint = $("steps-hint");
  if (!stepsEl || !q) return;
  const max = Number(q.steps_per_run) || 200000;
  stepsEl.value = max;
  stepsEl.min = 2000;
  stepsEl.max = max;
  if (q.steps_editable) {
    stepsEl.readOnly = false;
    stepsEl.disabled = false;
    if (hint) hint.textContent = `Training steps (instructor max: ${max.toLocaleString()})`;
  } else {
    stepsEl.readOnly = true;
    stepsEl.disabled = true;
    if (hint) hint.textContent = `Training steps set by instructor: ${max.toLocaleString()}`;
  }
}
function gatherTrainingSteps() {
  const q = ME && ME.quota;
  if (!q) return Number($("steps")?.value) || 200000;
  if (!q.steps_editable) return q.steps_per_run;
  const max = q.steps_per_run;
  return Math.min(Math.max(2000, Number($("steps")?.value) || max), max);
}

// ---------------------------------------------------------------- training
async function queueRun() {
  const enemies = gatherSelectedEnemies();
  if (!enemies.length) { alert("Select at least one opponent on the map."); return; }
  const weights = EnemyPicker.getWeights ? EnemyPicker.getWeights() : {};
  const active = enemies.filter((e) => (weights[e] ?? 1) > 0);
  if (!active.length) { alert("All opponents have priority 0. Raise at least one above 0."); return; }
  const steps = gatherTrainingSteps();
  const lockedEps = ME?.quota?.locked_eval_episodes_per_enemy ?? 30;
  const oppLine = active.map((e) => `${e} (${Number(weights[e] ?? 1).toFixed(2)})`).join(", ");
  const msg = [
    "Start a training run with these settings?",
    "",
    `Steps: ${steps}`,
    `Opponents — priority (${active.length}): ${oppLine}`,
    `Final locked score: ${lockedEps} episodes vs every locked opponent after training.`,
    "",
    "Continue?"
  ].join("\n");
  if (!confirm(msg)) return;
  const res = await postJSON("/api/train", { rewards: gatherRewards(), enemies, enemy_weights: weights, steps });
  if (!res.data.ok) { alert(res.data.error || "Could not start"); return; }
  watchedRun = res.data.run_id;
  history.t.length = 0; history.reward.length = 0; history.mission.length = 0; history.score.length = 0;
  resetEvalEpochs();
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

function resetEvalEpochs() {
  evalEpochs = [];
  selectedEpochIdx = -1;
  liveEpochId = null;
  evalLiveEpisodes = [];
  matchEpisodes = [];
  stopLiveReplay();
  renderEpochTabs();
  if ($("match-grid")) $("match-grid").innerHTML = "";
  if ($("eval-overall")) $("eval-overall").textContent = "";
}

function renderEpochTabs() {
  const bar = $("epoch-tabs");
  if (!bar) return;
  if (!evalEpochs.length && liveEpochId == null) {
    bar.classList.add("hidden");
    bar.innerHTML = "";
    return;
  }
  bar.classList.remove("hidden");
  bar.innerHTML = "";
  evalEpochs.forEach((ep, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "epoch-tab" + (i === selectedEpochIdx ? " active" : "");
    btn.textContent = `Epoch ${ep.id}`;
    btn.title = `${ep.timesteps.toLocaleString()} steps · score ${fmtScore(ep.score)}`;
    btn.onclick = () => selectEpoch(i);
    bar.appendChild(btn);
  });
  if (liveEpochId != null) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "epoch-tab live" + (selectedEpochIdx === -1 ? " active" : "");
    btn.textContent = `Epoch ${liveEpochId} · live`;
    btn.onclick = () => selectEpoch(-1);
    bar.appendChild(btn);
  }
}

function selectEpoch(idx) {
  stopLiveReplay();
  selectedEpochIdx = idx;
  if (idx === -1) {
    matchEpisodes = evalLiveEpisodes.map((e) => (e ? { ...e } : undefined));
  } else {
    matchEpisodes = evalEpochs[idx].episodes.map((e) => ({ ...e }));
  }
  buildMatchGrid(matchEpisodes);
  renderEpochTabs();
  updateEvalOverallForSelection();
}

function updateEvalOverallForSelection() {
  const el = $("eval-overall");
  if (!el) return;
  if (selectedEpochIdx === -1) {
    const avg = avgEvalScore(evalLiveEpisodes);
    const done = evalLiveEpisodes.filter(Boolean).length;
    if (avg != null) {
      el.textContent = `Average score: ${fmtScore(avg)} (${done} runs · live)`;
    }
    return;
  }
  const ep = evalEpochs[selectedEpochIdx];
  if (!ep) return;
  el.textContent =
    `Epoch ${ep.id} · ${ep.timesteps.toLocaleString()} steps · score ${fmtScore(ep.score)} (${ep.episodes.length} runs)`;
}

function finalizeLiveEpoch(timesteps, score) {
  const episodes = evalLiveEpisodes.filter(Boolean).map((e) => ({ ...e }));
  if (!episodes.length) return;
  evalEpochs.push({
    id: liveEpochId || evalEpochs.length + 1,
    timesteps,
    score,
    episodes,
  });
  liveEpochId = null;
  selectedEpochIdx = evalEpochs.length - 1;
  matchEpisodes = episodes;
  stopLiveReplay();
  buildMatchGrid(matchEpisodes);
  renderEpochTabs();
  updateEvalOverallForSelection();
}

function runLabel(r) {
  return r.run_uid || `R${String(r.id).padStart(6, "0")}`;
}
function appendHistoryPoint(t, reward, score) {
  if (t == null) return;
  const lastT = history.t[history.t.length - 1];
  const s = score ?? history.score[history.score.length - 1] ?? 0;
  const r = reward ?? history.reward[history.reward.length - 1] ?? 0;
  if (lastT === t && history.t.length) {
    history.reward[history.reward.length - 1] = r;
    history.score[history.score.length - 1] = s;
    history.mission[history.mission.length - 1] = s;
  } else {
    history.t.push(t);
    history.reward.push(r);
    history.score.push(s);
    history.mission.push(s);
  }
  drawCurve();
}
function showEvalProgress(ev) {
  const box = $("eval-progress");
  if (!box) return;
  box.classList.remove("hidden");
  if (ev.state === "starting") {
    evalLiveEpisodes = [];
    matchEpisodes = [];
    liveEpochId = ev.eval_epoch || evalEpochs.length + 1;
    selectedEpochIdx = -1;
    stopLiveReplay();
    if ($("match-grid")) $("match-grid").innerHTML = "";
    if ($("eval-overall")) $("eval-overall").textContent = "Running eval pass…";
    renderEpochTabs();
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
    evalLiveEpisodes[idx] = { ...ev.enemy_summary };
    if (selectedEpochIdx === -1) {
      matchEpisodes[idx] = { ...ev.enemy_summary };
      renderEvalCard(ev.enemy_summary, idx, "match-grid", "mc");
    }
    const avg = avgEvalScore(evalLiveEpisodes);
    const done = evalLiveEpisodes.filter(Boolean).length;
    if (avg != null) {
      $("m-mission").textContent = fmtScore(avg);
      if (selectedEpochIdx === -1 && $("eval-overall")) {
        $("eval-overall").textContent =
          `Average score: ${fmtScore(avg)} (${done}/${ev.enemy_total || ev.total} runs complete · live)`;
      }
    }
  }
}
function stopLiveReplay() {
  if (!liveReplayPlaying) return;
  const { cardIdx, canvasId, frames } = liveReplayPlaying;
  liveReplayPlaying = null;
  refreshLivePlayButtons();
  const cv = document.getElementById(canvasId);
  if (cv && frames?.length) drawFrame(cv.getContext("2d"), frames[0], cv.width, cv.height);
}
function playLiveReplay(i) {
  const ep = matchEpisodes[i];
  if (!ep?.frames?.length) return;
  if (liveReplayPlaying?.cardIdx === i) {
    stopLiveReplay();
    return;
  }
  stopLiveReplay();
  stopAnalysisFrameReplay();
  const cvId = `mc${i}`;
  liveReplayPlaying = { cardIdx: i, idx: 0, hold: 0, canvasId: cvId, frames: ep.frames };
  refreshLivePlayButtons();
}
function refreshLivePlayButtons() {
  for (let i = 0; i < matchEpisodes.length; i++) {
    const card = document.getElementById(`match-grid-card-${i}`);
    if (!card) continue;
    const playing = liveReplayPlaying?.cardIdx === i;
    card.classList.toggle("replay-playing", playing);
    const btn = card.querySelector(".live-play-btn");
    if (btn) btn.textContent = playing ? "■ Stop" : "▶ Play";
  }
}
function stopAnalysisFrameReplay(prefix) {
  if (analysisFrameReplay && (!prefix || analysisFrameReplay.prefix === prefix)) {
    const { prefix: p, canvasId, frames } = analysisFrameReplay;
    analysisFrameReplay = null;
    $(`${p}-analysis-replay-stop`)?.classList.add("hidden");
    const cv = $(canvasId);
    if (cv && frames?.length) drawFrame(cv.getContext("2d"), frames[0], cv.width, cv.height);
  }
}
function showAnalysisReplayPreview(prefix) {
  const enemy = $(`${prefix}-analysis-replay-enemy`)?.value;
  const ep = analysisReplayCache[prefix]?.replays?.[enemy];
  const info = $(`${prefix}-analysis-replay-info`);
  const cv = $(`${prefix}-analysis-replay`);
  if (!ep?.frames?.length) {
    if (info) info.textContent = enemy ? "No replay for this opponent." : "";
    if (cv) drawFrame(cv.getContext("2d"), { arena: 120 }, cv.width, cv.height);
    return;
  }
  if (info) {
    const worst = ep.picked_worst ? " (worst of eval runs)" : "";
    const evalMr = ep.mission_rate != null ? ` · eval ${(ep.mission_rate * 100).toFixed(0)}% mission` : "";
    info.textContent = `${ep.result}${evalMr}${worst} · ${Math.round(ep.steps || 0)} steps`;
  }
  if (cv) drawFrame(cv.getContext("2d"), ep.frames[0], cv.width, cv.height);
}
function playAnalysisFrameReplay(prefix) {
  const enemy = $(`${prefix}-analysis-replay-enemy`)?.value;
  const ep = analysisReplayCache[prefix]?.replays?.[enemy];
  if (!ep?.frames?.length) {
    showAnalysisReplayPreview(prefix);
    return;
  }
  if (analysisFrameReplay?.prefix === prefix && analysisFrameReplay?.enemy === enemy) {
    stopAnalysisFrameReplay(prefix);
    return;
  }
  stopLiveReplay();
  stopAnalysisFrameReplay();
  analysisFrameReplay = {
    prefix,
    enemy,
    idx: 0,
    hold: 0,
    canvasId: `${prefix}-analysis-replay`,
    frames: ep.frames,
  };
  $(`${prefix}-analysis-replay-stop`)?.classList.remove("hidden");
  showAnalysisReplayPreview(prefix);
}
function advanceFrameReplay(state) {
  if (!state?.frames?.length) return state;
  const cv = $(state.canvasId) || document.getElementById(state.canvasId);
  if (!cv) return state;
  const frameIdx = Math.min(state.idx, state.frames.length - 1);
  drawFrame(cv.getContext("2d"), state.frames[frameIdx], cv.width, cv.height);
  if (state.idx < state.frames.length - 1) return { ...state, idx: state.idx + 1, hold: 0 };
  const hold = state.hold + 1;
  if (hold > MATCH_HOLD_FRAMES) return { ...state, idx: 0, hold: 0 };
  return { ...state, hold };
}
function renderEvalCard(ep, i, gridId = "match-grid", idPrefix = "mc") {
  const grid = $(gridId);
  if (!grid) return;
  const cardId = `${gridId}-card-${i}`;
  const cvId = `${idPrefix}${i}`;
  const hasFrames = !!(ep.frames && ep.frames.length);
  const playing = liveReplayPlaying?.cardIdx === i;
  let wrap = document.getElementById(cardId);
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "match replay-card";
    wrap.id = cardId;
    wrap.innerHTML = `<div class="cap"><b>${ep.enemy}</b><span class="tag"></span></div>
      <div class="meta hint"></div>
      <button type="button" class="live-play-btn btn-sm primary">▶ Play</button>
      <canvas width="170" height="170" id="${cvId}"></canvas>`;
    grid.appendChild(wrap);
    wrap.querySelector(".live-play-btn").onclick = (e) => { e.stopPropagation(); playLiveReplay(i); };
  }
  wrap.classList.toggle("replay-playing", playing);
  const btn = wrap.querySelector(".live-play-btn");
  if (btn) {
    btn.disabled = !hasFrames;
    btn.textContent = !hasFrames ? "No data" : (playing ? "■ Stop" : "▶ Play");
  }
  const cap = wrap.querySelector(".cap b");
  if (cap) cap.textContent = ep.enemy;
  const tag = wrap.querySelector(".tag");
  const meta = wrap.querySelector(".meta");
  const tc = ep.result === "mission" ? "win" : (ep.result === "shot_down" ? "lost" : "timeout");
  tag.className = "tag " + tc;
  tag.textContent = ep.result;
  const sc = ep.score != null ? ` · ${fmtScore(ep.score)}` : "";
  const worst = ep.picked_worst ? " · worst replay" : "";
  meta.textContent = `${Math.round(ep.steps || ep.mean_steps || 0)} steps${sc}${worst}`;
  const cv = document.getElementById(cvId);
  if (cv && hasFrames && !playing) drawFrame(cv.getContext("2d"), ep.frames[0], cv.width, cv.height);
}
function buildMatchGrid(episodes, gridId = "match-grid", idPrefix = "mc") {
  const grid = $(gridId);
  if (!grid) return;
  grid.innerHTML = "";
  for (let i = 0; i < episodes.length; i++) {
    if (episodes[i]) renderEvalCard(episodes[i], i, gridId, idPrefix);
  }
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
      `Average score: ${fmtScore(score)} (${SCORE_FORMULA} over ${s.total_simulations} runs) · monitoring only`;
  }
  hideEvalProgress();
}
function handleEvent(ev) {
  if (watchedRun === null && ev.run_id) watchedRun = ev.run_id;
  if (ev.run_id && ev.run_id !== watchedRun) return;
  if (ev.type === "status") {
    $("m-state").textContent = ev.state;
    $("live-run").textContent = `#${ev.run_id} (${ev.state})`;
    if (ev.state === "running" && history.t.length === 0) {
      appendHistoryPoint(0, 0, 0);
    }
    if (ev.state === "error") alert("Run failed: " + (ev.message || ""));
    if (["done", "error", "stopped", "evaluating"].includes(ev.state)) loadRuns();
  } else if (ev.type === "eval_progress") {
    $("m-state").textContent = "evaluating";
    showEvalProgress(ev);
    if (ev.progress != null) {
      $("m-progress").textContent = Math.round(ev.progress * 100) + "%";
      $("progress-bar").style.width = ev.progress * 100 + "%";
    }
    if (ev.ep_rew_mean != null) $("m-reward").textContent = ev.ep_rew_mean;
    if (ev.timesteps != null) {
      const lastScore = history.score[history.score.length - 1] ?? 0;
      appendHistoryPoint(ev.timesteps, ev.ep_rew_mean, lastScore);
    }
  } else if (ev.type === "update") {
    $("m-progress").textContent = Math.round(ev.progress * 100) + "%";
    $("progress-bar").style.width = ev.progress * 100 + "%";
    const score = ev.eval_score != null ? ev.eval_score : ev.winrate;
    $("m-mission").textContent = fmtScore(score);
    $("m-reward").textContent = ev.ep_rew_mean;
    if (ev.eval_ran === false) hideEvalProgress();
    else {
      const score = ev.eval_score != null ? ev.eval_score : ev.winrate;
      finalizeLiveEpoch(ev.timesteps, score);
      applyEvalSummary(ev);
    }
    appendHistoryPoint(ev.timesteps, ev.ep_rew_mean, score);
  } else if (ev.type === "done") {
    $("m-state").textContent = "done";
    showPostRunPanel(ev.run_id);
    loadRuns();
  } else if (ev.type === "report") {
    const ps = $("post-report-status");
    if (ev.state === "pending" && ps) ps.textContent = "Generating analysis report…";
    if (ev.state === "ready") {
      if (lastDoneRunId === ev.run_id) loadReport(ev.run_id, "post");
      if (detailRunId === ev.run_id) loadReport(ev.run_id, "detail");
      loadRuns();
    }
    if (ev.state === "error" && ps) ps.textContent = "Analysis failed: " + (ev.message || "");
  } else if (ev.type === "final") {
    if (ev.state === "running") $("final-progress").textContent = "running... duels " + (ev.progress || "");
    else if (ev.state === "done") { $("final-progress").textContent = "done"; loadFinal(); }
    else if (ev.state === "error") $("final-progress").textContent = "error: " + (ev.message || "");
  }
}

// ---------------------------------------------------------------- my runs
async function loadRuns() {
  const data = await (await fetch("/api/runs")).json();
  if (data.quota) {
    if (ME) ME.quota = data.quota;
    updateQuota(data.quota);
    applyStepsConfig(data.quota);
  }
  const isAdmin = !!(data.is_admin || (ME && ME.is_admin));
  const rows = (data.runs || []).map((r) => {
    const sc = r.score != null ? (r.score * 100).toFixed(1) + "%" : "-";
    const miss = r.mission_rate != null ? (r.mission_rate * 100).toFixed(0) + "%" : "-";
    const kill = r.kill_rate != null ? (r.kill_rate * 100).toFixed(0) + "%" : "-";
    const eff = r.missile_efficiency != null ? r.missile_efficiency.toFixed(2) : "-";
    const isOwn = ME && r.user_id === ME.id;
    const canSubmit = r.status === "done" && r.score != null && isOwn;
    const submitBtn = canSubmit
      ? `<button class="${r.submitted ? "primary" : ""}" onclick="submitRun(${r.id})">${r.submitted ? "submitted" : "submit"}</button>`
      : "-";
    const stopBtn = (r.status === "running" || r.status === "queued")
      ? `<button class="danger btn-sm" onclick="stopRun(${r.id})">stop</button>` : "";
    const delBtn = isAdmin
      ? `<button class="danger btn-sm" onclick="deleteRun(${r.id}, '${runLabel(r)}')">delete</button>` : "";
    const reportLabel = { ready: "report ready", pending: "generating…", error: "failed" };
    const report = r.status === "done"
      ? `<span class="report-badge rb-${r.analysis_status || "pending"}">${reportLabel[r.analysis_status] || "generating…"}</span>`
      : "-";
    const reviewBtn = r.status === "done"
      ? `<button class="btn-sm" onclick="openRunDetail(${r.id})">${r.analysis_status === "ready" ? "view report" : "open"}</button>` : "";
    const userCell = isAdmin ? `<td class="name">${r.user_name || "-"}</td>` : "";
    return `<tr><td><b>${runLabel(r)}</b></td>${userCell}<td><span class="status-badge s-${r.status}">${r.status}</span></td>
      <td>${r.steps}</td><td class="score">${sc}</td><td>${miss}</td><td>${kill}</td><td>${eff}</td>
      <td>${r.mean_reward ?? "-"}</td><td>${report}</td><td>${submitBtn} ${reviewBtn} ${stopBtn} ${delBtn}</td></tr>`;
  }).join("");
  const userHdr = isAdmin ? "<th>user</th>" : "";
  const title = isAdmin ? "All runs" : "My runs";
  $("runs-table").innerHTML = `<p class="hint">${title} · score = ${SCORE_FORMULA}</p>
    <table><tr><th>id</th>${userHdr}<th>status</th><th>steps</th><th>score</th>
    <th>mission</th><th>kill</th><th>eff</th><th>reward</th><th>report</th><th>actions</th></tr>
    ${rows || `<tr><td colspan='${isAdmin ? 11 : 10}'>No runs yet.</td></tr>`}</table>`;
}
async function submitRun(id) { await postJSON(`/api/run/${id}/submit`); loadRuns(); }
async function stopRun(id) { await postJSON(`/api/run/${id}/stop`); loadRuns(); }
async function deleteRun(id, label) {
  if (!confirm(`Delete run ${label || id}? This removes the model, report, and job files.`)) return;
  const res = await fetch(`/api/admin/run/${id}`, { method: "DELETE" });
  const data = await res.json();
  if (!data.ok) { alert(data.error || "Could not delete"); return; }
  if (detailRunId === id) { detailRunId = null; $("run-detail")?.classList.add("hidden"); }
  loadRuns();
  loadAdmin();
}

function populateReplaySelect(selectId, enemies) {
  const sel = $(selectId);
  if (!sel) return;
  sel.innerHTML = "";
  const list = enemies && enemies.length ? enemies : (ME?.enemy_types || []);
  list.forEach((e) => {
    const opt = document.createElement("option");
    opt.value = e;
    opt.textContent = enemyLabel(e);
    sel.appendChild(opt);
  });
}

async function loadAnalysisReplays(runId, prefix) {
  stopAnalysisFrameReplay(prefix);
  const info = $(`${prefix}-analysis-replay-info`);
  if (info) info.textContent = "Loading replays…";
  const res = await fetch(`/api/run/${runId}/analysis/replays`);
  const data = await res.json();
  if (!data.ok || !data.replays) {
    if (info) info.textContent = data.error || "Replays not available for this run.";
    delete analysisReplayCache[prefix];
    return;
  }
  const enemies = data.enemies?.length ? data.enemies : Object.keys(data.replays);
  analysisReplayCache[prefix] = { enemies, replays: data.replays };
  populateReplaySelect(`${prefix}-analysis-replay-enemy`, enemies);
  showAnalysisReplayPreview(prefix);
}

function wireAnalysisReplayControls(prefix) {
  const btn = $(`${prefix}-analysis-replay-btn`);
  const stopBtn = $(`${prefix}-analysis-replay-stop`);
  const sel = $(`${prefix}-analysis-replay-enemy`);
  if (btn) btn.onclick = () => playAnalysisFrameReplay(prefix);
  if (stopBtn) stopBtn.onclick = () => stopAnalysisFrameReplay(prefix);
  if (sel) sel.onchange = () => { stopAnalysisFrameReplay(prefix); showAnalysisReplayPreview(prefix); };
}

function showPostRunPanel(runId) {
  lastDoneRunId = runId;
  const panel = $("post-run-panel");
  if (!panel) return;
  panel.classList.remove("hidden");
  $("post-run-label").textContent = `#${runId}`;
  $("post-report").innerHTML = "";
  $("post-report-status").textContent = "Analysis runs automatically when training finishes…";
  loadReport(runId, "post");
}

function openRunDetail(runId) {
  detailRunId = runId;
  showView("runs");
  $("run-detail").classList.remove("hidden");
  $("run-detail-label").textContent = `#${runId}`;
  $("detail-report").innerHTML = "";
  loadReport(runId, "detail");
}

const reportCache = {};  // prefix -> last loaded report (for "use parameters")
function prettyTerm(k) { return k.replace(/_/g, " "); }

async function loadReport(runId, prefix) {
  const status = $(prefix + "-report-status");
  const box = $(prefix + "-report");
  if (box) box.innerHTML = "";
  if (status) status.textContent = "Loading report…";
  const res = await fetch(`/api/run/${runId}/report`);
  const data = await res.json();
  if (!data.ok) { if (status) status.textContent = data.error || "Could not load report."; return; }
  renderReport(data, prefix, data.run?.id);
}

function renderReport(data, prefix, runId) {
  const box = $(prefix + "-report");
  const status = $(prefix + "-report-status");
  if (!box) return;
  reportCache[prefix] = data;
  const p = data.params || {};
  const rw = p.rewards || {};
  const st = data.analysis_status;

  // Selected parameters: enemies, steps, reward weights (non-zero first).
  const weightRows = Object.keys(rw).filter((k) => k !== "global_scale")
    .sort((a, b) => Math.abs(rw[b]) - Math.abs(rw[a]))
    .map((k) => `<tr><td>${prettyTerm(k)}</td><td class="${rw[k] > 0 ? "pos" : rw[k] < 0 ? "neg" : "zero"}">${rw[k]}</td></tr>`)
    .join("");
  const ew = p.enemy_weights || {};
  const oppStr = (p.enemies || [])
    .map((e) => (ew[e] != null ? `${e} (${Number(ew[e]).toFixed(2)})` : e))
    .join(", ") || "all";
  let html = `<div class="report-params">
    <h3>Selected parameters</h3>
    <p class="hint">steps <b>${p.steps}</b> · global scale <b>${rw.global_scale ?? 1}</b> · opponents — priority <b>${oppStr}</b></p>
    <button class="primary btn-sm" onclick="useReportParams('${prefix}')">Use these parameters in trainer</button>
    <table class="analysis-table report-weights"><tr><th>reward term</th><th>weight</th></tr>${weightRows}</table>
  </div>`;

  // Learning + reward curve.
  html += `<h3>Learning &amp; reward curve</h3>
    <canvas id="${prefix}-curve" class="chart-canvas" width="720" height="260"></canvas>`;

  // Analysis (auto-generated when the run finished).
  if (st === "ready" && data.stats) {
    const s = data.stats;
    html += `<h3>Analysis report</h3>
      <p>Score <b>${(s.score * 100).toFixed(0)}%</b> · mission <b>${(s.mission_rate * 100).toFixed(0)}%</b> · kill <b>${(s.kill_rate * 100).toFixed(0)}%</b> · survive <b>${(s.survival_rate * 100).toFixed(0)}%</b> · mean reward <b>${s.mean_reward}</b> · missile eff. <b>${s.missile_efficiency}</b></p>`;
    html += "<table class='analysis-table'><tr><th>enemy</th><th>mission</th><th>survive</th><th>kill</th><th>reward</th><th>missiles</th></tr>";
    for (const [enemy, e] of Object.entries(s.per_enemy || {})) {
      html += `<tr><td>${enemy}</td><td>${(e.mission_rate * 100).toFixed(0)}%</td><td>${(e.survival * 100).toFixed(0)}%</td><td>${(e.kills * 100).toFixed(0)}%</td><td>${e.mean_reward}</td><td>${e.missiles_used}</td></tr>`;
    }
    html += "</table>";
    if (data.agent_profile) {
      html += `<h3>Agent profile map</h3>
        <p class="hint">Opponent offense/defense profiles (same axes as the training picker). ★ estimates where your learned agent sits.</p>
        <div class="enemy-map-wrap report-profile-map"><canvas id="${prefix}-profile-map" width="520" height="360"></canvas></div>`;
    }
    html += "<div class='analysis-plots'>";
    for (const [name, path] of Object.entries(data.plots || {})) {
      if (name === "agent_profile") continue;
      html += `<img src="${path}?t=${Date.now()}" alt="${name}"/>`;
    }
    html += "</div>";
    html += `<h3>Eval replay</h3>
      <p class="hint">Select one opponent at a time. When eval results differ, the replay shows the worst episode.</p>
      <div class="run-tools">
        <select id="${prefix}-analysis-replay-enemy"></select>
        <button type="button" id="${prefix}-analysis-replay-btn" class="primary btn-sm">Watch replay</button>
        <button type="button" id="${prefix}-analysis-replay-stop" class="danger btn-sm hidden">Stop</button>
        <span id="${prefix}-analysis-replay-info" class="hint"></span>
      </div>
      <canvas id="${prefix}-analysis-replay" class="replay-canvas" width="420" height="420"></canvas>`;
  } else if (st === "pending") {
    html += `<p class="hint">Analysis is being generated automatically… reopen shortly.</p>`;
  } else if (st === "error") {
    html += `<p class="hint">Analysis failed: ${data.error || "unknown error"}.</p>`;
  }

  box.innerHTML = html;
  if (status) status.textContent = "";
  drawLearningCurve(`${prefix}-curve`, data.curve || { t: [], reward: [], score: [] });
  if (data.agent_profile && EnemyPicker.drawProfileMap) {
    EnemyPicker.drawProfileMap(`${prefix}-profile-map`, data.agent_profile);
  }
  wireAnalysisReplayControls(prefix);
  if (st === "ready" && runId) loadAnalysisReplays(runId, prefix);
}

function useReportParams(prefix) {
  const data = reportCache[prefix];
  if (!data) return;
  applyRunParamsToTrainer(data.params || {});
}

function applyRunParamsToTrainer(params) {
  const rewards = params.rewards || {};
  RewardEditor.renderAll(ME.reward_editor, rewards);
  if (params.enemies && params.enemies.length && ME.enemy_catalog) {
    EnemyPicker.render("enemy-picker", ME.enemy_catalog,
      { selected: params.enemies, weights: params.enemy_weights || {} });
  }
  if (ME?.quota?.steps_editable && params.steps) {
    const stepsEl = $("steps");
    const max = ME.quota.steps_per_run;
    if (stepsEl) stepsEl.value = Math.min(params.steps, max);
  } else {
    applyStepsConfig();
  }
  showView("train");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------- student report
const REPORT_FIELDS = [
  { key: "feedback_strategy",
    label: { en: "1. Feedback strategy", pt: "1. Estratégia de Feedbacks" },
    hint: {
      en: "How did you adjust reward/penalty weights between training runs? What behaviour did you expect, and what happened?",
      pt: "Explique a estratégia adotada para ajustar os pesos das recompensas e penalidades entre cada processo de treinamento, indicando comportamentos esperados e resultados obtidos.",
    } },
  { key: "final_analysis",
    label: { en: "2. Final results analysis", pt: "2. Análise dos Resultados Finais" },
    hint: {
      en: "Analyse your submitted run (score, charts, replays). What would make the agent more robust?",
      pt: "Analise o relatório do comportamento final submetido e sugira estratégias que você acredita que poderiam deixá-lo mais robusto.",
    } },
];

let reportLang = localStorage.getItem("reportLang") || "pt";
const REPORT_I18N = {
  en: {
    title: "Activity report — TE-276",
    subtitle: "2 short answers based on your submitted run",
    intro: "Use your runs, analysis charts and replays as evidence. Answer both sections briefly, then click Save report.",
    save: "Save report", placeholder: "Your answer…",
    notSaved: "Not saved yet.", saved: "Saved",
    contextHint: "No submission yet. Submit your best run on My runs to fill in the score box below.",
    contextTitle: "Submitted run",
    contextEvidence: "Use these numbers in your answers.",
    bestLabel: "Strongest vs", worstLabel: "Weakest vs",
  },
  pt: {
    title: "Relatório — TE-276",
    subtitle: "2 respostas com base no seu run submetido",
    intro: "Use suas execuções, os gráficos de análise e os replays como evidência. Responda as duas seções com objetividade e clique em Salvar relatório.",
    save: "Salvar relatório", placeholder: "Sua resposta…",
    notSaved: "Ainda não salvo.", saved: "Salvo",
    contextHint: "Nenhum run submetido ainda. Envie o melhor em Minhas execuções para preencher o quadro abaixo.",
    contextTitle: "Run submetido",
    contextEvidence: "Use estes números nas suas respostas.",
    bestLabel: "Melhor vs", worstLabel: "Pior vs",
  },
};
function reportT(k) { return (REPORT_I18N[reportLang] || REPORT_I18N.en)[k]; }

function migrateReportData(data) {
  if (!data || typeof data !== "object") return {};
  if (data.feedback_strategy || data.final_analysis) return data;
  if (data.rewards || data.results || data.improvement) {
    return {
      feedback_strategy: [data.rewards, data.improvement].filter(Boolean).join("\n\n").trim(),
      final_analysis: data.results || "",
    };
  }
  const parts = [];
  if (data.strategy) parts.push(data.strategy);
  if (data.reward_link) parts.push(data.reward_link);
  return {
    feedback_strategy: parts.join("\n\n").trim(),
    final_analysis: [data.curve, data.results, data.matchups, data.next_step].filter(Boolean).join("\n\n").trim(),
  };
}

function renderReportContext(ctx, containerId) {
  const box = $(containerId);
  if (!box) return;
  const t = reportT;
  if (!ctx || !ctx.has_submission) {
    box.innerHTML = `<p class="hint">${t("contextHint")}</p>`;
    return;
  }
  const pct = (v) => v == null ? "-" : (v * 100).toFixed(0) + "%";
  const matchRow = (m) => `<span class="chip">${m.enemy} ${pct(m.mission_rate)}</span>`;
  box.innerHTML = `
    <p class="hint"><b>${t("contextTitle")}</b> ${ctx.run_uid} — ${t("contextEvidence")}</p>
    <div class="report-metrics">
      <span><b>${pct(ctx.score)}</b> score</span>
      <span><b>${pct(ctx.mission_rate)}</b> missão</span>
      <span><b>${pct(ctx.kill_rate)}</b> abate</span>
    </div>
    ${(ctx.worst && ctx.worst.length) ? `<p class="hint">${t("worstLabel")}: ${ctx.worst.map(matchRow).join(" ")}</p>` : ""}
    ${(ctx.best && ctx.best.length) ? `<p class="hint">${t("bestLabel")}: ${ctx.best.map(matchRow).join(" ")}</p>` : ""}`;
}

function collectReportValues() {
  const out = {};
  REPORT_FIELDS.forEach((f) => { const el = $("rf-" + f.key); if (el) out[f.key] = el.value; });
  return out;
}

function buildReportForm(data) {
  const form = $("report-form");
  if (!form) return;
  data = migrateReportData(data);
  form.innerHTML = REPORT_FIELDS.map((f) => `
    <div class="report-field">
      <label for="rf-${f.key}">${f.label[reportLang] || f.label.en}</label>
      <p class="hint report-guide">${f.hint[reportLang] || f.hint.en}</p>
      <textarea id="rf-${f.key}" rows="3" placeholder="${reportT("placeholder")}">${escapeHtml(data[f.key] || "")}</textarea>
    </div>`).join("");
  applyReportI18n();
}

function applyReportI18n() {
  const t = REPORT_I18N[reportLang] || REPORT_I18N.en;
  if ($("report-title")) $("report-title").textContent = t.title;
  if ($("report-subtitle")) $("report-subtitle").textContent = t.subtitle;
  if ($("report-intro")) $("report-intro").textContent = t.intro;
  if ($("report-save")) $("report-save").textContent = t.save;
  document.querySelectorAll(".report-lang-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.lang === reportLang));
}

function setReportLang(lang) {
  if (lang === reportLang) return;
  const current = collectReportValues();
  reportLang = lang;
  localStorage.setItem("reportLang", lang);
  buildReportForm(current);
}

async function loadMyReport() {
  const res = await fetch("/api/report");
  const data = await res.json();
  if (!data.ok) return;
  renderReportContext(data.context, "report-context");
  buildReportForm(data.data || {});
  const saved = $("report-saved");
  if (saved) saved.textContent = data.updated_at
    ? `${reportT("saved")} ${new Date(data.updated_at * 1000).toLocaleString()}`
    : reportT("notSaved");
}

async function saveReport() {
  const out = collectReportValues();
  const res = await postJSON("/api/report", { data: out });
  const saved = $("report-saved");
  if (res.data && res.data.ok) {
    if (saved) saved.textContent = `${reportT("saved")} ${new Date().toLocaleString()}`;
  } else if (saved) {
    saved.textContent = "Could not save.";
  }
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

const TRAINING_CFG_KEYS = [
  "train_seed", "max_cycles", "train_device",
  "policy_hidden_size", "policy_n_layers",
  "ppo_learning_rate", "ppo_n_steps", "ppo_batch_size", "ppo_n_epochs",
  "ppo_gamma", "ppo_gae_lambda", "ppo_clip_range", "ppo_ent_coef", "ppo_vf_coef", "ppo_max_grad_norm",
];

async function loadAdmin() {
  const data = await (await fetch("/api/admin/config")).json();
  if (!data.config) return;
  adminRewardEditor = RewardEditor.normalizeEditor(data.reward_editor);
  for (const [k, v] of Object.entries(data.config)) { const el = $("cfg-" + k); if (el) el.value = v; }
  const z = $("cfg-rewards_start_zero");
  if (z) z.checked = data.config.rewards_start_zero === "1";
  const se = $("cfg-steps_editable");
  if (se) se.checked = data.config.steps_editable === "1";
  if (data.reward_editor && data.code_defaults) {
    RewardEditor.renderAdminTable("admin-reward-table", data.reward_editor, data.code_defaults, data.code_ranges);
  }
  const rows = (data.runs || []).map((r) =>
    `<tr><td><b>${runLabel(r)}</b></td><td class="name">${r.user_name}</td><td><span class="status-badge s-${r.status}">${r.status}</span></td>
     <td>${r.steps}</td><td class="score">${r.score != null ? (r.score * 100).toFixed(1) + "%" : "-"}</td>
     <td>${r.submitted ? "yes" : ""}</td>
     <td><button class="danger btn-sm" onclick="deleteRun(${r.id}, '${runLabel(r)}')">delete</button></td></tr>`).join("");
  $("admin-runs").innerHTML = `<table><tr><th>id</th><th>user</th><th>status</th><th>steps</th><th>score</th><th>submitted</th><th></th></tr>${rows}</table>`;
  if (data.training_defaults?.defaults) {
    const hint = $("training-config-hint");
    if (hint) {
      const d = data.training_defaults.defaults;
      hint.textContent =
        `Locked baseline: ${d.policy_hidden_size}×${d.policy_n_layers} MLP, lr=${d.ppo_learning_rate}, n_steps=${d.ppo_n_steps}, device=${d.train_device}. Values are clamped on save.`;
    }
  }
}
async function saveAdmin(includeTraining = false) {
  const keys = ["class_access_code", "runs_per_window", "steps_per_run", "window_hours", "max_concurrent", "registration_open",
    "eval_every_rollouts", "eval_episodes_per_enemy", "live_eval_max_enemies",
    "locked_eval_episodes_per_enemy", "analysis_episodes_per_enemy"];
  const body = {};
  keys.forEach((k) => (body[k] = $("cfg-" + k).value));
  if (includeTraining) TRAINING_CFG_KEYS.forEach((k) => (body[k] = $("cfg-" + k).value));
  body.rewards_start_zero = $("cfg-rewards_start_zero")?.checked ? "1" : "0";
  body.steps_editable = $("cfg-steps_editable")?.checked ? "1" : "0";
  if (adminRewardEditor) {
    const rc = RewardEditor.gatherAdminRewardConfig(adminRewardEditor.terms);
    body.reward_defaults_json = JSON.stringify(rc.defaults);
    body.reward_ranges_json = JSON.stringify(rc.ranges);
  }
  const res = await postJSON("/api/admin/config", body);
  alert("Saved.");
  loadAdmin();
  if (ME && ME.authenticated) {
    const me = await (await fetch("/api/me")).json();
    if (me.quota) { ME.quota = me.quota; applyStepsConfig(me.quota); }
  }
}
async function loadUsers() {
  const box = $("admin-users");
  if (!box) return;
  const data = await (await fetch("/api/admin/users")).json();
  if (!data.ok) { box.innerHTML = `<p class="hint">${data.error || "Could not load users."}</p>`; return; }
  const rows = (data.users || []).map((u) => {
    const score = u.submitted_score != null ? (u.submitted_score * 100).toFixed(1) + "%" : "-";
    const rep = u.has_report ? `<button class="btn-sm" onclick="openUserReport(${u.id}, '${escapeHtml(u.name)}')">view</button>` : `<span class="hint">none</span>`;
    return `<tr>
      <td class="name">${escapeHtml(u.name)}${u.is_admin ? ' <small>(admin)</small>' : ''}</td>
      <td>${u.total_runs || 0}</td>
      <td>${u.done_runs || 0}</td>
      <td>${u.quota_used}/${u.quota_per_window}</td>
      <td class="score">${score}</td>
      <td>${rep}</td>
      <td><button class="btn-sm" onclick="resetUserQuota(${u.id}, '${escapeHtml(u.name)}')">reset count</button></td>
    </tr>`;
  }).join("");
  box.innerHTML = `<div class="users-toolbar">
      <p class="hint">Quota window: ${data.window_hours}h. "Reset count" clears the rolling-window run count without deleting runs.</p>
      <button class="btn-sm" onclick="exportReports()">Export all reports (JSON)</button>
    </div>
    <table><tr><th>user</th><th>runs</th><th>done</th><th>quota used</th><th>submitted</th><th>report</th><th></th></tr>${rows}</table>`;
}

function exportReports() {
  window.open("/api/admin/reports/export", "_blank");
}

async function resetUserQuota(id, name) {
  if (!confirm(`Reset the run count for ${name}? Their used quota goes back to 0.`)) return;
  const res = await postJSON(`/api/admin/user/${id}/reset-quota`);
  if (res.data && res.data.ok) loadUsers();
  else alert((res.data && res.data.error) || "Could not reset.");
}

async function openUserReport(id, name) {
  const data = await (await fetch(`/api/admin/user/${id}/report`)).json();
  const wrap = $("admin-user-detail");
  if (!data.ok) { alert(data.error || "Could not load report."); return; }
  const updated = data.updated_at ? `<p class="hint">Last saved ${new Date(data.updated_at * 1000).toLocaleString()}</p>` : `<p class="hint">No written report saved.</p>`;
  const fields = REPORT_FIELDS.map((f) => {
    const val = (data.data || {})[f.key];
    const lbl = (f.label[reportLang] || f.label.en);
    return `<div class="report-field"><label>${lbl}</label>
      <div class="report-answer">${val ? escapeHtml(val).replace(/\n/g, "<br>") : '<span class="hint">— empty —</span>'}</div></div>`;
  }).join("");
  wrap.innerHTML = `<h3>Report — ${escapeHtml(name)}</h3>
    <div id="admin-user-report-context" class="report-context"></div>${updated}${fields}`;
  wrap.classList.remove("hidden");
  renderReportContext(data.context, "admin-user-report-context");
  wrap.scrollIntoView({ behavior: "smooth" });
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
  const res = frame.result;
  if (res && res !== "pending") {
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, h * 0.5 - 16, w, 32);
    ctx.fillStyle = res === "mission" ? "#5ec27a" : res === "shot_down" ? "#e8624c" : "#8b97ad";
    ctx.font = "bold 13px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(res).replace(/_/g, " "), w / 2, h * 0.5 + 5);
    ctx.textAlign = "left";
  }
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
  if (liveReplayPlaying) liveReplayPlaying = advanceFrameReplay(liveReplayPlaying);
  if (analysisFrameReplay) analysisFrameReplay = advanceFrameReplay(analysisFrameReplay);
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
  $("report-save").onclick = saveReport;
  $("admin-save").onclick = () => saveAdmin(false);
  $("admin-save-training")?.addEventListener("click", () => saveAdmin(true));
  $("final-btn").onclick = runFinal;
  setInterval(() => { if (ME && ME.authenticated && $("view-board").classList.contains("active")) loadBoard(); }, 15000);
});
