"use strict";

const $ = (id) => document.getElementById(id);
let CONFIG = null;
const history = { t: [], reward: [], winrate: [] };
let evalLiveEpisodes = [];
let evalEpochs = [];
let selectedEpochIdx = -1;
let liveEpochId = null;
let matchEpisodes = [];
let matchIdx = 0;
let matchHold = 0;
const MATCH_HOLD_FRAMES = 55;
const replayBuffer = [];
let replayIdx = 0;
let replayPlaying = false;

// --------------------------------------------------------------------------
// Config UI
// --------------------------------------------------------------------------
function enemyLabel(name, catalog) {
  const cat = catalog || (CONFIG && CONFIG.enemy_catalog) || (ME && ME.enemy_catalog) || {};
  const info = (cat.info && cat.info[name]) || {};
  return info.label || name;
}

async function fetchEnemyCatalog() {
  const res = await fetch("/api/enemies");
  if (!res.ok) throw new Error("Could not load enemy list");
  return res.json();
}

async function loadConfig() {
  const [cfgRes, catalog] = await Promise.all([fetch("/api/config"), fetchEnemyCatalog()]);
  CONFIG = await cfgRes.json();
  CONFIG.enemy_types = catalog.names;
  CONFIG.enemy_catalog = catalog;
  $("random_enemy_prob").value = CONFIG.scenario.random_enemy_prob ?? 0;
  $("train_timesteps").value = CONFIG.scenario.train_timesteps ?? 200000;
  RewardEditor.renderAll(CONFIG.reward_editor, CONFIG.rewards);

  const hint = $("enemy-hint");
  if (hint) {
    hint.textContent = (CONFIG.enemy_catalog && CONFIG.enemy_catalog.mode === "reference")
      ? "Click bubbles to toggle. Similar FSM profiles cluster — color = difficulty score."
      : "Click bubbles to toggle opponents. Position = offense vs defense style.";
  }

  const sel = $("replay-enemy");
  sel.innerHTML = "";
  const active = new Set(CONFIG.scenario.enemies || CONFIG.enemy_types || []);
  const initial = active.size ? [...active] : (CONFIG.enemy_types || []);
  if (CONFIG.enemy_types && CONFIG.enemy_types.length) {
    EnemyPicker.render("enemy-picker", catalog, { selected: initial });
  }
  (CONFIG.enemy_types || []).forEach((e) => {
    const opt = document.createElement("option");
    opt.value = e; opt.textContent = enemyLabel(e, catalog);
    sel.appendChild(opt);
  });
}

function renderTerms(containerId, terms, values) {
  /* legacy — reward editor handles rendering in loadConfig */
}

function gatherConfig() {
  const rewards = RewardEditor.gather();
  let enemies = EnemyPicker.getSelected();
  if (!enemies.length && CONFIG && CONFIG.enemy_types) enemies = [...CONFIG.enemy_types];
  const scenario = {
    enemies,
    random_enemy_prob: Number($("random_enemy_prob").value),
    train_timesteps: Number($("train_timesteps").value),
  };
  return { rewards, scenario };
}

async function saveConfig() {
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(gatherConfig()),
  });
  flash($("save-btn"), "Saved!");
}

function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1200);
}

// --------------------------------------------------------------------------
// Actions
// --------------------------------------------------------------------------
async function startTraining() {
  await saveConfig();
  history.t.length = 0; history.reward.length = 0; history.winrate.length = 0;
  resetEvalEpochs();
  await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quick: $("quick").checked }),
  });
}
async function stopJob() { await fetch("/api/train/stop", { method: "POST" }); }

async function startReplay() {
  replayBuffer.length = 0; replayIdx = 0; replayPlaying = true;
  $("replay-info").textContent = "loading model...";
  const res = await fetch("/api/replay/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enemy: $("replay-enemy").value }),
  });
  const data = await res.json();
  if (!data.ok) $("replay-info").textContent = data.error || "error";
}

async function runAnalysis() {
  const btn = $("analysis-btn");
  btn.disabled = true; btn.textContent = "Running...";
  const res = await fetch("/api/analysis", { method: "POST" });
  const data = await res.json();
  btn.disabled = false; btn.textContent = "Run analysis on saved model";
  const out = $("analysis-out");
  if (!data.ok) { out.innerHTML = `<p class="hint">${data.error}</p>`; return; }
  const s = data.stats;
  let html = `<p>Score <b>${(s.score * 100).toFixed(0)}%</b> &middot; mission <b>${(s.mission_rate * 100).toFixed(0)}%</b> &middot; kill <b>${(s.kill_rate * 100).toFixed(0)}%</b> &middot; survive <b>${(s.survival_rate * 100).toFixed(0)}%</b> &middot; mean reward <b>${s.mean_reward}</b> &middot; missile eff. <b>${s.missile_efficiency}</b></p>`;
  html += "<table class='analysis-table'><tr><th>enemy</th><th>mission</th><th>survive</th><th>kill</th><th>reward</th><th>missiles</th></tr>";
  for (const [enemy, e] of Object.entries(s.per_enemy)) {
    html += `<tr><td>${enemy}</td><td>${(e.mission_rate*100).toFixed(0)}%</td><td>${(e.survival*100).toFixed(0)}%</td><td>${(e.kills*100).toFixed(0)}%</td><td>${e.mean_reward}</td><td>${e.missiles_used}</td></tr>`;
  }
  html += "</table><div>";
  for (const path of Object.values(data.plots)) html += `<img src="${path}?t=${Date.now()}"/>`;
  html += "</div>";
  out.innerHTML = html;
}

// --------------------------------------------------------------------------
// WebSocket
// --------------------------------------------------------------------------
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => handleEvent(JSON.parse(ev.data));
  ws.onclose = () => setTimeout(connectWS, 1500);
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
}

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (cls ? " " + cls : "");
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
  matchIdx = 0;
  matchHold = 0;
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
  selectedEpochIdx = idx;
  if (idx === -1) {
    matchEpisodes = evalLiveEpisodes.filter(Boolean).map((e) => ({ ...e }));
  } else {
    matchEpisodes = evalEpochs[idx].episodes.map((e) => ({ ...e }));
  }
  matchIdx = 0;
  matchHold = 0;
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
  matchIdx = 0;
  matchHold = 0;
  buildMatchGrid(matchEpisodes);
  renderEpochTabs();
  updateEvalOverallForSelection();
}

function showEvalProgress(ev) {
  const box = $("eval-progress");
  if (!box) return;
  box.classList.remove("hidden");
  if (ev.state === "starting") {
    evalLiveEpisodes = [];
    liveEpochId = ev.eval_epoch || evalEpochs.length + 1;
    selectedEpochIdx = -1;
    matchIdx = 0;
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
    evalLiveEpisodes[idx] = ev.enemy_summary;
    if (selectedEpochIdx === -1) {
      matchEpisodes[idx] = ev.enemy_summary;
      renderEvalCard(ev.enemy_summary, idx);
    }
    const avg = avgEvalScore(evalLiveEpisodes);
    const done = evalLiveEpisodes.filter(Boolean).length;
    if (avg != null) {
      $("m-winrate").textContent = fmtScore(avg);
      if (selectedEpochIdx === -1 && $("eval-overall")) {
        $("eval-overall").textContent =
          `Average score: ${fmtScore(avg)} (${done}/${ev.enemy_total || ev.total} runs complete · live)`;
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
      <div class="meta hint"></div><canvas width="180" height="180" id="mc${i}"></canvas>`;
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
  $("m-winrate").textContent = fmtScore(score);
  $("m-eval-sims").textContent = `${s.finished}/${s.total_simulations}`;
  if ($("eval-overall")) {
    $("eval-overall").textContent =
      `Average score: ${fmtScore(score)} (0.6×mission + 0.25×kill + 0.15×missile eff. over ${s.total_simulations} runs) · monitoring only`;
  }
  hideEvalProgress();
}

function handleEvent(ev) {
  if (ev.type === "status") {
    if (ev.state === "training") setStatus("training...", "training");
    else if (ev.state === "done") setStatus("training complete", "done");
    else if (ev.state === "stopping") setStatus("stopping...", "training");
    else if (ev.state === "error") setStatus("error: " + (ev.message || ""), "error");
  } else if (ev.type === "eval_progress") {
    showEvalProgress(ev);
    if (ev.timesteps) $("m-timesteps").textContent = ev.timesteps;
    if (ev.progress != null) {
      $("m-progress").textContent = Math.round(ev.progress * 100) + "%";
      $("progress-bar").style.width = (ev.progress * 100) + "%";
    }
    if (ev.ep_rew_mean != null) $("m-reward").textContent = ev.ep_rew_mean;
  } else if (ev.type === "update") {
    $("m-progress").textContent = Math.round(ev.progress * 100) + "%";
    $("progress-bar").style.width = (ev.progress * 100) + "%";
    const score = ev.eval_score != null ? ev.eval_score : ev.winrate;
    $("m-winrate").textContent = fmtScore(score);
    $("m-reward").textContent = ev.ep_rew_mean;
    $("m-timesteps").textContent = ev.timesteps;
    if (ev.eval_ran === false) hideEvalProgress();
    else {
      const score = ev.eval_score != null ? ev.eval_score : ev.winrate;
      finalizeLiveEpoch(ev.timesteps, score);
      applyEvalSummary(ev);
    }
    history.t.push(ev.timesteps);
    history.reward.push(ev.ep_rew_mean);
    history.winrate.push(ev.eval_score != null ? ev.eval_score : ev.winrate);
    drawCurve();
  } else if (ev.type === "replay_start") {
    $("replay-info").textContent = "vs " + ev.enemy;
  } else if (ev.type === "replay_frame") {
    replayBuffer.push(ev.frame);
  } else if (ev.type === "replay_end") {
    replayPlaying = false;
    const r = ev.result || {};
    $("replay-info").textContent = `result: ${r.result || "?"} (reward ${r.reward ?? "-"})`;
  }
}

// --------------------------------------------------------------------------
// Rendering helpers
// --------------------------------------------------------------------------
function w2c(x, y, arena, w, h) { return [x / arena * w, h - y / arena * h]; }

function drawFrame(ctx, frame, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1322"; ctx.fillRect(0, 0, w, h);
  const arena = frame.arena || 100;
  // grid
  ctx.strokeStyle = "#1b2438"; ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    ctx.beginPath(); ctx.moveTo(i / 5 * w, 0); ctx.lineTo(i / 5 * w, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i / 5 * h); ctx.lineTo(w, i / 5 * h); ctx.stroke();
  }
  const b = frame.blue, rd = frame.red;
  // goal
  if (b && b.goal) {
    const [gx, gy] = w2c(b.goal[0], b.goal[1], arena, w, h);
    ctx.strokeStyle = "#5ec27a"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(gx, gy, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx - 9, gy); ctx.lineTo(gx + 9, gy);
    ctx.moveTo(gx, gy - 9); ctx.lineTo(gx, gy + 9); ctx.stroke();
  }
  // tracking line
  if (b && rd && rd.tracked && rd.alive && b.alive) {
    const [bx, by] = w2c(b.x, b.y, arena, w, h);
    const [rx, ry] = w2c(rd.x, rd.y, arena, w, h);
    ctx.strokeStyle = "rgba(232,98,76,0.35)"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(rx, ry); ctx.stroke();
    ctx.setLineDash([]);
  }
  // missiles
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
    ctx.beginPath(); ctx.moveTo(cx - 5, cy - 5); ctx.lineTo(cx + 5, cy + 5);
    ctx.moveTo(cx + 5, cy - 5); ctx.lineTo(cx - 5, cy + 5); ctx.stroke();
    return;
  }
  const a = -ac.hdg; // screen y is flipped
  const s = 7;
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
      const frameIdx = Math.min(matchIdx, maxLen - 1);
      matchEpisodes.forEach((ep, i) => {
        const cv = $("mc" + i);
        if (!cv || !ep || !ep.frames || !ep.frames.length) return;
        drawFrame(cv.getContext("2d"), ep.frames[frameIdx], cv.width, cv.height);
      });
      if (matchIdx < maxLen - 1) {
        matchIdx++;
        matchHold = 0;
      } else {
        matchHold++;
        if (matchHold > MATCH_HOLD_FRAMES) {
          matchIdx = 0;
          matchHold = 0;
        }
      }
    }
  }
  if (replayBuffer.length) {
    const cv = $("replay");
    if (cv) {
      const idx = Math.min(replayIdx, replayBuffer.length - 1);
      drawFrame(cv.getContext("2d"), replayBuffer[idx], cv.width, cv.height);
      if (replayIdx < replayBuffer.length - 1) replayIdx++;
    }
  }
  setTimeout(tickVisuals, 90);
}

function drawCurve() {
  drawLearningCurve("curve", history);
}

// --------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  connectWS();
  tickVisuals();
  $("save-btn").onclick = saveConfig;
  $("train-btn").onclick = startTraining;
  $("stop-btn").onclick = stopJob;
  $("replay-btn").onclick = startReplay;
  $("analysis-btn").onclick = runAnalysis;
});
