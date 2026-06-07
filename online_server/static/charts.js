"use strict";

function formatSteps(t) {
  if (t >= 1e6) return (t / 1e6).toFixed(1) + "M";
  if (t >= 1e3) return Math.round(t / 1e3) + "k";
  return String(Math.round(t));
}

/** Dual-axis learning curve: train reward (left) + avg score 0..1 (right). */
function drawLearningCurve(canvas, history, opts) {
  opts = opts || {};
  const cv = typeof canvas === "string" ? document.getElementById(canvas) : canvas;
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const w = cv.width;
  const h = cv.height;
  const padL = 54;
  const padR = 46;
  const padT = 28;
  const padB = 38;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.fillStyle = opts.bg || "#0d1322";
  ctx.fillRect(0, 0, w, h);

  const ts = history.t || [];
  const reward = history.reward || [];
  const score = history.score || history.mission || history.winrate || [];
  if (ts.length < 2) {
    ctx.fillStyle = "#8b97ad";
    ctx.font = "12px system-ui, Segoe UI, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Waiting for training data…", padL, padT + 18);
    return;
  }

  const n = ts.length;
  const tMin = ts[0];
  const tMax = ts[n - 1];
  const rMin = Math.min(...reward);
  const rMax = Math.max(...reward);
  const rSpan = (rMax - rMin) || 1;
  const X = (i) => padL + (i / (n - 1)) * plotW;
  const yReward = (v) => padT + plotH - ((v - rMin) / rSpan) * plotH;
  const yScore = (v) => padT + plotH - Math.max(0, Math.min(1, v)) * plotH;

  ctx.strokeStyle = "#1a2438";
  ctx.lineWidth = 1;
  const gridY = 5;
  for (let g = 0; g <= gridY; g++) {
    const y = padT + (g / gridY) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
  }
  const gridX = Math.min(6, Math.max(2, n - 1));
  for (let g = 0; g <= gridX; g++) {
    const x = padL + (g / gridX) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
  }

  ctx.strokeStyle = "#3d4a63";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(padL + plotW, padT);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  ctx.font = "10px system-ui, Segoe UI, sans-serif";
  ctx.fillStyle = "#8b97ad";
  ctx.textAlign = "right";
  for (let g = 0; g <= gridY; g++) {
    const y = padT + (g / gridY) * plotH;
    const val = rMax - (g / gridY) * rSpan;
    ctx.fillText(val.toFixed(1), padL - 8, y + 3);
  }
  ctx.textAlign = "left";
  for (let g = 0; g <= gridX; g++) {
    const x = padL + (g / gridX) * plotW;
    const t = tMin + (g / gridX) * (tMax - tMin);
    ctx.textAlign = "center";
    ctx.fillText(formatSteps(t), x, padT + plotH + 16);
  }
  ctx.textAlign = "right";
  for (let g = 0; g <= gridY; g++) {
    const y = padT + (g / gridY) * plotH;
    const pct = Math.round((1 - g / gridY) * 100);
    ctx.fillText(pct + "%", w - 6, y + 3);
  }

  ctx.save();
  ctx.translate(14, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = "#4c9be8";
  ctx.font = "11px system-ui, Segoe UI, sans-serif";
  ctx.fillText("train reward", 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(w - 10, padT + plotH / 2);
  ctx.rotate(Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = "#5ec27a";
  ctx.fillText("avg score", 0, 0);
  ctx.restore();
  ctx.textAlign = "center";
  ctx.fillStyle = "#6b7890";
  ctx.fillText("timesteps", padL + plotW / 2, h - 8);

  ctx.lineWidth = 2.25;
  ctx.strokeStyle = "#4c9be8";
  ctx.beginPath();
  reward.forEach((v, i) => (i ? ctx.lineTo(X(i), yReward(v)) : ctx.moveTo(X(i), yReward(v))));
  ctx.stroke();

  ctx.strokeStyle = "#5ec27a";
  ctx.beginPath();
  score.forEach((v, i) => (i ? ctx.lineTo(X(i), yScore(v)) : ctx.moveTo(X(i), yScore(v))));
  ctx.stroke();

  ctx.font = "11px system-ui, Segoe UI, sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = "#4c9be8";
  ctx.fillRect(padL, 8, 10, 10);
  ctx.fillStyle = "#c8d4e8";
  ctx.fillText("train reward", padL + 14, 17);
  ctx.fillStyle = "#5ec27a";
  ctx.fillRect(padL + 100, 8, 10, 10);
  ctx.fillStyle = "#c8d4e8";
  ctx.fillText("avg score", padL + 114, 17);
}
