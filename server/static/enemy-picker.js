"use strict";

/** Enemy cluster map: pick opponents by offense/defense profile and difficulty score. */
const EnemyPicker = (() => {
  let catalog = null;
  let selected = new Set();
  let hits = [];
  let canvas = null;
  let ctx = null;
  let onChange = null;

  function offense(params) {
    const p = params || {};
    const ag = p.aggressive ? 1.12 : 0.88;
    return (p.shot_frac ?? 0.9) * ag * (p.can_fire === false ? 0.2 : 1.0);
  }

  function defense(params) {
    const p = params || {};
    return (p.crank_frac ?? 0.9) * 0.55 + (p.break_dist ?? 30) / 90;
  }

  function scoreColor(score, minS, maxS) {
    const t = maxS > minS ? (score - minS) / (maxS - minS) : 0.5;
    const r = Math.round(80 + t * 160);
    const g = Math.round(200 - t * 130);
    return `rgb(${r},${g},70)`;
  }

  function draw() {
    if (!canvas || !ctx || !catalog) return;
    const names = catalog.names || [];
    const w = canvas.width;
    const h = canvas.height;
    const pad = { l: 36, r: 12, t: 14, b: 28 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d1322";
    ctx.fillRect(0, 0, w, h);

    const points = names.map((name) => {
      const info = (catalog.info && catalog.info[name]) || {};
      const p = info.params || {};
      return { name, info, x: offense(p), y: defense(p), score: Number(info.score ?? 5) };
    });
    if (!points.length) {
      ctx.fillStyle = "#8b97ad";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.fillText("No opponents loaded", pad.l, h / 2);
      return;
    }

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs) - 0.08;
    const maxX = Math.max(...xs) + 0.08;
    const minY = Math.min(...ys) - 0.08;
    const maxY = Math.max(...ys) + 0.08;
    const scores = points.map((p) => p.score);
    const minS = Math.min(...scores);
    const maxS = Math.max(...scores);

    const toPx = (x, y) => ({
      px: pad.l + ((x - minX) / (maxX - minX || 1)) * plotW,
      py: pad.t + plotH - ((y - minY) / (maxY - minY || 1)) * plotH,
    });

    ctx.strokeStyle = "#28324a";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, plotW, plotH);
    ctx.fillStyle = "#8b97ad";
    ctx.font = "10px Segoe UI, sans-serif";
    ctx.fillText("Defense ↑", 4, pad.t + 10);
    ctx.fillText("Offense →", pad.l, h - 8);

    hits = [];
    points.forEach((pt) => {
      const { px, py } = toPx(pt.x, pt.y);
      const on = selected.has(pt.name);
      const r = on ? 11 : 9;
      hits.push({ name: pt.name, x: px, y: py, r: r + 4 });

      ctx.beginPath();
      ctx.arc(px, py, r + 3, 0, Math.PI * 2);
      ctx.fillStyle = on ? "rgba(76,155,232,0.15)" : "rgba(0,0,0,0.2)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = scoreColor(pt.score, minS, maxS);
      ctx.globalAlpha = on ? 1 : 0.35;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = on ? "#4c9be8" : "#28324a";
      ctx.lineWidth = on ? 2.5 : 1;
      ctx.stroke();

      ctx.fillStyle = on ? "#e6ecf5" : "#8b97ad";
      ctx.font = "bold 10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(pt.name, px, py + 3);
      ctx.textAlign = "left";
    });

    updateCount();
  }

  function updateCount() {
    const el = document.getElementById("enemy-count");
    if (el && catalog) {
      el.textContent = `${selected.size} / ${(catalog.names || []).length} selected`;
    }
  }

  function toggle(name) {
    if (selected.has(name)) selected.delete(name);
    else selected.add(name);
    draw();
    if (onChange) onChange(getSelected());
  }

  function selectAll() {
    (catalog.names || []).forEach((n) => selected.add(n));
    draw();
    if (onChange) onChange(getSelected());
  }

  function selectNone() {
    selected.clear();
    draw();
    if (onChange) onChange(getSelected());
  }

  function onClick(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = (ev.clientX - rect.left) * sx;
    const y = (ev.clientY - rect.top) * sy;
    for (let i = hits.length - 1; i >= 0; i--) {
      const h = hits[i];
      const dx = x - h.x;
      const dy = y - h.y;
      if (dx * dx + dy * dy <= h.r * h.r) {
        toggle(h.name);
        return;
      }
    }
  }

  function render(containerId, cat, opts) {
    catalog = cat || { names: [], info: {} };
    opts = opts || {};
    selected = new Set(opts.selected || catalog.names || []);
    onChange = opts.onChange || null;

    const box = document.getElementById(containerId);
    if (!box) return;
    canvas = box.querySelector("canvas");
    if (!canvas) return;
    ctx = canvas.getContext("2d");

    canvas.onclick = onClick;
    const allBtn = document.getElementById("enemy-all");
    const noneBtn = document.getElementById("enemy-none");
    if (allBtn) allBtn.onclick = () => selectAll();
    if (noneBtn) noneBtn.onclick = () => selectNone();

    draw();
  }

  function getSelected() {
    return [...selected];
  }

  function setSelected(names) {
    selected = new Set(names || []);
    draw();
  }

  return { render, getSelected, setSelected, selectAll, selectNone };
})();
