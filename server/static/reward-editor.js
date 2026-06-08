"use strict";

/** Compact reward-term UI: slider + number, sign = reward vs penalty. */
const RewardEditor = (() => {
  const $ = (id) => document.getElementById(id);

  const EVENT_TERMS = [
    "mission_completed", "hit_enemy", "was_hit", "fire_missile", "miss_missile",
  ];
  const SHAPING_TERMS = [
    "mission_shaping", "maintain_track", "lost_track", "closing_bonus", "wez_advantage",
  ];

  const TERM_LABELS = {
    global_scale: "Global scale",
    mission_completed: "Mission done",
    hit_enemy: "Hit enemy",
    was_hit: "Was hit",
    fire_missile: "Fire missile",
    miss_missile: "Miss missile",
    mission_shaping: "Mission shaping",
    maintain_track: "Maintain track",
    lost_track: "Lost track",
    closing_bonus: "Closing bonus",
    wez_advantage: "WEZ advantage",
  };

  const DEFAULT_HELP = {
    global_scale: "Multiplies every term.",
    mission_completed: "Reach the goal area alive.",
    hit_enemy: "Shoot down the enemy.",
    was_hit: "Negative penalizes; positive rewards getting hit.",
    fire_missile: "Negative = launch cost; positive = firing bonus.",
    miss_missile: "Negative = miss penalty; positive = miss bonus.",
    mission_shaping: "Per-step progress toward the goal.",
    maintain_track: "Per-step while enemy on radar.",
    lost_track: "Negative when lock lost; positive when lock lost.",
    closing_bonus: "Per-step for closing on the enemy.",
    wez_advantage: "Per-step when you have WEZ advantage.",
  };

  const WIDE = { min: -100, max: 100, step: 0.5 };
  const WIDE_SM = { min: -10, max: 10, step: 0.01 };
  const DEFAULT_RANGES = {
    global_scale: { min: 0, max: 10, step: 0.1 },
    mission_completed: WIDE,
    hit_enemy: WIDE,
    was_hit: WIDE,
    fire_missile: { min: -10, max: 10, step: 0.05 },
    miss_missile: { min: -10, max: 10, step: 0.1 },
    mission_shaping: WIDE_SM,
    maintain_track: { min: -1, max: 1, step: 0.001 },
    lost_track: { min: -10, max: 10, step: 0.05 },
    closing_bonus: WIDE_SM,
    wez_advantage: WIDE_SM,
  };

  function termLabel(term) {
    return TERM_LABELS[term] || term.replace(/_/g, " ");
  }

  function termKind(term, val) {
    if (term === "global_scale") return "scale";
    if (val === 0) return "off";
    return val < 0 ? "penalty" : "reward";
  }

  function decimalsForStep(step) {
    if (!step || step >= 1) return 0;
    return Math.min(4, Math.ceil(-Math.log10(step)));
  }

  function formatVal(v, step) {
    return Number(Number(v).toFixed(decimalsForStep(step)));
  }

  function clamp(v, spec) {
    return Math.max(spec.min, Math.min(spec.max, v));
  }

  function normalizeEditor(api, savedValues) {
    const values = savedValues || (api && api.defaults) || {};
    return {
      start_zero: !!(api && api.start_zero),
      defaults: (api && api.defaults) || values,
      configured_defaults: (api && api.configured_defaults) || (api && api.defaults) || values,
      ranges: (api && api.ranges) || DEFAULT_RANGES,
      help: { ...DEFAULT_HELP, ...((api && api.help) || {}) },
      terms: (api && api.terms) || { event: EVENT_TERMS, shaping: SHAPING_TERMS },
    };
  }

  function renderAll(editor, values) {
    const ed = normalizeEditor(editor, values);
    const vals = values || ed.defaults;
    renderGlobalScale("global-scale-wrap", ed, vals);
    renderTerms("event-terms", ed.terms.event || EVENT_TERMS, ed, vals);
    renderTerms("shaping-terms", ed.terms.shaping || SHAPING_TERMS, ed, vals);
    return ed;
  }

  function renderGlobalScale(containerId, editor, values) {
    const wrap = $(containerId);
    if (!wrap) return;
    const ed = normalizeEditor(editor, values);
    const term = "global_scale";
    const spec = ed.ranges[term] || DEFAULT_RANGES[term];
    const val = (values && values[term]) ?? ed.defaults[term] ?? 0;
    wrap.innerHTML = "";
    wrap.className = "rw-scale-wrap";
    wrap.appendChild(buildRow(term, val, spec, ed.help[term]));
  }

  function renderTerms(containerId, terms, editor, values) {
    const c = $(containerId);
    if (!c || !terms) return;
    const ed = normalizeEditor(editor, values);
    c.innerHTML = "";
    c.className = "rw-list";
    terms.forEach((term) => {
      const spec = ed.ranges[term] || DEFAULT_RANGES[term] || { min: -100, max: 100, step: 0.1 };
      const val = (values && values[term]) ?? ed.defaults[term] ?? 0;
      c.appendChild(buildRow(term, val, spec, ed.help[term]));
    });
  }

  function buildRow(term, val, spec, helpText) {
    const v0 = formatVal(val, spec.step);
    const row = document.createElement("article");
    row.className = `rw-row rw-row--${termKind(term, v0)}`;
    row.dataset.term = term;

    const label = document.createElement("label");
    label.className = "rw-label";
    label.textContent = termLabel(term);
    label.title = helpText || term;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "rw-slider";
    slider.min = spec.min;
    slider.max = spec.max;
    slider.step = spec.step ?? 0.1;
    slider.value = v0;
    slider.title = helpText || term;

    const number = document.createElement("input");
    number.type = "number";
    number.className = "rw-number";
    number.step = spec.step ?? 0.1;
    number.min = spec.min;
    number.max = spec.max;
    number.value = v0;
    number.title = helpText || term;
    if (term === "global_scale") number.id = "global_scale";
    else number.dataset.term = term;

    function refreshVisuals() {
      const v = formatVal(number.value, spec.step);
      row.className = `rw-row rw-row--${termKind(term, v)}`;
    }

    slider.addEventListener("input", () => {
      number.value = formatVal(slider.value, spec.step);
      refreshVisuals();
    });
    number.addEventListener("input", () => {
      const v = formatVal(clamp(Number(number.value), spec), spec.step);
      number.value = v;
      slider.value = v;
      refreshVisuals();
    });
    number.addEventListener("change", () => {
      const v = formatVal(clamp(Number(number.value), spec), spec.step);
      number.value = v;
      slider.value = v;
      refreshVisuals();
    });

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(number);
    return row;
  }

  function applyDefaults(editor) {
    const ed = normalizeEditor(editor);
    document.querySelectorAll(".rw-row").forEach((row) => {
      const term = row.dataset.term;
      const def = ed.defaults[term];
      if (def === undefined) return;
      const num = row.querySelector(".rw-number");
      const slider = row.querySelector(".rw-slider");
      if (!num || !slider) return;
      const spec = ed.ranges[term] || DEFAULT_RANGES[term] || { step: 0.1 };
      const v = formatVal(def, spec.step);
      num.value = v;
      slider.value = v;
      num.dispatchEvent(new Event("input"));
    });
  }

  function gather() {
    const gs = $("global_scale");
    const rewards = { global_scale: gs ? Number(gs.value) : 1.0 };
    document.querySelectorAll(".rw-number[data-term]").forEach((inp) => {
      rewards[inp.dataset.term] = Number(inp.value);
    });
    return rewards;
  }

  function adminKind(term, defaults) {
    const v = defaults[term] ?? 0;
    return termKind(term, v);
  }

  function renderAdminTable(containerId, editor, codeDefaults, codeRanges) {
    const box = $(containerId);
    if (!box) return;
    const ed = normalizeEditor(editor);
    const defaults = ed.configured_defaults || ed.defaults;
    const ranges = ed.ranges;
    const terms = ["global_scale", ...(ed.terms.event || EVENT_TERMS), ...(ed.terms.shaping || SHAPING_TERMS)];
    let html = `<table class="admin-reward-table"><thead><tr>
      <th>Term</th><th>Default</th><th>Min</th><th>Max</th><th>Step</th>
    </tr></thead><tbody>`;
    terms.forEach((term) => {
      const d = defaults[term] ?? (codeDefaults && codeDefaults[term]) ?? 0;
      const r = ranges[term] || (codeRanges && codeRanges[term]) || DEFAULT_RANGES[term] || { min: -100, max: 100, step: 0.1 };
      const kind = adminKind(term, defaults);
      html += `<tr class="rw-admin-row rw-admin-row--${kind}">
        <td class="name"><span>${termLabel(term)}</span><code>${term}</code></td>
        <td><input type="number" step="any" id="ad-def-${term}" value="${d}" /></td>
        <td><input type="number" step="any" id="ad-min-${term}" value="${r.min}" /></td>
        <td><input type="number" step="any" id="ad-max-${term}" value="${r.max}" /></td>
        <td><input type="number" step="any" id="ad-step-${term}" value="${r.step}" /></td>
      </tr>`;
    });
    html += "</tbody></table>";
    box.innerHTML = html;
  }

  function gatherAdminRewardConfig(termLists) {
    const defaults = {};
    const ranges = {};
    const terms = ["global_scale", ...(termLists.event || EVENT_TERMS), ...(termLists.shaping || SHAPING_TERMS)];
    terms.forEach((term) => {
      const d = document.getElementById(`ad-def-${term}`);
      const mn = document.getElementById(`ad-min-${term}`);
      const mx = document.getElementById(`ad-max-${term}`);
      const st = document.getElementById(`ad-step-${term}`);
      if (d) defaults[term] = Number(d.value);
      if (mn && mx && st) {
        ranges[term] = { min: Number(mn.value), max: Number(mx.value), step: Number(st.value) };
      }
    });
    return { defaults, ranges };
  }

  return {
    normalizeEditor, renderAll, renderGlobalScale, renderTerms,
    applyDefaults, gather, renderAdminTable, gatherAdminRewardConfig,
  };
})();
