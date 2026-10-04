"use strict";

const $ = (selector) => document.querySelector(selector);
const dom = {
  event: $("#event-select"), eventCard: $("#event-card"), daily: $("#daily-button"), hourly: $("#hourly-button"),
  coverage: $("#coverage-copy"), before: $("#before-range"), after: $("#after-range"), beforeValue: $("#before-value"),
  afterValue: $("#after-value"), windowTotal: $("#window-total"), canvas: $("#network-canvas"), graphWrap: $("#graph-wrap"),
  graphEmpty: $("#graph-empty"), tooltip: $("#node-tooltip"), play: $("#play-button"), playIcon: $("#play-icon"),
  stateLabel: $("#play-state-label"), stepLabel: $("#play-step-label"), timeline: $("#timeline-range"),
  frameDate: $("#frame-date"), assetCount: $("#asset-count"), prev: $("#prev-button"), next: $("#next-button"),
  speed: $("#speed-select"), activity: $("#activity-canvas"), chartEmpty: $("#chart-empty"),
  assetTable: $("#asset-table"), search: $("#asset-search"), sector: $("#sector-filter"), toast: $("#toast"),
  source: $("#source-link"), method: $("#method-copy"), dialog: $("#method-dialog"),
};

const palette = ["#55dfc0", "#78a9ff", "#f5c66a", "#c59aff", "#ff8295", "#67ddeb", "#a7d875", "#fda4af", "#ffa972", "#a5b4fc"];
const sectorColors = new Map();
const state = { events: [], event: null, resolution: null, replay: null, frame: 0, playing: false, timer: null, speed: 1, selected: null, hovered: null, toastTimer: null };
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function sectorColor(sector) {
  if (!sectorColors.has(sector)) sectorColors.set(sector, palette[sectorColors.size % palette.length]);
  return sectorColors.get(sector);
}

function showError(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => dom.toast.classList.remove("show"), 5200);
}

function formatTimestamp(value, options = {}) {
  const date = new Date(value);
  return new Intl.DateTimeFormat("en", { timeZone: "UTC", month: "short", day: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false, ...options }).format(date) + " UTC";
}

function formatDuration(hours) {
  if (hours === null || hours === undefined || !Number.isFinite(Number(hours))) return "Not observed";
  const n = Number(hours);
  if (n < 24) return `${n} h`;
  const days = n / 24;
  return Number.isInteger(days) ? `${days} d` : `${days.toFixed(1)} d`;
}

function eventAvailable(event) {
  return ["hourly", "daily"].some((resolution) => event.available?.[resolution]);
}

function fillEventOptions() {
  dom.event.replaceChildren();
  state.events.forEach((event, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    const available = [event.available?.hourly && "1h", event.available?.daily && "1d"].filter(Boolean).join(" · ");
    option.textContent = `${event.date}  /  ${event.name}${available ? `  ·  ${available}` : "  ·  unavailable"}`;
    option.disabled = !available;
    dom.event.append(option);
  });
  let preferred = -1, fallback = -1;
  state.events.forEach((event, index) => {
    if (eventAvailable(event)) fallback = index;
    if (eventAvailable(event) && event.is_crisis) preferred = index;
  });
  const index = preferred >= 0 ? preferred : fallback;
  if (index < 0) {
    showError("No event date overlaps the available processed datasets.");
    return;
  }
  dom.event.value = String(index);
  chooseEvent(index);
}

function chooseEvent(index) {
  stopPlayback();
  state.event = state.events[index];
  state.selected = null;
  renderEventCard();
  const canHourly = !!state.event.available?.hourly;
  const canDaily = !!state.event.available?.daily;
  dom.hourly.disabled = !canHourly;
  dom.daily.disabled = !canDaily;
  const preferred = canHourly ? "hourly" : canDaily ? "daily" : null;
  if (!preferred) {
    state.resolution = null;
    showError("This event is outside the coverage of the available datasets.");
    return;
  }
  state.resolution = preferred;
  updateResolutionButtons();
  loadReplay();
}

function renderEventCard() {
  const event = state.event;
  if (!event) return;
  dom.eventCard.classList.remove("skeleton");
  const date = document.createElement("div");
  date.className = "event-date";
  date.textContent = `${event.date}  ·  ${event.is_crisis ? "MARKET STRESS" : "HISTORICAL EVENT"}`;
  const title = document.createElement("div");
  title.className = "event-title";
  title.textContent = event.name;
  const description = document.createElement("p");
  description.textContent = event.description || "A catalogued historical market event.";
  dom.eventCard.replaceChildren(date, title, description);
  if (event.source_url?.startsWith("https://")) {
    const link = document.createElement("a");
    link.className = "event-link";
    link.textContent = "SOURCE ↗";
    link.href = event.source_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    dom.eventCard.append(link);
    dom.source.href = event.source_url;
    dom.source.style.visibility = "visible";
  } else {
    dom.source.removeAttribute("href");
    dom.source.style.visibility = "hidden";
  }
}

function updateResolutionButtons() {
  dom.daily.classList.toggle("selected", state.resolution === "daily");
  dom.hourly.classList.toggle("selected", state.resolution === "hourly");
}

function updateWindowLabels() {
  const before = Number(dom.before.value);
  const after = Number(dom.after.value);
  dom.beforeValue.textContent = `${before} ${before === 1 ? "day" : "days"}`;
  dom.afterValue.textContent = `${after} ${after === 1 ? "day" : "days"}`;
  dom.windowTotal.textContent = `${before + after} DAYS`;
}

async function loadReplay() {
  if (!state.event || !state.resolution) return;
  stopPlayback();
  dom.graphEmpty.textContent = "Building event replay…";
  dom.graphEmpty.style.display = "grid";
  dom.chartEmpty.style.display = "grid";
  dom.coverage.textContent = "Preparing the selected event window…";
  updateWindowLabels();
  const query = new URLSearchParams({
    event: String(state.events.indexOf(state.event)), resolution: state.resolution,
    before: String(Number(dom.before.value) * 24), after: String(Number(dom.after.value) * 24),
  });
  try {
    const response = await fetch(`/api/replay?${query}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not build replay.");
    state.replay = payload;
    state.frame = Math.max(0, payload.frames.findIndex((frame) => frame.index === payload.anchor_index));
    dom.coverage.textContent = `${payload.coverage.start.slice(0, 10)} — ${payload.coverage.end.slice(0, 10)} · ${payload.nodes.length} assets`;
    dom.assetCount.textContent = `${payload.nodes.length} ASSETS  ·  ${payload.edges.length} PRIOR EDGES`;
    dom.graphEmpty.style.display = "none";
    dom.chartEmpty.style.display = "none";
    dom.timeline.min = "0";
    dom.timeline.max = String(Math.max(0, payload.frames.length - 1));
    dom.timeline.value = String(state.frame);
    const firstFrame = payload.frames[0];
    const lastFrame = payload.frames[payload.frames.length - 1];
    $("#scrubber-start").textContent = new Date(firstFrame.timestamp).toISOString().slice(5, 16).replace("T", " ") + " UTC";
    $("#scrubber-end").textContent = new Date(lastFrame.timestamp).toISOString().slice(5, 16).replace("T", " ") + " UTC";
    $("#scrubber-anchor").textContent = `EVENT · ${payload.event.date}`;
    $("#chart-window-label").textContent = `${state.resolution.toUpperCase()} EVENT WINDOW`;
    $("#assets-subtitle").textContent = `Event-active assets with post-event shock observations · ${payload.frames.length} replay bars`;
    updateSectorFilter();
    updateFrame();
    renderMetrics();
    renderTable();
    renderIntelligence();
    dom.method.textContent = payload.method + " Event timing and correlation are descriptive; neither establishes causal transmission.";
  } catch (error) {
    state.replay = null;
    dom.graphEmpty.textContent = error.message;
    dom.graphEmpty.style.display = "grid";
    dom.coverage.textContent = "Replay unavailable";
    showError(error.message);
  }
}

function updateFrame() {
  if (!state.replay) return;
  const frame = state.replay.frames[state.frame];
  if (!frame) return;
  dom.timeline.value = String(state.frame);
  dom.frameDate.textContent = formatTimestamp(frame.timestamp);
  dom.stepLabel.textContent = `${frame.offset_hours >= 0 ? "+" : ""}${frame.offset_hours} HOURS FROM EVENT`;
  dom.stateLabel.textContent = state.playing ? "REPLAY IN MOTION" : "INSPECTING REPLAY";
  dom.playIcon.textContent = state.playing ? "Ⅱ" : "▶";
  dom.play.setAttribute("aria-label", state.playing ? "Pause replay" : "Play replay");
  const shockedCount = frame.shocked.length;
  $("#metric-current").textContent = `${shockedCount} / ${frame.active.length}`;
  $("#metric-current-note").textContent = `${frame.active.length} assets active · ${formatTimestamp(frame.timestamp, { weekday: "short" })}`;
  drawNetwork();
  drawActivity();
  if (state.selected) renderSelectedAssetDetail();
}

function renderMetrics() {
  const metrics = state.replay.metrics;
  $("#metric-reach").textContent = `${Math.round(metrics.reach_fraction * 100)}%`;
  $("#metric-reach-note").textContent = `${metrics.affected_assets} of ${metrics.event_active_assets} assets active at event later registered a shock`;
  $("#reach-track").style.width = `${Math.min(100, Math.max(0, metrics.reach_fraction * 100))}%`;
  $("#metric-lag").textContent = metrics.first_to_next_shock_hours === null ? "—" : formatDuration(metrics.first_to_next_shock_hours);
  $("#metric-lag-note").textContent = metrics.first_to_next_shock_hours === null ? "No two distinct post-event shock times observed" : "gap between first two distinct post-event shock times";
  $("#metric-duration").textContent = metrics.median_contiguous_duration_hours === null ? "—" : formatDuration(metrics.median_contiguous_duration_hours);
  $("#metric-duration-note").textContent = metrics.median_contiguous_duration_hours === null ? "No post-event shock runs observed" : "median across affected assets active at the event";
}

function graphLayout() {
  const canvas = dom.canvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width, height = rect.height;
  const radius = Math.max(65, Math.min(width, height) * 0.37);
  const center = { x: width / 2, y: height / 2 + 2 };
  const points = state.replay.nodes.map((node) => ({
    x: center.x + node.x * radius,
    y: center.y + node.y * radius,
    node,
  }));
  return { ctx, width, height, points };
}

function drawNetwork() {
  if (!state.replay) return;
  const { ctx, width, height, points } = graphLayout();
  ctx.clearRect(0, 0, width, height);
  const current = state.replay.frames[state.frame];
  const active = new Set(current.active);
  const shocked = new Set(current.shocked);
  const seen = new Set();
  for (let i = 0; i <= state.frame; i++) state.replay.frames[i].shocked.forEach((node) => seen.add(node));

  // Edges are deliberately undirected: they represent pre-event correlation only.
  ctx.lineWidth = 1;
  for (const edge of state.replay.edges) {
    const a = points[edge.source], b = points[edge.target];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = `rgba(120, 151, 176, ${0.06 + edge.strength * 0.19})`;
    ctx.lineWidth = 0.65 + edge.strength * 0.8;
    ctx.stroke();
  }

  for (const point of points) {
    const node = point.node;
    const isActive = active.has(node.id);
    const isNow = shocked.has(node.id);
    const isPast = !isNow && seen.has(node.id);
    const isSelected = state.selected === node.asset;
    const base = Math.abs(current.returns_pct[node.id] || 0);
    const r = (isNow ? 8.5 : 6.5) + Math.min(3.6, base * 0.12);
    const color = !isActive ? "#4b5b6d" : isNow ? "#fb5668" : isPast ? "#f3b54b" : "#60a5fa";
    ctx.globalAlpha = isActive ? 1 : 0.36;
    if (isNow && !reducedMotion) {
      const glow = ctx.createRadialGradient(point.x, point.y, r * 0.2, point.x, point.y, r * 2.6);
      glow.addColorStop(0, "rgba(251,86,104,.24)");
      glow.addColorStop(1, "rgba(251,86,104,0)");
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(point.x, point.y, r * 2.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(point.x, point.y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = isNow ? "#fb5668" : "transparent";
    ctx.shadowBlur = isNow ? 13 : 0;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = isSelected ? 2.5 : 1.2;
    ctx.strokeStyle = isSelected ? "#ffe08a" : "#0a1220";
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (width > 540 || points.length <= 20) {
      ctx.font = `${isSelected ? 700 : 500} 9px "DM Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = isActive ? "#c5d4e3" : "#64768a";
      ctx.fillText(node.asset, point.x, point.y + r + 5);
    }
  }

  ctx.save();
  ctx.textAlign = "center";
  ctx.font = "500 8px 'DM Mono', monospace";
  const sectors = [...new Set(state.replay.nodes.map((node) => node.sector))];
  sectors.forEach((sector, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / sectors.length;
    const x = width / 2 + Math.cos(angle) * (Math.min(width, height) * 0.47);
    const y = height / 2 + Math.sin(angle) * (Math.min(width, height) * 0.47);
    ctx.fillStyle = sectorColor(sector) + "99";
    if (width > 460) ctx.fillText(sector.toUpperCase(), x, y);
  });
  ctx.restore();
  state.hitPoints = points;
}

function drawActivity() {
  if (!state.replay) return;
  const canvas = dom.activity;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const counts = state.replay.frames.map((frame) => frame.shocked.length);
  const max = Math.max(1, ...counts);
  const pad = { left: 28, right: 7, top: 10, bottom: 23 };
  const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
  ctx.font = "8px 'DM Mono', monospace";
  ctx.textAlign = "right";
  for (let tick = 0; tick <= 3; tick++) {
    const y = pad.top + (plotH * tick) / 3;
    const label = Math.round(max * (1 - tick / 3));
    ctx.strokeStyle = "rgba(110,132,158,.13)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    ctx.fillStyle = "#6c7e93"; ctx.fillText(String(label), pad.left - 7, y + 3);
  }
  const xAt = (i) => pad.left + (counts.length <= 1 ? 0 : (i / (counts.length - 1)) * plotW);
  const yAt = (value) => pad.top + plotH - (value / max) * plotH;
  const anchor = state.replay.frames.findIndex((frame) => frame.index === state.replay.anchor_index);
  if (anchor >= 0) {
    ctx.save(); ctx.setLineDash([3, 4]); ctx.strokeStyle = "#fb7182";
    ctx.beginPath(); ctx.moveTo(xAt(anchor), pad.top); ctx.lineTo(xAt(anchor), h - pad.bottom); ctx.stroke(); ctx.restore();
  }
  const fill = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
  fill.addColorStop(0, "rgba(85,223,192,.24)"); fill.addColorStop(1, "rgba(85,223,192,0)");
  ctx.beginPath();
  counts.forEach((count, i) => { const x = xAt(i), y = yAt(count); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.lineTo(xAt(counts.length - 1), h - pad.bottom); ctx.lineTo(xAt(0), h - pad.bottom); ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  ctx.beginPath();
  counts.forEach((count, i) => { const x = xAt(i), y = yAt(count); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = "#55dfc0"; ctx.lineWidth = 2; ctx.stroke();
  const x = xAt(state.frame), y = yAt(counts[state.frame] || 0);
  ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = "#d9fff5"; ctx.fill();
  ctx.strokeStyle = "#41c8ac"; ctx.lineWidth = 2; ctx.stroke();
  const first = new Date(state.replay.frames[0].timestamp);
  const last = new Date(state.replay.frames[state.replay.frames.length - 1].timestamp);
  ctx.textAlign = "left"; ctx.fillStyle = "#71849a";
  ctx.fillText(first.toISOString().slice(5, 10), pad.left, h - 5);
  ctx.textAlign = "right"; ctx.fillText(last.toISOString().slice(5, 10), w - pad.right, h - 5);
}

function updateSectorFilter() {
  const current = dom.sector.value;
  const sectors = [...new Set(state.replay.nodes.map((node) => node.sector))].sort();
  dom.sector.replaceChildren(new Option("All sectors", "all"));
  sectors.forEach((sector) => dom.sector.add(new Option(sector, sector)));
  if (sectors.includes(current)) dom.sector.value = current;
}

function renderTable() {
  if (!state.replay) return;
  const needle = dom.search.value.trim().toLowerCase();
  const selectedSector = dom.sector.value;
  const rows = state.replay.nodes
    .filter((node) => node.active_at_event && node.first_shock_offset_hours !== null)
    .filter((node) => selectedSector === "all" || node.sector === selectedSector)
    .filter((node) => !needle || node.asset.toLowerCase().includes(needle) || node.sector.toLowerCase().includes(needle))
    .sort((a, b) => a.first_shock_offset_hours - b.first_shock_offset_hours || b.contiguous_duration_hours - a.contiguous_duration_hours);
  dom.assetTable.replaceChildren();
  if (!rows.length) {
    const tr = document.createElement("tr"), td = document.createElement("td");
    td.colSpan = 6; td.className = "empty-table"; td.textContent = "No matching post-event shock observations in this replay window.";
    tr.append(td); dom.assetTable.append(tr); return;
  }
  for (const node of rows) {
    const tr = document.createElement("tr");
    tr.classList.toggle("row-selected", state.selected === node.asset);
    tr.addEventListener("click", () => { state.selected = state.selected === node.asset ? null : node.asset; drawNetwork(); renderTable(); renderSelectedAssetDetail(); });
    const offset = node.first_shock_offset_hours;
    const frame = state.replay.frames.find((item) => item.offset_hours === offset);
    const absReturns = state.replay.frames.map((item) => Math.abs(item.returns_pct[node.id] || 0));
    const peak = Math.max(0, ...absReturns);
    const cells = [
      `<span class="asset-cell"><span class="asset-icon">${escapeHtml(node.asset.slice(0, 3))}</span>${escapeHtml(node.asset)}</span>`,
      `<span class="sector-pill">${escapeHtml(node.sector)}</span>`,
      frame ? escapeHtml(formatTimestamp(frame.timestamp, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })) : "—",
      `<span class="delay-cell">+${escapeHtml(formatDuration(offset))}</span>`,
      escapeHtml(formatDuration(node.contiguous_duration_hours)),
      `<span class="${frame && frame.returns_pct[node.id] < 0 ? "return-negative" : "return-positive"}">${peak.toFixed(2)}%</span>`,
    ];
    cells.forEach((html) => { const td = document.createElement("td"); td.innerHTML = html; tr.append(td); });
    dom.assetTable.append(tr);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

function renderIntelligence() {
  $("#assets-subtitle").textContent = `Assets that registered a shock after the event anchor · ${state.replay.frames.length} replay bars`;
  const event = state.replay.event;
  $("#impact-event-title").textContent = `${event.date} · ${event.name}`;
  const type = event.is_crisis ? "Market stress event" : "Historical market event";
  const body = document.createElement("div");
  body.className = "intelligence-body";
  const headline = document.createElement("div"); headline.className = "intel-headline"; headline.textContent = event.name;
  const copy = document.createElement("p"); copy.className = "intel-copy"; copy.textContent = `${type} recorded on ${event.date}. The event date is the replay anchor; metrics summarize detected post-anchor shock labels in the selected window.`;
  const grid = document.createElement("div"); grid.className = "intel-grid";
  const vals = [
    ["OBSERVATION", state.resolution === "hourly" ? "Hourly candle close" : "Daily bar"],
    ["EVENT WINDOW", `${dom.before.value}d before · ${dom.after.value}d after`],
    ["RELATIONSHIPS", `${state.replay.edges.length} prior correlations`],
    ["INTERPRETATION", "Descriptive, not causal"],
  ];
  vals.forEach(([label, value]) => { const cell = document.createElement("div"); cell.className = "intel-cell"; const span = document.createElement("span"); span.textContent = label; const strong = document.createElement("strong"); strong.textContent = value; cell.append(span, strong); grid.append(cell); });
  body.append(headline, copy, grid);
  const container = $("#event-intelligence"); container.replaceChildren(body);
}

function renderSelectedAssetDetail() {
  if (!state.selected || !state.replay) return;
  const node = state.replay.nodes.find((item) => item.asset === state.selected);
  if (!node) return;
  const current = state.replay.frames[state.frame];
  const returnPct = current.returns_pct[node.id] || 0;
  const status = !current.active.includes(node.id) ? "Inactive / unavailable" : current.shocked.includes(node.id) ? "Shock detected now" : "No shock at this bar";
  const info = `Selected: ${node.asset} · ${node.sector} · ${status} · log return ${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(3)}%`;
  $("#assets-subtitle").textContent = info;
}

function advanceFrame(step) {
  if (!state.replay) return;
  const length = state.replay.frames.length;
  state.frame = (state.frame + step + length) % length;
  updateFrame();
}

function stopPlayback() {
  state.playing = false;
  clearInterval(state.timer);
  state.timer = null;
  if (state.replay) updateFrame();
}

function togglePlayback() {
  if (!state.replay || state.replay.frames.length < 2) return;
  if (state.playing) { stopPlayback(); return; }
  if (state.frame >= state.replay.frames.length - 1) state.frame = 0;
  state.playing = true;
  updateFrame();
  const interval = reducedMotion ? 650 : Math.max(45, 360 / state.speed);
  state.timer = setInterval(() => {
    if (!state.replay || state.frame >= state.replay.frames.length - 1) { stopPlayback(); return; }
    state.frame += 1;
    updateFrame();
  }, interval);
}

function pointerMove(event) {
  if (!state.replay || !state.hitPoints) return;
  const rect = dom.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let nearest = null, distance = Infinity;
  for (const point of state.hitPoints) {
    const d = Math.hypot(point.x - x, point.y - y);
    if (d < distance) { nearest = point; distance = d; }
  }
  if (!nearest || distance > 19) { dom.tooltip.style.display = "none"; state.hovered = null; return; }
  const node = nearest.node, frame = state.replay.frames[state.frame];
  const status = !frame.active.includes(node.id) ? "Inactive / unavailable" : frame.shocked.includes(node.id) ? "Shock detected now" : node.first_shock_offset_hours !== null && node.first_shock_offset_hours < frame.offset_hours ? "Shocked earlier" : "Not yet shocked";
  const value = frame.returns_pct[node.id] || 0;
  dom.tooltip.innerHTML = `<strong>${escapeHtml(node.asset)}</strong><br><span>${escapeHtml(node.sector)} · ${escapeHtml(status)}<br>Log return ${value >= 0 ? "+" : ""}${value.toFixed(3)}%</span>`;
  dom.tooltip.style.display = "block";
  dom.tooltip.style.left = `${Math.min(rect.width - 170, Math.max(8, nearest.x + 13))}px`;
  dom.tooltip.style.top = `${Math.max(8, nearest.y - 52)}px`;
}

function initInteractions() {
  dom.event.addEventListener("change", () => chooseEvent(Number(dom.event.value)));
  dom.daily.addEventListener("click", () => { if (state.event?.available?.daily) { state.resolution = "daily"; updateResolutionButtons(); loadReplay(); } });
  dom.hourly.addEventListener("click", () => { if (state.event?.available?.hourly) { state.resolution = "hourly"; updateResolutionButtons(); loadReplay(); } });
  [dom.before, dom.after].forEach((input) => input.addEventListener("input", updateWindowLabels));
  [dom.before, dom.after].forEach((input) => input.addEventListener("change", loadReplay));
  dom.timeline.addEventListener("input", () => { if (state.playing) stopPlayback(); state.frame = Number(dom.timeline.value); updateFrame(); });
  dom.play.addEventListener("click", togglePlayback);
  dom.prev.addEventListener("click", () => { stopPlayback(); advanceFrame(-1); });
  dom.next.addEventListener("click", () => { stopPlayback(); advanceFrame(1); });
  dom.speed.addEventListener("change", () => { state.speed = Number(dom.speed.value); if (state.playing) { stopPlayback(); togglePlayback(); } });
  dom.search.addEventListener("input", renderTable);
  dom.sector.addEventListener("change", renderTable);
  document.querySelectorAll(".nav-button[data-view], .back-button[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $("#method-button").addEventListener("click", () => dom.dialog.showModal());
  dom.canvas.addEventListener("pointermove", pointerMove);
  dom.canvas.addEventListener("pointerleave", () => { dom.tooltip.style.display = "none"; });
  dom.canvas.addEventListener("click", (event) => {
    if (!state.hitPoints) return;
    const rect = dom.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const point = state.hitPoints.map((p) => ({ p, d: Math.hypot(p.x - x, p.y - y) })).sort((a, b) => a.d - b.d)[0];
    if (point?.d < 18) {
      state.selected = state.selected === point.p.node.asset ? null : point.p.node.asset;
      drawNetwork(); renderTable();
      if (state.selected) renderSelectedAssetDetail(); else renderIntelligence();
    }
  });
  new ResizeObserver(() => { drawNetwork(); drawActivity(); }).observe(dom.graphWrap);
  new ResizeObserver(drawActivity).observe(dom.activity.parentElement);
}

function setView(viewName) {
  document.querySelectorAll(".view").forEach((view) => {
    const selected = view.id === `${viewName}-view`;
    view.hidden = !selected;
    view.classList.toggle("view-active", selected);
  });
  document.querySelectorAll(".nav-button[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewName);
  });
  if (viewName === "impact" && state.replay) {
    renderTable();
    drawActivity();
  } else if (viewName === "replay") {
    requestAnimationFrame(() => { drawNetwork(); drawActivity(); });
  }
}

async function init() {
  initInteractions();
  updateWindowLabels();
  try {
    const response = await fetch("/api/events");
    if (!response.ok) throw new Error("Could not load local event catalogue.");
    state.events = await response.json();
    fillEventOptions();
  } catch (error) {
    dom.graphEmpty.textContent = error.message;
    showError(`${error.message} Start the local server with python web/server.py.`);
  }
}

init();
