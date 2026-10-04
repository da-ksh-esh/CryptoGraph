"use strict";

const researchCache = new Map();
const researchViews = {
  network: { endpoint: "/api/research/network", render: renderNetwork },
  research: { endpoint: "/api/research/spillover", render: renderSpillover },
  security: { endpoint: "/api/research/security", render: renderSecurity },
};
let forecastOptions = null;

function safe(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
  return String(value);
}

function addCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value === null || value === undefined ? "—" : String(value);
  if (className) cell.className = className;
  row.append(cell);
}

function tableRows(target, rows, columns, emptyMessage) {
  const body = document.querySelector(target);
  body.replaceChildren();
  if (!rows?.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.colSpan = columns.length;
    cell.className = "empty-table";
    cell.textContent = emptyMessage;
    row.append(cell); body.append(row); return;
  }
  rows.forEach((record) => {
    const row = document.createElement("tr");
    columns.forEach((column) => addCell(row, column(record)));
    body.append(row);
  });
}

function summaryCards(target, cards) {
  const host = document.querySelector(target);
  host.replaceChildren();
  cards.forEach(([label, value, note]) => {
    const card = document.createElement("article"); card.className = "research-metric";
    const title = document.createElement("span"); title.textContent = label;
    const strong = document.createElement("strong"); strong.textContent = value;
    const detail = document.createElement("small"); detail.textContent = note;
    card.append(title, strong, detail); host.append(card);
  });
}

function renderNetwork(data) {
  const rows = data.rankings || [];
  const search = document.querySelector("#network-search").value.trim().toLowerCase();
  const filtered = rows.filter((item) => !search || String(item.coin).toLowerCase().includes(search));
  tableRows("#network-table", filtered, [
    (r) => r.coin, (r) => `${safe(r.dy_net_spillover)}%`, (r) => safe(r.dy_rank, 0),
    (r) => safe(r.pagerank, 5), (r) => safe(r.pagerank_rank, 0),
    (r) => safe(r.eigenvector, 5), (r) => safe(r.eigenvector_rank, 0),
  ], "No saved rankings report. Run the network analysis pipeline to populate this view.");
  summaryCards("#network-summary", [
    ["ASSETS RANKED", safe(rows.length, 0), "Saved ranking report"],
    ["TOP NET TRANSMITTER", rows[0]?.coin || "—", "By DY net spillover rank"],
    ["TOP SPILLOVER", rows.length ? `${safe(rows[0].dy_net_spillover)}%` : "—", "Diebold–Yilmaz net value"],
  ]);
  drawNetworkSnapshot(data.network);
}

function drawNetworkSnapshot(network) {
  const canvas = document.querySelector("#snapshot-canvas");
  const empty = document.querySelector("#snapshot-empty");
  if (!network?.nodes?.length) { empty.textContent = "Hourly processed data is unavailable; showing saved rankings only."; empty.style.display = "grid"; return; }
  empty.style.display = "none";
  const rect = canvas.getBoundingClientRect(), dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width, height = rect.height, radius = Math.max(65, Math.min(width, height) * 0.36);
  const points = network.nodes.map((node) => ({x: width / 2 + node.x * radius, y: height / 2 + node.y * radius, node}));
  ctx.clearRect(0, 0, width, height);
  network.edges.forEach((edge) => {
    const a = points[edge.source], b = points[edge.target]; if (!a || !b) return;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = `rgba(120,151,176,${0.05 + edge.strength * 0.18})`; ctx.lineWidth = 0.6 + edge.strength * 0.6; ctx.stroke();
  });
  const maxRank = Math.max(0.000001, ...network.nodes.map((item) => Number(item.pagerank || 0)));
  points.forEach(({x, y, node}) => {
    const radiusPx = 5 + 5 * Math.sqrt(Number(node.pagerank || 0) / maxRank);
    const value = Number(node.dy_net_spillover || 0);
    ctx.beginPath(); ctx.arc(x, y, radiusPx, 0, Math.PI * 2);
    ctx.fillStyle = value >= 0 ? "#e7847d" : "#75a9dc"; ctx.fill();
    ctx.strokeStyle = "#111416"; ctx.lineWidth = 1.3; ctx.stroke();
    ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillStyle = "#c5d0d2"; ctx.fillText(node.asset, x, y + radiusPx + 4);
  });
  document.querySelector("#snapshot-count").textContent = `${network.nodes.length} ASSETS · ${network.edges.length} EDGES`;
}

function renderSpillover(data) {
  const rows = data.persistence || [], explanations = data.explanations || [];
  const radius = rows.length ? rows.reduce((sum, r) => sum + Number(r.blast_radius_pct || 0), 0) / rows.length : null;
  const delay = rows.length ? rows.reduce((sum, r) => sum + Number(r.propagation_delay_hours || 0), 0) / rows.length : null;
  const halfLife = rows.length ? rows.reduce((sum, r) => sum + Number(r.mean_persistence_half_life_hours || 0), 0) / rows.length : null;
  summaryCards("#spillover-summary", [
    ["EVENT STUDIES", safe(rows.length, 0), "Rows in saved persistence report"],
    ["MEAN BLAST RADIUS", radius === null ? "—" : `${safe(radius * 100, 1)}%`, "Mean reported share of universe"],
    ["MEAN PROPAGATION DELAY", delay === null ? "—" : `${safe(delay, 1)} h`, "Report-defined delay to peak"],
    ["MEAN HALF-LIFE", halfLife === null ? "—" : `${safe(halfLife, 1)} h`, "Estimated persistence summary"],
  ]);
  tableRows("#spillover-table", rows, [
    (r) => r.event_name, (r) => r.initial_transmitters,
    (r) => `${safe(r.blast_radius_count, 0)} · ${safe(Number(r.blast_radius_pct) * 100, 1)}%`,
    (r) => `${safe(r.propagation_delay_hours, 1)} h`,
    (r) => `${safe(r.mean_persistence_half_life_hours, 1)} h`, (r) => r.most_impacted_sector,
  ], "No persistence report is available.");
  tableRows("#explanation-table", explanations, [
    (r) => safe(r.timestep, 0), (r) => r.target_asset,
    (r) => r.statistical_status, (r) => safe(r.fidelity_plus), (r) => safe(r.fidelity_minus),
    (r) => `${safe(Number(r.sparsity) * 100, 1)}%`, (r) => safe(r.spearman_dy_net), (r) => r.top_explanatory_edges,
  ], "No saved TempME explanation records.");
}

function renderSecurity(data) {
  const anomalies = data.anomalies || [], robustness = data.robustness || [];
  const flagged = anomalies.filter((r) => r.is_statistical_anomaly).length;
  const manifestCount = data.manifest?.checkpoints?.length || 0;
  summaryCards("#security-summary", [
    ["EVENTS AUDITED", safe(anomalies.length, 0), "Saved T-GAE event records"],
    ["STATISTICAL FLAGS", safe(flagged, 0), "Not evidence of malicious activity"],
    ["ROBUSTNESS CASES", safe(robustness.length, 0), "Saved perturbation rows"],
    ["CHECKPOINTS IN MANIFEST", safe(manifestCount, 0), manifestCount ? "Manifest records, not deployment status" : "No checkpoint manifest found"],
  ]);
  tableRows("#anomaly-table", anomalies, [
    (r) => r.event_name, (r) => r.event_date, (r) => safe(r.event_reconstruction_error, 5),
    (r) => safe(r.baseline_reconstruction_error, 5), (r) => `${safe(r.anomaly_ratio, 2)}×`,
    (r) => safe(r.z_score), (r) => safe(r.p_value, 4), (r) => r.most_anomalous_asset,
    (r) => r.is_statistical_anomaly ? "Flag" : "No flag",
  ], "No T-GAE anomaly report is available.");
  tableRows("#robustness-table", robustness, [
    (r) => r.perturbation_type, (r) => safe(r.epsilon),
    (r) => safe(r.prediction_drift_mse, 7), (r) => safe(r.explanation_jaccard_stability, 4),
  ], "No robustness report is available.");
}

async function loadResearchView(name) {
  const view = researchViews[name];
  if (!view) return;
  if (!researchCache.has(name)) {
    try {
      const response = await fetch(view.endpoint);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load report data.");
      researchCache.set(name, data);
    } catch (error) {
      document.querySelector(`#${name}-view .secondary-heading p`).textContent = error.message;
      return;
    }
  }
  view.render(researchCache.get(name));
}

async function initForecast() {
  const status = document.querySelector("#forecast-status");
  try {
    const response = await fetch("/api/forecast/options");
    forecastOptions = await response.json();
    if (!forecastOptions.available) { status.textContent = forecastOptions.reason; return; }
    const model = document.querySelector("#forecast-model");
    forecastOptions.models.forEach((value) => model.add(new Option(value.toUpperCase(), value)));
    const horizon = document.querySelector("#forecast-horizon");
    forecastOptions.horizons.forEach((value) => horizon.add(new Option(`+${value} hour${value === 1 ? "" : "s"}`, value)));
    const time = document.querySelector("#forecast-time");
    time.max = String(forecastOptions.test_indices.length - 1);
    time.value = time.max;
    updateForecastTime();
    status.textContent = `${forecastOptions.test_indices.length.toLocaleString()} held-out test timestamps available · inference runs locally.`;
    document.querySelector("#forecast-run").disabled = false;
  } catch (error) { status.textContent = error.message; }
}

function updateForecastTime() {
  if (!forecastOptions) return;
  const pos = Number(document.querySelector("#forecast-time").value);
  document.querySelector("#forecast-time-label").textContent = forecastOptions.timestamps[pos] || "—";
}

async function runForecast() {
  const button = document.querySelector("#forecast-run"), status = document.querySelector("#forecast-status");
  button.disabled = true; status.textContent = "Running historical model inference… This may take a moment.";
  try {
    const pos = Number(document.querySelector("#forecast-time").value);
    const query = new URLSearchParams({
      model: document.querySelector("#forecast-model").value,
      timestep: String(forecastOptions.test_indices[pos]),
      horizon: document.querySelector("#forecast-horizon").value,
      threshold: document.querySelector("#forecast-threshold").value,
      top_k: "30",
    });
    const response = await fetch(`/api/forecast?${query}`), data = await response.json();
    if (!response.ok) throw new Error(data.error || "Forecast failed.");
    renderForecast(data);
    status.textContent = "Inference complete. These are model outputs for the selected historical test timestamp.";
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
}

function renderForecast(data) {
  const meta = data.metadata;
  summaryCards("#forecast-summary", [
    ["RISK REGIME", meta.market_risk_regime, `${meta.model_type} · ${meta.timestamp}`],
    ["MEAN SHOCK PROBABILITY", `${safe(meta.market_mean_shock_probability * 100, 1)}%`, `+${meta.horizon_hours}h horizon`],
    ["ABOVE THRESHOLD", `${meta.assets_in_alert} / ${meta.total_assets_evaluated}`, `Alert threshold ${safe(meta.threshold * 100, 0)}%`],
    ["DEVICE", meta.device, `timestep ${meta.timestep_index}`],
  ]);
  tableRows("#forecast-table", data.predictions || [], [
    (r) => safe(r.rank, 0), (r) => r.asset, (r) => r.sector,
    (r) => `${safe(r.shock_probability * 100, 2)}%`, (r) => r.status,
    (r) => `${safe(r.persistence_halflife_hours, 1)} h`, (r) => r.explanation,
  ], "No forecast output.");
  document.querySelector("#forecast-results").hidden = false;
}

document.querySelectorAll(".nav-button[data-view]").forEach((button) => button.addEventListener("click", () => {
  if (researchViews[button.dataset.view]) loadResearchView(button.dataset.view);
}));
document.querySelector("#network-search").addEventListener("input", () => {
  if (researchCache.has("network")) renderNetwork(researchCache.get("network"));
});
new ResizeObserver(() => {
  if (researchCache.has("network")) drawNetworkSnapshot(researchCache.get("network").network);
}).observe(document.querySelector("#snapshot-canvas"));
document.querySelector("#forecast-time").addEventListener("input", updateForecastTime);
document.querySelector("#forecast-run").addEventListener("click", runForecast);
document.querySelector("#forecast-run").disabled = true;
initForecast();
