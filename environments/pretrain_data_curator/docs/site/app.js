const state = {
  manifest: null,
  runs: [],
  filtered: [],
  selectedId: null,
  metricKey: "reward",
};

async function init() {
  const res = await fetch("data/manifest.json");
  state.manifest = await res.json();
  state.runs = state.manifest.runs || [];
  state.metricKey = state.manifest.metric_columns?.[0]?.key || "reward";
  document.getElementById("run-count").textContent = state.manifest.run_count ?? state.runs.length;
  document.getElementById("generated-at").textContent = formatDate(state.manifest.generated_at);
  populateFilters();
  populateMetricSelect();
  bindControls();
  applyFilters();
}

function bindControls() {
  ["filter-model", "filter-harness", "filter-trace-only"].forEach((id) => {
    document.getElementById(id).addEventListener("change", applyFilters);
  });
  document.getElementById("metric-select").addEventListener("change", (e) => {
    state.metricKey = e.target.value;
    renderMetricChart();
  });
}

function populateFilters() {
  const models = unique(state.runs.map((r) => r.model)).sort();
  const harnesses = unique(state.runs.map((r) => r.harness)).sort();
  fillSelect("filter-model", models);
  fillSelect("filter-harness", harnesses);
}

function populateMetricSelect() {
  const select = document.getElementById("metric-select");
  select.innerHTML = "";
  for (const col of state.manifest.metric_columns || []) {
    const opt = document.createElement("option");
    opt.value = col.key;
    opt.textContent = col.label;
    select.appendChild(opt);
  }
  select.value = state.metricKey;
}

function fillSelect(id, values) {
  const select = document.getElementById(id);
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.appendChild(opt);
  }
}

function applyFilters() {
  const model = document.getElementById("filter-model").value;
  const harness = document.getElementById("filter-harness").value;
  const traceOnly = document.getElementById("filter-trace-only").checked;
  state.filtered = state.runs.filter((run) => {
    if (model && run.model !== model) return false;
    if (harness && run.harness !== harness) return false;
    if (traceOnly && !run.has_trace) return false;
    return true;
  });
  renderLeaderboard();
  renderMetricChart();
  renderTimeChart();
}

function renderLeaderboard() {
  const tbody = document.querySelector("#leaderboard tbody");
  tbody.innerHTML = "";
  state.filtered.forEach((run, index) => {
    const tr = document.createElement("tr");
    if (run.id === state.selectedId) tr.classList.add("selected");
    tr.innerHTML = `
      <td class="rank">${index + 1}</td>
      <td>${escapeHtml(shortModel(run.model))}</td>
      <td>${escapeHtml(run.harness)}</td>
      <td class="num">${formatBudget(run.token_budget)}</td>
      <td class="num good">${formatNum(run.reward, 3)}</td>
      <td class="num">${formatNum(metric(run, "perf_loss"), 3)}</td>
      <td class="num">${formatNum(metric(run, "leakage_score"), 3)}</td>
      <td class="num">${formatPct(metric(run, "budget_fill_ratio"))}</td>
      <td class="num">${formatDuration(run.timing?.total)}</td>
    `;
    tr.addEventListener("click", () => selectRun(run.id));
    tbody.appendChild(tr);
  });
}

function renderMetricChart() {
  const root = document.getElementById("metric-chart");
  root.innerHTML = "";
  const top = [...state.filtered]
    .sort((a, b) => (metric(b, state.metricKey) ?? -Infinity) - (metric(a, state.metricKey) ?? -Infinity))
    .slice(0, 12);
  const values = top.map((r) => metric(r, state.metricKey)).filter((v) => v != null);
  const max = values.length ? Math.max(...values) : 1;
  for (const run of top) {
    const value = metric(run, state.metricKey);
    const row = document.createElement("div");
    row.className = "bar-row";
    const width = value == null ? 0 : (Math.abs(value) / (max || 1)) * 100;
    row.innerHTML = `
      <div class="bar-label" title="${escapeHtml(run.model)}">${escapeHtml(shortModel(run.model))}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
      <div class="bar-value">${value == null ? "—" : formatNum(value, 3)}</div>
    `;
    root.appendChild(row);
  }
  if (!top.length) root.innerHTML = '<p class="muted">No runs match the current filters.</p>';
}

function renderTimeChart() {
  const root = document.getElementById("time-chart");
  root.innerHTML = "";
  const top = [...state.filtered]
    .filter((r) => (r.timing?.total || 0) > 0)
    .sort((a, b) => (b.timing?.total || 0) - (a.timing?.total || 0))
    .slice(0, 10);
  const max = top.length ? Math.max(...top.map((r) => r.timing?.total || 0)) : 1;
  for (const run of top) {
    const gen = run.timing?.generation || 0;
    const score = run.timing?.scoring || 0;
    const total = run.timing?.total || gen + score || 1;
    const row = document.createElement("div");
    row.className = "time-row";
    row.innerHTML = `
      <div class="bar-label" title="${escapeHtml(run.model)}">${escapeHtml(shortModel(run.model))}</div>
      <div class="time-stack" style="width:${((total / max) * 100).toFixed(1)}%">
        <div class="time-gen" style="width:${((gen / total) * 100).toFixed(1)}%"></div>
        <div class="time-score" style="width:${((score / total) * 100).toFixed(1)}%"></div>
      </div>
      <div class="bar-value">${formatDuration(total)}</div>
    `;
    root.appendChild(row);
  }
  if (!top.length) root.innerHTML = '<p class="muted">No timing data for the current filters.</p>';
}

async function selectRun(id) {
  state.selectedId = id;
  renderLeaderboard();
  const run = state.runs.find((r) => r.id === id);
  const subtitle = document.getElementById("trace-subtitle");
  const meta = document.getElementById("trace-meta");
  const view = document.getElementById("trace-view");
  if (!run) return;
  subtitle.textContent = `${run.model} · ${run.harness} · ${run.rel_path}`;
  if (!run.has_trace) {
    meta.classList.add("hidden");
    view.classList.add("empty");
    view.textContent = "This run has no stored trace nodes.";
    return;
  }
  view.classList.remove("empty");
  view.textContent = "Loading trace…";
  const res = await fetch(`data/traces/${id}.json`);
  const payload = await res.json();
  renderTraceMeta(payload);
  renderTrace(payload.trace || []);
}

function renderTraceMeta(payload) {
  const meta = document.getElementById("trace-meta");
  meta.classList.remove("hidden");
  meta.innerHTML = `
    <div class="chip"><span>Reward</span><strong>${formatNum(payload.reward, 3)}</strong></div>
    <div class="chip"><span>Perf Loss</span><strong>${formatNum(metricPayload(payload, "perf_loss"), 3)}</strong></div>
    <div class="chip"><span>Leakage</span><strong>${formatNum(metricPayload(payload, "leakage_score"), 3)}</strong></div>
    <div class="chip"><span>Corpus</span><strong>${formatTokens(metricPayload(payload, "corpus_tokens"))}</strong></div>
    <div class="chip"><span>Steps</span><strong>${(payload.trace || []).length}</strong></div>
    <div class="chip"><span>Stop</span><strong>${escapeHtml(payload.stop_condition || "—")}</strong></div>
  `;
}

function renderTrace(trace) {
  const view = document.getElementById("trace-view");
  view.innerHTML = "";
  trace.forEach((step, index) => {
    const el = document.createElement("article");
    el.className = "step";

    const head = document.createElement("div");
    head.className = "step-head";
    head.innerHTML = `
      <span class="role ${escapeHtml(step.role)}">${escapeHtml(step.role)}</span>
      <span class="step-index">#${index + 1}</span>
    `;
    el.appendChild(head);

    if (step.content) {
      const body = document.createElement("div");
      body.className = "step-body";
      body.innerHTML = renderRichContent(step.content, { role: step.role });
      el.appendChild(body);
    }

    if (step.reasoning) {
      const details = document.createElement("details");
      details.className = "reasoning";
      details.open = true;
      details.innerHTML = `<summary>Reasoning</summary>`;
      const body = document.createElement("div");
      body.className = "step-body";
      body.innerHTML = renderRichContent(step.reasoning, { role: "assistant", hint: "markdown" });
      details.appendChild(body);
      el.appendChild(details);
    }

    for (const tool of step.tool_calls || []) {
      const details = document.createElement("details");
      details.className = "tool-block";
      details.innerHTML = `<summary>${escapeHtml(tool.name || "tool")}</summary>`;
      const body = document.createElement("div");
      body.className = "step-body";
      body.innerHTML = renderToolArguments(tool.arguments);
      details.appendChild(body);
      el.appendChild(details);
    }

    view.appendChild(el);
  });
}

function metric(run, key) {
  const value = run.metrics?.[key];
  return value == null ? null : Number(value);
}

function metricPayload(payload, key) {
  const value = payload.metrics?.[key];
  return value == null ? null : Number(value);
}

function shortModel(model) {
  if (!model) return "unknown";
  const parts = String(model).split("/");
  return parts[parts.length - 1];
}

function formatBudget(value) {
  if (value == null) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(0)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return String(value);
}

function formatTokens(value) {
  if (value == null) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

function formatPct(value) {
  if (value == null) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNum(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function formatDuration(seconds) {
  if (!seconds) return "—";
  const s = Number(seconds);
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  return d.toLocaleString();
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

init();
