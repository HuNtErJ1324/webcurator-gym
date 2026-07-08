const state = {
  manifest: null,
  runs: [],
  filtered: [],
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
    const traceLink = run.has_trace
      ? `<a class="trace-link" href="${runTraceUrl(run.id)}">View trace</a>`
      : `<span class="muted">—</span>`;
    const status = run.source === "debug" ? `<span class="badge debug">debug</span>` : "";
    tr.innerHTML = `
      <td class="rank">${index + 1}</td>
      <td>${escapeHtml(shortModel(run.model))} ${status}</td>
      <td>${escapeHtml(run.harness)}</td>
      <td class="num">${formatBudget(run.token_budget)}</td>
      <td class="num good">${formatNum(run.reward, 3)}</td>
      <td class="num">${formatNum(metric(run, "perf_loss"), 3)}</td>
      <td class="num">${formatMetricValue("leakage_score", metric(run, "leakage_score"))}</td>
      <td class="num">${formatPct(metric(run, "budget_fill_ratio"))}</td>
      <td class="num">${formatDuration(run.timing?.total)}</td>
      <td class="trace-col">${traceLink}</td>
    `;
    tr.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      if (run.has_trace) window.location.href = runTraceUrl(run.id);
    });
    tbody.appendChild(tr);
  });
}

function renderMetricChart() {
  const root = document.getElementById("metric-chart");
  root.innerHTML = "";
  const higherIsBetter = metricColumn(state.manifest, state.metricKey)?.higher_is_better !== false;
  const scored = state.filtered.filter((r) => metric(r, state.metricKey) != null);
  const top = [...scored]
    .sort((a, b) => compareMetric(a, b, state.metricKey, higherIsBetter))
    .slice(0, 12);
  const values = top.map((r) => metric(r, state.metricKey)).filter((v) => v != null);
  const max = values.length ? Math.max(...values.map((v) => Math.abs(v))) : 1;
  for (const run of top) {
    const value = metric(run, state.metricKey);
    const row = document.createElement("div");
    row.className = "bar-row";
    const width = value == null ? 0 : (Math.abs(value) / (max || 1)) * 100;
    row.innerHTML = `
      <div class="bar-label" title="${escapeHtml(run.model)}">${escapeHtml(shortModel(run.model))}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
      <div class="bar-value">${value == null ? "—" : formatMetricValue(state.metricKey, value)}</div>
    `;
    root.appendChild(row);
  }
  if (!top.length) root.innerHTML = '<p class="muted">No scored runs match the current filters.</p>';
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

init();
