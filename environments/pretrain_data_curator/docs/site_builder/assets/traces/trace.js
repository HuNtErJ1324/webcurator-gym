const state = {
  runId: null,
  run: null,
  payload: null,
  activeTab: "trace",
  activeArtifact: 0,
};

async function init() {
  const params = new URLSearchParams(window.location.search);
  state.runId = params.get("id");
  if (!state.runId) {
    showError("Missing run id. Open this page from the leaderboard.");
    return;
  }

  state.activeTab = parseTabHash() || "trace";
  bindTabs();

  const [manifestRes, traceRes] = await Promise.all([
    fetch("../data/manifest.json"),
    fetch(`../data/traces/${state.runId}.json`),
  ]);
  const manifest = await manifestRes.json();
  state.run = (manifest.runs || []).find((r) => r.id === state.runId);
  state.payload = await traceRes.json();

  document.title = `${shortModel(state.payload.model || state.run?.model)} · trace`;
  document.getElementById("crumb-run").textContent = shortModel(state.payload.model || state.run?.model);
  document.getElementById("run-title").textContent =
    `${state.payload.model || state.run?.model || "unknown"} · ${state.payload.harness || state.run?.harness || "codex"}`;
  document.getElementById("run-subtitle").textContent =
    state.payload.rel_path || state.run?.rel_path || state.run?.run_group || "";

  document.getElementById("run-meta").innerHTML = renderMetricChips(state.run, state.payload);
  document.getElementById("download-trace").href = `../data/traces/${state.runId}.json`;
  document.getElementById("download-trace").download = `${state.runId}.json`;

  renderTraceTab();
  renderMetricsTab();
  renderArtifactsTab();
  renderLogTab();
  activateTab(state.activeTab);
}

function parseTabHash() {
  const match = window.location.hash.match(/^#tab=(\w+)/);
  return match ? match[1] : null;
}

function bindTabs() {
  document.querySelectorAll(".tab-bar .tab").forEach((button) => {
    button.addEventListener("click", () => {
      activateTab(button.dataset.tab);
      history.replaceState(null, "", `${window.location.pathname}?id=${encodeURIComponent(state.runId)}#tab=${button.dataset.tab}`);
    });
  });
  window.addEventListener("hashchange", () => {
    const tab = parseTabHash();
    if (tab) activateTab(tab);
  });
}

function activateTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab-bar .tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tab}`);
  });
}

function renderTraceTab() {
  const root = document.getElementById("trace-view");
  const trace = state.payload.trace || [];
  const note = state.payload.trace_note;
  const kind = state.payload.trace_kind;

  root.innerHTML = "";
  if (kind === "log_reconstruction" && note) {
    const banner = document.createElement("div");
    banner.className = "trace-banner";
    banner.innerHTML = `<p>${escapeHtml(note)}</p>`;
    root.appendChild(banner);
  }

  const stepsRoot = document.createElement("div");
  stepsRoot.className = "trace-steps";
  root.appendChild(stepsRoot);
  renderTraceSteps(trace, stepsRoot);
  if (trace.length) {
    root.classList.remove("empty");
  }
}

function renderMetricsTab() {
  const root = document.getElementById("metrics-table");
  const metrics = { ...(state.run?.metrics || {}), ...(state.payload.metrics || {}) };
  const timing = state.payload.timing || state.run?.timing || {};
  const rows = [
    ["Reward", formatNum(state.payload.reward ?? state.run?.reward, 3)],
    ["Perf loss", formatNum(metrics.perf_loss, 3)],
    ["Perf vs baseline", formatNum(metrics.perf_vs_baseline, 3)],
    ["Leakage", formatMetricValue("leakage_score", metrics.leakage_score)],
    ["Budget fill", formatPct(metrics.budget_fill_ratio)],
    ["Corpus tokens", formatTokens(metrics.corpus_tokens)],
    ["Sources", formatNum(metrics.num_sources, 0)],
    ["Agent turns", formatNum(metrics.agent_turns, 0)],
    ["Failed sources", formatNum(metrics.failed_sources, 0)],
    ["Generation", formatDuration(timing.generation)],
    ["Scoring", formatDuration(timing.scoring)],
    ["Total", formatDuration(timing.total)],
    ["Stop condition", state.payload.stop_condition || state.run?.stop_condition || "—"],
    ["Completed", String(state.payload.is_completed ?? state.run?.is_completed ?? false)],
  ];
  root.innerHTML = `
    <table>
      <tbody>
        ${rows
          .map(
            ([label, value]) =>
              `<tr><th>${escapeHtml(label)}</th><td class="num">${escapeHtml(String(value))}</td></tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;

  const notes = document.getElementById("run-notes");
  const text = state.payload.notes || "";
  if (text.trim()) {
    notes.classList.remove("hidden");
    notes.innerHTML = `<h3>Notes</h3><pre class="notes-block">${escapeHtml(text)}</pre>`;
  }
}

function renderArtifactsTab() {
  const list = document.getElementById("artifact-list");
  const view = document.getElementById("artifact-view");
  const artifacts = state.payload.artifacts || [];
  list.innerHTML = "";
  if (!artifacts.length) {
    view.classList.add("empty");
    view.textContent = "No artifacts captured for this run.";
    return;
  }
  artifacts.forEach((artifact, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "artifact-item";
    button.innerHTML = `
      <strong>${escapeHtml(artifact.label || artifact.path)}</strong>
      <span>${escapeHtml(artifact.path || "")}</span>
    `;
    button.addEventListener("click", () => selectArtifact(index));
    list.appendChild(button);
  });
  selectArtifact(0);
}

function selectArtifact(index) {
  state.activeArtifact = index;
  const artifacts = state.payload.artifacts || [];
  const artifact = artifacts[index];
  const view = document.getElementById("artifact-view");
  document.querySelectorAll(".artifact-item").forEach((el, i) => {
    el.classList.toggle("active", i === index);
  });
  if (!artifact) return;
  view.classList.remove("empty");
  const language = artifact.language || (artifact.path?.endsWith(".json") ? "json" : "plaintext");
  view.innerHTML = renderCodeBlock(artifact.content || "", language);
  view.querySelectorAll("pre code").forEach((block) => {
    if (typeof hljs !== "undefined") hljs.highlightElement(block);
  });
}

function renderLogTab() {
  const log = state.payload.log || "";
  const root = document.getElementById("log-view");
  if (!log.trim()) {
    root.textContent = "No log captured for this run.";
    return;
  }
  root.innerHTML = highlightCode(log, "bash");
}

function showError(message) {
  document.getElementById("run-title").textContent = "Trace unavailable";
  document.getElementById("trace-view").textContent = message;
}

init();
