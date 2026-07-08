function metricColumn(manifest, key) {
  return (manifest?.metric_columns || []).find((col) => col.key === key);
}

function compareMetric(a, b, key, higherIsBetter) {
  const av = metric(a, key);
  const bv = metric(b, key);
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  return higherIsBetter ? bv - av : av - bv;
}

function metric(run, key) {
  const fromMetrics = run.metrics?.[key];
  if (fromMetrics != null) return Number(fromMetrics);
  if (run[key] != null) return Number(run[key]);
  return null;
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

function formatScientific(value, fractionDigits = 4) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toExponential(fractionDigits);
}

function formatMetricValue(key, value, digits = 3) {
  if (key === "leakage_score") return formatScientific(value);
  return formatNum(value, digits);
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

function runTraceUrl(id, tab = "trace") {
  return `traces/run.html?id=${encodeURIComponent(id)}#tab=${tab}`;
}

function renderTraceSteps(trace, root) {
  root.innerHTML = "";
  if (!trace?.length) {
    root.classList.add("empty");
    root.textContent = "No conversation trace stored for this run.";
    return;
  }
  root.classList.remove("empty");
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
      details.open = true;
      details.innerHTML = `<summary>${escapeHtml(tool.name || "tool")}</summary>`;
      const body = document.createElement("div");
      body.className = "step-body";
      body.innerHTML = renderToolCall(tool);
      details.appendChild(body);
      el.appendChild(details);
    }

    root.appendChild(el);
  });
}

function renderMetricChips(run, payload) {
  const reward = payload?.reward ?? run?.reward;
  const metrics = payload?.metrics || run?.metrics || {};
  const chips = [
    ["Reward", formatNum(reward, 3)],
    ["Perf Loss", formatNum(metrics.perf_loss ?? metricPayload(payload, "perf_loss"), 3)],
    ["Leakage", formatMetricValue("leakage_score", metrics.leakage_score ?? metricPayload(payload, "leakage_score"))],
    ["Corpus", formatTokens(metrics.corpus_tokens ?? metricPayload(payload, "corpus_tokens"))],
    ["Sources", formatNum(metrics.num_sources ?? metricPayload(payload, "num_sources"), 0)],
    ["Agent Turns", formatNum(metrics.agent_turns ?? metricPayload(payload, "agent_turns"), 0)],
    ["Stop", payload?.stop_condition || run?.stop_condition || "—"],
  ];
  return chips
    .map(
      ([label, value]) =>
        `<div class="chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`
    )
    .join("");
}
