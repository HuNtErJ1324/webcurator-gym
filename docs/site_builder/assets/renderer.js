/* Rich trace rendering: markdown, JSON, and shell output. */

function configureRenderer() {
  if (typeof marked !== "undefined") {
    marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false,
    });
  }
}

function looksLikeJson(text) {
  const trimmed = text.trim();
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return false;
  try {
    JSON.parse(trimmed);
    return true;
  } catch {
    return false;
  }
}

function looksLikeMarkdown(text) {
  return /(^|\n)\s{0,3}#{1,6}\s|```|(^|\n)[-*]\s|(^|\n)\d+\.\s|\*\*[^*]+\*\*|`[^`]+`/m.test(text);
}

function looksLikeShellOutput(text) {
  return /^(Chunk ID:|Process exited|Original token count:|Wall time:|\$ |# )/m.test(text);
}

function highlightCode(code, language) {
  if (typeof hljs === "undefined") {
    return escapeHtml(code);
  }
  if (language && language !== "plaintext" && hljs.getLanguage(language)) {
    return hljs.highlight(code, { language }).value;
  }
  return hljs.highlightAuto(code).value;
}

function renderCodeBlock(code, language = "plaintext") {
  let body = code;
  if (language === "json") {
    try {
      body = JSON.stringify(JSON.parse(code), null, 2);
    } catch {
      /* keep original */
    }
  }
  return `<pre class="hljs"><code class="language-${escapeHtml(language)}">${highlightCode(body, language)}</code></pre>`;
}

function renderMarkdown(text) {
  if (typeof marked === "undefined") {
    return `<pre class="fallback">${escapeHtml(text)}</pre>`;
  }
  const html = marked.parse(text);
  const container = document.createElement("div");
  container.className = "rich-md";
  container.innerHTML = html;
  container.querySelectorAll("pre code").forEach((block) => {
    if (typeof hljs !== "undefined") {
      hljs.highlightElement(block);
    }
  });
  return container.outerHTML;
}

function renderRichContent(text, { role = "assistant", hint = null } = {}) {
  if (!text) return "";
  const trimmed = String(text).trim();
  if (!trimmed) return "";

  if (hint === "json" || looksLikeJson(trimmed)) {
    return renderCodeBlock(trimmed, "json");
  }

  if (role === "tool" && (hint === "shell" || looksLikeShellOutput(trimmed))) {
    return renderCodeBlock(trimmed, "bash");
  }

  if (hint === "markdown" || looksLikeMarkdown(trimmed)) {
    return renderMarkdown(trimmed);
  }

  if ((role === "user" || role === "assistant") && trimmed.length > 120) {
    return renderMarkdown(trimmed);
  }

  if (trimmed.includes("\n")) {
    return renderCodeBlock(trimmed, role === "tool" ? "bash" : "plaintext");
  }

  return `<div class="rich-plain">${escapeHtml(trimmed)}</div>`;
}

function renderToolArguments(args) {
  if (args == null || args === "") {
    return '<p class="muted">No arguments recorded.</p>';
  }
  if (typeof args === "string") {
    return renderRichContent(args, { role: "tool", hint: looksLikeJson(args) ? "json" : null });
  }
  return renderCodeBlock(JSON.stringify(args, null, 2), "json");
}

function renderToolCall(tool) {
  const args = tool?.arguments;
  if (tool?.name === "exec_command" && args && typeof args === "object") {
    const cmd = args.cmd ?? args.command;
    if (typeof cmd === "string" && cmd.trim()) {
      return renderCodeBlock(cmd, "bash");
    }
    if (Array.isArray(cmd)) {
      return renderCodeBlock(cmd.map((part) => String(part)).join(" "), "bash");
    }
  }
  return renderToolArguments(args);
}

configureRenderer();
