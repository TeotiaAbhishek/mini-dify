// Traces: every chat answer, workflow run and eval judge call, as a waterfall of spans.
// Each bar starts at the span's offset and is as wide as its duration, so you can see
// at a glance which step took the time.

import { api } from "../api.js";
import { ago, badge, fill, h, json, ms, notice, pageHead, panel, short, stat, table } from "../ui.js";

export async function tracesPage() {
  const status = new URLSearchParams(location.hash.split("?")[1]).get("status") ?? "";
  const traces = await api(`/v1/traces?limit=100${status ? `&status=${status}` : ""}`);
  const filter = h("select", {
    onchange: (e) => (location.hash = e.target.value ? `#/traces?status=${e.target.value}` : "#/traces"),
  }, [["", "All"], ["ok", "OK"], ["error", "Errors"]].map(([v, t]) => h("option", { value: v, selected: v === status }, t)));
  return h("div", {},
    pageHead("Traces", filter),
    panel(table(traces, [
      ["Name", (t) => h("strong", {}, t.name)],
      ["Status", (t) => badge(t.status)],
      ["Input", (t) => h("span", { class: "muted" }, preview(t.inputs))],
      ["Duration", (t) => ms(t.duration_ms)],
      ["LLM calls", (t) => t.llm_calls],
      ["Tokens", (t) => `${t.prompt_tokens} + ${t.completion_tokens}`],
      ["When", (t) => ago(t.started_at)],
    ], { onclick: (t) => (location.hash = `#/traces/${t.trace_id}`), empty: "No traces yet. Chat with an app or run a workflow." })),
  );
}

function preview(inputs) {
  const text = inputs.query ?? inputs.question ?? JSON.stringify(inputs.inputs ?? inputs);
  return String(text).length > 70 ? `${String(text).slice(0, 70)}…` : String(text);
}

export async function tracePage({ id }) {
  const t = await api(`/v1/traces/${id}`);
  const total = t.root.duration_ms || 1;
  const details = h("div");
  const rows = [];

  const select = (node, row) => {
    rows.forEach((r) => r.classList.remove("selected"));
    row.classList.add("selected");
    fill(details, panel(
      h("h2", {}, node.name, " ", badge(node.kind, node.kind === "llm" ? "llm" : ""), " ", badge(node.status)),
      h("p", { class: "muted small" }, `starts at +${ms(node.offset_ms)}, takes ${ms(node.duration_ms)}`),
      node.error ? notice(node.error) : null,
      h("div", { class: "cols" },
        h("div", {}, h("div", { class: "small muted" }, "Inputs"), json(node.inputs)),
        h("div", {}, h("div", { class: "small muted" }, "Outputs"), json(node.outputs))),
      Object.keys(node.attributes).length ? [h("div", { class: "small muted" }, "Attributes"), json(node.attributes)] : null,
    ));
  };

  // Walk the tree depth first: one row per span, indented by its depth.
  const waterfall = h("div");
  const walk = (node, depth) => {
    const left = (node.offset_ms / total) * 100;
    const width = Math.max((node.duration_ms / total) * 100, 0.3);
    const row = h("div", { class: "span-row" },
      h("div", { class: "span-name", style: `padding-left: ${depth * 16}px`, title: `${node.kind}: ${node.name}` }, node.name),
      h("div", { class: "span-track" },
        h("div", { class: `span-bar ${node.status === "error" ? "error" : node.kind}`, style: `left: ${left}%; width: ${width}%` })),
      h("div", { class: "span-ms" }, ms(node.duration_ms)));
    row.addEventListener("click", () => select(node, row));
    rows.push(row);
    waterfall.append(row);
    node.children.forEach((child) => walk(child, depth + 1));
  };
  walk(t.root, 0);
  select(t.root, rows[0]);

  return h("div", {},
    pageHead(t.name, badge(t.status), h("a", { href: "#/traces" }, "All traces")),
    h("div", { class: "stats" },
      stat("Duration", ms(t.duration_ms)),
      stat("Spans", t.span_count),
      stat("LLM calls", t.llm_calls),
      stat("Prompt tokens", t.prompt_tokens),
      stat("Completion tokens", t.completion_tokens)),
    t.error ? notice(t.error) : null,
    panel(h("h2", {}, "Waterfall"), h("p", { class: "small muted" }, `Trace ${short(t.trace_id)} · click a span to see what went in and out`), waterfall),
    details,
  );
}
