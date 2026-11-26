// Workflows: the list, a JSON editor for the graph, and a run form that shows each node
// start and finish live, from the streamed events.

import { api, stream } from "../api.js";
import { ago, badge, busy, field, fill, h, json, ms, notice, pageHead, panel, parseJson, table } from "../ui.js";

const EXAMPLE = {
  nodes: [
    { id: "start", type: "start", data: { variables: [{ name: "topic" }] } },
    { id: "llm", type: "llm", data: { prompt: "Write one short paragraph about {{#start.topic#}}." } },
    { id: "end", type: "end", data: { outputs: [{ name: "text", value: "{{#llm.text#}}" }] } },
  ],
  edges: [
    { source: "start", target: "llm" },
    { source: "llm", target: "end" },
  ],
};

const nodeCount = (w) => `${w.graph.nodes.length} nodes`;
const types = (w) => [...new Set(w.graph.nodes.map((n) => n.type))].join(", ");

export async function workflowsPage() {
  const workflows = await api("/v1/workflows");
  return h("div", {},
    pageHead("Workflows", h("a", { class: "button", href: "#/workflows/new" }, "New workflow")),
    panel(table(workflows, [
      ["Name", (w) => h("strong", {}, w.name)],
      ["Description", (w) => w.description],
      ["Graph", (w) => `${nodeCount(w)}: ${types(w)}`],
      ["Updated", (w) => ago(w.updated_at)],
    ], { onclick: (w) => (location.hash = `#/workflows/${w.id}`), empty: "No workflows yet." })),
  );
}

// Name, description and the graph as JSON. Used for new and existing workflows.
function editor(workflow, onSave, saveText) {
  const name = h("input", { value: workflow?.name ?? "", required: true });
  const description = h("input", { value: workflow?.description ?? "" });
  const graph = h("textarea", { class: "code", spellcheck: false }, JSON.stringify(workflow?.graph ?? EXAMPLE, null, 2));
  const errors = h("div");
  const save = h("button", { type: "submit" }, saveText);
  return h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(save, errors, () => onSave({
        name: name.value.trim(), description: description.value, graph: parseJson(graph.value, "The graph"),
      }));
    },
  }, errors,
    h("div", { class: "row" }, field("Name", name), field("Description", description)),
    field("Graph (nodes: start, llm, knowledge_retrieval, if_else, template, http_request, end)", graph),
    h("div", { class: "actions" }, save));
}

export async function newWorkflowPage() {
  return h("div", {},
    pageHead("New workflow"),
    panel(editor(null, async (body) => {
      const workflow = await api("/v1/workflows", { method: "POST", json: body });
      location.hash = `#/workflows/${workflow.id}`;
    }, "Create workflow")),
  );
}

export async function workflowPage({ id }) {
  const workflow = await api(`/v1/workflows/${id}`);
  const saved = h("div");
  const runBox = h("div", {}, runForm(workflow));
  return h("div", {},
    pageHead(workflow.name, h("a", { href: "#/workflows" }, "All workflows")),
    h("div", { class: "cols" },
      panel(h("h2", {}, "Run"), runBox),
      panel(h("h2", {}, "Graph"), saved, editor(workflow, async (body) => {
        Object.assign(workflow, await api(`/v1/workflows/${id}`, { method: "PUT", json: body }));
        fill(runBox, runForm(workflow)); // the inputs may have changed
        fill(saved, notice("Saved.", "info"));
      }, "Save graph")),
    ),
  );
}

function runForm(workflow) {
  const start = workflow.graph.nodes.find((n) => n.type === "start");
  const variables = start?.data?.variables ?? [];
  const inputs = Object.fromEntries(variables.map((v) => [v.name, h("input", { placeholder: v.required === false ? "optional" : "" })]));
  const nodes = h("div");
  const result = h("div");
  const run = h("button", { type: "submit" }, "Run");

  const go = async () => {
    const values = {};
    for (const [name, input] of Object.entries(inputs)) if (input.value !== "") values[name] = input.value;
    // One row per node, filled in as its node_started and node_finished events arrive.
    const rows = new Map();
    const body = h("tbody");
    fill(nodes, h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Node", "Type", "Status", "Time"].map((t) => h("th", {}, t)))), body)));
    fill(result);

    const rowFor = (event) => {
      if (!rows.has(event.node_id)) {
        const status = h("td", {}, badge("running"));
        const time = h("td");
        const details = h("tr", { hidden: true }, h("td", { colspan: 4 }));
        const row = h("tr", { class: "clickable", onclick: () => (details.hidden = !details.hidden) },
          h("td", {}, event.title || event.node_id), h("td", {}, event.node_type), status, time);
        rows.set(event.node_id, { status, time, details });
        body.append(row, details);
      }
      return rows.get(event.node_id);
    };

    await stream(`/v1/workflows/${workflow.id}/run`, { inputs: values }, (event) => {
      if (event.event === "node_started") {
        rowFor(event);
      } else if (event.event === "node_finished") {
        const row = rowFor(event); // a skipped node finishes without starting
        fill(row.status, badge(event.status, event.status === "skipped" ? "" : event.status));
        row.time.textContent = event.status === "skipped" ? "" : ms(event.elapsed_ms);
        fill(row.details.firstChild,
          event.error ? notice(event.error) : null,
          h("div", { class: "small muted" }, "inputs"), json(event.inputs),
          h("div", { class: "small muted" }, "outputs"), json(event.outputs));
      } else if (event.event === "workflow_finished") {
        fill(result,
          h("p", {}, badge(event.status), ` in ${ms(event.elapsed_ms)}`,
            event.trace_id ? [" · ", h("a", { href: `#/traces/${event.trace_id}` }, "trace")] : null),
          event.error ? notice(event.error) : null,
          h("h2", {}, "Outputs"), json(event.outputs));
      }
    });
  };

  return h("form", { onsubmit: (e) => { e.preventDefault(); busy(run, result, go); } },
    variables.length ? Object.entries(inputs).map(([name, input]) => field(name, input)) : h("p", { class: "muted" }, "This workflow has no inputs."),
    h("div", { class: "actions" }, run),
    h("p", { class: "small muted" }, "Click a node to see its inputs and outputs."),
    nodes, result);
}
