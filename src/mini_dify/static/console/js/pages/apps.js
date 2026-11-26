// Apps: the list, a form to create one, and one app's page with four tabs:
// Settings (edit, delete), Try it (chat), API keys, Logs (its conversations).

import { api, stream } from "../api.js";
import {
  ago, badge, busy, field, fill, h, ms, notice, pageHead, panel, short, table, tabs, time,
} from "../ui.js";

const mode = (app) => (app.tools.length || app.mcp_server_ids.length ? "agent" : "chat");

export async function appsPage() {
  const apps = await api("/v1/apps");
  return h("div", {},
    pageHead("Apps", h("a", { class: "button", href: "#/apps/new" }, "New app")),
    panel(table(apps, [
      ["Name", (a) => h("strong", {}, a.name)],
      ["Mode", (a) => badge(mode(a), mode(a) === "agent" ? "llm" : "")],
      ["Model", (a) => a.model ?? h("span", { class: "muted" }, "default")],
      ["Knowledge", (a) => (a.dataset_ids.length ? `${a.dataset_ids.length} dataset(s)` : "")],
      ["Tools", (a) => a.tools.join(", ")],
      ["Created", (a) => ago(a.created_at)],
    ], { onclick: (a) => (location.hash = `#/apps/${a.id}`), empty: "No apps yet. Create one to start." })),
  );
}

// ---------- the settings form, used to create and to edit ----------

async function appForm(app, onSave, saveText) {
  // What can be attached to an app: datasets, built-in tools and MCP servers.
  const [datasets, tools, servers] = await Promise.all([
    api("/v1/datasets"), api("/v1/tools"), api("/v1/mcp-servers"),
  ]);
  const value = (name, fallback) => app?.[name] ?? fallback;

  const name = h("input", { value: value("name", ""), required: true });
  const prompt = h("textarea", {}, value("system_prompt", "You are a helpful assistant."));
  const model = h("input", { value: value("model", "") ?? "", placeholder: "default model" });
  const temperature = h("input", { type: "number", step: "0.1", min: "0", max: "2", value: value("temperature", 0.7) });
  const memory = h("input", { type: "number", min: "0", max: "100", value: value("memory_window", 10) });
  const topK = h("input", { type: "number", min: "1", max: "20", value: value("retrieval_top_k", 4) });
  const agentMode = h("select", {},
    ["function_calling", "react"].map((m) => h("option", { value: m, selected: value("agent_mode") === m }, m)));
  const iterations = h("input", { type: "number", min: "1", max: "10", value: value("max_iterations", 5) });

  // A row of checkboxes; picked() gives back the values that are ticked.
  const checks = (items, chosen, label, valueOf) => {
    const boxes = items.map((item) => h("input", { type: "checkbox", value: valueOf(item), checked: chosen.includes(valueOf(item)) }));
    const node = items.length
      ? h("div", { class: "checks" }, items.map((item, i) => h("label", { class: "check" }, boxes[i], label(item))))
      : h("div", { class: "muted small" }, "None yet");
    return { node, picked: () => boxes.filter((b) => b.checked).map((b) => b.value) };
  };
  const datasetBoxes = checks(datasets, value("dataset_ids", []), (d) => d.name, (d) => d.id);
  const toolBoxes = checks(tools, value("tools", []), (t) => t.name, (t) => t.name);
  const serverBoxes = checks(servers, value("mcp_server_ids", []), (s) => `${s.name} (${s.tools.length} tools)`, (s) => s.id);

  const errors = h("div");
  const save = h("button", { type: "submit" }, saveText);
  const form = h("form", {
    onsubmit: (event) => {
      event.preventDefault();
      busy(save, errors, () => onSave({
        name: name.value.trim(),
        system_prompt: prompt.value,
        model: model.value.trim() || null,
        temperature: Number(temperature.value),
        memory_window: Number(memory.value),
        dataset_ids: datasetBoxes.picked(),
        retrieval_top_k: Number(topK.value),
        tools: toolBoxes.picked(),
        agent_mode: agentMode.value,
        max_iterations: Number(iterations.value),
        mcp_server_ids: serverBoxes.picked(),
      }));
    },
  },
    errors,
    field("Name", name),
    field("System prompt (use {{name}} for a variable the user fills in)", prompt),
    h("div", { class: "row" }, field("Model", model), field("Temperature", temperature), field("Memory (past messages)", memory)),
    h("h2", {}, "Knowledge"),
    datasetBoxes.node,
    h("div", { class: "row" }, field("Chunks to retrieve", topK)),
    h("h2", {}, "Tools"),
    h("p", { class: "muted small" }, "Picking a tool or an MCP server makes this an agent app."),
    toolBoxes.node,
    h("label", {}, "MCP servers", serverBoxes.node),
    h("div", { class: "row" }, field("Agent strategy", agentMode), field("Max iterations", iterations)),
    h("div", { class: "actions" }, save),
  );
  return form;
}

export async function newAppPage() {
  const form = await appForm(null, async (body) => {
    const app = await api("/v1/apps", { method: "POST", json: body });
    location.hash = `#/apps/${app.id}`;
  }, "Create app");
  return h("div", {}, pageHead("New app"), panel(form));
}

// ---------- one app ----------

export async function appPage({ id }) {
  const app = await api(`/v1/apps/${id}`);
  const settings = await settingsTab(app);
  return h("div", {},
    pageHead(app.name, badge(mode(app), mode(app) === "agent" ? "llm" : "")),
    tabs({
      "Try it": () => chatTab(app),
      Settings: () => settings,
      "API keys": () => later(keysTab(app)),
      Logs: () => later(logsTab(app)),
    }),
  );
}

// A tab whose contents need a request: show "Loading…" and fill it in when it's ready.
function later(promise) {
  const box = h("div", {}, h("p", { class: "muted" }, "Loading…"));
  promise.then((node) => fill(box, node), (error) => fill(box, notice(error.detail ?? error.message)));
  return box;
}

async function settingsTab(app) {
  const saved = h("div");
  const form = await appForm(app, async (body) => {
    await api(`/v1/apps/${app.id}`, { method: "PUT", json: body });
    fill(saved, notice("Saved.", "info"));
  }, "Save changes");
  const remove = h("button", {
    class: "danger",
    onclick: async () => {
      if (!confirm(`Delete "${app.name}" with its conversations, API keys and eval runs?`)) return;
      await api(`/v1/apps/${app.id}`, { method: "DELETE" });
      location.hash = "#/apps";
    },
  }, "Delete app");
  return h("div", {}, saved, panel(form), panel(h("h2", {}, "Danger zone"), remove));
}

// ---------- Try it: chat with the app, streamed ----------

function chatTab(app) {
  let conversationId = null;
  const log = h("div", { class: "chat-log" });
  const inputs = Object.fromEntries(app.variables.map((v) => [v, h("input", { placeholder: v })]));
  const query = h("textarea", { placeholder: "Ask something…", rows: 2, style: "min-height: 60px" });
  const send = h("button", { type: "submit" }, "Send");
  const restart = h("button", {
    type: "button", class: "secondary",
    onclick: () => {
      conversationId = null;
      fill(log);
      Object.values(inputs).forEach((input) => (input.disabled = false));
    },
  }, "New conversation");

  const ask = async () => {
    const text = query.value.trim();
    if (!text) return;
    query.value = "";
    log.append(h("div", { class: "msg user" }, text));
    const answer = h("span");
    const steps = h("div");
    const meta = h("div", { class: "meta" });
    const bubble = h("div", { class: "msg assistant" }, steps, answer, meta);
    log.append(bubble);

    const values = Object.fromEntries(Object.entries(inputs).map(([k, input]) => [k, input.value]));
    const body = { query: text, inputs: values, conversation_id: conversationId };
    const started = performance.now();
    try {
      await stream(`/v1/apps/${app.id}/chat-messages`, body, (event) => {
        if (event.conversation_id) {
          conversationId = event.conversation_id;
          Object.values(inputs).forEach((input) => (input.disabled = true)); // fixed for this conversation
        } else if (event.agent_step) {
          const s = event.agent_step;
          steps.append(h("div", { class: "step" }, `🔧 ${s.tool}(${JSON.stringify(s.arguments)}) → ${s.observation.slice(0, 160)}`));
        } else if (event.error) {
          answer.append(h("div", { class: "notice" }, event.error));
        } else if (event.message_id) {
          meta.append(`${ms(performance.now() - started)}`);
          if (event.citations.length) meta.append(` · ${event.citations.map((c) => `[${c.number}] ${c.filename}`).join(", ")}`);
          if (event.trace_id) meta.append(" · ", h("a", { href: `#/traces/${event.trace_id}` }, "trace"));
        } else if (event.delta !== undefined) {
          answer.append(event.delta);
        }
      });
    } catch (error) {
      answer.append(h("div", { class: "notice" }, error.detail ?? error.message));
    }
    bubble.scrollIntoView({ block: "end" });
  };

  const form = h("form", { onsubmit: (e) => { e.preventDefault(); busy(send, null, ask); } },
    app.variables.length ? h("div", { class: "row" }, Object.entries(inputs).map(([k, input]) => h("label", {}, k, input))) : null,
    query,
    h("div", { class: "actions", style: "margin-top: 8px" }, send, restart),
  );
  query.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  return panel(log, form);
}

// ---------- API keys ----------

async function keysTab(app) {
  const box = h("div");
  const shown = h("div");

  const render = async () => {
    const keys = await api(`/v1/apps/${app.id}/api-keys`);
    fill(box, table(keys, [
      ["Name", (k) => k.name],
      ["Key", (k) => h("code", {}, `${k.prefix}…`)],
      ["Limit", (k) => (k.rate_limit_per_minute ? `${k.rate_limit_per_minute}/min` : "default")],
      ["Last used", (k) => ago(k.last_used_at)],
      ["", (k) => (k.revoked_at ? badge("revoked") : h("button", {
        class: "danger",
        onclick: async () => {
          if (!confirm(`Revoke "${k.name}"? Apps using it stop working at once.`)) return;
          await api(`/v1/api-keys/${k.id}`, { method: "DELETE" });
          render();
        },
      }, "Revoke"))],
    ], { empty: "No keys yet." }));
  };

  const name = h("input", { placeholder: "website", required: true });
  const limit = h("input", { type: "number", min: "1", placeholder: "default" });
  const errors = h("div");
  const create = h("button", { type: "submit" }, "Create key");
  const form = h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(create, errors, async () => {
        const key = await api(`/v1/apps/${app.id}/api-keys`, {
          method: "POST",
          json: { name: name.value.trim(), rate_limit_per_minute: limit.value ? Number(limit.value) : null },
        });
        name.value = "";
        fill(shown, panel(
          h("h2", {}, "Copy this key now"),
          h("p", { class: "muted" }, "It's shown only once. mini-dify keeps only its hash."),
          h("div", { class: "secret" }, key.key),
          h("p", { class: "small muted" }, "Use it with the public API or the ", h("a", { href: "/ui/", target: "_blank" }, "chat page"), "."),
        ));
        render();
      });
    },
  }, errors, h("div", { class: "row" }, field("Name", name), field("Requests per minute", limit), h("div", {}, create)));

  await render();
  return h("div", {}, shown, panel(h("h2", {}, "New key"), form), panel(box));
}

// ---------- Logs: every conversation, from the console, the public API or evals ----------

async function logsTab(app) {
  const conversations = await api(`/v1/apps/${app.id}/conversations`);
  const detail = h("div");
  const open = async (c) => {
    const messages = await api(`/v1/conversations/${c.id}/messages`);
    fill(detail, panel(
      h("h2", {}, `Conversation ${short(c.id)}`),
      h("p", { class: "muted small" }, `${c.end_user ? `User ${c.end_user} · ` : ""}${time(c.created_at)}${Object.keys(c.inputs).length ? ` · inputs ${JSON.stringify(c.inputs)}` : ""}`),
      h("div", { class: "chat-log" }, messages.map((m) => h("div", { class: `msg ${m.role}` },
        (m.agent_steps ?? []).map((s) => h("div", { class: "step" }, `🔧 ${s.tool}(${JSON.stringify(s.arguments)})`)),
        m.content,
        m.role === "assistant" ? h("div", { class: "meta" },
          `${m.prompt_tokens ?? 0} + ${m.completion_tokens ?? 0} tokens`,
          m.trace_id ? [" · ", h("a", { href: `#/traces/${m.trace_id}` }, "trace")] : null) : null,
      ))),
    ));
    detail.scrollIntoView({ behavior: "smooth" });
  };
  return h("div", {},
    panel(table(conversations, [
      ["First question", (c) => c.first_query ?? ""],
      ["User", (c) => c.end_user ?? h("span", { class: "muted" }, "none")],
      ["Messages", (c) => c.message_count],
      ["Started", (c) => ago(c.created_at)],
    ], { onclick: open, empty: "No conversations yet. Try the app, or call it with an API key." })),
    detail,
  );
}
