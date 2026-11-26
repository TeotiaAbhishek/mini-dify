// MCP servers: register a remote server, see the tools it offers, refresh or remove it.
// Apps pick servers in their settings; their tools then show up as name__tool.

import { api } from "../api.js";
import { ago, busy, field, fill, h, pageHead, panel, parseJson, table } from "../ui.js";

export async function mcpPage() {
  const list = h("div");
  const render = async () => {
    const servers = await api("/v1/mcp-servers");
    fill(list, (servers.length ? servers.map(serverPanel) : [panel(h("div", { class: "empty" }, "No MCP servers yet."))]));
  };

  const serverPanel = (s) => {
    const errors = h("div");
    const refresh = h("button", { class: "secondary", onclick: () => busy(refresh, errors, async () => { await api(`/v1/mcp-servers/${s.id}/refresh`, { method: "POST" }); await render(); }) }, "Refresh tools");
    const remove = h("button", {
      class: "danger",
      onclick: () => confirm(`Remove ${s.name}? Apps using it lose its tools.`) && busy(remove, errors, async () => {
        await api(`/v1/mcp-servers/${s.id}`, { method: "DELETE" });
        await render();
      }),
    }, "Remove");
    return panel(
      h("div", { class: "page-head" }, h("h2", { style: "margin: 0" }, s.name), h("span", { class: "muted small" }, s.url), h("span", { class: "grow" }), refresh, remove),
      errors,
      h("p", { class: "muted small" }, `${s.server_info.name ?? "server"} ${s.server_info.version ?? ""} · protocol ${s.protocol_version} · updated ${ago(s.updated_at)}${s.header_names.length ? ` · headers: ${s.header_names.join(", ")}` : ""}`),
      table(s.tools, [
        ["Tool", (t) => h("code", {}, `${s.name}__${t.name}`)],
        ["Description", (t) => t.description],
        ["Arguments", (t) => Object.keys(t.input_schema.properties ?? {}).join(", ")],
      ], { empty: "This server offers no tools." }),
    );
  };

  const name = h("input", { placeholder: "weather", pattern: "[a-z][a-z0-9_]{0,29}", required: true });
  const url = h("input", { placeholder: "https://example.com/mcp", required: true });
  const headers = h("textarea", { class: "code", style: "min-height: 70px", spellcheck: false }, "{}");
  const errors = h("div");
  const add = h("button", { type: "submit" }, "Connect");
  const form = h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(add, errors, async () => {
        await api("/v1/mcp-servers", { method: "POST", json: { name: name.value, url: url.value, headers: parseJson(headers.value, "Headers") } });
        name.value = url.value = "";
        await render();
      });
    },
  }, errors,
    h("div", { class: "row" }, field("Name (lowercase, used as the tool prefix)", name), field("URL", url)),
    field("Headers, as JSON (sent to the server, never shown again)", headers),
    h("div", { class: "actions" }, add));

  await render();
  return h("div", {},
    pageHead("MCP servers"),
    h("p", { class: "muted" }, "mini-dify is an MCP server too: every workflow is a tool at ", h("code", {}, "/mcp"), "."),
    panel(h("h2", {}, "Connect a server"), form),
    list,
  );
}
