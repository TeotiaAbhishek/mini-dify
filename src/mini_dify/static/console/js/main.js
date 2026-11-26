// The console is one page. The part after # in the address says what to show:
// #/apps, #/apps/<id>, #/traces/<id>... Changing it doesn't reload, it fires "hashchange",
// and route() swaps the contents of <main id="view">.

import { ApiError, getToken, setToken } from "./api.js";
import { appsPage, appPage, newAppPage } from "./pages/apps.js";
import { evalSetPage, evalRunPage, evalsPage } from "./pages/evals.js";
import { datasetPage, knowledgePage } from "./pages/knowledge.js";
import { mcpPage } from "./pages/mcp.js";
import { tracePage, tracesPage } from "./pages/traces.js";
import { newWorkflowPage, workflowPage, workflowsPage } from "./pages/workflows.js";
import { $, fill, h, leave, notice, pageHead, panel } from "./ui.js";

// Each route: a pattern, which nav section it belongs to, and the page function.
// ":id" matches one part of the path and is passed to the page.
const routes = [
  ["/apps", "apps", appsPage],
  ["/apps/new", "apps", newAppPage],
  ["/apps/:id", "apps", appPage],
  ["/knowledge", "knowledge", knowledgePage],
  ["/knowledge/:id", "knowledge", datasetPage],
  ["/workflows", "workflows", workflowsPage],
  ["/workflows/new", "workflows", newWorkflowPage],
  ["/workflows/:id", "workflows", workflowPage],
  ["/traces", "traces", tracesPage],
  ["/traces/:id", "traces", tracePage],
  ["/evals", "evals", evalsPage],
  ["/evals/:id", "evals", evalSetPage],
  ["/eval-runs/:id", "evals", evalRunPage],
  ["/mcp", "mcp", mcpPage],
];

function match(path) {
  const parts = path.split("/").filter(Boolean);
  for (const [pattern, section, page] of routes) {
    const want = pattern.split("/").filter(Boolean);
    if (want.length !== parts.length) continue;
    const params = {};
    if (want.every((p, i) => (p.startsWith(":") ? (params[p.slice(1)] = parts[i]) : p === parts[i]))) {
      return { section, page, params };
    }
  }
  return null;
}

function tokenForm(message) {
  const input = h("input", { type: "password", placeholder: "CONSOLE_TOKEN", value: getToken() });
  const save = () => {
    setToken(input.value.trim());
    route();
  };
  return panel(
    h("h2", {}, "Console token"),
    notice(message),
    h("p", { class: "muted" }, "This server has CONSOLE_TOKEN set. Paste it here. It's kept in this browser only."),
    h("div", { class: "actions" }, input, h("button", { onclick: save }, "Save")),
  );
}

async function route() {
  leave(); // stop the old page's timers
  const hash = location.hash;
  const path = hash.slice(1).split("?")[0] || "/apps"; // "?status=error" is read by the page
  const view = $("view");
  const found = match(path);
  for (const link of document.querySelectorAll("#nav a[data-section]")) {
    link.classList.toggle("active", link.dataset.section === found?.section);
  }
  if (!found) {
    fill(view, pageHead("Not found"), notice(`No page at #${path}`));
    return;
  }
  fill(view, h("p", { class: "muted" }, "Loading…"));
  try {
    const node = await found.page(found.params);
    // If the address changed while this page was loading, the newer page wins.
    if (location.hash === hash) fill(view, node);
  } catch (error) {
    if (location.hash !== hash) return;
    if (error instanceof ApiError && error.status === 401) {
      fill(view, pageHead("Sign in"), tokenForm(error.detail));
    } else {
      fill(view, pageHead("Something went wrong"), notice(error.detail ?? error.message));
      console.error(error);
    }
  }
}

window.addEventListener("hashchange", route);
route();
