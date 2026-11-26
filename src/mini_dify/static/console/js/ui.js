// Small helpers for building the page. Everything is made with h(), which sets text with
// textContent, never innerHTML: names, answers and documents can't inject HTML.

// h("button", { class: "secondary", onclick: save }, "Save")
// Children can be nodes, strings, arrays of them, or null/false (left out).
const clean = (children) => children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false);

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (name.startsWith("on")) el.addEventListener(name.slice(2), value);
    else if (name === "class") el.className = value;
    else if (name in el && typeof value !== "string") el[name] = value; // value, checked...
    else el.setAttribute(name, value === true ? "" : value);
  }
  el.append(...clean(children));
  return el;
}

// Replace what's inside el, with the same rules as h().
export function fill(el, ...children) {
  el.replaceChildren(...clean(children));
}

export const $ = (id) => document.getElementById(id);

// ---------- formatting ----------

export function ago(iso) {
  if (!iso) return "never";
  const seconds = (Date.now() - new Date(iso)) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

export const time = (iso) => (iso ? new Date(iso).toLocaleString() : "");
export const ms = (n) => (n === null || n === undefined ? "–" : n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`);
export const pct = (n) => (n === null || n === undefined ? "–" : `${Math.round(n * 100)}%`);
export const num = (n, digits = 2) => (n === null || n === undefined ? "–" : Number(n.toFixed(digits)).toString());
export const short = (id) => String(id).slice(0, 8);

export const badge = (text, kind = text) => h("span", { class: `badge ${kind}` }, text);
export const json = (value) => h("pre", {}, JSON.stringify(value, null, 2));

// ---------- building blocks ----------

export function pageHead(title, ...actions) {
  return h("div", { class: "page-head" }, h("h1", {}, title), h("span", { class: "grow" }), ...actions);
}

export const panel = (...children) => h("div", { class: "panel" }, ...children);
export const field = (text, input) => h("label", {}, text, input);
export const notice = (text, kind = "") => h("div", { class: `notice ${kind}` }, text);
export const stat = (label, value) => h("div", { class: "stat" }, h("div", { class: "value" }, value), h("div", { class: "label" }, label));

// table(rows, [["Name", (r) => r.name], ...], { onclick: (r) => ..., empty: "No apps yet" })
export function table(rows, columns, { onclick, empty = "Nothing here yet" } = {}) {
  if (!rows.length) return h("div", { class: "empty" }, empty);
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, columns.map(([title]) => h("th", {}, title)))),
    h("tbody", {}, rows.map((row) => h("tr", {
      class: onclick ? "clickable" : "",
      onclick: onclick && (() => onclick(row)),
    }, columns.map(([, cell]) => h("td", {}, cell(row) ?? ""))))),
  ));
}

// Tabs: tabs({ Settings: () => node, Logs: () => node }, "Settings")
export function tabs(pages, first) {
  const body = h("div");
  const bar = h("div", { class: "tabs" });
  const show = (name) => {
    for (const button of bar.children) button.classList.toggle("active", button.textContent === name);
    fill(body, pages[name]());
  };
  for (const name of Object.keys(pages)) bar.append(h("button", { type: "button", onclick: () => show(name) }, name));
  show(first ?? Object.keys(pages)[0]);
  return h("div", {}, bar, body);
}

// Run an async action from a button: disable it while it works, show errors in `box`.
export async function busy(button, box, action) {
  button.disabled = true;
  box?.replaceChildren();
  try {
    await action();
  } catch (error) {
    if (box) fill(box, notice(error.detail ?? error.message));
    else throw error;
  } finally {
    button.disabled = false;
  }
}

// Parse a JSON textarea, saying which one is wrong.
export function parseJson(text, what) {
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${what} isn't valid JSON: ${error.message}`);
  }
}

// ---------- leaving a page ----------
// Pages that poll register a cleanup here. The router calls leave() before showing the next page,
// so a timer from the old page never keeps running.

let cleanups = [];
export const onLeave = (fn) => cleanups.push(fn);
export function leave() {
  for (const fn of cleanups) fn();
  cleanups = [];
}

// Call fn every `every` ms until it returns false or the page changes.
export function poll(fn, every = 1500) {
  let timer;
  const tick = async () => {
    if ((await fn()) !== false) timer = setTimeout(tick, every);
  };
  timer = setTimeout(tick, every);
  onLeave(() => clearTimeout(timer));
}
