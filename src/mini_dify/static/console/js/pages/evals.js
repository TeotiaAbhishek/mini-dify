// Evals: eval sets with their cases, runs against an app, and a run's results,
// compared with an earlier run to find regressions.

import { api } from "../api.js";
import {
  ago, badge, busy, field, fill, h, ms, notice, num, pageHead, panel, parseJson, pct, poll, short, stat, table,
} from "../ui.js";

const EXAMPLE = {
  name: "Smoke test",
  description: "",
  cases: [{ query: "How many vacation days do I get?", must_include: ["25"] }],
};

export async function evalsPage() {
  const sets = await api("/v1/eval-sets");
  const text = h("textarea", { class: "code", spellcheck: false, style: "min-height: 200px" }, JSON.stringify(EXAMPLE, null, 2));
  const file = h("input", {
    type: "file", accept: ".json",
    onchange: async () => file.files[0] && (text.value = await file.files[0].text()),
  });
  const errors = h("div");
  const create = h("button", { type: "submit" }, "Create eval set");
  const form = h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(create, errors, async () => {
        const set = await api("/v1/eval-sets", { method: "POST", json: parseJson(text.value, "The eval set") });
        location.hash = `#/evals/${set.id}`;
      });
    },
  }, errors, field("Or load a file, like examples/hr/eval-set.json", file), text, h("div", { class: "actions" }, create));

  return h("div", {},
    pageHead("Evals"),
    panel(table(sets, [
      ["Name", (s) => h("strong", {}, s.name)],
      ["Description", (s) => s.description],
      ["Cases", (s) => s.cases.length],
      ["Created", (s) => ago(s.created_at)],
    ], { onclick: (s) => (location.hash = `#/evals/${s.id}`), empty: "No eval sets yet." })),
    panel(h("h2", {}, "New eval set"), form),
  );
}

const passRate = (r) => (r.summary ? pct(r.summary.pass_rate) : "");

export async function evalSetPage({ id }) {
  const [set, apps] = await Promise.all([api(`/v1/eval-sets/${id}`), api("/v1/apps")]);
  const appNames = Object.fromEntries(apps.map((a) => [a.id, a.name]));

  const runs = h("div");
  const render = async () => {
    const list = await api(`/v1/eval-runs?eval_set_id=${id}&limit=50`);
    fill(runs, table(list, [
      ["Run", (r) => short(r.id)],
      ["App", (r) => appNames[r.app_id] ?? h("span", { class: "muted" }, "deleted")],
      ["Status", (r) => badge(r.status)],
      ["Pass rate", passRate],
      ["Judge", (r) => (r.judge_model ? `${num(r.summary?.judge)} (${r.judge_model})` : "")],
      ["p50", (r) => ms(r.summary?.latency_p50_ms)],
      ["Started", (r) => ago(r.created_at)],
    ], { onclick: (r) => (location.hash = `#/eval-runs/${r.id}`), empty: "No runs yet." }));
    return list.some((r) => r.status === "running");
  };

  const app = h("select", {}, apps.map((a) => h("option", { value: a.id }, a.name)));
  const judge = h("input", { type: "checkbox" });
  const judgeModel = h("input", { placeholder: "default model" });
  const concurrency = h("input", { type: "number", value: 1, min: 1, max: 4 });
  const errors = h("div");
  const start = h("button", { type: "submit", disabled: !apps.length }, "Start run");
  const form = h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(start, errors, async () => {
        await api(`/v1/eval-sets/${id}/runs`, {
          method: "POST",
          json: { app_id: app.value, judge: judge.checked, judge_model: judgeModel.value || null, concurrency: Number(concurrency.value) },
        });
        if (await render()) poll(render);
      });
    },
  }, errors,
    apps.length ? null : notice("Create an app first.", "info"),
    h("div", { class: "row" },
      field("App", app),
      field("Concurrency", concurrency),
      field("Judge model", judgeModel),
      h("label", { class: "check" }, judge, "Grade with an LLM judge")),
    h("div", { class: "actions" }, start));

  if (await render()) poll(render);
  return h("div", {},
    pageHead(set.name, h("a", { href: "#/evals" }, "All eval sets")),
    set.description ? h("p", { class: "muted" }, set.description) : null,
    panel(h("h2", {}, "Run it"), form),
    panel(h("h2", {}, "Runs"), runs),
    panel(h("h2", {}, `Cases (${set.cases.length})`), table(set.cases, [
      ["#", (c) => c.position + 1],
      ["Question", (c) => c.query],
      ["Must include", (c) => c.must_include.join(", ")],
      ["Expected sources", (c) => c.expected_sources.join(", ")],
      ["Expected answer", (c) => h("span", { class: "muted" }, c.expected_answer ?? "")],
    ])),
  );
}

const scoreText = (scores) => Object.entries(scores).map(([k, v]) => `${k} ${num(v)}`).join(" · ");

export async function evalRunPage({ id }) {
  const box = h("div", {}, h("p", { class: "muted" }, "Loading…"));
  const render = async () => {
    const run = await api(`/v1/eval-runs/${id}`);
    fill(box, await runView(run));
    return run.status === "running";
  };
  if (await render()) poll(render);
  return box;
}

async function runView(run) {
  const s = run.summary ?? {};
  const head = pageHead(`Run ${short(run.id)}`, badge(run.status), h("a", { href: `#/evals/${run.eval_set_id}` }, "Eval set"));
  const parts = [head, h("p", { class: "muted small" }, `App "${run.app_config.name}" as it was when the run started${run.judge_model ? ` · judged by ${run.judge_model}` : ""}`)];
  if (run.error) parts.push(notice(run.error));
  if (run.status === "running") parts.push(notice(`Running… ${run.results.length} cases done so far.`, "info"));
  if (run.summary) {
    parts.push(h("div", { class: "stats" },
      stat("Pass rate", `${pct(s.pass_rate)} (${s.passed}/${s.cases})`),
      stat("Contains", num(s.contains)),
      stat("Source recall", num(s.source_recall)),
      stat("Judge (1-5)", num(s.judge)),
      stat("Latency p50", ms(s.latency_p50_ms)),
      stat("Latency p95", ms(s.latency_p95_ms)),
      stat("Tokens", `${s.prompt_tokens} + ${s.completion_tokens}`),
      stat("Errors", s.errors)));
    parts.push(await compareBox(run));
  }
  parts.push(panel(h("h2", {}, "Results"), table(run.results, [
    ["#", (r) => r.position + 1],
    ["", (r) => badge(r.passed ? "pass" : "fail")],
    ["Answer", (r) => [r.error ? notice(r.error) : null, h("div", {}, r.answer ?? ""), r.judge_reason ? h("div", { class: "small muted" }, `Judge: ${r.judge_reason}`) : null]],
    ["Scores", (r) => h("span", { class: "small" }, scoreText(r.scores))],
    ["Time", (r) => ms(r.latency_ms)],
    ["", (r) => (r.trace_id ? h("a", { href: `#/traces/${r.trace_id}` }, "trace") : "")],
  ])));
  return h("div", {}, parts);
}

// Pick an earlier finished run of the same set and see what changed.
async function compareBox(run) {
  const others = (await api(`/v1/eval-runs?eval_set_id=${run.eval_set_id}&limit=50`))
    .filter((r) => r.id !== run.id && r.status === "succeeded");
  if (!others.length) return panel(h("h2", {}, "Compare"), h("p", { class: "muted" }, "Start another run of this set to compare the two."));

  const baseline = h("select", {}, others.map((r) => h("option", { value: r.id }, `${short(r.id)} · ${passRate(r)} · ${ago(r.created_at)}`)));
  const out = h("div");
  const compare = h("button", { type: "button" }, "Compare");
  compare.addEventListener("click", () => busy(compare, out, async () => {
    const c = await api(`/v1/eval-runs/${run.id}/compare?baseline=${baseline.value}`);
    const changes = (list, title) => [h("h2", { style: "margin-top: 18px" }, `${title} (${list.length})`), table(list, [
      ["#", (x) => x.position + 1],
      ["Question", (x) => x.query],
      ["Before", (x) => h("span", { class: "small" }, scoreText(x.baseline_scores))],
      ["Now", (x) => h("span", { class: "small" }, scoreText(x.scores))],
      ["", (x) => (x.trace_id ? h("a", { href: `#/traces/${x.trace_id}` }, "trace") : "")],
    ], { empty: "None" })];
    fill(out,
      table(Object.entries(c.summary), [
        ["Number", ([name]) => name],
        ["Baseline", ([, v]) => num(v.baseline)],
        ["This run", ([, v]) => num(v.run)],
        ["Change", ([, v]) => (v.change === null ? "–" : `${v.change > 0 ? "+" : ""}${num(v.change)}`)],
      ]),
      ...changes(c.regressions, "Regressions: passed before, fail now"),
      ...changes(c.fixes, "Fixes: failed before, pass now"),
      h("p", { class: "muted small" }, `${c.unchanged} cases unchanged.`));
  }));
  return panel(h("h2", {}, "Compare with an earlier run"), h("div", { class: "actions" }, baseline, compare), out);
}
