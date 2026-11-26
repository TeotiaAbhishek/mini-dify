// Knowledge: datasets, the documents in them, their chunks, and a retrieval test
// that shows what an app would find for a question.

import { api } from "../api.js";
import { ago, badge, busy, field, fill, h, notice, num, pageHead, panel, poll, table } from "../ui.js";

export async function knowledgePage() {
  const datasets = await api("/v1/datasets");
  const name = h("input", { required: true, placeholder: "HR handbook" });
  const description = h("input", { placeholder: "optional" });
  const size = h("input", { type: "number", value: 1000, min: 100, max: 4000 });
  const overlap = h("input", { type: "number", value: 150, min: 0, max: 1000 });
  const errors = h("div");
  const create = h("button", { type: "submit" }, "Create dataset");
  const form = h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(create, errors, async () => {
        const dataset = await api("/v1/datasets", {
          method: "POST",
          json: { name: name.value.trim(), description: description.value, chunk_size: Number(size.value), chunk_overlap: Number(overlap.value) },
        });
        location.hash = `#/knowledge/${dataset.id}`;
      });
    },
  }, errors,
    h("div", { class: "row" }, field("Name", name), field("Description", description)),
    h("div", { class: "row" }, field("Chunk size (characters)", size), field("Chunk overlap", overlap), h("div", {}, create)));

  return h("div", {},
    pageHead("Knowledge"),
    panel(table(datasets, [
      ["Name", (d) => h("strong", {}, d.name)],
      ["Description", (d) => d.description],
      ["Chunks", (d) => `${d.chunk_size} / ${d.chunk_overlap} overlap`],
      ["Created", (d) => ago(d.created_at)],
    ], { onclick: (d) => (location.hash = `#/knowledge/${d.id}`), empty: "No datasets yet." })),
    panel(h("h2", {}, "New dataset"), form),
  );
}

export async function datasetPage({ id }) {
  // There's no GET /v1/datasets/{id}: the list has everything we need.
  const dataset = (await api("/v1/datasets")).find((d) => d.id === id);
  if (!dataset) throw new Error("Dataset not found");

  const docs = h("div");
  const chunks = h("div");
  const showChunks = async (doc) => {
    const list = await api(`/v1/documents/${doc.id}/chunks`);
    fill(chunks, panel(
      h("h2", {}, `${doc.filename}: ${list.length} chunks`),
      list.map((c) => h("div", {}, h("div", { class: "small muted" }, `#${c.position}`), h("pre", {}, c.content))),
    ));
  };

  // Indexing runs in the background, so poll the list until no document is still working.
  const render = async () => {
    const list = await api(`/v1/datasets/${id}/documents`);
    fill(docs, table(list, [
      ["File", (d) => d.filename],
      ["Status", (d) => [badge(d.status), d.error ? h("div", { class: "small muted" }, d.error) : null]],
      ["Characters", (d) => d.char_count.toLocaleString()],
      ["Chunks", (d) => d.chunk_count],
      ["Added", (d) => ago(d.created_at)],
    ], { onclick: (d) => d.status === "completed" && showChunks(d), empty: "No documents yet. Upload a .txt, .md or .pdf file." }));
    return list.some((d) => d.status === "pending" || d.status === "indexing");
  };

  const file = h("input", { type: "file", accept: ".txt,.md,.pdf", multiple: true });
  const uploadErrors = h("div");
  const upload = h("button", { type: "submit" }, "Upload");
  const uploadForm = h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(upload, uploadErrors, async () => {
        for (const f of file.files) {
          const form = new FormData();
          form.append("file", f);
          await api(`/v1/datasets/${id}/documents`, { method: "POST", form });
        }
        file.value = "";
        if (await render()) poll(render);
      });
    },
  }, uploadErrors, h("div", { class: "actions" }, file, upload));

  if (await render()) poll(render);
  return h("div", {},
    pageHead(dataset.name, h("a", { href: "#/knowledge" }, "All datasets")),
    dataset.description ? h("p", { class: "muted" }, dataset.description) : null,
    panel(h("h2", {}, "Documents"), uploadForm, docs),
    chunks,
    panel(h("h2", {}, "Retrieval test"), h("p", { class: "muted small" }, "See which chunks an app would get for a question, and their scores."), retrievalTest(id)),
  );
}

function retrievalTest(id) {
  const query = h("input", { placeholder: "How many vacation days do I get?", required: true });
  const topK = h("input", { type: "number", value: 4, min: 1, max: 20 });
  const mode = h("select", {}, ["hybrid", "vector", "keyword"].map((m) => h("option", { value: m }, m)));
  const results = h("div");
  const run = h("button", { type: "submit" }, "Search");
  return h("form", {
    onsubmit: (e) => {
      e.preventDefault();
      busy(run, results, async () => {
        const hits = await api(`/v1/datasets/${id}/retrieve`, {
          method: "POST", json: { query: query.value, top_k: Number(topK.value), mode: mode.value },
        });
        fill(results, (hits.length
          ? hits.map((c) => h("div", {}, h("div", { class: "small muted" }, `${c.filename} #${c.position} · score ${num(c.score, 3)}`), h("pre", {}, c.content)))
          : [notice("Nothing found.", "info")]));
      });
    },
  }, h("div", { class: "row" }, field("Question", query), field("Top k", topK), field("Mode", mode), h("div", {}, run)), results);
}
