(function () {
  const $ = (id) => document.getElementById(id);
  const form = $("predict-form");
  const datasetSelect = $("dataset");
  let formats = {};

  // datetime fields use a datetime-local input, which wants "2019-01-01T00:00:18".
  const toInput = (field, value) => field.kind === "datetime" ? String(value).replace(" ", "T") : value;

  function inputFor(field) {
    const id = `in-${field.name}`;
    let control;
    if (field.kind === "choice") {
      control = SB.el("select", { id, name: field.name, required: true },
        SB.el("option", { value: "" }, "Choose…"),
        field.options.map((o) => SB.el("option", { value: o.value }, o.label)));
    } else {
      const type = { number: "number", date: "date", datetime: "datetime-local" }[field.kind];
      control = SB.el("input", {
        id, name: field.name, type, required: true, step: field.kind === "datetime" ? "1" : "any",
        min: field.minimum ?? null, max: field.maximum ?? null, title: field.help || null,
      });
    }
    return SB.el("div", { class: "feature" }, SB.el("label", { for: id }, field.label), control);
  }

  function showFields(key) {
    const format = formats[key];
    $("inputs").replaceChildren(...format.fields.map(inputFor));
    $("dataset-help").textContent = format.description;
    resetResult();
  }

  function resetResult() {
    $("result").replaceChildren(SB.el("p", { class: "muted" }, "Fill in the values and press Check."));
  }

  function showResult(r) {
    const max = Math.max(1, r.risk_score, r.threshold);
    const fill = Math.min(r.risk_score / max, 1) * 100;
    $("result").replaceChildren(
      SB.el("p", {}, r.predicted_label === 1
        ? SB.el("span", { class: "badge badge-critical badge-lg" }, "⚠ Suspicious")
        : SB.el("span", { class: "badge badge-good badge-lg" }, "✓ Low risk")),
      SB.el("div", { class: "score-meter", role: "img", "aria-label": `Risk score ${SB.score(r.risk_score)}, alert threshold ${SB.score(r.threshold)}` },
        SB.el("div", { class: "fill", style: `width:${fill}%` }),
        SB.el("div", { class: "threshold", style: `left:${(r.threshold / max) * 100}%`, title: "Alert threshold" })),
      SB.el("dl", { class: "dl" },
        SB.el("dt", {}, "Risk score"), SB.el("dd", {}, SB.score(r.risk_score)),
        SB.el("dt", {}, "Alert threshold"), SB.el("dd", {}, SB.score(r.threshold)),
        SB.el("dt", {}, "Model"), SB.el("dd", {}, r.model_version)),
      SB.el("h3", { class: "mt-16" }, "What raised the score"),
      SB.reasons(r.explanation),
      SB.el("p", { class: "muted" }, r.note),
    );
  }

  async function loadFrom(id) {
    try {
      const { transaction } = await SB.api(`/api/transactions/${id}`);
      if (!formats[transaction.dataset]) {
        SB.message($("load-message"), `No trained model for ${SB.datasetName(transaction.dataset)} data yet.`);
        return;
      }
      datasetSelect.value = transaction.dataset;
      showFields(transaction.dataset);
      for (const field of formats[transaction.dataset].fields) {
        form.elements[field.name].value = toInput(field, transaction.features[field.name]);
      }
      SB.message($("load-message"), `Loaded transaction #${id}.`, "success");
    } catch (error) {
      SB.message($("load-message"), error.status === 404 ? `Transaction #${id} does not exist.` : error.message);
    }
  }

  $("load-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if ($("load-id").value) loadFrom($("load-id").value);
  });

  datasetSelect.addEventListener("change", () => showFields(datasetSelect.value));

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    SB.message($("predict-error"), "");
    const format = formats[datasetSelect.value];
    const payload = { dataset: format.key };
    for (const field of format.fields) {
      const raw = form.elements[field.name].value;
      if (raw === "") {
        SB.message($("predict-error"), `Fill in ${field.label}.`);
        form.elements[field.name].focus();
        return;
      }
      payload[field.name] = field.kind === "number" ? Number(raw) : raw;
    }
    try {
      showResult(await SB.api("/api/predict", { method: "POST", json: payload }));
    } catch (error) {
      SB.message($("predict-error"), error.details ? error.details.join(". ") : error.message);
    }
  });

  // Reset clears the values but keeps the chosen format.
  form.addEventListener("reset", () => {
    const keep = datasetSelect.value;
    setTimeout(() => { datasetSelect.value = keep; }, 0);
    resetResult();
  });

  (async function start() {
    try {
      const { items } = await SB.api("/api/models");
      const trained = items.filter((m) => m.model);
      if (!trained.length) {
        SB.message($("predict-error"), "No model has been trained yet. Run python -m ml.train first.");
        return;
      }
      formats = Object.fromEntries(trained.map((m) => [m.key, m]));
      datasetSelect.replaceChildren(...trained.map((m) => SB.el("option", { value: m.key }, m.title)));
      showFields(trained[0].key);
      const from = new URLSearchParams(window.location.search).get("from");
      if (from) {
        $("load-id").value = from;
        loadFrom(from);
      }
    } catch (error) {
      SB.message($("predict-error"), error.message);
    }
  })();
})();
