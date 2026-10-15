(function () {
  const $ = (id) => document.getElementById(id);
  const FEATURES = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];
  const HELP = { Time: "Seconds since the first transaction", Amount: "Amount in €" };
  const form = $("predict-form");

  $("inputs").replaceChildren(...FEATURES.map((name) =>
    SB.el("div", { class: "feature" },
      SB.el("label", { for: `in-${name}` }, name),
      SB.el("input", { id: `in-${name}`, name, type: "number", step: "any", required: true,
                       min: name === "Amount" || name === "Time" ? 0 : null, title: HELP[name] || null }))));

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
      SB.el("p", { class: "muted" }, r.note),
    );
  }

  async function loadFrom(id) {
    try {
      const { transaction } = await SB.api(`/api/transactions/${id}`);
      for (const name of FEATURES) form.elements[name].value = transaction.features[name];
      SB.message($("load-message"), `Loaded transaction #${id}.`, "success");
    } catch (error) {
      SB.message($("load-message"), error.status === 404 ? `Transaction #${id} does not exist.` : error.message);
    }
  }

  $("load-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if ($("load-id").value) loadFrom($("load-id").value);
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    SB.message($("predict-error"), "");
    const payload = {};
    for (const name of FEATURES) {
      const raw = form.elements[name].value;
      if (raw === "") {
        SB.message($("predict-error"), `Fill in ${name}.`);
        form.elements[name].focus();
        return;
      }
      payload[name] = Number(raw);
    }
    try {
      showResult(await SB.api("/api/predict", { method: "POST", json: payload }));
    } catch (error) {
      SB.message($("predict-error"), error.details ? error.details.join(". ") : error.message);
    }
  });

  form.addEventListener("reset", () => {
    $("result").replaceChildren(SB.el("p", { class: "muted" }, "Fill in the values and press Check."));
  });

  const from = new URLSearchParams(window.location.search).get("from");
  if (from) {
    $("load-id").value = from;
    loadFrom(from);
  }
})();
