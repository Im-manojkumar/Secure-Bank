(async function () {
  const $ = (id) => document.getElementById(id);
  try {
    const { transaction: t } = await SB.api(`/api/transactions/${window.TRANSACTION_ID}`);
    $("tx-sub").textContent = `${SB.money(t.amount)} on ${SB.dateTime(t.timestamp)}`;

    $("tx-details").replaceChildren(
      SB.el("dt", {}, "Amount"), SB.el("dd", {}, SB.money(t.amount)),
      SB.el("dt", {}, "Time"), SB.el("dd", {}, SB.dateTime(t.timestamp)),
      SB.el("dt", {}, "Type"), SB.el("dd", {}, t.transaction_type),
      SB.el("dt", {}, "Source record"), SB.el("dd", {}, t.raw_data_reference || "–"),
      SB.el("dt", {}, "Dataset label"),
      SB.el("dd", {}, t.dataset_label === null ? "None" : t.dataset_label === 1 ? "Fraud (from the dataset)" : "Legitimate (from the dataset)"),
      SB.el("dt", {}, "Case"),
      SB.el("dd", {}, t.case ? SB.el("a", { href: `/cases/${t.case.id}` }, `#${t.case.id} `, SB.statusBadge(t.case.status)) : "No case"),
    );

    const p = t.prediction;
    $("tx-prediction").replaceChildren(p
      ? SB.el("div", {},
          SB.predictionBadge(p),
          SB.el("dl", { class: "dl", style: "margin-top:12px" },
            SB.el("dt", {}, "Risk score"), SB.el("dd", {}, SB.score(p.risk_score)),
            SB.el("dt", {}, "Model"), SB.el("dd", {}, p.model_version),
            SB.el("dt", {}, "Scored"), SB.el("dd", {}, SB.dateTime(p.created_at))),
          SB.el("p", { class: "muted" }, "A prediction is not proof of fraud. Suspicious transactions are sent for review."))
      : SB.el("p", { class: "muted" }, "This transaction has not been scored."));

    if (t.case) {
      $("tx-actions").append(SB.el("a", { class: "btn btn-primary", href: `/cases/${t.case.id}` }, "Open case"));
    }
    $("tx-actions").append(SB.el("a", { class: "btn", href: `/predict?from=${t.id}` }, "Re-check with the model"));

    $("tx-features").replaceChildren(...Object.entries(t.features).map(([name, value]) =>
      SB.el("div", { class: "feature" }, SB.el("span", {}, name), value.toFixed(4))));
  } catch (error) {
    SB.message($("tx-error"), error.status === 404 ? "This transaction does not exist." : error.message);
  }
})();
