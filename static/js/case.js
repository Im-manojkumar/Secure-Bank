(function () {
  const $ = (id) => document.getElementById(id);
  const form = $("review-form");

  // An outcome is only chosen when resolving a case.
  function toggleOutcome() {
    $("outcome-field").hidden = form.status.value !== "Resolved";
  }

  function render(c) {
    $("case-status").replaceChildren(SB.statusBadge(c.status));
    $("case-sub").textContent = `Opened ${SB.dateTime(c.created_at)} · last updated ${SB.dateTime(c.updated_at)}`;
    const t = c.transaction;
    $("case-transaction").replaceChildren(
      SB.el("dt", {}, "Transaction"), SB.el("dd", {}, `#${t.id}`),
      SB.el("dt", {}, "Amount"), SB.el("dd", {}, SB.money(t.amount)),
      SB.el("dt", {}, "Time"), SB.el("dd", {}, SB.dateTime(t.timestamp)),
      SB.el("dt", {}, "Risk score"), SB.el("dd", {}, SB.score(t.risk_score)),
      SB.el("dt", {}, "Outcome"), SB.el("dd", {}, SB.outcomeText(c.outcome)),
    );
    $("case-tx-link").href = `/transactions/${t.id}`;
    form.status.value = c.status;
    form.outcome.value = c.outcome || "";
    form.review_notes.value = c.review_notes || "";
    $("review-meta").textContent = c.reviewed_by ? `Last reviewed by ${c.reviewed_by}` : "Not reviewed yet";
    toggleOutcome();
  }

  form.status.addEventListener("change", toggleOutcome);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = { status: form.status.value, review_notes: form.review_notes.value };
    if (payload.status === "Resolved") {
      if (!form.outcome.value) {
        SB.message($("review-message"), "Choose an outcome before resolving the case.");
        form.outcome.focus();
        return;
      }
      payload.outcome = form.outcome.value;
    }
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      const { case: updated } = await SB.api(`/api/cases/${window.CASE_ID}`, { method: "PATCH", json: payload });
      render(updated);
      SB.message($("review-message"), "Review saved.", "success");
    } catch (error) {
      SB.message($("review-message"), error.message);
    } finally {
      button.disabled = false;
    }
  });

  SB.api(`/api/cases/${window.CASE_ID}`)
    .then(({ case: c }) => render(c))
    .catch((error) => {
      form.hidden = true;
      SB.message($("case-error"), error.status === 404 ? "This case does not exist." : error.message);
    });
})();
