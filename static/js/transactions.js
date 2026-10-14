(function () {
  const form = document.getElementById("filters");
  const body = document.getElementById("tx-body");
  const pager = document.getElementById("tx-pager");
  const errorBox = document.getElementById("tx-error");

  // Filters live in the page URL, so a search can be bookmarked or shared.
  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    for (const element of form.elements) {
      if (element.name && params.has(element.name)) element.value = params.get(element.name);
    }
    return Number(params.get("page") || 1);
  }

  function query(page) {
    const params = new URLSearchParams();
    for (const element of form.elements) {
      if (element.name && element.value) params.set(element.name, element.value);
    }
    if (page > 1) params.set("page", page);
    return params;
  }

  async function load(page = 1, pushState = true) {
    const params = query(page);
    if (pushState) history.replaceState(null, "", `?${params}`);
    params.set("per_page", 25);
    SB.message(errorBox, "");
    try {
      const data = await SB.api(`/api/transactions?${params}`);
      if (!data.items.length) {
        body.replaceChildren(SB.el("tr", {}, SB.el("td", { colspan: 6, class: "empty" }, "No transactions match these filters.")));
      } else {
        body.replaceChildren(...data.items.map((t) => SB.linkRow(`/transactions/${t.id}`, [
          SB.el("td", {}, t.id),
          SB.el("td", {}, SB.dateTime(t.timestamp)),
          SB.el("td", { class: "num" }, SB.money(t.amount, t.currency)),
          SB.el("td", {}, SB.predictionBadge(t.prediction)),
          SB.el("td", { class: "num" }, SB.score(t.prediction && t.prediction.risk_score)),
          SB.el("td", {}, t.case ? SB.statusBadge(t.case.status) : "–"),
        ])));
      }
      SB.renderPager(pager, data, (p) => load(p));
    } catch (error) {
      body.replaceChildren();
      pager.replaceChildren();
      SB.message(errorBox, error.message);
    }
  }

  async function loadSources() {
    try {
      const { items } = await SB.api("/api/sources");
      const select = document.getElementById("f-source");
      for (const s of items.filter((s) => s.status === "completed")) {
        select.append(SB.el("option", { value: s.id }, `${s.name} (#${s.id})`));
      }
    } catch (e) { /* the filter is optional */ }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); load(1); });
  form.addEventListener("reset", () => setTimeout(() => load(1), 0));

  loadSources().then(() => load(readUrl(), false));
})();
