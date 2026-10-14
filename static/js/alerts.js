(function () {
  const tabs = document.getElementById("status-tabs");
  const body = document.getElementById("alerts-body");
  const pager = document.getElementById("alerts-pager");
  const errorBox = document.getElementById("alerts-error");
  const params = new URLSearchParams(window.location.search);
  let status = params.has("status") ? params.get("status") : "Open";

  function syncTabs() {
    for (const b of tabs.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.status === status));
  }

  async function load(page = 1) {
    const url = new URLSearchParams({ page, per_page: 25 });
    if (status) url.set("status", status);
    history.replaceState(null, "", `?${new URLSearchParams({ status, ...(page > 1 ? { page } : {}) })}`);
    SB.message(errorBox, "");
    try {
      const data = await SB.api(`/api/alerts?${url}`);
      if (!data.items.length) {
        body.replaceChildren(SB.el("tr", {}, SB.el("td", { colspan: 7, class: "empty" }, "No alerts with this status.")));
      } else {
        body.replaceChildren(...data.items.map((c) => SB.linkRow(`/cases/${c.id}`, [
          SB.el("td", {}, `#${c.id}`),
          SB.el("td", {}, SB.statusBadge(c.status)),
          SB.el("td", {}, SB.dateTime(c.transaction.timestamp)),
          SB.el("td", { class: "num" }, SB.money(c.transaction.amount, c.transaction.currency)),
          SB.el("td", { class: "num" }, SB.score(c.transaction.risk_score)),
          SB.el("td", {}, SB.outcomeText(c.outcome)),
          SB.el("td", {}, c.reviewed_by || "–"),
        ])));
      }
      SB.renderPager(pager, data, load);
    } catch (error) {
      SB.message(errorBox, error.message);
    }
  }

  tabs.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    status = button.dataset.status;
    syncTabs();
    load(1);
  });

  syncTabs();
  load(Number(params.get("page") || 1));
})();
