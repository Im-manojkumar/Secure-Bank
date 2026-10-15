(function () {
  const body = document.getElementById("reports-body");

  async function load() {
    try {
      const { items } = await SB.api("/api/reports");
      if (!items.length) {
        body.replaceChildren(SB.el("tr", {}, SB.el("td", { colspan: 4, class: "empty" }, "No reports saved yet.")));
        return;
      }
      body.replaceChildren(...items.map((r) => SB.linkRow(`/reports/${r.id}`, [
        SB.el("td", {}, SB.el("strong", {}, r.name)),
        SB.el("td", {}, SB.describeFilters(r.filters)),
        SB.el("td", {}, r.created_by || "–"),
        SB.el("td", {}, SB.dateTime(r.created_at)),
      ])));
    } catch (error) {
      body.replaceChildren();
      SB.message(document.getElementById("reports-error"), error.message);
    }
  }

  load();
})();
