(function () {
  const $ = (id) => document.getElementById(id);
  const REPORT_ID = $("page-title").dataset.id;

  function render(report, s) {
    document.title = `${report.name} · SecureBank`;
    $("page-title").textContent = report.name;
    $("report-filters").textContent = SB.describeFilters(report.filters);
    $("report-meta").textContent =
      `Saved by ${report.created_by || "unknown"} on ${SB.dateTime(report.created_at)} · figures as of ${SB.dateTime(s.generated_at)}`;
    $("report-open").href = `/transactions?${SB.filterQuery(report.filters)}`;
    $("report-download").href = `/api/reports/${report.id}/export`;
    $("report-delete").hidden = !report.can_delete;

    const pending = s.cases_by_status.Open + s.cases_by_status["Under Review"];
    $("r-total").textContent = SB.number(s.transactions);
    $("r-flagged").textContent = SB.number(s.suspicious);
    const rate = s.transactions ? s.suspicious / s.transactions : 0;
    $("r-rate").textContent = s.transactions ? `${SB.percent(rate, rate >= 0.01 ? 1 : 3)} of them` : "\u00a0";
    $("r-pending").textContent = SB.number(pending);
    $("r-confirmed").textContent = SB.number(s.outcomes["Confirmed fraud"] || 0);
    $("r-legit").textContent = `${SB.number(s.outcomes.Legitimate || 0)} found legitimate`;

    $("r-datasets").replaceChildren(...(s.by_dataset.length
      ? s.by_dataset.map((d) => SB.el("tr", {},
        SB.el("td", {}, SB.datasetName(d.dataset)),
        SB.el("td", { class: "num" }, SB.number(d.transactions)),
        SB.el("td", { class: "num" }, SB.money(d.amount, d.currency)),
        SB.el("td", { class: "num" }, SB.number(d.suspicious)),
        SB.el("td", { class: "num" }, SB.money(d.suspicious_amount, d.currency))))
      : [SB.el("tr", {}, SB.el("td", { colspan: 5, class: "empty" }, "No transactions match these filters."))]));

    $("r-cases").replaceChildren(
      ...["Open", "Under Review", "Resolved"].flatMap((status) => [
        SB.el("dt", {}, SB.statusBadge(status)), SB.el("dd", {}, SB.number(s.cases_by_status[status] || 0))]),
      SB.el("dt", {}, "Confirmed fraud"), SB.el("dd", {}, SB.number(s.outcomes["Confirmed fraud"] || 0)),
      SB.el("dt", {}, "Found legitimate"), SB.el("dd", {}, SB.number(s.outcomes.Legitimate || 0)),
    );
  }

  SB.guardDownload($("report-download"), $("report-error"));

  $("report-delete").addEventListener("click", async () => {
    if (!window.confirm("Delete this report? The transactions themselves are not affected.")) return;
    try {
      await SB.api(`/api/reports/${REPORT_ID}`, { method: "DELETE" });
      window.location.href = "/reports";
    } catch (error) {
      SB.message($("report-error"), error.message);
    }
  });

  SB.api(`/api/reports/${REPORT_ID}`)
    .then(({ report, summary }) => render(report, summary))
    .catch((error) => SB.message($("report-error"), error.message));
})();
