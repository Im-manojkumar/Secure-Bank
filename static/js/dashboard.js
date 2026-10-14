(async function () {
  const $ = (id) => document.getElementById(id);

  // "2013-09-01 14:00" -> "14:00" on the axis ("1 Sep" at midnight), "1 Sep, 14:00" in tooltips.
  const hourLabel = (period, short) => {
    const [date, time] = period.split(" ");
    const day = new Date(`${date}T00:00:00Z`).toLocaleDateString(undefined, { day: "numeric", month: "short", timeZone: "UTC" });
    if (short) return time === "00:00" ? day : time;
    return `${day}, ${time}`;
  };
  const STATUSES = ["Open", "Under Review", "Resolved"];

  try {
    const [summary, alerts] = await Promise.all([
      SB.api("/api/dashboard/summary?granularity=hour"),
      SB.api("/api/alerts?status=Open&per_page=8"),
    ]);

    $("t-total").textContent = SB.number(summary.total_transactions);
    $("t-flagged").textContent = SB.number(summary.predicted_fraud);
    $("t-rate").textContent = `${SB.percent(summary.predicted_fraud_rate, 3)} of transactions`;
    $("t-pending").textContent = SB.number(summary.pending_review);
    $("t-amount").textContent = SB.money(summary.predicted_fraud_amount);

    const common = { labelFormat: hourLabel, periodName: "Hour (UTC)" };
    SB.barChart($("chart-volume"), summary.trend.map((t) => ({ label: t.period, value: t.transactions })),
      { ...common, name: "Transactions" });
    SB.barChart($("chart-suspicious"), summary.trend.map((t) => ({ label: t.period, value: t.suspicious })),
      { ...common, name: "Suspicious" });

    const counts = summary.cases_by_status;
    const totalCases = Object.values(counts).reduce((a, b) => a + b, 0);
    $("status-bars").replaceChildren(...STATUSES.map((status) => [status, counts[status] || 0]).map(([status, n]) =>
      SB.el("div", { class: "status-row" },
        SB.statusBadge(status),
        SB.el("div", { class: "status-track", role: "presentation" },
          SB.el("div", { class: "status-fill", style: `width:${totalCases ? (n / totalCases) * 100 : 0}%` })),
        SB.el("span", { class: "num" }, SB.number(n)))));

    const outcomes = summary.resolved_outcomes;
    $("outcomes").replaceChildren(
      SB.el("dt", {}, "Confirmed fraud"), SB.el("dd", {}, SB.number(outcomes["Confirmed fraud"] || 0)),
      SB.el("dt", {}, "Found legitimate"), SB.el("dd", {}, SB.number(outcomes["Legitimate"] || 0)),
    );

    const body = $("top-alerts");
    if (!alerts.items.length) {
      body.replaceChildren(SB.el("tr", {}, SB.el("td", { colspan: 4, class: "empty" }, "No open alerts.")));
    } else {
      body.replaceChildren(...alerts.items.map((c) => SB.linkRow(`/cases/${c.id}`, [
        SB.el("td", {}, `#${c.id}`),
        SB.el("td", {}, SB.dateTime(c.transaction.timestamp)),
        SB.el("td", { class: "num" }, SB.money(c.transaction.amount)),
        SB.el("td", { class: "num" }, SB.score(c.transaction.risk_score)),
      ])));
    }
  } catch (error) {
    SB.message($("dashboard-error"), `Could not load the dashboard: ${error.message}`);
  }
})();
