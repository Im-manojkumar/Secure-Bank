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

  // The dataset filter lives in the page URL, so a view can be bookmarked.
  const select = $("dash-dataset");
  const dataset = new URLSearchParams(window.location.search).get("dataset") || "";
  select.value = dataset;
  select.addEventListener("change", () => {
    window.location.search = select.value ? `?dataset=${select.value}` : "";
  });
  // Sparkov and "all data" span months or years, so they are charted by day; the others by hour.
  const byDay = dataset === "" || dataset === "sparkov";
  const scope = dataset ? `&dataset=${dataset}` : "";
  for (const id of ["volume-title", "suspicious-title"]) {
    $(id).textContent = $(id).textContent.replace(byDay ? "per hour" : "per day", byDay ? "per day" : "per hour");
  }
  // Axis labels show the year too when the data spans several years (all datasets together).
  const dayLabeller = (periods) => {
    const manyYears = new Set(periods.map((p) => p.slice(0, 4))).size > 1;
    return (period, short) => {
      const d = new Date(`${period}T00:00:00Z`);
      return d.toLocaleDateString(undefined, short && !manyYears
        ? { day: "numeric", month: "short", timeZone: "UTC" }
        : short
          ? { month: "short", year: "numeric", timeZone: "UTC" }
          : { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
    };
  };

  try {
    const [summary, alerts] = await Promise.all([
      SB.api(`/api/dashboard/summary?granularity=${byDay ? "day" : "hour"}${scope}`),
      SB.api(`/api/alerts?status=Open&per_page=8${scope}`),
    ]);

    $("t-total").textContent = SB.number(summary.total_transactions);
    $("t-flagged").textContent = SB.number(summary.predicted_fraud);
    $("t-rate").textContent = `${SB.percent(summary.predicted_fraud_rate, 3)} of transactions`;
    $("t-pending").textContent = SB.number(summary.pending_review);
    // One total per currency: the datasets do not share one.
    const amounts = summary.predicted_fraud_amounts;
    $("t-amount").classList.toggle("multi", amounts.length > 1);
    $("t-amount").replaceChildren(...(amounts.length
      ? amounts.map((a) => SB.el("div", {}, a.currency ? SB.money(a.amount, a.currency) : `${SB.money(a.amount, null)} units`))
      : [SB.money(0)]));

    const common = byDay
      ? { labelFormat: dayLabeller(summary.trend.map((t) => t.period)), periodName: "Day (UTC)" }
      : { labelFormat: hourLabel, periodName: "Hour (UTC)" };
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
        SB.el("td", { class: "num" }, SB.money(c.transaction.amount, c.transaction.currency)),
        SB.el("td", { class: "num" }, SB.score(c.transaction.risk_score)),
      ])));
    }
  } catch (error) {
    SB.message($("dashboard-error"), `Could not load the dashboard: ${error.message}`);
  }
})();
