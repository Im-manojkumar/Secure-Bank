// A small dependency-free SVG bar chart with hover tooltips and a table view.
// SB.barChart(container, [{label, value}], {name, format, labelFormat})

(function () {
  const NS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs = {}) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  // Round the axis maximum up to a tidy number and return evenly spaced ticks.
  function niceTicks(max, count = 4) {
    if (max <= 0) return [0, 1];
    const raw = max / count;
    const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw);
    const ticks = [];
    for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(v);
    if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);
    return ticks;
  }

  // Bar with 4px rounded top corners and a square base on the axis.
  function barPath(x, y, w, h) {
    const r = Math.min(4, w / 2, h);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  SB.barChart = function (container, data, options = {}) {
    const format = options.format || SB.number;
    const labelFormat = options.labelFormat || ((l) => l);
    container.replaceChildren();
    if (!data.length) {
      container.append(SB.el("p", { class: "empty" }, "No data yet. Import transactions to see this chart."));
      return;
    }

    const width = 560, height = 220;
    const m = { top: 8, right: 8, bottom: 26, left: 44 };
    const plotW = width - m.left - m.right, plotH = height - m.top - m.bottom;
    const ticks = niceTicks(Math.max(...data.map((d) => d.value)));
    const yMax = ticks[ticks.length - 1];
    const slot = plotW / data.length;
    const gap = Math.min(2, slot * 0.2);
    const y = (v) => m.top + plotH - (v / yMax) * plotH;

    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.name || "Bar chart" });

    for (const t of ticks) {
      root.append(svg("line", { class: "grid-line", x1: m.left, x2: width - m.right, y1: y(t), y2: y(t) }));
      const label = svg("text", { class: "tick", x: m.left - 6, y: y(t) + 4, "text-anchor": "end" });
      label.textContent = format(t);
      root.append(label);
    }

    const bars = data.map((d, i) => {
      const h = Math.max((d.value / yMax) * plotH, d.value > 0 ? 1 : 0);
      const bar = svg("path", { class: "bar", d: barPath(m.left + i * slot + gap / 2, m.top + plotH - h, Math.max(slot - gap, 1), h) });
      root.append(bar);
      return bar;
    });
    root.append(svg("line", { class: "axis-line", x1: m.left, x2: width - m.right, y1: m.top + plotH, y2: m.top + plotH }));

    // About six x-axis labels, whatever the number of bars.
    const every = Math.max(1, Math.ceil(data.length / 6));
    data.forEach((d, i) => {
      if (i % every) return;
      const label = svg("text", { class: "tick", x: m.left + i * slot + slot / 2, y: height - 6, "text-anchor": "middle" });
      label.textContent = labelFormat(d.label, true);
      root.append(label);
    });

    const tooltip = SB.el("div", { class: "chart-tooltip", hidden: true });
    data.forEach((d, i) => {
      // Hit area covers the whole column, bigger than the bar itself.
      const hit = svg("rect", { class: "hit", x: m.left + i * slot, y: m.top, width: slot, height: plotH });
      hit.addEventListener("mouseenter", () => {
        bars.forEach((b, j) => b.classList.toggle("dim", j !== i));
        tooltip.replaceChildren(SB.el("span", { class: "muted" }, labelFormat(d.label, false)), SB.el("strong", {}, format(d.value)));
        tooltip.hidden = false;
        const box = container.getBoundingClientRect();
        const colX = ((m.left + i * slot + slot / 2) / width) * box.width;
        tooltip.style.left = `${Math.min(Math.max(colX - tooltip.offsetWidth / 2, 0), box.width - tooltip.offsetWidth)}px`;
        // Pin to the top of the plot so it never covers the hovered bar.
        tooltip.style.top = "0px";
      });
      hit.addEventListener("mouseleave", () => {
        bars.forEach((b) => b.classList.remove("dim"));
        tooltip.hidden = true;
      });
      root.append(hit);
    });

    const rows = data.map((d) => SB.el("tr", {}, SB.el("td", {}, labelFormat(d.label, false)), SB.el("td", { class: "num" }, format(d.value))));
    const table = SB.el("details", { class: "table-view" },
      SB.el("summary", {}, "View as table"),
      SB.el("div", { class: "table-wrap" }, SB.el("table", {},
        SB.el("thead", {}, SB.el("tr", {}, SB.el("th", {}, options.periodName || "Period"), SB.el("th", { class: "num" }, options.name || "Value"))),
        SB.el("tbody", {}, rows))));

    container.append(root, tooltip, table);
  };
})();
