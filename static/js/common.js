// Shared helpers for every page: API calls, safe DOM building and formatting.
// All user-supplied text is inserted with textContent, never innerHTML.

const SB = {};

// Call the JSON API. Sends the browser to the login page if the session expired.
SB.api = async function (path, options = {}) {
  const opts = { credentials: "same-origin", ...options, headers: { ...(options.headers || {}) } };
  if (opts.json !== undefined) {
    opts.body = JSON.stringify(opts.json);
    opts.headers["Content-Type"] = "application/json";
    delete opts.json;
  }
  const response = await fetch(path, opts);
  if (response.status === 401 && !path.startsWith("/api/auth/login")) {
    window.location.href = "/login?next=" + encodeURIComponent(window.location.pathname + window.location.search);
    throw new Error("Please log in first.");
  }
  let body = null;
  try { body = await response.json(); } catch (e) { /* empty or non-JSON body */ }
  if (!response.ok) {
    const error = new Error((body && body.error) || `Request failed (${response.status})`);
    error.details = body && body.details;
    error.status = response.status;
    throw error;
  }
  return body;
};

// el("a", {href: "/x", class: "link"}, "text", childNode)
SB.el = function (tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    // CSSOM, not a style attribute: the Content-Security-Policy blocks inline style attributes.
    else if (key === "style") node.style.cssText = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
};

// Amounts carry their dataset's currency; PaySim has none, so it shows a plain number.
SB.money = (n, currency = "EUR") => currency
  ? (n ?? 0).toLocaleString(undefined, { style: "currency", currency })
  : (n ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
SB.number = (n) => (n ?? 0).toLocaleString();
SB.percent = (n, digits = 2) => `${((n ?? 0) * 100).toFixed(digits)}%`;
SB.score = (n) => (n === null || n === undefined ? "–" : n.toFixed(3));
SB.dateTime = (iso) => {
  if (!iso) return "–";
  const d = new Date(iso + (iso.endsWith("Z") ? "" : "Z"));
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }) + " UTC";
};

// Badge for a prediction: "Suspicious" or "Low risk". Text + icon, never colour alone.
SB.predictionBadge = function (prediction) {
  if (!prediction) return SB.el("span", { class: "badge" }, "Not scored");
  return prediction.predicted_label === 1
    ? SB.el("span", { class: "badge badge-critical" }, "⚠ Suspicious")
    : SB.el("span", { class: "badge badge-good" }, "✓ Low risk");
};

SB.statusBadge = function (status) {
  const cls = { Open: "badge-critical", "Under Review": "badge-warning", Resolved: "badge-good" }[status] || "";
  const icon = { Open: "●", "Under Review": "◐", Resolved: "✓" }[status] || "";
  return SB.el("span", { class: `badge ${cls}` }, `${icon} ${status}`);
};

SB.outcomeText = (outcome) => outcome || "–";

SB.DATASETS = { creditcard: "Kaggle credit card", sparkov: "Sparkov card payments", paysim: "PaySim mobile money" };
SB.datasetName = (key) => SB.DATASETS[key] || key || "–";

// A search's filters in words, e.g. "Sparkov card payments · Suspicious · Amount from 500".
SB.describeFilters = function (filters) {
  const f = filters || {};
  const day = (v) => SB.dateTime(v.length === 10 ? `${v}T00:00:00` : v);
  const parts = [];
  if (f.dataset) parts.push(SB.datasetName(f.dataset));
  if (f.prediction) parts.push(f.prediction === "suspicious" ? "Suspicious" : "Low risk");
  if (f.status) parts.push(`Case ${f.status.toLowerCase()}`);
  if (f.min_amount && f.max_amount) parts.push(`Amount ${f.min_amount} to ${f.max_amount}`);
  else if (f.min_amount) parts.push(`Amount from ${f.min_amount}`);
  else if (f.max_amount) parts.push(`Amount up to ${f.max_amount}`);
  if (f.date_from) parts.push(`From ${day(f.date_from)}`);
  if (f.date_to) parts.push(`Until ${day(f.date_to)}`);
  if (f.source_id) parts.push(`Import #${f.source_id}`);
  return parts.length ? parts.join(" · ") : "All transactions";
};

// The query string for a set of filters, as /transactions and the API read it.
SB.filterQuery = (filters) => new URLSearchParams(Object.entries(filters || {}).filter(([, v]) => v)).toString();

// "Download CSV" links: check the row limit first, so a refusal shows as a message
// rather than the browser saving the error as a file.
SB.guardDownload = function (link, messageBox) {
  link.addEventListener("click", async (e) => {
    e.preventDefault();
    const url = link.getAttribute("href");
    try {
      const check = await SB.api(`${url}${url.includes("?") ? "&" : "?"}check=1`);
      if (!check.allowed) {
        SB.message(messageBox, `This matches ${SB.number(check.rows)} transactions. Narrow the filters to ` +
          `${SB.number(check.limit)} or fewer to download it.`);
        return;
      }
      SB.message(messageBox, "");
      window.location.href = url;
    } catch (error) {
      SB.message(messageBox, error.message);
    }
  });
};

// A stored input value: numbers to 4 decimal places unless whole, text as is.
SB.inputValue = (value) => typeof value === "number"
  ? (Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4))
  : String(value);

// "Why it was flagged": the inputs that raised the score most, each against a typical genuine value.
SB.reasons = function (explanation) {
  if (!explanation || !explanation.length) {
    return SB.el("p", { class: "muted" }, "No single input stands out; the score comes from several small differences.");
  }
  return SB.el("ol", { class: "reasons" }, explanation.map((r) =>
    SB.el("li", {},
      SB.el("strong", {}, r.feature), ": ", r.value,
      SB.el("span", { class: "muted" }, ` (typical ${r.typical}) · +${r.effect.toFixed(3)} to the score`))));
};

// Show a message in a [role=status] or [role=alert] element.
SB.message = function (target, text, kind = "error") {
  target.textContent = text;
  target.className = `message message-${kind}`;
  target.hidden = !text;
};

// Previous / next pagination under a table.
SB.renderPager = function (container, data, onPage) {
  container.replaceChildren();
  if (!data.total) return;
  const first = (data.page - 1) * data.per_page + 1;
  const last = Math.min(data.page * data.per_page, data.total);
  container.append(
    SB.el("span", { class: "muted" }, `${SB.number(first)}–${SB.number(last)} of ${SB.number(data.total)}`),
    SB.el("button", { type: "button", class: "btn", disabled: data.page <= 1, onclick: () => onPage(data.page - 1) }, "Previous"),
    SB.el("button", { type: "button", class: "btn", disabled: data.page >= data.pages, onclick: () => onPage(data.page + 1) }, "Next"),
  );
};

// A table row that behaves like a link: click or Enter opens href.
SB.linkRow = function (href, cells) {
  const row = SB.el("tr", { class: "row-link", tabindex: "0", role: "link" }, cells);
  row.addEventListener("click", () => { window.location.href = href; });
  row.addEventListener("keydown", (e) => { if (e.key === "Enter") window.location.href = href; });
  return row;
};

document.addEventListener("DOMContentLoaded", () => {
  const logout = document.getElementById("logout");
  if (logout) {
    logout.addEventListener("click", async () => {
      try { await SB.api("/api/auth/logout", { method: "POST" }); } finally { window.location.href = "/login"; }
    });
  }
});
