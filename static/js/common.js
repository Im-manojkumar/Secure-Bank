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

SB.money = (n) => (n ?? 0).toLocaleString(undefined, { style: "currency", currency: "EUR" });
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
