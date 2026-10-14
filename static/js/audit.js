(function () {
  const form = document.getElementById("audit-filters");
  const body = document.getElementById("audit-body");
  const pager = document.getElementById("audit-pager");
  const errorBox = document.getElementById("audit-error");

  const LABELS = {
    login_success: "Login", login_failed: "⚠ Failed login", login_blocked: "⛔ Blocked login", logout: "Logout",
    import_completed: "Import completed", import_failed: "⚠ Import failed", case_updated: "Case updated", user_created: "User created",
  };

  // Plain-language summary of the details object for each kind of entry.
  function describe(entry) {
    const d = entry.details || {};
    if (entry.action === "case_updated") {
      const before = d.before || {}, after = d.after || {};
      const parts = [];
      if (before.status !== after.status) parts.push(`${before.status} → ${after.status}`);
      if (before.outcome !== after.outcome && after.outcome) parts.push(`outcome: ${after.outcome}`);
      if (d.notes_changed) parts.push("notes edited");
      return parts.join(", ") || "no change";
    }
    if (entry.action === "import_completed") return `${d.file}: ${SB.number(d.rows_imported)} imported, ${SB.number(d.flagged)} flagged`;
    if (entry.action === "import_failed") return `${d.file}: ${d.error}`;
    if (entry.action === "user_created") return `role ${d.role}`;
    return "";
  }

  function subjectCell(subject) {
    const match = /^case:(\d+)$/.exec(subject || "");
    return match ? SB.el("a", { href: `/cases/${match[1]}` }, subject) : (subject || "–");
  }

  async function load(page = 1) {
    const params = new URLSearchParams({ page, per_page: 50 });
    for (const el of form.elements) if (el.name && el.value) params.set(el.name, el.value);
    SB.message(errorBox, "");
    try {
      const data = await SB.api(`/api/audit?${params}`);
      body.replaceChildren(...(data.items.length ? data.items.map((e) => SB.el("tr", {},
        SB.el("td", {}, SB.dateTime(e.created_at)),
        SB.el("td", {}, LABELS[e.action] || e.action),
        SB.el("td", {}, e.user || "–"),
        SB.el("td", {}, subjectCell(e.subject)),
        SB.el("td", { class: "wrap" }, describe(e)),
        SB.el("td", {}, e.ip_address || "–"),
      )) : [SB.el("tr", {}, SB.el("td", { colspan: 6, class: "empty" }, "No entries match."))]));
      SB.renderPager(pager, data, load);
    } catch (error) {
      SB.message(errorBox, error.message);
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); load(1); });
  form.addEventListener("reset", () => setTimeout(() => load(1), 0));
  load();
})();
