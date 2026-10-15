(function () {
  const $ = (id) => document.getElementById(id);
  const form = $("import-form");
  const drop = $("drop");
  const progress = $("progress");

  async function loadHistory() {
    try {
      const { items } = await SB.api("/api/sources");
      $("history").replaceChildren(...(items.length ? items.map((s) => SB.el("tr", {},
        SB.el("td", {}, s.id),
        SB.el("td", {}, s.name),
        SB.el("td", {}, SB.dateTime(s.created_at)),
        SB.el("td", {}, statusBadge(s.status)),
        SB.el("td", { class: "num" }, SB.number(s.rows_imported)),
        SB.el("td", { class: "num" }, SB.number(s.rows_skipped)),
        SB.el("td", { class: "wrap" }, s.message || (RUNNING.includes(s.status) ? progressText(s) : "")),
      )) : [SB.el("tr", {}, SB.el("td", { colspan: 7, class: "empty" }, "Nothing imported yet."))]));
      // Keep the table current while an import started elsewhere is still running.
      clearTimeout(historyTimer);
      if (items.some((s) => RUNNING.includes(s.status))) historyTimer = setTimeout(loadHistory, 3000);
    } catch (error) {
      SB.message($("import-message"), error.message);
    }
  }

  const RUNNING = ["queued", "processing"];
  let historyTimer = null;

  function statusBadge(status) {
    if (status === "completed") return SB.el("span", { class: "badge badge-good" }, "✓ Completed");
    if (status === "failed") return SB.el("span", { class: "badge badge-critical" }, "✕ Failed");
    return SB.el("span", { class: "badge badge-info" }, status === "queued" ? "… Queued" : "↻ Importing");
  }

  function progressText(s) {
    if (s.status === "queued") return "Waiting to start…";
    if (!s.rows_received) return "Reading and checking the file…";
    if (!s.rows_processed) return `Scoring ${SB.number(s.rows_received)} rows…`;
    return `Saved ${SB.number(s.rows_processed)} new rows of ${SB.number(s.rows_received)}…`;
  }

  // Poll an import until it finishes. Returns the finished source.
  async function follow(id) {
    for (;;) {
      const { source } = await SB.api(`/api/sources/${id}`);
      if (!RUNNING.includes(source.status)) return source;
      SB.message($("import-message"), progressText(source), "info");
      if (source.rows_received && source.rows_processed) {
        progress.value = (source.rows_processed / source.rows_received) * 100;
      } else {
        progress.removeAttribute("value");  // An indeterminate bar while reading and scoring.
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  // XMLHttpRequest rather than fetch, because it reports upload progress.
  function upload(file) {
    return new Promise((resolve, reject) => {
      const data = new FormData();
      data.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/transactions/import");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) progress.value = (e.loaded / e.total) * 100; };
      xhr.upload.onload = () => SB.message($("import-message"), "Uploaded. Checking the file…", "info");
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
        if (xhr.status === 401) window.location.href = "/login?next=/import";
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body.error || `Import failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("The upload was interrupted."));
      xhr.send(data);
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("file").files[0];
    if (!file) { SB.message($("import-message"), "Choose a CSV file first."); return; }
    const button = form.querySelector("button");
    button.disabled = true;
    progress.hidden = false;
    progress.value = 0;
    SB.message($("import-message"), `Uploading ${file.name}…`, "info");
    try {
      let { source } = await upload(file);
      form.reset();
      loadHistory();
      // The import runs in the background; the page can be left and the history table shows the result.
      if (RUNNING.includes(source.status)) source = await follow(source.id);
      if (source.status === "failed") throw new Error(source.message);
      SB.message($("import-message"), source.message, "success");
    } catch (error) {
      SB.message($("import-message"), error.message);
    } finally {
      button.disabled = false;
      progress.hidden = true;
      loadHistory();
    }
  });

  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) $("file").files = e.dataTransfer.files; });

  loadHistory();
})();
