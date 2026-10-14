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
        SB.el("td", {}, s.status === "completed"
          ? SB.el("span", { class: "badge badge-good" }, "✓ Completed")
          : SB.el("span", { class: "badge badge-critical" }, "✕ Failed")),
        SB.el("td", { class: "num" }, SB.number(s.rows_imported)),
        SB.el("td", { class: "num" }, SB.number(s.rows_skipped)),
        SB.el("td", { class: "wrap" }, s.message || ""),
      )) : [SB.el("tr", {}, SB.el("td", { colspan: 7, class: "empty" }, "Nothing imported yet."))]));
    } catch (error) {
      SB.message($("import-message"), error.message);
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
      xhr.upload.onload = () => SB.message($("import-message"), "Uploaded. Scoring transactions…", "info");
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
      const { source } = await upload(file);
      SB.message($("import-message"), source.message, "success");
      form.reset();
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
