(async function () {
  const $ = (id) => document.getElementById(id);
  const messageBox = $("tf-message");

  function show(user, mustSetUp) {
    const on = user.two_factor_enabled;
    $("tf-status").textContent = on ? "Two-factor login is on." : "Two-factor login is off.";
    $("tf-required").hidden = on || !mustSetUp;
    $("tf-start").hidden = on;
    $("tf-confirm").hidden = true;
    // Admins who must keep it on get no way to turn it off.
    $("tf-off").hidden = !on || mustSetUp;
  }

  let mustSetUp = false;
  try {
    const { user } = await SB.api("/api/auth/me");
    mustSetUp = user.role === "admin" && document.body.dataset.require2fa !== "0";
    show(user, mustSetUp);
  } catch (error) {
    SB.message(messageBox, error.message);
  }

  $("tf-setup").addEventListener("click", async () => {
    try {
      const { secret, uri } = await SB.api("/api/auth/2fa/setup", { method: "POST", json: {} });
      $("tf-secret").textContent = secret.replace(/(.{4})/g, "$1 ").trim();
      $("tf-uri").textContent = uri;
      $("tf-start").hidden = true;
      $("tf-confirm").hidden = false;
      $("tf-code").focus();
      SB.message(messageBox, "");
    } catch (error) {
      SB.message(messageBox, error.message);
    }
  });

  $("tf-confirm").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const { user } = await SB.api("/api/auth/2fa/enable", { method: "POST", json: { code: $("tf-code").value } });
      show(user, mustSetUp);
      SB.message(messageBox, "Two-factor login is on. You will be asked for a code each time you log in.", "success");
    } catch (error) {
      SB.message(messageBox, error.message);
    }
  });

  $("tf-off").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const { user } = await SB.api("/api/auth/2fa/disable", {
        method: "POST", json: { password: $("tf-password").value, code: $("tf-off-code").value },
      });
      show(user, mustSetUp);
      SB.message(messageBox, "Two-factor login is off.", "success");
    } catch (error) {
      SB.message(messageBox, error.message);
    }
  });
})();
