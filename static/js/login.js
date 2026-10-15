(function () {
  const form = document.getElementById("login-form");
  const errorBox = document.getElementById("login-error");
  const codeField = document.getElementById("code-field");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    try {
      const body = { email: form.email.value, password: form.password.value };
      if (!codeField.hidden) body.code = form.code.value;
      const result = await SB.api("/api/auth/login", { method: "POST", json: body });
      // An admin without two-factor login goes to set it up first.
      // The server only puts same-site paths in data-next.
      window.location.href = result.must_set_up_two_factor ? "/account" : form.dataset.next;
    } catch (error) {
      const askingForCode = error.data && error.data.two_factor_required && codeField.hidden;
      if (askingForCode) {
        codeField.hidden = false;
        form.code.focus();
      }
      // The first request for a code is a prompt, not a failure.
      SB.message(errorBox, error.message, askingForCode ? "info" : "error");
      button.disabled = false;
    }
  });
})();
