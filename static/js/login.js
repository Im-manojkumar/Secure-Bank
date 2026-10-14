(function () {
  const form = document.getElementById("login-form");
  const errorBox = document.getElementById("login-error");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    try {
      await SB.api("/api/auth/login", { method: "POST", json: { email: form.email.value, password: form.password.value } });
      // The server only puts same-site paths in data-next.
      window.location.href = form.dataset.next;
    } catch (error) {
      SB.message(errorBox, error.message);
      button.disabled = false;
    }
  });
})();
