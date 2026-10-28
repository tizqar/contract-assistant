/* Contract Assistant: the browser side of the file manager.
 *
 * The rows and the toolbar load their dialogs (name, move, share, delete for good) into the
 * page's one <dialog>, which opens when they arrive. A change answers with an HX-Trigger
 * header naming "files-changed" and {message, keep}: the message becomes a toast, the dialog
 * closes unless kept, and the view reloads itself on the same event (see files/page.html).
 * A row menu closes when one of its actions is chosen or when the reader clicks elsewhere.
 */
(function () {
  "use strict";

  const CHANGED = "files-changed";

  function app() {
    return window.contractAssistant || { toast() {} };
  }

  function dialog() {
    return document.getElementById("files-dialog");
  }

  function closeDialog() {
    const box = dialog();
    if (box && box.open) box.close();
  }

  function closeMenus(except) {
    document.querySelectorAll("details.files-menu[open]").forEach((menu) => {
      if (menu !== except) menu.open = false;
    });
  }

  document.addEventListener("htmx:afterSwap", (event) => {
    const target = event.detail.target;
    if (!target || target.id !== "files-dialog-body") return;
    const box = dialog();
    if (box && !box.open) box.showModal();
    const field = target.querySelector("[autofocus]") || target.querySelector("input:not([type=hidden]), select, button");
    if (field) field.focus();
  });

  document.addEventListener(CHANGED, (event) => {
    const detail = event.detail || {};
    if (detail.message) app().toast(detail.message);
    if (!detail.keep) closeDialog();
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-dialog-close]")) {
      closeDialog();
      return;
    }
    const action = event.target.closest("[data-files-action]");
    if (action) {
      const menu = action.closest("details");
      if (menu) menu.open = false;
      return;
    }
    closeMenus(event.target.closest("details.files-menu"));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenus(null);
  });

  document.addEventListener("DOMContentLoaded", () => {
    const box = dialog();
    if (!box) return;
    box.addEventListener("close", () => {
      const body = document.getElementById("files-dialog-body");
      if (body) body.innerHTML = "";
    });
    // A click on the backdrop, outside the dialog's box, closes it.
    box.addEventListener("click", (event) => {
      if (event.target === box) box.close();
    });
  });
})();
