/* A contract's review: the history panel, the menus of the steps, and the notice of a step.
 *
 * A step reloads the contract's page (HX-Redirect). The server sends what the step did in an
 * HX-Trigger event first; it is kept for this tab and shown as a toast on the next page.
 */
(function () {
  "use strict";

  const NOTICE_KEY = "contract-assistant.review-notice";

  function keep(text) {
    try {
      sessionStorage.setItem(NOTICE_KEY, text);
    } catch (error) {
      /* private mode: the page reloads without the notice */
    }
  }

  function takeKept() {
    try {
      const text = sessionStorage.getItem(NOTICE_KEY);
      sessionStorage.removeItem(NOTICE_KEY);
      return text;
    } catch (error) {
      return null;
    }
  }

  function openHistory(url) {
    const dialog = document.getElementById("history-dialog");
    const body = document.getElementById("history-body");
    if (!dialog || !body || !url) return;
    if (!dialog.open) dialog.showModal();
    if (window.htmx) window.htmx.ajax("GET", url, { target: body, swap: "innerHTML" });
  }

  function closeMenus(except) {
    document.querySelectorAll("details.review-menu[open]").forEach((menu) => {
      if (menu !== except) menu.open = false;
    });
  }

  document.addEventListener("review-done", (event) => {
    const detail = event.detail || {};
    if (detail.value) keep(detail.value);
  });

  document.addEventListener("click", (event) => {
    const opener = event.target.closest("[data-history-open]");
    if (opener) {
      openHistory(opener.dataset.historyOpen);
      return;
    }
    const dialog = event.target.closest("dialog.history-dialog");
    // A click on the backdrop lands on the dialog itself.
    if (event.target.closest("[data-history-close]") || event.target === dialog) {
      dialog.close();
      return;
    }
    const cancel = event.target.closest("[data-review-cancel]");
    if (cancel) {
      closeMenus(null);
      return;
    }
    closeMenus(event.target.closest("details.review-menu"));
  });

  // Opening one step's menu closes the other; the note gets the focus.
  document.addEventListener(
    "toggle",
    (event) => {
      const menu = event.target;
      if (!(menu instanceof HTMLDetailsElement) || !menu.classList.contains("review-menu")) return;
      if (!menu.open) return;
      closeMenus(menu);
      const field = menu.querySelector("textarea, button[type='submit']");
      if (field) field.focus();
    },
    true,
  );

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const open = document.querySelector("details.review-menu[open]");
    if (!open) return;
    open.open = false;
    const summary = open.querySelector("summary");
    if (summary) summary.focus();
  });

  document.addEventListener("DOMContentLoaded", () => {
    const text = takeKept();
    if (text && window.contractAssistant) window.contractAssistant.toast(text);
  });
})();
