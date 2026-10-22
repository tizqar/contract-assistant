/* Contract Assistant: the browser side of the contracts pages.
 *
 * Copying a quote, the filters of the comparison checklist and of the portfolio table, the
 * portfolio rows that open a contract, and page navigation that keeps the thresholds. The
 * words of the copy toasts come from the button (data-copied, data-not-copied), in the
 * language of the page.
 */
(function () {
  "use strict";

  const COPIED = "Quote copied.";
  const NOT_COPIED = "The quote could not be copied; select it and copy it by hand.";

  function app() {
    return window.contractAssistant || { toast() {}, withThresholds: (url) => url };
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(() => copyWithSelection(text));
    }
    return copyWithSelection(text);
  }

  // Without a secure context, or when the browser refuses, a hidden textarea still works.
  function copyWithSelection(text) {
    return new Promise((resolve, reject) => {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.className = "sr-only";
      document.body.appendChild(area);
      area.select();
      const done = document.execCommand("copy");
      area.remove();
      if (done) resolve();
      else reject(new Error("copy failed"));
    });
  }

  function pressOne(group, selector, chosen, activeClass) {
    group.querySelectorAll(selector).forEach((button) => {
      const on = button === chosen;
      button.classList.toggle(activeClass, on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function filterRows(body) {
    const search = document.querySelector('[data-row-search="#' + body.id + '"]');
    const chips = document.querySelector('[data-row-filters="#' + body.id + '"]');
    const active = chips && chips.querySelector('[aria-pressed="true"]');
    const flag = active ? active.dataset.rowFilter : "";
    const words = search ? search.value.trim().toLowerCase().split(/\s+/).filter(Boolean) : [];
    let shown = 0;
    body.querySelectorAll("tr[data-search]").forEach((row) => {
      const text = row.dataset.search || "";
      const match = words.every((word) => text.includes(word)) && (!flag || row.dataset[flag] === "1");
      row.hidden = !match;
      if (match) shown += 1;
    });
    const empty = document.getElementById("portfolio-empty");
    if (empty) empty.hidden = shown > 0;
  }

  function navigate(url) {
    window.location.assign(app().withThresholds(url));
  }

  document.addEventListener("click", (event) => {
    const copy = event.target.closest("[data-copy]");
    if (copy) {
      copyText(copy.dataset.copy).then(
        () => app().toast(copy.dataset.copied || COPIED),
        () => app().toast(copy.dataset.notCopied || NOT_COPIED)
      );
      return;
    }
    const show = event.target.closest("[data-checklist-filter] [data-show]");
    if (show) {
      const group = show.closest("[data-checklist-filter]");
      const list = document.querySelector(group.dataset.checklistFilter);
      if (list) list.dataset.show = show.dataset.show;
      pressOne(group, "[data-show]", show, "seg-item-active");
      return;
    }
    const chip = event.target.closest("[data-row-filters] [data-row-filter]");
    if (chip) {
      const group = chip.closest("[data-row-filters]");
      pressOne(group, "[data-row-filter]", chip, "chip-active");
      const body = document.querySelector(group.dataset.rowFilters);
      if (body) filterRows(body);
      return;
    }
    const link = event.target.closest("a[data-keep-thresholds]");
    if (link && !event.metaKey && !event.ctrlKey && !event.shiftKey && event.button === 0) {
      event.preventDefault();
      navigate(link.getAttribute("href"));
      return;
    }
    const row = event.target.closest("tr[data-href]");
    if (row && !event.target.closest("a, button, input, select")) {
      navigate(row.dataset.href);
    }
  });

  document.addEventListener("input", (event) => {
    const search = event.target.closest("[data-row-search]");
    if (!search) return;
    const body = document.querySelector(search.dataset.rowSearch);
    if (body) filterRows(body);
  });

  document.addEventListener("change", (event) => {
    const select = event.target.closest("select[data-submit-on-change]");
    if (select && select.form) select.form.requestSubmit();
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("form[data-keep-thresholds]");
    if (!form) return;
    event.preventDefault();
    const query = new URLSearchParams(new FormData(form)).toString();
    navigate((form.getAttribute("action") || window.location.pathname) + "?" + query);
  });
})();
