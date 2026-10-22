/* The overview: Stop clears unchecked text, a decision reloads only what it changes, and the
 * section chips follow the scroll. */
(function () {
  "use strict";

  // The page's word comes from the button (data-stopped, in the interface language).
  const STOPPED = "Stopped.";
  const PANE = "overview-pane";
  let spy = null;

  document.addEventListener("click", (event) => {
    const stop = event.target.closest(".stream-form [data-stop]");
    if (!stop) return;
    const form = stop.closest("form");
    const live = form.dataset.target ? document.querySelector(form.dataset.target) : null;
    if (!live) return;
    // The text written so far was never checked against the contract, so it is not kept.
    const note = document.createElement("p");
    note.className = "muted small";
    note.textContent = stop.dataset.stopped || STOPPED;
    live.replaceChildren(note);
  });

  function currentTau() {
    const low = document.getElementById("tau-low");
    const high = document.getElementById("tau-high");
    return low && high ? low.value + "-" + high.value : null;
  }

  // After a decision app.js triggers "refresh" on the overview. A decision changes the counts,
  // the things to check and the clauses, not the generated texts: reloading the whole overview
  // would restart a summary still being written. So only the parts marked data-refresh reload,
  // each into its container. New thresholds change everything, so then the whole overview does.
  document.addEventListener("htmx:confirm", (event) => {
    const elt = event.detail.elt;
    const trigger = event.detail.triggeringEvent;
    if (!elt || elt.id !== PANE || !trigger || trigger.type !== "refresh" || !window.htmx) return;
    const overview = elt.querySelector("[data-overview]");
    if (!overview || overview.dataset.tau !== currentTau()) return;
    event.preventDefault();
    elt.querySelectorAll("[data-refresh]").forEach((part) => {
      const into = part.parentElement;
      if (!into) return;
      // Each request needs its own source: htmx queues requests from one element and keeps
      // only the last, so with the default source (the body) all but two would be dropped.
      window.htmx.ajax("GET", part.dataset.refresh, { source: into, target: into, swap: "innerHTML" });
    });
  });

  document.addEventListener("thresholds-changed", () => {
    const pane = document.getElementById(PANE);
    if (pane && window.htmx) window.htmx.trigger(pane, "refresh");
  });

  function follow(root) {
    if (spy) spy.disconnect();
    spy = null;
    const chips = Array.from(root.querySelectorAll(".section-chip"));
    if (!chips.length || !("IntersectionObserver" in window)) return;
    const byId = new Map(chips.map((chip) => [chip.getAttribute("href").slice(1), chip]));
    spy = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const active = byId.get(entry.target.id);
        chips.forEach((chip) => {
          chip.classList.toggle("chip-active", chip === active);
          if (chip === active) chip.setAttribute("aria-current", "true");
          else chip.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-130px 0px -60% 0px" });
    byId.forEach((chip, id) => {
      const section = document.getElementById(id);
      if (section) spy.observe(section);
    });
  }

  document.addEventListener("htmx:afterSwap", (event) => {
    const target = event.detail.target;
    if (target && target.id === PANE) follow(target);
  });
})();
