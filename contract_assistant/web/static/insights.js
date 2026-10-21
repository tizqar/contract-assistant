/* Insights tabs: Stop clears the text written so far, which was never checked. */
(function () {
  "use strict";

  const STOPPED = "Stopped.";

  document.addEventListener("click", (event) => {
    const stop = event.target.closest(".stream-form [data-stop]");
    if (!stop) return;
    const form = stop.closest("form");
    const live = form.dataset.target ? document.querySelector(form.dataset.target) : null;
    if (!live) return;
    // The text written so far was never checked against the contract, so it is not kept.
    const note = document.createElement("p");
    note.className = "muted small";
    note.textContent = STOPPED;
    live.replaceChildren(note);
  });
})();
