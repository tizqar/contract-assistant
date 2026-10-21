/* Chat: a reply the user stops is marked as stopped, so that its status line does not keep
 * spinning. The server keeps nothing of a stopped exchange.
 */
(function () {
  "use strict";

  const STOPPED = "Stopped. This question was not answered.";

  document.addEventListener("click", (event) => {
    const stop = event.target.closest("[data-stop]");
    const form = stop && stop.closest("form");
    if (!form || form.dataset.target !== "#insight-body") return;
    const pending = document.querySelector("#chat-pending .bubble-pending");
    if (!pending) return;
    const note = document.createElement("p");
    note.className = "chat-stopped";
    note.textContent = STOPPED;
    pending.replaceChildren(note);
    pending.classList.remove("bubble-pending");
    pending.removeAttribute("aria-busy");
  });
})();
