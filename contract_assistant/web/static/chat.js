/* Chat: three things app.js does not do.
 *
 * - A reply the user stops is marked as stopped, so that its status line does not keep
 *   spinning. The server keeps nothing of a stopped exchange.
 * - At the end of a reply the server sends a `data-follow` marker with the passage it quoted;
 *   the document pane loads it as its own request, so the passage is ready in the document
 *   while the centre stays on what the reader is looking at.
 * - After the reader confirms, rejects or undoes a passage anywhere, the chat is drawn again,
 *   so its cards show the decision. The scroll position is kept.
 */
(function () {
  "use strict";

  const STOPPED = "Stopped. This question was not answered.";
  const CHAT_ROUTE = /\/(ask|explain)$/;

  function chatBody() {
    return document.getElementById("chat-body");
  }

  function writing() {
    return document.querySelector("#chat-pending .bubble-pending");
  }

  document.addEventListener("click", (event) => {
    const stop = event.target.closest("[data-stop]");
    const form = stop && stop.closest("form");
    if (!form || !CHAT_ROUTE.test(form.getAttribute("action") || "")) return;
    const pending = writing();
    if (!pending) return;
    const note = document.createElement("p");
    note.className = "chat-stopped";
    note.textContent = STOPPED;
    pending.replaceChildren(note);
    pending.classList.remove("bubble-pending");
    pending.removeAttribute("aria-busy");
  });

  function follow(root) {
    root.querySelectorAll("[data-follow]").forEach((marker) => {
      const url = marker.dataset.follow;
      marker.remove();
      const pane = document.getElementById("document-pane");
      if (!pane || !url || !window.htmx) return;
      window.htmx.ajax("GET", url, { source: pane, target: pane, swap: "innerHTML" });
    });
  }

  const body = chatBody();
  if (body) new MutationObserver(() => follow(body)).observe(body, { childList: true, subtree: true });

  let keptScroll = null;

  document.addEventListener("htmx:afterRequest", (event) => {
    const path = event.detail.pathInfo ? event.detail.pathInfo.requestPath : "";
    if (!path || path.indexOf("/decide") < 0 || !event.detail.successful) return;
    const chat = chatBody();
    const url = chat && chat.getAttribute("hx-get");
    // A reply being written draws the chat again when it ends.
    if (!url || writing() || !window.htmx) return;
    keptScroll = chat.scrollTop;
    window.htmx.ajax("GET", url, { source: chat, target: chat, swap: "innerHTML" });
  });

  document.addEventListener("htmx:afterSettle", (event) => {
    const chat = chatBody();
    if (keptScroll === null || !chat || event.detail.target !== chat) return;
    chat.scrollTop = keptScroll;
    keptScroll = null;
  });
})();
