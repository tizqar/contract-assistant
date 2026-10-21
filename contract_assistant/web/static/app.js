/* Contract Assistant: the browser side.
 *
 * htmx does the partial updates. This file adds what htmx does not: the thresholds on every
 * request, the tab state, the streamed generation (newline-delimited JSON frames from the
 * server, see web/core.py `frame`), the Stop button, toasts and a few conveniences.
 */
(function () {
  "use strict";

  const streams = new Map(); // form -> AbortController

  function thresholds() {
    const low = document.getElementById("tau-low");
    const high = document.getElementById("tau-high");
    return { low: low ? low.value : null, high: high ? high.value : null };
  }

  function withThresholds(url) {
    const { low, high } = thresholds();
    if (low === null || high === null) return url;
    const target = new URL(url, window.location.href);
    target.searchParams.set("low", low);
    target.searchParams.set("high", high);
    return target.pathname + target.search + target.hash;
  }

  function toast(text) {
    const box = document.getElementById("toasts");
    if (!box || !text) return;
    const item = document.createElement("div");
    item.className = "toast";
    item.textContent = text;
    box.appendChild(item);
    setTimeout(() => item.remove(), 6000);
  }

  function activateTab(name) {
    const tab = document.querySelector('.tab[data-tab="' + name + '"]');
    if (tab) markActive(tab);
  }

  function markActive(tab) {
    const bar = tab.closest(".tabs");
    if (!bar) return;
    bar.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("tab-active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
  }

  function settle(target) {
    if (!target) return;
    if (window.htmx) window.htmx.process(target);
    const chat = target.classList.contains("chat-list") ? target : target.querySelector(".chat-list");
    if (chat) {
      const body = chat.closest(".insight-body") || chat;
      body.scrollTop = body.scrollHeight;
    }
    const mark = target.querySelector("mark.highlight");
    if (mark) mark.scrollIntoView({ block: "center" });
    autosubmit(target);
  }

  function applyFrame(frame, target) {
    if (frame.html !== undefined && target) target.innerHTML = frame.html;
    if (frame.oob) {
      for (const [id, html] of Object.entries(frame.oob)) {
        const el = document.getElementById(id);
        if (el) {
          el.innerHTML = html;
          settle(el);
        }
      }
    }
    if (frame.load && window.htmx) {
      for (const [id, url] of Object.entries(frame.load)) {
        const el = document.getElementById(id);
        if (el) window.htmx.ajax("GET", withThresholds(url), { target: "#" + id, swap: "innerHTML" });
      }
    }
    if (frame.notice) toast(frame.notice);
    if (frame.activate) activateTab(frame.activate);
    if (frame.redirect) window.location.assign(frame.redirect);
  }

  async function streamForm(form) {
    const target = document.querySelector(form.dataset.target || "#insight-body");
    const previous = streams.get(form);
    if (previous) previous.abort();
    const controller = new AbortController();
    streams.set(form, controller);
    const submit = form.querySelector('[type="submit"]');
    const stop = form.querySelector("[data-stop]");
    if (submit) submit.disabled = true;
    if (stop) stop.hidden = false;
    if (form.dataset.activate) activateTab(form.dataset.activate);
    const body = new FormData(form);
    if (form.dataset.reset !== undefined) form.reset();
    try {
      const response = await fetch(withThresholds(form.getAttribute("action") || window.location.href), {
        method: (form.getAttribute("method") || "post").toUpperCase(),
        body,
        signal: controller.signal,
        headers: { Accept: "application/x-ndjson, text/html" },
      });
      if (!response.ok) {
        toast(await response.text());
        return;
      }
      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let newline;
        while ((newline = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (!line) continue;
          try {
            applyFrame(JSON.parse(line), target);
          } catch (error) {
            console.error("bad frame", line, error);
          }
          if (target) settle(target);
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error(error);
        toast("The connection to the server was lost.");
      }
    } finally {
      if (streams.get(form) === controller) streams.delete(form);
      if (submit) submit.disabled = false;
      if (stop) stop.hidden = true;
      if (target) settle(target);
    }
  }

  function autosubmit(root) {
    root.querySelectorAll("form[data-stream][data-autosubmit]").forEach((form) => {
      if (form.dataset.submitted) return;
      form.dataset.submitted = "1";
      streamForm(form);
    });
  }

  function setThreshold(which, value) {
    const input = document.getElementById(which === "low" ? "tau-low" : "tau-high");
    const output = document.getElementById(which === "low" ? "tau-low-out" : "tau-high-out");
    const range = document.querySelector('[data-threshold="' + which + '"]');
    if (input) input.value = value;
    if (output) output.textContent = value;
    if (range) range.value = value;
  }

  function thresholdsChanged() {
    const active = document.querySelector(".tab.tab-active");
    if (active && window.htmx) window.htmx.trigger(active, "click");
    document.body.dispatchEvent(new CustomEvent("thresholds-changed", { bubbles: true }));
  }

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("form[data-stream]");
    if (!form) return;
    event.preventDefault();
    streamForm(form);
  });

  document.addEventListener("click", (event) => {
    const stop = event.target.closest("[data-stop]");
    if (stop) {
      const form = stop.closest("form");
      const controller = form && streams.get(form);
      if (controller) controller.abort();
      return;
    }
    const chip = event.target.closest("[data-fill]");
    if (chip) {
      const input = document.getElementById("question");
      if (input) {
        input.value = chip.dataset.fill;
        const form = input.closest("form");
        if (form) form.requestSubmit();
      }
      return;
    }
    const tab = event.target.closest(".tab[data-tab]");
    if (tab) markActive(tab);
  });

  document.addEventListener("change", (event) => {
    const select = event.target.closest("[data-navigate]");
    if (select && select.value) {
      window.location.assign(select.value);
      return;
    }
    const range = event.target.closest("[data-threshold]");
    if (range) {
      const which = range.dataset.threshold;
      let value = Number(range.value);
      const other = Number(document.getElementById(which === "low" ? "tau-high" : "tau-low").value);
      if (which === "low" && value > other) value = other;
      if (which === "high" && value < other) value = other;
      setThreshold(which, String(value));
      thresholdsChanged();
    }
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-threshold-reset]")) return;
    const low = document.getElementById("tau-low");
    const high = document.getElementById("tau-high");
    if (low) setThreshold("low", low.dataset.default);
    if (high) setThreshold("high", high.dataset.default);
    thresholdsChanged();
  });

  document.addEventListener("htmx:configRequest", (event) => {
    event.detail.path = withThresholds(event.detail.path);
  });

  document.addEventListener("htmx:responseError", (event) => {
    const xhr = event.detail.xhr;
    toast(xhr && xhr.responseText ? xhr.responseText : "Something went wrong; see the server log.");
  });

  document.addEventListener("htmx:sendError", () => {
    toast("The server did not answer.");
  });

  document.addEventListener("htmx:afterSwap", (event) => {
    settle(event.detail.target);
  });

  document.addEventListener("DOMContentLoaded", () => {
    autosubmit(document);
  });

  window.contractAssistant = { toast, activateTab, withThresholds, streamForm };
})();
