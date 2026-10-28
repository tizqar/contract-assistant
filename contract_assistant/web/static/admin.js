/* Contract Assistant: the admin pages and the account page.
 *
 * A form with data-confirm asks before it is sent (deactivating an account, resetting a
 * password); the question comes from the page, in its language. Copying a temporary password
 * uses the [data-copy] buttons of library.js.
 */
(function () {
  "use strict";

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("form[data-confirm]");
    if (form && !window.confirm(form.dataset.confirm)) event.preventDefault();
  });
})();
