(function () {
  'use strict';

  // ---- WhatsApp -----------------------------------------------------------
  // Put the number here in international form, digits only: 91 then the
  // ten-digit number, e.g. '919876543210'. Leave it empty and the WhatsApp
  // tile quietly turns into an email link instead of a dead button.
  var WHATSAPP = '919910144125';

  var wa = document.getElementById('waLink');
  if (wa) {
    if (/^\d{10,15}$/.test(WHATSAPP)) {
      var hello = encodeURIComponent(
        'नमस्ते! मैंने आपकी वेबसाइट देखी। '
      );
      wa.href = 'https://wa.me/' + WHATSAPP + '?text=' + hello;
      wa.target = '_blank';
      wa.rel = 'noopener';
    } else {
      // No number set yet. Hide the tile rather than show a second one
      // saying Email, or a button that goes nowhere.
      wa.hidden = true;
      var reach = wa.parentElement;
      if (reach) reach.style.gridTemplateColumns = 'repeat(2, 1fr)';
    }
  }

  // ---- The enquiry form ---------------------------------------------------
  // Posted with fetch so the person stays on the page: a redirect to the form
  // service's own thank-you screen loses them, and this one has to end with
  // them still looking at the site.
  var form = document.getElementById('enquiry');
  var status = document.getElementById('formStatus');
  if (!form || !status) return;

  var key = form.querySelector('input[name="access_key"]').value;

  form.addEventListener('submit', function (e) {
    e.preventDefault();

    if (key.indexOf('WEB3FORMS') === 0) {
      // Not configured yet. Say so rather than appearing to send.
      status.className = 'status bad';
      status.textContent =
        'The form is not connected yet. Please write to pallavsharmaofficial@gmail.com ' +
        'or use WhatsApp.';
      return;
    }

    var button = form.querySelector('button[type="submit"]');
    button.disabled = true;
    status.className = 'status';
    status.textContent = 'Sending…';

    fetch(form.action, { method: 'POST', body: new FormData(form) })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.success) {
          form.reset();
          status.className = 'status ok';
          status.textContent =
            'Thank you — it has reached me. I answer the same day. ' +
            'धन्यवाद, संदेश मिल गया।';
        } else {
          throw new Error(data.message || 'rejected');
        }
      })
      .catch(function () {
        status.className = 'status bad';
        status.textContent =
          'That did not go through. Please write to pallavsharmaofficial@gmail.com ' +
          'or use WhatsApp.';
      })
      .then(function () { button.disabled = false; });
  });
})();

// ---- Tabbed screenshots on the case-study pages --------------------------
(function () {
  'use strict';
  var gallery = document.querySelector('.gallery');
  if (!gallery) return;
  var tabs = [].slice.call(gallery.querySelectorAll('.tab'));
  var panes = [].slice.call(gallery.querySelectorAll('.pane'));

  function show(i) {
    tabs.forEach(function (t, n) { t.classList.toggle('on', n === i); });
    panes.forEach(function (p, n) { p.classList.toggle('on', n === i); });
  }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { show(i); });
  });
  // Arrow keys, because someone flipping through six screens will try them.
  gallery.addEventListener('keydown', function (e) {
    var at = tabs.findIndex(function (t) { return t.classList.contains('on'); });
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      show((at + 1) % tabs.length); tabs[(at + 1) % tabs.length].focus();
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      var p = (at - 1 + tabs.length) % tabs.length;
      show(p); tabs[p].focus();
    }
  });
})();
