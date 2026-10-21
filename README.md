# pallavsharmaofficial.github.io

My portfolio. Plain HTML, CSS and one small script — no build step, no
framework, nothing to install. Edit a file, push, and it is live.

## Two things need filling in

Both are marked with a placeholder, and both fail safely until they are set.

**1. WhatsApp number** — `main.js`, the `WHATSAPP` constant. International
form, digits only: `91` then the ten digits. Until it is set, the WhatsApp
tile turns itself into an email link rather than sitting there dead.

**2. The enquiry form** — `index.html`, the `access_key` field.

Get a free key from [web3forms.com](https://web3forms.com): enter the inbox
address, and the key arrives by email. No account, no card, 250 submissions a
month. The key is meant to be public — all it can do is post a form to that
one inbox.

Until it is set, the form says plainly that it is not connected and points at
email and WhatsApp instead, rather than pretending to send.

## Checking it locally

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Layout

| | |
| --- | --- |
| `index.html` | the whole site; one page, anchored sections |
| `style.css` | light and dark, both handled |
| `main.js` | WhatsApp link, and the form posted without leaving the page |
| `assets/` | app icons, a screenshot, the UPI QR |
