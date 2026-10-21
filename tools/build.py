#!/usr/bin/env python3
"""Builds the case-study pages from one table, so adding an app is one entry.

Plain files, no framework: edit, run this, push, and it is live.
"""
import html
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

APPS = [
    dict(
        slug='kundlisaar', name='KundliSaar', hindi='कुंडलीसार',
        tag='Flutter · Dart · own ephemeris',
        one='A Vedic astrology app that computes the planets itself, on the '
            'handset, with the radio off.',
        body=[
            ('The problem', [
                'Every astrology app leans on Swiss Ephemeris. Its licence is '
                'AGPL, which would have forced this whole project open and '
                'made the App Store a dead end. Paying for the commercial '
                'licence was not realistic either.',
                'So I built the ephemeris. A Poisson series fitted to NASA '
                'JPL’s DE440s and emitted as Dart tables: planetary '
                'longitudes land within about an arcsecond from 1900 to 2070, '
                'computed on the phone in milliseconds, with no network.',
            ]),
            ('What it does', [
                'Sixteen divisional charts, Vimshottari to three levels with '
                'dated periods, Tajika varshphal with sahams and Mudda dasha, '
                'ashtakavarga, shadbala, Ghat and Sarvatobhadra chakras, and '
                'a full panchang.',
                'A predictive layer that names the chart factor behind every '
                'sentence it writes, says how many factors agree, and where '
                'the classical texts disagree shows both positions with the '
                'argument for each.',
            ]),
            ('How it is kept honest', [
                'Over 450 tests. The ephemeris is checked against skyfield '
                'and DE440s; the panchang against published almanacs, '
                'matching to the minute.',
                'It refuses to predict death, illness, pregnancy, exam '
                'results, court outcomes or investments, and it never sells a '
                'remedy out of fear.',
                'The app will also hand you the whole reading as a PDF: 127 '
                'pages of it, in Hindi, typeset in the app with an OpenType '
                'shaper written for the purpose, because the PDF library does '
                'no Indic shaping and Hindi would otherwise print as loose '
                'letters. Most free kundli apps give you 25 to 30 pages.',
            ]),
        ],
        facts=['450+ tests', '~1 arcsecond, 1900–2070', '7,084 Indian towns',
               'Hindi and English throughout', 'Nothing leaves the device'],
        shots=[('01-chart', 'The birth chart, in three styles'),
               ('02-panchang', 'A full panchang for any day and place'),
               ('03-strengths', 'Shadbala and ashtakavarga'),
               ('04-ask', 'Questions answered from the chart'),
               ('05-hastrekha', 'Palm reading from a photograph'),
               ('06-home', 'Everything on one screen')],
        links=[('Open the app', 'https://pallavsharmaofficial.github.io/KundliSaar/app/'),
               ('Sample report, 127 pages', '../assets/sample-kundli.pdf'),
               ('Source', 'https://github.com/pallavsharmaofficial/KundliSaar')],
    ),
    dict(
        slug='sursaar', name='SurSaar', hindi='सुरसार',
        tag='Flutter · Dart · on-device audio',
        one='A guitar teacher that listens to you play, with no server to '
            'send the audio to.',
        body=[
            ('The idea', [
                'Learning an instrument needs someone to tell you the chord '
                'was wrong. That feedback usually means streaming your '
                'microphone to somebody’s server.',
                'Here the pitch detection and chord tracking run on the '
                'handset, so it works on a train with no signal and the audio '
                'never leaves the room.',
            ]),
            ('What it does', [
                'A tuner, a metronome, chord and strumming practice with live '
                'feedback, a song library with capo charts, and progress kept '
                'on the device.',
            ]),
        ],
        facts=['Pitch detection on device', 'Works with no signal',
               'Hindi and English', 'Left-handed and mirror modes'],
        shots=[('01-home', 'Where a practice session starts'),
               ('02-practice', 'Live feedback while you play'),
               ('03-learn', 'Chords and strumming patterns'),
               ('04-tuner', 'The tuner'),
               ('05-songs', 'Songs with chords and capo'),
               ('06-progress', 'What you have practised')],
        links=[('Source', 'https://github.com/pallavsharmaofficial/SurSaar')],
    ),
    dict(
        slug='vyuhbhed', name='Vyuhbhed', hindi='व्यूहभेद',
        tag='Flutter · Dart',
        one='A job hunt run like a campaign, for the weeks of a notice period.',
        body=[
            ('The idea', [
                'A notice period is a fixed, countable window, and most people '
                'spend it anxious rather than organised. This turns it into a '
                'plan with a date on it.',
            ]),
            ('What it does', [
                'A daily plan that adapts to how many days are left, mock '
                'interviews that are scored rather than just recorded, and a '
                'resume review. Offline by default, English and Hindi.',
            ]),
        ],
        facts=['Daily plan against your last date', 'Scored mock interviews',
               'Resume review', 'Offline by default'],
        shots=[],
        links=[('Source', 'https://github.com/pallavsharmaofficial/vyuhbhed')],
    ),
    dict(
        slug='saath', name='Saath', hindi='साथ',
        tag='Flutter · Dart · private by design',
        one='A relationship counsellor that runs entirely on the phone.',
        body=[
            ('Why it is built this way', [
                'What people would type into a relationship counsellor is the '
                'last thing that should sit on somebody’s server. So '
                'none of it does: the whole thing runs on the handset, with '
                'no account and nothing uploaded.',
            ]),
            ('What it does', [
                'Guided conversations, reflection prompts and exercises drawn '
                'from counselling practice, in English and Hindi, free and '
                'offline.',
            ]),
        ],
        facts=['Runs entirely on the device', 'No account, nothing uploaded',
               'Hindi and English', 'Free'],
        shots=[],
        links=[('Source', 'https://github.com/pallavsharmaofficial/saath')],
    ),
    dict(
        slug='launchkit', name='Launchkit', hindi='',
        tag='Python · Google Play API',
        one='The parts of shipping a Flutter app to Google Play that a '
            'machine can do, done by a machine.',
        body=[
            ('Why', [
                'Publishing an app means the same forty steps every time: '
                'store listings in each language, graphics at the right '
                'sizes, screenshots, signing, release tracks. Doing that by '
                'hand once is tedious; doing it for every app is a waste.',
            ]),
            ('What it does', [
                'Scaffolds a new app, generates the store assets, and drives '
                'the Google Play Developer API to write listings, upload '
                'graphics and push a release, without opening the console.',
                'It is what put the apps on this page into testing.',
            ]),
        ],
        facts=['Play Developer API v3', 'Listings and graphics in every locale',
               'Release tracks from the command line'],
        shots=[],
        links=[],
        private=True,
    ),
]


def esc(t):
    return html.escape(t, quote=False)


def page(app):
    shots = app['shots']
    tabs = buttons = ''
    if shots:
        buttons = ''.join(
            f'<button class="tab{" on" if i == 0 else ""}" data-i="{i}">'
            f'{esc(c)}</button>'
            for i, (_, c) in enumerate(shots))
        tabs = ''.join(
            f'<figure class="pane{" on" if i == 0 else ""}" data-i="{i}">'
            f'<img src="../assets/shots/{app["slug"]}-{f}.png" alt="{esc(c)}" '
            f'loading="lazy"><figcaption>{esc(c)}</figcaption></figure>'
            for i, (f, c) in enumerate(shots))
        tabs = (f'<div class="gallery"><div class="tabs">{buttons}</div>'
                f'<div class="panes">{tabs}</div></div>')
    elif not app.get('private'):
        tabs = ('<p class="note">Screenshots go up when this one reaches the '
                'store. The source is public in the meantime.</p>')

    sections = ''.join(
        f'<h2>{esc(h)}</h2>' + ''.join(f'<p>{esc(p)}</p>' for p in ps)
        for h, ps in app['body'])
    facts = ''.join(f'<li>{esc(f)}</li>' for f in app['facts'])
    links = ''.join(f'<a href="{u}">{esc(t)}</a>' for t, u in app['links'])
    if app.get('private'):
        links += '<span class="muted">Private repository</span>'
    hindi = f' <span class="hi">{app["hindi"]}</span>' if app['hindi'] else ''

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(app["name"])} — Pallav Sharma</title>
<meta name="description" content="{esc(app["one"])}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=Mukta:wght@400;600&display=swap" rel="stylesheet">
<link rel="icon" href="../assets/{app["slug"]}.png">
<meta name="theme-color" content="#111317">
<link rel="stylesheet" href="../style.css">
</head>
<body>
<nav class="topbar"><div class="wrap">
  <a href="../" class="back">← Pallav Sharma</a>
  <a href="../#contact" class="btn small primary">Start a project</a>
</div></nav>

<header class="case">
  <div class="wrap">
    <img src="../assets/{app["slug"]}.png" alt="" class="icon">
    <p class="kicker">{esc(app["tag"])}</p>
    <h1>{esc(app["name"])}{hindi}</h1>
    <p class="lede">{esc(app["one"])}</p>
    <p class="links">{links}</p>
  </div>
</header>

<section><div class="wrap narrow">
  {sections}
  <ul class="facts big">{facts}</ul>
</div></section>

<section class="alt"><div class="wrap">
  {tabs}
</div></section>

<section><div class="wrap narrow center">
  <h2>Want something like this?</h2>
  <p class="sub center">Tell me what you have in mind. I answer the same day.</p>
  <a class="btn primary" href="../proposal.html">Send your requirements</a>
</div></section>

<footer><div class="wrap">
  <p><a href="../">← All work</a></p>
</div></footer>
<script src="../main.js"></script>
</body>
</html>
'''


def main():
    out = ROOT / 'work'
    out.mkdir(exist_ok=True)
    for app in APPS:
        (out / f'{app["slug"]}.html').write_text(page(app), encoding='utf-8')
    print(f'{len(APPS)} case-study pages written to work/')


if __name__ == '__main__':
    main()
