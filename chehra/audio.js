// Eyes-free guidance: soft directional tones plus a calm voice that speaks only when it has something new to say.
//
// Tones (each fades in and out, nothing clicks):
//   rising chirp   = move up            falling chirp  = move down
//   two high taps  = toward the ear     two low taps   = away from the ear
//   wobble         = twist              warm chord     = on target
// The cue repeats faster as the device gets closer, like a parking sensor, and goes quiet once it is on target.

let ac = null, master = null;
function ctx() {
  if (!ac) {
    ac = new (window.AudioContext || window.webkitAudioContext)();
    master = ac.createGain(); master.gain.value = 0.85;
    const comp = ac.createDynamicsCompressor();
    master.connect(comp).connect(ac.destination);
  }
  if (ac.state === 'suspended') ac.resume();
  return ac;
}
export function unlockAudio() { try { ctx(); } catch {} }

function tone(f0, f1, t0, dur, vol = 0.07, type = 'sine') {
  const a = ctx(), o = a.createOscillator(), o2 = a.createOscillator(), g = a.createGain(), g2 = a.createGain();
  o.type = type; o.frequency.setValueAtTime(f0, t0); o.frequency.exponentialRampToValueAtTime(f1, t0 + dur);
  // a quiet octave above gives the tone some warmth without harshness
  o2.type = 'sine'; o2.frequency.setValueAtTime(f0 * 2, t0); o2.frequency.exponentialRampToValueAtTime(f1 * 2, t0 + dur); g2.gain.value = 0.18;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(vol, t0 + Math.min(0.03, dur * 0.3));
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g); o2.connect(g2).connect(g); g.connect(master);
  o.start(t0); o2.start(t0); o.stop(t0 + dur + 0.03); o2.stop(t0 + dur + 0.03);
}

const CUES = {
  up: t => tone(440, 680, t, 0.2),
  down: t => tone(680, 440, t, 0.2),
  forward: t => { tone(698, 698, t, 0.08, 0.06); tone(698, 698, t + 0.13, 0.08, 0.06); },
  back: t => { tone(392, 392, t, 0.08, 0.06); tone(392, 392, t + 0.13, 0.08, 0.06); },
  cw: t => { tone(523, 587, t, 0.1, 0.055); tone(587, 659, t + 0.11, 0.1, 0.055); },
  ccw: t => { tone(659, 587, t, 0.1, 0.055); tone(587, 523, t + 0.11, 0.1, 0.055); },
  ok: t => [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(f, f, t + i * 0.08, 0.5 - i * 0.06, 0.05)),
  done: t => [659.25, 783.99, 1046.5].forEach((f, i) => tone(f, f, t + i * 0.11, 0.6, 0.05)),
  soft: t => tone(660, 660, t, 0.12, 0.04),
};
export function cue(name) { try { const a = ctx(); CUES[name]?.(a.currentTime + 0.01); } catch {} }

// ---------- phrasing ----------
const pick = a => a[(Math.random() * a.length) | 0];
const PHRASES = {
  up: [['Just a touch higher', 'A tiny bit up'], ['A little higher', 'Slide it up a little'], ['Move it up', 'Bring it higher']],
  down: [['Just a touch lower', 'A tiny bit down'], ['A little lower', 'Slide it down a little'], ['Move it down', 'Bring it lower']],
  forward: [['A touch closer to your ear'], ['A little closer to your ear', 'Slide it toward your ear a little'], ['Move it toward your ear']],
  back: [['A touch away from your ear'], ['A little further from your ear', 'Slide it back a little'], ['Move it away from your ear']],
  cw: [['Twist it very slightly clockwise'], ['Turn it a little clockwise'], ['Turn it clockwise']],
  ccw: [['Twist it very slightly anticlockwise'], ['Turn it a little anticlockwise'], ['Turn it anticlockwise']],
};
const level = mm => (mm > 12 ? 2 : mm > 5 ? 1 : 0);

// ---------- the guide ----------
// opts: { mode(): 'all' | 'tones' | 'voice' | 'off', say(text), vibrate(pattern) }
export function createGuide(opts) {
  const st = { main: null, mainSince: 0, lastCue: 0, lastSaid: '', lastSaidT: -1e9, lastSaidMain: null, lastSaidDist: null,
    speakingUntil: 0, onTarget: false, unseenSince: 0 };
  const tonesOn = () => ['all', 'tones'].includes(opts.mode());
  const voiceOn = () => ['all', 'voice'].includes(opts.mode());

  function say(text, now, { force = false, gap = 3800 } = {}) {
    if (!voiceOn() || !text) return false;
    if (!force && now - st.lastSaidT < gap) return false;
    if (!force && text === st.lastSaid && now - st.lastSaidT < 9000) return false;
    st.lastSaid = text; st.lastSaidT = now; st.speakingUntil = now + 450 + text.length * 62;
    opts.say(text); return true;
  }

  // the single correction that matters most right now, with hysteresis so it does not flip between axes
  function chooseMain(s) {
    const need = {};
    if (Math.abs(s.dv) > s.tol * 0.6) need[s.dv > 0 ? 'up' : 'down'] = Math.abs(s.dv);
    if (Math.abs(s.du) > s.tol * 0.6) need[s.du > 0 ? 'forward' : 'back'] = Math.abs(s.du);
    if (s.turn && (s.posOk || Object.keys(need).length === 0)) need[s.turn] = Math.abs(s.rotErr) / 3;   // ~3° per mm of effort
    const keys = Object.keys(need);
    if (!keys.length) return null;
    const best = keys.reduce((a, b) => (need[a] >= need[b] ? a : b));
    if (st.main && need[st.main] && need[st.main] * 1.4 >= need[best]) return { name: st.main, mag: need[st.main] };
    return { name: best, mag: need[best] };
  }

  return {
    say: (text, now = performance.now(), o) => say(text, now, o),
    reset() { st.main = null; st.onTarget = false; st.lastSaidDist = null; st.unseenSince = 0; },
    // s: { found, ok, dist, du, dv, rotErr, turn, posOk, tol }
    update(s, now = performance.now()) {
      if (!s.found) {
        st.main = null;
        if (!st.unseenSince) st.unseenSince = now;
        if (now - st.unseenSince > 1500) say(pick(["I can't see the device yet. Place it just below your ear lobe.", 'Place the device just below your ear lobe, on the outline.']), now, { gap: 10000 });
        if (st.onTarget && now - st.unseenSince > 800) st.onTarget = false;
        return;
      }
      st.unseenSince = 0;
      if (s.ok) {
        if (!st.onTarget) {
          st.onTarget = true; st.main = null;
          if (tonesOn()) cue('ok');
          opts.vibrate?.([60, 40, 60]);
          say(pick(["That's it. Hold it there.", 'Perfect. Hold it right there.', 'Good, that is the spot. Hold still.']), now, { force: true });
        }
        return;                                  // quiet while the user holds it
      }
      if (st.onTarget && s.dist < s.tol * 1.5) return;   // small wobble around the target: stay calm
      st.onTarget = false;

      const m = chooseMain(s);
      if (!m) return;
      if (m.name !== st.main) { st.main = m.name; st.mainSince = now; }

      // tones: repeat the direction cue, faster when close; never over the voice
      const interval = Math.max(320, Math.min(1300, 260 + s.dist * 70));
      if (tonesOn() && now - st.lastCue > interval && now > st.speakingUntil) {
        st.lastCue = now; cue(m.name);
        if (s.dist < s.tol * 2.5) opts.vibrate?.(12);
      }

      // voice: when the advice settles on something new, or every ~7 s if nothing is improving
      if (now - st.mainSince < 600) return;
      const lv = level(m.name === 'cw' || m.name === 'ccw' ? m.mag * 1.2 : m.mag);
      const improving = st.lastSaidDist != null && st.lastSaidDist - s.dist > 2.5;
      const changed = m.name !== st.lastSaidMain;
      if (!changed && !(now - st.lastSaidT > 7000)) return;
      let text = pick(PHRASES[m.name][lv]);
      if (improving && !changed) text = (s.dist < s.tol * 2 ? 'Almost there. ' : 'Good. ') + text;
      else if (improving && Math.random() < 0.5) text = 'Good. ' + text;
      if (say(text, now)) { st.lastSaidMain = m.name; st.lastSaidDist = s.dist; }
    },
    preview: cue,
  };
}
