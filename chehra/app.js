// Chehra: real-time face reading. Module 1: Dusq placement behind the ear lobe.
//
// Pipeline per frame:
//   MediaPipe Face Landmarker (478 3D points) -> a head frame (out / down / back axes)
//   -> target spot = visible-side cheek edge + offsets in face-width units
//   -> a pose-independent patch is resampled on the plane spanned by back/down
//   -> compared with a bare-skin baseline (or skin colour) to find the device
//   -> centroid, size and rotation in mm/deg -> live guidance.
import { FaceLandmarker, FilesetResolver } from './vendor/vision_bundle.mjs';
import { greyPyramid, corners, lk, similarity } from './track.js';

const $ = s => document.querySelector(s);
const cv = $('#view'), ctx = cv.getContext('2d');
const video = Object.assign(document.createElement('video'), { playsInline: true, muted: true, autoplay: true });
const grab = document.createElement('canvas'), gctx = grab.getContext('2d', { willReadFrequently: true });
const gcnv = document.createElement('canvas'), gc2d = gcnv.getContext('2d', { willReadFrequently: true });

// ---------- config ----------
const KEY = 'chehra.dusq.v1';
const DEF = { down: 0.17, back: 0.2, angle: 40, size: 24, tol: 3, tolDeg: 10, voice: true, mesh: false, faceMM: 140 };
const store = { get() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } },
  set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch {} } };
let cfg = { ...DEF, ...store.get() };

// ---------- vector helpers ----------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const norm = a => mul(a, 1 / (len(a) || 1));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const DEG = 180 / Math.PI;
const wrap90 = a => ((a % 90) + 135) % 90 - 45;

// ---------- state ----------
const S = {
  step: 'intro', facing: 'user', stream: null, track: null, lm: null, torch: false,
  base: {}, scanAcc: null, scanN: 0, ema: null, mmPx: 0, head: null, det: null,
  goodSince: 0, lastVT: -1, tapSet: false, lastSpeak: '', lastSpeakT: 0, lastBeep: 0, xf: null,
};
const N = 44;          // patch grid resolution
const HALF = 0.16;     // patch half-size in face widths
const SCAN_FRAMES = 14;

// ---------- model ----------
async function loadModel() {
  const fs = await FilesetResolver.forVisionTasks(new URL('./vendor/wasm', import.meta.url).href);
  const opts = d => ({ baseOptions: { modelAssetPath: './vendor/face_landmarker.task', delegate: d },
    runningMode: 'VIDEO', numFaces: 1, minFaceDetectionConfidence: 0.4, minFacePresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
  try { S.lm = await FaceLandmarker.createFromOptions(fs, opts('GPU')); }
  catch { S.lm = await FaceLandmarker.createFromOptions(fs, opts('CPU')); }
}

async function startCamera() {
  S.stream?.getTracks().forEach(t => t.stop());
  S.stream = await navigator.mediaDevices.getUserMedia({ audio: false,
    video: { facingMode: S.facing, width: { ideal: 1280 }, height: { ideal: 720 } } });
  S.track = S.stream.getVideoTracks()[0];
  video.srcObject = S.stream; await video.play();
  const caps = S.track.getCapabilities?.() || {};
  $('#torchBtn').style.visibility = caps.torch ? 'visible' : 'hidden';
  S.torch = false; S.lastVT = -1;
}

// ---------- head frame ----------
function headFrame(f, vw, vh) {
  const P = i => [f[i].x * vw, f[i].y * vh, f[i].z * vw];
  const L = P(234), R = P(454), mid = mul(add(L, R), 0.5);
  const X = norm(sub(R, L)), W = len(sub(R, L));
  let U = sub(P(10), P(152)); U = norm(sub(U, mul(X, dot(U, X))));
  let F = cross(X, U); if (dot(F, sub(P(1), mid)) < 0) F = mul(F, -1);
  const yaw = Math.atan2(F[0], -F[2]) * DEG, pitch = Math.asin(clamp(-F[1], -1, 1)) * DEG;
  const rightSide = L[2] < R[2];               // subject's right (234) is nearer the camera
  const ear = rightSide ? L : R, out = norm(sub(ear, mid));
  const down = mul(U, -1), back = mul(F, -1);
  const T = add(add(ear, mul(down, cfg.down * W)), mul(back, cfg.back * W));
  // scale: vertical iris diameter (11.7 mm) on the visible eye, blended with face width
  const ir = rightSide ? [470, 472] : [475, 477];
  const a = P(ir[0]), b = P(ir[1]), irisPx = Math.hypot(a[0] - b[0], a[1] - b[1]);
  let mmPx = cfg.faceMM / W;
  if (irisPx > 5) mmPx = 0.5 * mmPx + 0.5 * (11.7 / irisPx);
  S.mmPx = S.mmPx ? S.mmPx * 0.85 + mmPx * 0.15 : mmPx;
  return { P, W, ear, down, back, out, T, yaw, pitch, side: rightSide ? 'right' : 'left' };
}
const sideOk = h => h.tracked || Math.abs(h.yaw) >= 40;
const sideIdeal = h => h.tracked || Math.abs(h.yaw) >= 55;
const TRACK_MAX = 450; // frames the tracker may run without the mesh (~15 s)

// ---------- ear tracker hand-off ----------
function seedTracker(h, pyr) {
  const k = pyr[0].k, cx = (h.ear[0] + h.T[0]) / 2 * k, cy = (h.ear[1] + h.T[1]) / 2 * k;
  const pts = corners(pyr, cx, cy, 0.32 * h.W * k, h.T[0] * k, h.T[1] * k, cfg.size * 0.8 / S.mmPx * k);
  S.trk = pts.length >= 10 ? { pts, h: { ...h, P: null }, age: 0 } : null;
}
function stepTracker(pyr) {
  if (!S.trk || !S.prevPyr || S.prevPyr[0].w !== pyr[0].w) { S.trk = null; return; }
  const nq = lk(S.prevPyr, pyr, S.trk.pts), p = [], q = [];
  nq.forEach((v, i) => { if (v) { p.push(S.trk.pts[i]); q.push(v); } });
  const m = similarity(p, q);
  if (!m || m.n < 8 || Math.abs(Math.hypot(m.a, m.b) - 1) > 0.15) { S.trk = null; return; }
  const k = pyr[0].k, P = v => [(m.a * v[0] * k - m.b * v[1] * k + m.tx) / k, (m.b * v[0] * k + m.a * v[1] * k + m.ty) / k, v[2]];
  const V = v => [m.a * v[0] - m.b * v[1], m.b * v[0] + m.a * v[1], v[2]], h = S.trk.h;
  S.trk.h = { ...h, ear: P(h.ear), T: P(h.T), down: V(h.down), back: V(h.back) };
  S.trk.pts = q; S.trk.age++;
}

// ---------- patch sampling on the back/down plane ----------
function samplePatch(h, vw, vh) {
  const R = HALF * h.W, pts = new Float32Array(N * N * 2);
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const u = (i / (N - 1) * 2 - 1) * R, v = (j / (N - 1) * 2 - 1) * R;
    const x = h.T[0] + h.back[0] * u + h.down[0] * v, y = h.T[1] + h.back[1] * u + h.down[1] * v;
    const k = (j * N + i) * 2; pts[k] = x; pts[k + 1] = y;
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  x0 = Math.max(0, Math.floor(x0) - 1); y0 = Math.max(0, Math.floor(y0) - 1);
  x1 = Math.min(vw - 1, Math.ceil(x1) + 1); y1 = Math.min(vh - 1, Math.ceil(y1) + 1);
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  if (bw < 8 || bh < 8) return null;
  grab.width = bw; grab.height = bh;
  gctx.drawImage(video, x0, y0, bw, bh, 0, 0, bw, bh);
  const d = gctx.getImageData(0, 0, bw, bh).data, out = new Float32Array(N * N * 3);
  let inside = 0;
  for (let k = 0; k < N * N; k++) {
    const x = Math.round(pts[k * 2] - x0), y = Math.round(pts[k * 2 + 1] - y0);
    if (x < 0 || y < 0 || x >= bw || y >= bh) { out[k * 3] = out[k * 3 + 1] = out[k * 3 + 2] = -1; continue; }
    inside++;
    // 3x3 box average for noise
    let r = 0, g = 0, b = 0, c = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = clamp(x + dx, 0, bw - 1), yy = clamp(y + dy, 0, bh - 1), o = (yy * bw + xx) * 4;
      r += d[o]; g += d[o + 1]; b += d[o + 2]; c++;
    }
    out[k * 3] = r / c; out[k * 3 + 1] = g / c; out[k * 3 + 2] = b / c;
  }
  return inside > N * N * 0.7 ? out : null;
}

function skinRef(h, vw, vh) {
  if (!h.P) return S.skin;
  const ids = h.side === 'right' ? [50, 101, 187] : [280, 330, 411];
  let r = 0, g = 0, b = 0, c = 0;
  grab.width = 7; grab.height = 7;
  for (const id of ids) {
    const p = h.P(id), x = Math.round(p[0]) - 3, y = Math.round(p[1]) - 3;
    if (x < 0 || y < 0 || x + 7 > vw || y + 7 > vh) continue;
    gctx.drawImage(video, x, y, 7, 7, 0, 0, 7, 7);
    const d = gctx.getImageData(0, 0, 7, 7).data;
    for (let o = 0; o < d.length; o += 4) { r += d[o]; g += d[o + 1]; b += d[o + 2]; c++; }
  }
  return c ? (S.skin = [r / c, g / c, b / c]) : null;
}

// ---------- device detection ----------
function detect(h, cur, vw, vh) {
  const base = S.base[h.side];
  const diff = new Float32Array(N * N);
  if (base) {
    // lighting gain from the outer ring, where the device should not be
    let sc = 0, sb = 0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      if (i > 2 && j > 2 && i < N - 3 && j < N - 3) continue;
      const k = (j * N + i) * 3; if (cur[k] < 0 || base[k] < 0) continue;
      sc += cur[k] + cur[k + 1] + cur[k + 2]; sb += base[k] + base[k + 1] + base[k + 2];
    }
    const g = sb > 0 ? clamp(sc / sb, 0.6, 1.6) : 1;
    const cd = (k, kb) => { const o = k * 3, q = kb * 3; if (cur[o] < 0 || base[q] < 0) return 0;
      return Math.hypot(cur[o] - base[q] * g, cur[o + 1] - base[q + 1] * g, cur[o + 2] - base[q + 2] * g); };
    // undo tracking/mesh drift: best global shift (±3 cells), scored on the patch border
    let bx = 0, by = 0, bestE = Infinity;
    for (let sy = -3; sy <= 3; sy++) for (let sx = -3; sx <= 3; sx++) {
      let e = 0;
      for (let j = 3; j < N - 3; j++) for (let i = 3; i < N - 3; i++) {
        if (i > 9 && j > 9 && i < N - 10 && j < N - 10) continue;
        e += cd(j * N + i, (j + sy) * N + i + sx);
      }
      if (e < bestE) { bestE = e; bx = sx; by = sy; }
    }
    // then each cell takes its closest baseline neighbour, so edges that wobble by a cell do not count
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      let m = Infinity;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const ii = i + bx + dx, jj = j + by + dy; if (ii < 0 || jj < 0 || ii >= N || jj >= N) continue;
        m = Math.min(m, cd(j * N + i, jj * N + ii));
      }
      diff[j * N + i] = m === Infinity ? 0 : m;
    }
  } else {
    const s = skinRef(h, vw, vh); if (!s) return null;
    for (let k = 0; k < N * N; k++) {
      const o = k * 3; if (cur[o] < 0) { diff[k] = 0; continue; }
      diff[k] = Math.hypot(cur[o] - s[0], cur[o + 1] - s[1], cur[o + 2] - s[2]);
    }
  }
  if (!S.ema || S.ema.length !== diff.length) S.ema = diff;
  else for (let k = 0; k < diff.length; k++) S.ema[k] = S.ema[k] * 0.55 + diff[k] * 0.45;
  const e = S.ema, sorted = Float32Array.from(e).sort();
  const med = sorted[sorted.length >> 1], mad = sorted[(sorted.length * 0.75) | 0] - med;
  const thr = Math.max(base ? 26 : 45, med + 3 * mad);
  const mask = new Uint8Array(N * N); for (let k = 0; k < N * N; k++) mask[k] = e[k] > thr;
  // connected components; keep the biggest blob whose centre is near the target
  const lab = new Int32Array(N * N).fill(-1), q = new Int32Array(N * N);
  let best = null;
  for (let s0 = 0; s0 < N * N; s0++) {
    if (!mask[s0] || lab[s0] >= 0) continue;
    let qh = 0, qt = 0, n = 0, sx = 0, sy = 0; q[qt++] = s0; lab[s0] = s0;
    const cells = [];
    while (qh < qt) {
      const k = q[qh++], i = k % N, j = (k / N) | 0; n++; sx += i; sy += j; cells.push(k);
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= N || jj >= N) continue;
        const kk = jj * N + ii; if (mask[kk] && lab[kk] < 0) { lab[kk] = s0; q[qt++] = kk; }
      }
    }
    const cx = sx / n, cy = sy / n, rc = Math.hypot(cx - (N - 1) / 2, cy - (N - 1) / 2) / ((N - 1) / 2);
    if (rc > 0.85) continue;
    const score = n * (1 - 0.5 * rc);
    if (!best || score > best.score) best = { n, cx, cy, cells, score };
  }
  const cellPx = (2 * HALF * h.W) / (N - 1), cellMM = cellPx * S.mmPx;
  const expected = (cfg.size / cellMM) ** 2;
  if (!best || best.n < expected * 0.25) return { found: false, mask };
  // rotation: 4th-order angular moment gives the corner direction of a square, mod 90
  let s4 = 0, c4 = 0, fill = 0, rmax = 0;
  for (const k of best.cells) {
    const du = (k % N) - best.cx, dv = ((k / N) | 0) - best.cy, r2 = du * du + dv * dv, th = Math.atan2(dv, du); // from back toward down, as squarePath rotates
    s4 += r2 * r2 * Math.sin(4 * th); c4 += r2 * r2 * Math.cos(4 * th); rmax = Math.max(rmax, r2);
  }
  fill = best.n / (Math.PI * rmax || 1);
  const rot = wrap90(Math.atan2(s4, c4) / 4 * DEG - 45);
  const half = (N - 1) / 2;
  const du = (best.cx - half) * cellMM, dv = (best.cy - half) * cellMM; // + = behind / below target
  return { found: true, mask, du, dv, dist: Math.hypot(du, dv), rot, sizeMM: Math.sqrt(best.n) * cellMM,
    fill, cx: best.cx, cy: best.cy, cellPx, conf: clamp((best.n / expected) * fill * 1.4, 0, 1) };
}

// ---------- guidance text ----------
function guidance(h, d) {
  if (!d?.found) return { msg: 'Device not seen yet. Place it on the outline.', cls: 'warn', ok: false };
  const parts = [];
  const tol = cfg.tol;
  if (Math.abs(d.dv) > tol * 0.7) parts.push(`${Math.abs(d.dv).toFixed(0)} mm ${d.dv > 0 ? 'up' : 'down'}`);
  if (Math.abs(d.du) > tol * 0.7) parts.push(`${Math.abs(d.du).toFixed(0)} mm ${d.du > 0 ? 'toward the ear' : 'toward the back of the neck'}`);
  const rotErr = wrap90(d.rot - cfg.angle);
  const posOk = d.dist <= tol, rotOk = Math.abs(rotErr) <= cfg.tolDeg;
  if (posOk && rotOk) return { msg: `Perfect — ${d.dist.toFixed(1)} mm off. Hold still.`, cls: 'ok', ok: true, rotErr };
  let msg = parts.length ? 'Move ' + parts.join(' and ') : '';
  if (!rotOk) {
    // turn the head-frame rotation into a clockwise/anticlockwise instruction on screen
    const ds = [h.down[0], h.down[1]], bs = [h.back[0], h.back[1]];
    let cr = bs[0] * ds[1] - bs[1] * ds[0]; if (S.facing === 'user') cr = -cr;
    const cw = (-rotErr) * cr > 0;
    msg += (msg ? ', then ' : '') + `rotate ${Math.abs(rotErr).toFixed(0)}° ${cw ? 'clockwise' : 'anticlockwise'}`;
  }
  return { msg, cls: d.dist > tol * 3 ? 'bad' : 'warn', ok: false, rotErr };
}

// ---------- drawing ----------
function layout(vw, vh) {
  const dpr = Math.min(devicePixelRatio || 1, 2), cw = innerWidth * dpr, ch = innerHeight * dpr;
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const s = Math.min(Math.max(cw / vw, ch / vh), Math.min(cw / vw, ch / vh) * 1.35), ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
  const m = S.facing === 'user';
  S.xf = { s, ox, oy, m, cw, dpr };
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#0b0b0c'; ctx.fillRect(0, 0, cw, ch);
  ctx.setTransform(m ? -s : s, 0, 0, s, m ? cw - ox : ox, oy);
}
const toVideo = (sx, sy) => { const { s, ox, oy, m, cw, dpr } = S.xf; sx *= dpr; sy *= dpr;
  return [m ? (cw - sx - ox) / s : (sx - ox) / s, (sy - oy) / s]; };

function squarePath(h, cu, cv_, sizePx, rotDeg) {
  const r = rotDeg / DEG, hs = sizePx / 2, rr = hs * 0.28, pts = [];
  // rounded square in (u = back, v = down) plane
  const corner = (sx, sy) => {
    const u = sx * Math.cos(r) - sy * Math.sin(r), v = sx * Math.sin(r) + sy * Math.cos(r);
    return [h.T[0] + h.back[0] * (cu + u) + h.down[0] * (cv_ + v), h.T[1] + h.back[1] * (cu + u) + h.down[1] * (cv_ + v)];
  };
  for (let a = 0; a < 4; a++) {
    const sx = a === 0 || a === 3 ? 1 : -1, sy = a < 2 ? 1 : -1;
    for (let t = 0; t <= 6; t++) {
      const ang = (a * 90 + t * 15) / DEG;
      pts.push(corner(sx * (hs - rr) + Math.cos(ang) * rr, sy * (hs - rr) + Math.sin(ang) * rr));
    }
  }
  ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
}

function drawOverlay(h, d, g, t) {
  const lw = 2.5 / S.xf.s * S.xf.dpr, sizePx = cfg.size / S.mmPx;
  if (cfg.mesh && h.P) { ctx.fillStyle = '#7fd3ff88'; for (let i = 0; i < 478; i += 2) { const p = h.P(i); ctx.fillRect(p[0] - lw / 2, p[1] - lw / 2, lw, lw); } }
  const pulse = 0.5 + 0.5 * Math.sin(t / 300);
  // target outline
  squarePath(h, 0, 0, sizePx, cfg.angle);
  ctx.fillStyle = g?.ok ? 'rgba(61,220,132,.35)' : `rgba(236,214,196,${0.35 + 0.2 * pulse})`; ctx.fill();
  ctx.setLineDash([6 * lw, 4 * lw]); ctx.lineWidth = lw; ctx.strokeStyle = g?.ok ? '#3ddc84' : '#fff'; ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(h.T[0], h.T[1], lw * 1.6, 0, 7); ctx.fillStyle = '#fff'; ctx.fill();
  if (d?.found && S.step !== 'scan') {
    const half = (N - 1) / 2, cu = (d.cx - half) * d.cellPx, cvv = (d.cy - half) * d.cellPx;
    squarePath(h, cu, cvv, d.sizeMM / S.mmPx, d.rot);
    ctx.lineWidth = lw * 1.2; ctx.strokeStyle = g.ok ? '#3ddc84' : g.cls === 'bad' ? '#ff5a5a' : '#ffc24a'; ctx.stroke();
    if (!g.ok) { // arrow from device to target
      const x0 = h.T[0] + h.back[0] * cu + h.down[0] * cvv, y0 = h.T[1] + h.back[1] * cu + h.down[1] * cvv;
      const x1 = h.T[0], y1 = h.T[1], a = Math.atan2(y1 - y0, x1 - x0), hl = lw * 5;
      if (Math.hypot(x1 - x0, y1 - y0) > hl * 1.5) {
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        ctx.lineTo(x1 - hl * Math.cos(a - 0.5), y1 - hl * Math.sin(a - 0.5)); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - hl * Math.cos(a + 0.5), y1 - hl * Math.sin(a + 0.5)); ctx.stroke();
      }
    }
  }
}

// ---------- audio ----------
let actx;
function beep(f, ms) {
  try { actx ||= new AudioContext(); const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = f; g.gain.value = 0.06; o.connect(g).connect(actx.destination); o.start(); o.stop(actx.currentTime + ms / 1000); } catch {}
}
function speak(msg, force) {
  if (!cfg.voice || !('speechSynthesis' in window)) return;
  const now = performance.now();
  if (!force && (msg === S.lastSpeak ? now - S.lastSpeakT < 6000 : now - S.lastSpeakT < 2800)) return;
  S.lastSpeak = msg; S.lastSpeakT = now; speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(msg));
}
function proximityBeep(d, ok) {
  if (!cfg.voice || !d?.found) return;
  const now = performance.now(), gap = ok ? 140 : clamp(d.dist * 60, 180, 1200);
  if (now - S.lastBeep > gap) { S.lastBeep = now; beep(ok ? 1320 : 880, 60); }
}

// ---------- UI ----------
const ui = { title: $('#title'), sub: $('#sub'), live: $('#live'), hud: $('#hud'), primary: $('#primary'), secondary: $('#secondary') };
function setLive(msg, cls = '') { if (ui.live.textContent !== msg) ui.live.textContent = msg; ui.live.className = cls; }
function hud(items) { const h = items.map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join(''); if (ui.hud.innerHTML !== h) ui.hud.innerHTML = h; }
function toast(m, ms = 2200) { const t = $('#toast'); t.textContent = m; t.style.display = 'block'; clearTimeout(toast.t); toast.t = setTimeout(() => t.style.display = 'none', ms); }

function go(step) {
  S.step = step; S.goodSince = 0;
  const side = S.head?.side;
  $('#camSeg').style.display = step === 'intro' ? '' : 'none';
  $('#shutterRow').style.display = step === 'check' ? 'flex' : 'none';
  $('#brackets').style.display = step === 'check' ? 'block' : 'none';
  $('#ring').style.display = step === 'scan' ? 'block' : 'none';
  $('#result').style.display = step === 'result' ? 'flex' : 'none';
  ui.secondary.style.display = 'none'; ui.primary.style.display = '';
  $('#hdr').textContent = step === 'intro' ? 'Chehra' : 'Placement guide / Fit check';
  if (step === 'intro') { ui.title.textContent = 'Chehra · Dusq placement'; ui.sub.textContent = 'We find the spot behind your ear lobe and guide the device there, live.'; ui.primary.textContent = 'Start camera'; }
  if (step === 'scan') {
    S.scanAcc = null; S.scanN = 0;
    ui.title.textContent = 'Show us the spot'; ui.sub.textContent = 'Face the camera, then slowly turn your head to show the side of your neck. Device off for this scan.';
    ui.primary.style.display = 'none'; ui.secondary.style.display = ''; ui.secondary.textContent = 'Device already on? Skip scan';
    speak('Turn your head to show the side of your neck, behind the ear.', true);
  }
  if (step === 'guide') {
    ui.title.textContent = 'Place it here'; ui.sub.textContent = 'Match the outline, then confirm';
    ui.primary.textContent = "I've placed it"; ui.secondary.style.display = ''; ui.secondary.textContent = 'Rescan bare skin';
    speak('Place the device on the outline behind your ear lobe.', true);
  }
  if (step === 'check') {
    const dir = side === 'left' ? 'right' : side === 'right' ? 'left' : 'left/right';
    ui.title.textContent = `Turn your neck towards your ${dir} and look at your shoulder`; ui.sub.textContent = 'Captures by itself when the fit is right';
    ui.primary.style.display = 'none';
    speak(`Turn your neck towards your ${dir} and look at your shoulder.`, true);
  }
  setLive('');
}

function capture(h, d, g) {
  const img = cv.toDataURL('image/jpeg', 0.9);
  $('#shot').src = img; S.shot = img;
  const ok = g?.ok;
  $('#verdict').textContent = !d?.found ? 'Device not detected' : ok ? '✓ Placed correctly' : 'Needs adjustment';
  $('#verdict').style.color = ok ? 'var(--ok)' : 'var(--warn)';
  const rows = [['Side', h ? `${h.side} ear` : '—'],
    ['Offset from target', d?.found ? `${d.dist.toFixed(1)} mm` : '—'],
    ['Vertical', d?.found ? `${Math.abs(d.dv).toFixed(1)} mm ${d.dv > 0 ? 'low' : 'high'}` : '—'],
    ['Horizontal', d?.found ? `${Math.abs(d.du).toFixed(1)} mm ${d.du > 0 ? 'too far back' : 'too close to ear'}` : '—'],
    ['Rotation error', d?.found ? `${Math.abs(g.rotErr).toFixed(0)}°` : '—'],
    ['Detected size', d?.found ? `${d.sizeMM.toFixed(0)} mm` : '—'],
    ['Detection confidence', d?.found ? `${Math.round(d.conf * 100)}%` : '—'],
    ['Head turn', h ? `${Math.abs(h.yaw).toFixed(0)}°` : '—'], ['Advice', g?.msg || 'Show the side of your neck']];
  $('#rows').innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  speak(ok ? 'Placed correctly.' : g?.msg || 'Device not detected.', true);
  go('result');
}

// ---------- main loop ----------
function frame(t) {
  requestAnimationFrame(frame);
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || S.step === 'result') return;
  layout(vw, vh);
  ctx.drawImage(video, 0, 0, vw, vh);
  if (!S.lm || video.currentTime === S.lastVT) { if (S.head && S.step !== 'intro') drawOverlay(S.head, S.det, S.g, t); return; }
  S.lastVT = video.currentTime;
  const res = S.lm.detectForVideo(video, performance.now());
  const f = res.faceLandmarks?.[0];
  const pyr = S.step === 'intro' ? null : greyPyramid(video, vw, vh, gcnv, gc2d);
  if (pyr) stepTracker(pyr);
  let h = null;
  if (f) {
    h = headFrame(f, vw, vh); h.tracked = false;
    if (pyr) { if (!S.trk || S.trk.age > 8 || S.trk.pts.length < 30) seedTracker(h, pyr); else S.trk.h = { ...h, P: null }; }
  } else if (S.trk && S.trk.age < TRACK_MAX) h = { ...S.trk.h, tracked: true };
  S.prevPyr = pyr;
  if (!h) {
    S.head = null; S.det = null; S.trk = null; hud([]);
    if (S.step !== 'intro') { const m = 'Face not found. Start facing the camera, then turn slowly.'; setLive(m, 'bad'); speak(m); }
    return;
  }
  S.head = h;
  hud([['Side', h.side], ['Turn', h.tracked ? 'full profile · ear-locked' : `${Math.abs(h.yaw).toFixed(0)}°`], ['Scale', `${(1 / S.mmPx).toFixed(1)} px/mm`]]);
  if (S.step === 'intro') { if (cfg.mesh) drawOverlay(h, null, null, t); return; }
  if (!sideOk(h)) {
    S.det = null; S.g = null; drawOverlay(h, null, null, t);
    const m = `Turn your head more to the side (${Math.abs(h.yaw).toFixed(0)}° of ~60°).`; setLive(m, 'warn'); speak('Turn your head more to the side.');
    return;
  }
  const patch = samplePatch(h, vw, vh);
  if (!patch) { setLive('Move a bit back so the ear and neck are in view.', 'warn'); drawOverlay(h, null, null, t); return; }

  if (S.step === 'scan') {
    if (!sideIdeal(h)) { setLive('A little more…', 'warn'); }
    else {
      if (!S.scanAcc || S.scanSide !== h.side) { S.scanAcc = new Float32Array(patch.length); S.scanN = 0; S.scanSide = h.side; }
      for (let k = 0; k < patch.length; k++) S.scanAcc[k] += patch[k];
      S.scanN++; setLive('Hold still…', 'ok');
      $('#ringArc').style.strokeDashoffset = 176 * (1 - S.scanN / SCAN_FRAMES);
      if (S.scanN >= SCAN_FRAMES) {
        S.base[h.side] = S.scanAcc.map(v => v / S.scanN); S.ema = null; beep(1320, 120); go('guide');
      }
    }
    drawOverlay(h, null, null, t); return;
  }
  const d = S.det = detect(h, patch, vw, vh), g = S.g = guidance(h, d);
  drawOverlay(h, d, g, t);
  setLive(g.msg, g.cls); speak(g.msg); proximityBeep(d, g.ok);
  $('#shutter').classList.toggle('ok', g.ok);
  if (S.step === 'check') {
    if (g.ok && sideIdeal(h)) { S.goodSince ||= t; if (t - S.goodSince > 1200) capture(h, d, g); }
    else S.goodSince = 0;
  }
}

// ---------- events ----------
$('#primary').onclick = async () => {
  if (S.step === 'intro') {
    ui.primary.disabled = true; ui.primary.textContent = 'Loading face model…';
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API not available — open this page over https.');
      await Promise.all([startCamera(), S.lm ? null : loadModel()]);
      beep(660, 1); go('scan');
    } catch (e) { toast(e.message || 'Could not start camera'); ui.primary.textContent = 'Try again'; }
    ui.primary.disabled = false;
  } else if (S.step === 'guide') go('check');
};
$('#secondary').onclick = () => {
  if (S.step === 'scan') { go('guide'); toast('Using skin colour instead of a scan — less precise near hair.'); }
  else if (S.step === 'guide') { if (S.head) delete S.base[S.head.side]; go('scan'); }
};
const PREV = { scan: 'intro', guide: 'scan', check: 'guide', result: 'check' };
$('#backBtn').onclick = () => PREV[S.step] && go(PREV[S.step]);
$('#shutter').onclick = () => capture(S.head, S.det, S.g);
$('#retake').onclick = () => go('check');
$('#restart').onclick = () => { S.base = {}; go('scan'); };
$('#save').onclick = () => { const a = document.createElement('a'); a.href = S.shot; a.download = `chehra-fit-${Date.now()}.jpg`; a.click(); };
$('#flipBtn').onclick = async () => { S.facing = S.facing === 'user' ? 'environment' : 'user'; S.base = {}; await startCamera(); toast('Camera switched — rescan the bare skin for best precision.'); };
$('#torchBtn').onclick = async () => { S.torch = !S.torch; try { await S.track.applyConstraints({ advanced: [{ torch: S.torch }] }); } catch {} };
document.querySelectorAll('#camSeg button').forEach(b => b.onclick = () => {
  S.facing = b.dataset.f; document.querySelectorAll('#camSeg button').forEach(x => x.classList.toggle('on', x === b));
});

// settings
const sliders = [['sDown', 'down', v => `${(v * 100).toFixed(1)}% fw`], ['sBack', 'back', v => `${(v * 100).toFixed(1)}% fw`],
  ['sAngle', 'angle', v => `${v}°`], ['sSize', 'size', v => `${v} mm`], ['sTol', 'tol', v => `±${v} mm`], ['sTolDeg', 'tolDeg', v => `±${v}°`]];
function syncSettings() {
  for (const [id, k, fmt] of sliders) { $('#' + id).value = cfg[k]; $('#v' + id.slice(1)).textContent = fmt(cfg[k]); }
  $('#sVoice').checked = cfg.voice; $('#sMesh').checked = cfg.mesh;
}
for (const [id, k, fmt] of sliders) $('#' + id).oninput = e => { cfg[k] = +e.target.value; $('#v' + id.slice(1)).textContent = fmt(cfg[k]); store.set(cfg); };
$('#sVoice').onchange = e => { cfg.voice = e.target.checked; store.set(cfg); };
$('#sMesh').onchange = e => { cfg.mesh = e.target.checked; store.set(cfg); };
$('#setBtn').onclick = () => { syncSettings(); $('#settings').classList.add('open'); };
$('#closeSet').onclick = () => $('#settings').classList.remove('open');
$('#resetCfg').onclick = () => { cfg = { ...DEF }; store.set(cfg); syncSettings(); };
$('#tapSet').onclick = () => { $('#settings').classList.remove('open'); S.tapSet = true; toast('Turn your head to the side, then tap the exact spot behind the ear lobe.', 4000); };
cv.addEventListener('pointerdown', e => {
  if (!S.tapSet) return;
  const h = S.head; if (!h || !sideOk(h)) { toast('Turn to the side first so the ear is in view.'); return; }
  const [x, y] = toVideo(e.clientX, e.clientY), dx = x - h.ear[0], dy = y - h.ear[1];
  // solve [down back] * [a b] * W = d  in the image plane
  const a1 = h.down[0], a2 = h.down[1], b1 = h.back[0], b2 = h.back[1], det = a1 * b2 - b1 * a2;
  if (Math.abs(det) < 1e-3) { toast('Turn further to the side and tap again.'); return; }
  cfg.down = clamp(((dx * b2 - b1 * dy) / det) / h.W, 0, 0.4); cfg.back = clamp(((a1 * dy - dx * a2) / det) / h.W, 0, 0.4);
  store.set(cfg); S.tapSet = false; S.base = {}; S.ema = null;
  toast('Target saved. Rescan the bare skin for best precision.'); if (S.step !== 'intro') go('scan');
});

window.__chehra = { S, cfg: () => cfg, seedTracker }; // test hook
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
go('intro');
requestAnimationFrame(frame);
