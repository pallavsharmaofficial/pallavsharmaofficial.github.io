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
const KEY = 'chehra.dusq.v2';   // v2: defaults measured from a correct placement; v1 targets were for the old detector
const DEF = { down: 0.33, back: 0.19, angle: 10, size: 34, tol: 4, tolDeg: 15, voice: true, mesh: false, tone: 'dark', faceMM: 140 };
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
const N = 48;          // patch grid resolution
const HALF = 0.28;     // patch half-size in face widths
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
  const side = rightSide ? 'right' : 'left', sm = S.hs && S.hs.side === side ? S.hs : null, A = 0.4;
  const f1 = (k, v) => sm ? add(sm[k], mul(sub(v, sm[k]), A)) : v;
  const o = { P, W: sm ? sm.W + (W - sm.W) * A : W, ear: f1('ear', ear), down: norm(f1('down', down)), back: norm(f1('back', back)), out, T: f1('T', T), yaw, pitch, side };
  // axes keep their projected length (foreshortening) after normalising in 3D
  return (S.hs = o);
}
const sideOk = h => h.tracked || Math.abs(h.yaw) >= 28;
const sideIdeal = h => h.tracked || Math.abs(h.yaw) >= 36;
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
  S.img = { d, x0, y0, bw, bh };
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
// Per cell "device evidence" = change from the bare-skin scan (or from skin colour when there is no scan),
// gated by the device tone (a dark device must be darker than skin, which rules out ear edges and glare).
// The threshold comes from the outer ring only, so a large device cannot raise its own threshold.
const lum = (a, o) => 0.299 * a[o] + 0.587 * a[o + 1] + 0.114 * a[o + 2];
function detect(h, cur, vw, vh, useBase = true) {
  const base = useBase ? S.base[h.side] : null, ek = base ? 'emaB' : 'emaS', skin = skinRef(h, vw, vh) || S.skin;
  const skinL = skin ? 0.299 * skin[0] + 0.587 * skin[1] + 0.114 * skin[2] : 140;
  const diff = new Float32Array(N * N), C = (N - 1) / 2;
  const ringOf = k => Math.hypot((k % N) - C, ((k / N) | 0) - C) / C > 0.72;
  let g = 1, bx = 0, by = 0;
  if (base) {
    let sc = 0, sb = 0;
    for (let k = 0; k < N * N; k++) if (ringOf(k) && cur[k * 3] >= 0 && base[k * 3] >= 0) { sc += lum(cur, k * 3); sb += lum(base, k * 3); }
    g = sb > 0 ? clamp(sc / sb, 0.6, 1.6) : 1;
  }
  const cd = (k, kb) => { const o = k * 3, q = kb * 3; if (cur[o] < 0 || base[q] < 0) return 0;
    return Math.hypot(cur[o] - base[q] * g, cur[o + 1] - base[q + 1] * g, cur[o + 2] - base[q + 2] * g); };
  if (base) { // undo mesh/tracker drift: best global shift (±3 cells), scored on the ring
    let bestE = Infinity;
    for (let sy = -3; sy <= 3; sy++) for (let sx = -3; sx <= 3; sx++) {
      let e = 0;
      for (let j = 3; j < N - 3; j += 2) for (let i = 3; i < N - 3; i += 2) { const k = j * N + i; if (ringOf(k)) e += cd(k, (j + sy) * N + i + sx); }
      if (e < bestE) { bestE = e; bx = sx; by = sy; }
    }
  }
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const k = j * N + i, o = k * 3; if (cur[o] < 0) continue;
    let v;
    if (base) { // nearest baseline neighbour, so edges that wobble by a cell do not count
      v = Infinity;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const ii = i + bx + dx, jj = j + by + dy; if (ii >= 0 && jj >= 0 && ii < N && jj < N) v = Math.min(v, cd(k, jj * N + ii));
      }
      if (v === Infinity) v = 0;
    } else v = skin ? Math.hypot(cur[o] - skin[0], cur[o + 1] - skin[1], cur[o + 2] - skin[2]) : 0;
    const L = lum(cur, o) / g;
    if (cfg.tone === 'dark' && L > skinL * (base ? 0.72 : 0.55)) v *= 0.15;
    if (cfg.tone === 'light' && L < skinL * 1.12) v *= 0.15;
    diff[k] = v;
  }
  if (!S[ek] || S[ek].length !== diff.length) S[ek] = diff;
  else for (let k = 0; k < diff.length; k++) S[ek][k] = S[ek][k] * 0.5 + diff[k] * 0.5;
  const e = S[ek], ring = [];
  for (let k = 0; k < N * N; k++) if (ringOf(k)) ring.push(e[k]);
  ring.sort((a, b) => a - b);
  const rMed = ring[ring.length >> 1], rMad = ring[(ring.length * 0.75) | 0] - rMed;
  const thr = clamp(rMed + 4 * rMad + 8, base ? 22 : 35, 90);
  const mask = new Uint8Array(N * N); for (let k = 0; k < N * N; k++) mask[k] = e[k] > thr;
  // 3x3 majority filter removes speckle and thin hair strands
  const m2 = new Uint8Array(N * N);
  for (let j = 1; j < N - 1; j++) for (let i = 1; i < N - 1; i++) {
    let c = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) c += mask[(j + dy) * N + i + dx];
    m2[j * N + i] = c >= 5;
  }
  const lab = new Int32Array(N * N).fill(-1), q = new Int32Array(N * N);
  const cands = [];
  for (let s0 = 0; s0 < N * N; s0++) {
    if (!m2[s0] || lab[s0] >= 0) continue;
    let qh = 0, qt = 0, n = 0, sx = 0, sy = 0, edge = 0; q[qt++] = s0; lab[s0] = s0;
    const cells = [];
    while (qh < qt) {
      const k = q[qh++], i = k % N, j = (k / N) | 0; n++; sx += i; sy += j; cells.push(k);
      if (i < 2 || j < 2 || i > N - 3 || j > N - 3) edge++;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= N || jj >= N) continue;
        const kk = jj * N + ii; if (m2[kk] && lab[kk] < 0) { lab[kk] = s0; q[qt++] = kk; }
      }
    }
    if (n < 6) continue;
    const cx = sx / n, cy = sy / n, rc = Math.hypot(cx - C, cy - C) / C;
    if (rc > 0.68) continue;               // the device is looked for within ~25 mm of the target
    cands.push({ n, cx, cy, cells, score: n * (1 - 0.5 * rc) * (1 - Math.min(0.8, edge / n * 3)) });
  }
  const cellPx = (2 * HALF * h.W) / (N - 1), cellMM = cellPx * S.mmPx, expected = (cfg.size / cellMM) ** 2;
  const side = cfg.size / cellMM, win = side * 0.8;
  cands.sort((x, y) => y.score - x.score);
  let reject = null;
  for (const cand of cands.slice(0, 3)) {
    if (cand.n < expected * 0.15) continue;
    // mean-shift a device-sized window onto the densest mass, so beard or hair touching the device is cut away
    let mx = cand.cx, my = cand.cy;
    if (S.sd?.found && Math.hypot(S.sd.cx - cand.cx, S.sd.cy - cand.cy) < side * 1.5) { mx = S.sd.cx; my = S.sd.cy; }
    for (let it = 0; it < 8; it++) {
      let sx = 0, sy = 0, n = 0;
      for (const k of cand.cells) { const i = k % N, j = (k / N) | 0; if ((i - mx) ** 2 + (j - my) ** 2 <= win * win) { sx += i; sy += j; n++; } }
      if (!n) break;
      const nx = sx / n, ny = sy / n, moved = Math.hypot(nx - mx, ny - my); mx = nx; my = ny; if (moved < 0.05) break;
    }
    const cells = cand.cells.filter(k => ((k % N) - mx) ** 2 + (((k / N) | 0) - my) ** 2 <= win * win);
    if (cells.length < expected * 0.15) continue;
    let s4 = 0, c4 = 0;
    for (const k of cells) {
      const du = (k % N) - mx, dv = ((k / N) | 0) - my, r2 = du * du + dv * dv, th = Math.atan2(dv, du); // from back toward down, as squarePath rotates
      s4 += r2 * r2 * Math.sin(4 * th); c4 += r2 * r2 * Math.cos(4 * th);
    }
    const rot0 = S.sd?.found ? S.sd.rot : wrap90(Math.atan2(s4, c4) / 4 * DEG - 45);
    const fit = fitDevice(h, (mx - C) * cellPx, (my - C) * cellPx, rot0, g, skinL, skin);
    if (!fit.ok) { reject = reject || fit; continue; }
    return { found: true, mask: m2, thr, fit, du: fit.u * S.mmPx, dv: fit.v * S.mmPx, rot: fit.rot, sizeMM: fit.size,
      cx: C + fit.u / cellPx, cy: C + fit.v / cellPx, cellPx, conf: fit.conf };
  }
  return { found: false, mask: m2, thr, fit: reject };
}

// ---------- device verification: rotated rounded-square template at full camera resolution ----------
// The Dusq device is a dark, smooth rounded square with a bright rim. Hair and beard are dark too, but textured,
// rimless and dark on at least one side, so they fail this fit. The fit also refines centre, rotation and size.
const RGB = [0, 0, 0];
function lumAt(x, y) {   // 2x2 mean; also leaves the colour in RGB
  const I = S.img; if (!I) return -1;
  const xi = Math.round(x - I.x0), yi = Math.round(y - I.y0);
  if (xi < 0 || yi < 0 || xi >= I.bw - 1 || yi >= I.bh - 1) return -1;
  let r = 0, gg = 0, b = 0;
  for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) { const o = ((yi + dy) * I.bw + xi + dx) * 4; r += I.d[o]; gg += I.d[o + 1]; b += I.d[o + 2]; }
  RGB[0] = r / 4; RGB[1] = gg / 4; RGB[2] = b / 4;
  return 0.299 * RGB[0] + 0.587 * RGB[1] + 0.114 * RGB[2];
}
function fitDevice(h, u0, v0, rotDeg, g, skinL, skin) {
  const mm = 1 / S.mmPx;   // axis units per millimetre
  const at = (cu, cv, cs, sn, a, b) => lumAt(h.T[0] + h.back[0] * (cu + a * cs - b * sn) + h.down[0] * (cv + a * sn + b * cs),
    h.T[1] + h.back[1] * (cu + a * cs - b * sn) + h.down[1] * (cv + a * sn + b * cs)) / g;
  const evalAt = (cu, cv, r, s) => {
    const cs = Math.cos(r / DEG), sn = Math.sin(r / DEG), hs = s / 2 * mm, ins = [];
    for (let i = -3; i <= 3; i++) for (let j = -3; j <= 3; j++) { const L = at(cu, cv, cs, sn, i / 3 * 0.62 * hs, j / 3 * 0.62 * hs); if (L < 0) return null; ins.push(L); }
    ins.sort((x, y) => x - y);
    const body = ins.slice(0, (ins.length * 0.8) | 0), Li = body[body.length >> 1];   // brightest 20% is the logo
    let tex = 0; for (const L of body) tex += Math.abs(L - Li); tex /= body.length;
    const sides = []; let skinSides = 0;
    for (let k = 0; k < 4; k++) {
      let sum = 0, n = 0, cr = 0, cg = 0, cb = 0;
      for (let t = -3; t <= 3; t++) for (const off of [1.0, 2.2]) {
        const o = hs + off * mm, w = t / 3 * 0.6 * hs;
        const [a, b] = k === 0 ? [o, w] : k === 1 ? [-o, w] : k === 2 ? [w, o] : [w, -o];
        const L = at(cu, cv, cs, sn, a, b); if (L < 0) return null; sum += L; n++; cr += RGB[0]; cg += RGB[1]; cb += RGB[2];
      }
      sides.push(sum / n - Li);
      // is this side skin? compare chromaticity, so light and shadow on the skin both count
      if (skin) { const t = cr + cg + cb || 1, ts = skin[0] + skin[1] + skin[2] || 1;
        if (Math.hypot(cr / t - skin[0] / ts, cg / t - skin[1] / ts) < 0.035 && (cr / n) / g > skin[0] * 0.55) skinSides++; }
    }
    sides.sort((x, y) => y - x);
    const ctr = (sides[0] + sides[1] + sides[2]) / 3;
    return { Li, tex, s3: sides[2], s4: sides[3], ctr, skinSides, J: sides[2] + 0.5 * sides[3] + 0.3 * ctr - 2 * tex + 6 * skinSides };
  };
  let p = { u: u0, v: v0, rot: rotDeg, size: cfg.size }, best = evalAt(p.u, p.v, p.rot, p.size);
  if (!best) return { ok: false, why: 'edge' };
  // coordinate search, steps halve each round
  let st = { u: 2 * mm, v: 2 * mm, rot: 8, size: 3 };
  for (let round = 0; round < 4; round++) {
    let improved = true, guard = 0;
    while (improved && guard++ < 6) {
      improved = false;
      for (const k of ['u', 'v', 'rot', 'size']) for (const sgn of [1, -1]) {
        const q = { ...p, [k]: p[k] + sgn * st[k] };
        if (q.size < cfg.size * 0.8 || q.size > cfg.size * 1.2 || Math.hypot(q.u - u0, q.v - v0) > cfg.size * 0.45 * mm) continue;
        const e = evalAt(q.u, q.v, q.rot, q.size);
        if (e && e.J > best.J) { best = e; p = q; improved = true; }
      }
    }
    for (const k in st) st[k] /= 2;
  }
  const texR = best.tex / Math.max(1, best.ctr);
  const onSkin = !skin || best.skinSides >= 2, near = Math.hypot(p.u, p.v) * S.mmPx <= 24;
  const ok = best.s3 > 18 && best.Li < skinL * 0.62 && texR < 0.28 && onSkin && near;
  const conf = clamp((best.s3 - 12) / 40, 0, 1) * clamp(1 - texR / 0.4, 0, 1) * clamp(0.6 + best.s4 / 50, 0, 1);
  return { ok, conf, u: p.u, v: p.v, rot: wrap90(p.rot), size: p.size, s3: best.s3, s4: best.s4, texR, Li: best.Li, skinSides: best.skinSides,
    why: ok ? '' : !near ? 'too far from target' : !onSkin ? 'not on skin' : best.Li >= skinL * 0.62 ? 'not dark' : texR >= 0.28 ? 'textured (hair?)' : 'no clear edge' };
}

// temporal smoothing + hysteresis so the readout does not flicker
function smoothDet(d) {
  const p = S.sd;
  if (d?.found) {
    if (p?.found) {
      const a = 0.25, dr = wrap90(d.rot - p.rot);
      for (const k of ['du', 'dv', 'cx', 'cy', 'sizeMM', 'conf']) d[k] = p[k] + (d[k] - p[k]) * a;
      d.rot = wrap90(p.rot + dr * a);
    }
    d.dist = Math.hypot(d.du, d.dv); d.miss = 0; return (S.sd = d);
  }
  if (p?.found && (p.miss = (p.miss || 0) + 1) < 8) return p;   // hold through short dropouts
  return (S.sd = d);
}

// ---------- guidance text ----------
function guidance(h, d) {
  if (!d?.found) return { msg: 'Device not seen yet. Place it on the outline.', say: 'Device not seen', cls: 'warn', ok: false };
  const parts = [];
  const tol = cfg.tol;
  if (Math.abs(d.dv) > tol * 0.7) parts.push(`${Math.abs(d.dv).toFixed(0)} mm ${d.dv > 0 ? 'up' : 'down'}`);
  if (Math.abs(d.du) > tol * 0.7) parts.push(`${Math.abs(d.du).toFixed(0)} mm ${d.du > 0 ? 'toward the ear' : 'toward the back of the neck'}`);
  let rotErr = wrap90(d.rot - cfg.angle);
  // a square looks the same every 90°, so near ±45° keep the direction we said last time
  if (Math.abs(rotErr) > 38 && S.rotSign && Math.sign(rotErr) !== S.rotSign) rotErr += S.rotSign * 90;
  S.rotSign = Math.abs(rotErr) > 3 ? Math.sign(rotErr) : 0;
  const posOk = d.dist <= tol, rotOk = Math.abs(rotErr) <= cfg.tolDeg;
  if (posOk && rotOk) return { msg: `Perfect — ${d.dist.toFixed(1)} mm off. Hold still.`, say: 'Perfect. Hold.', cls: 'ok', ok: true, rotErr };
  // short spoken form, biggest correction first, for when the screen cannot be seen
  const sayParts = [];
  const vert = Math.abs(d.dv) > tol * 0.7 && `${d.dv > 0 ? 'Up' : 'Down'} ${Math.round(Math.abs(d.dv))}`;
  const horz = Math.abs(d.du) > tol * 0.7 && `${d.du > 0 ? 'Forward' : 'Back'} ${Math.round(Math.abs(d.du))}`;
  (Math.abs(d.dv) >= Math.abs(d.du) ? [vert, horz] : [horz, vert]).forEach(x => x && sayParts.push(x));
  let msg = parts.length ? 'Move ' + parts.join(' and ') : '';
  if (!rotOk) {
    // turn the head-frame rotation into a clockwise/anticlockwise instruction on screen
    const ds = [h.down[0], h.down[1]], bs = [h.back[0], h.back[1]];
    let cr = bs[0] * ds[1] - bs[1] * ds[0]; if (S.facing === 'user') cr = -cr;
    const cw = (-rotErr) * cr > 0;
    msg += (msg ? ', then ' : '') + `rotate ${Math.abs(rotErr).toFixed(0)}° ${cw ? 'clockwise' : 'anticlockwise'}`;
    if (posOk || sayParts.length < 2) sayParts.push(`Turn ${cw ? 'clockwise' : 'anticlockwise'}`);
  }
  return { msg, say: sayParts.slice(0, 2).join('. '), cls: d.dist > tol * 3 ? 'bad' : 'warn', ok: false, rotErr };
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
  if (cfg.mesh && d?.mask) { // detection debug: search area and cells read as device
    const c = (N - 1) / 2, cp = (2 * HALF * h.W) / (N - 1); ctx.fillStyle = '#ff3b6b99';
    for (let k = 0; k < N * N; k++) if (d.mask[k]) { const u = (k % N - c) * cp, v = (((k / N) | 0) - c) * cp;
      ctx.fillRect(h.T[0] + h.back[0] * u + h.down[0] * v - lw, h.T[1] + h.back[1] * u + h.down[1] * v - lw, 2 * lw, 2 * lw); }
    ctx.strokeStyle = '#ff3b6b'; ctx.lineWidth = lw * 0.6; const R = HALF * h.W; ctx.beginPath();
    [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, b], i) => { const x = h.T[0] + (h.back[0] * a + h.down[0] * b) * R, y = h.T[1] + (h.back[1] * a + h.down[1] * b) * R; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke(); }
  const pulse = 0.5 + 0.5 * Math.sin(t / 300);
  // target outline
  squarePath(h, 0, 0, sizePx, cfg.angle);
  if (!d?.found) { ctx.fillStyle = `rgba(236,214,196,${0.12 + 0.1 * pulse})`; ctx.fill(); }
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
  const native = window.ChehraAndroid;
  if (!cfg.voice || (!native && !('speechSynthesis' in window))) return;
  const now = performance.now();
  if (!force && (msg === S.lastSpeak ? now - S.lastSpeakT < 6000 : now - S.lastSpeakT < 2800)) return;
  S.lastSpeak = msg; S.lastSpeakT = now;
  if (native) { native.speak(msg); return; }
  speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(msg));
}
// pitch tells the direction (high = up, low = down, middle = sideways or turn), the rate tells the distance,
// and the phone in the hand vibrates, so the user can place the device without seeing the screen
function proximityBeep(d, ok) {
  if (!d?.found) return;
  const now = performance.now(), gap = ok ? 150 : clamp(d.dist * 70, 160, 1100);
  if (now - S.lastBeep < gap) return;
  S.lastBeep = now;
  const vertical = Math.abs(d.dv) > cfg.tol * 0.7 && Math.abs(d.dv) >= Math.abs(d.du);
  if (cfg.voice) beep(ok ? 1320 : vertical ? (d.dv > 0 ? 1050 : 520) : 760, ok ? 50 : 70);
  if (navigator.vibrate) { if (ok && !S.wasOk) navigator.vibrate([70, 50, 70]); else if (!ok && d.dist < cfg.tol * 3) navigator.vibrate(15); }
  S.wasOk = ok;
}

// glanceable cues: coloured screen edge + one huge arrow and number, readable from the corner of the eye
function glance(h, d, g) {
  const stage = $('#stage'), arrow = $('#bigArrow');
  stage.dataset.state = !d?.found ? 'none' : g.ok ? 'ok' : g.cls;
  if (!d?.found || g.ok || !['guide', 'check'].includes(S.step)) { arrow.style.display = g?.ok ? 'grid' : 'none'; arrow.className = g?.ok ? 'ok' : ''; arrow.querySelector('b').textContent = g?.ok ? '✓' : ''; arrow.querySelector('i').style.display = g?.ok ? 'none' : ''; return; }
  const c = (N - 1) / 2, cu = (d.cx - c) * d.cellPx, cv_ = (d.cy - c) * d.cellPx;
  let dx = -(h.back[0] * cu + h.down[0] * cv_), dy = -(h.back[1] * cu + h.down[1] * cv_); if (S.facing === 'user') dx = -dx;
  const posOk = d.dist <= cfg.tol;
  arrow.style.display = 'grid'; arrow.className = g.cls;
  arrow.querySelector('i').style.display = posOk ? 'none' : ''; arrow.querySelector('i').style.transform = `rotate(${Math.atan2(dy, dx) * DEG}deg)`;
  arrow.querySelector('b').textContent = posOk ? '⟳' : `${Math.round(d.dist)} mm`;
}

// ---------- UI ----------
const ui = { title: $('#title'), sub: $('#sub'), live: $('#live'), hud: $('#hud'), primary: $('#primary'), secondary: $('#secondary') };
function setLive(msg, cls = '') { if (ui.live.textContent !== msg) ui.live.textContent = msg; ui.live.className = cls; }
function hud(items) { const h = items.map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join(''); if (ui.hud.innerHTML !== h) ui.hud.innerHTML = h; }
function toast(m, ms = 2200) { const t = $('#toast'); t.textContent = m; t.style.display = 'block'; clearTimeout(toast.t); toast.t = setTimeout(() => t.style.display = 'none', ms); }

function go(step) {
  S.step = step; S.goodSince = 0; S.wasOk = false; glance(null, null, null);
  const side = S.head?.side;
  $('#camSeg').style.display = step === 'intro' ? '' : 'none';
  $('#shutterRow').style.display = step === 'check' ? 'flex' : 'none';
  $('#brackets').style.display = step === 'check' ? 'block' : 'none';
  $('#ring').style.display = step === 'scan' ? 'block' : 'none';
  $('#result').style.display = step === 'result' ? 'flex' : 'none';
  ui.secondary.style.display = 'none'; ui.primary.style.display = '';
  $('#hdr').textContent = step === 'intro' ? 'Chehra' : 'Placement guide / Fit check';
  $('#history').style.display = step === 'intro' ? '' : 'none'; if (step === 'intro') showHistory();
  $('#getApp').style.display = step === 'intro' && /Android/i.test(navigator.userAgent) && !window.ChehraAndroid ? '' : 'none';
  if (step === 'intro') { ui.title.textContent = 'Chehra · Dusq placement'; ui.sub.textContent = 'We find the spot behind your ear lobe and guide the device there, live. Turn the sound up: once you turn your head you will hear and feel where to move.'; ui.primary.textContent = 'Start camera'; }
  if (step === 'scan') {
    S.scanAcc = null; S.scanN = 0; S.scanDev = 0;
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
  const hist = loadHist(); hist.unshift({ t: Date.now(), ok: !!ok, found: !!d?.found, dist: d?.found ? +d.dist.toFixed(1) : null, rot: d?.found ? Math.round(Math.abs(g.rotErr)) : null, side: h?.side });
  try { localStorage.setItem(HKEY, JSON.stringify(hist.slice(0, 30))); } catch {}
  speak(ok ? 'Placed correctly.' : g?.msg || 'Device not detected.', true);
  go('result');
}

const HKEY = 'chehra.dusq.history';
const loadHist = () => { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } };
function showHistory() {
  const h = loadHist(), el = $('#history'); if (!el) return;
  el.innerHTML = h.length ? '<h4>Recent fit checks</h4>' + h.slice(0, 5).map(x => `<div><span>${new Date(x.t).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })} · ${x.side || ''}</span><b style="color:${x.ok ? 'var(--ok)' : 'var(--warn)'}">${x.ok ? '✓ ' : ''}${x.found ? x.dist + ' mm · ' + x.rot + '°' : 'not detected'}</b></div>`).join('') : '';
}

// ---------- calibration sharing: a clinician calibrates once and sends the link ----------
const CAL = ['down', 'back', 'angle', 'size', 'tol', 'tolDeg', 'tone'];
function calLink() { const o = {}; CAL.forEach(k => o[k] = cfg[k]); return location.origin + location.pathname + '#cal=' + btoa(JSON.stringify(o)).replace(/=+$/, ''); }
function readCalFromLink() {
  const m = location.hash.match(/cal=([A-Za-z0-9+/_-]+)/); if (!m) return;
  try { const o = JSON.parse(atob(m[1].replace(/-/g, '+').replace(/_/g, '/'))); CAL.forEach(k => { if (k in o) cfg[k] = o[k]; }); store.set(cfg); toast('Calibration loaded from link.', 3000); } catch { toast('That calibration link is damaged.'); }
  history.replaceState(null, '', location.pathname + location.search);
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
    S.head = null; S.det = null; S.trk = null; hud([]); glance(null, null, null);
    if (S.step !== 'intro') { const m = 'Face not found. Start facing the camera, then turn slowly.'; setLive(m, 'bad'); speak(m); }
    return;
  }
  S.head = h;
  hud([['Side', h.side], ['Turn', h.tracked ? 'full profile · ear-locked' : `${Math.abs(h.yaw).toFixed(0)}°`], ['Scale', `${(1 / S.mmPx).toFixed(1)} px/mm`]]);
  if (S.step === 'intro') { if (cfg.mesh) drawOverlay(h, null, null, t); return; }
  if (!sideOk(h)) {
    S.det = null; S.g = null; drawOverlay(h, null, null, t); glance(h, null, null);
    const m = `Turn your head more to the side (${Math.abs(h.yaw).toFixed(0)}° of ~45°).`; setLive(m, 'warn'); speak('Turn your head more to the side.');
    return;
  }
  const patch = samplePatch(h, vw, vh);
  if (!patch) { setLive('Move a bit back so the ear and neck are in view.', 'warn'); drawOverlay(h, null, null, t); return; }

  if (S.step === 'scan') {
    if (!sideIdeal(h)) { setLive('A little more…', 'warn'); }
    else {
      const pre = detect(h, patch, vw, vh, false);
      S.scanDev = pre.found && pre.conf > 0.45 ? (S.scanDev || 0) + 1 : 0;
      if (S.scanDev >= 5) { S.base[h.side] = null; S.sd = null; toast('Device already on, so no scan is needed. Using live detection.', 3500); go('guide'); drawOverlay(h, null, null, t); return; }
      if (!S.scanAcc || S.scanSide !== h.side) { S.scanAcc = new Float32Array(patch.length); S.scanN = 0; S.scanSide = h.side; }
      for (let k = 0; k < patch.length; k++) S.scanAcc[k] += patch[k];
      S.scanN++; setLive('Hold still…', 'ok');
      $('#ringArc').style.strokeDashoffset = 176 * (1 - S.scanN / SCAN_FRAMES);
      if (S.scanN >= SCAN_FRAMES) {
        S.base[h.side] = S.scanAcc.map(v => v / S.scanN); S.emaB = S.emaS = null; S.sd = null; beep(1320, 120); go('guide');
      }
    }
    drawOverlay(h, null, null, t); return;
  }
  let raw = detect(h, patch, vw, vh);
  if (!raw.found && S.base[h.side]) { const alt = detect(h, patch, vw, vh, false); if (alt.found && alt.conf > 0.4) raw = alt; }
  const d = S.det = smoothDet(raw), g = S.g = guidance(h, d);
  drawOverlay(h, d, g, t);
  setLive(g.msg, g.cls); speak(g.say || g.msg); proximityBeep(d, g.ok); glance(h, d, g);
  if (S.step === 'guide') { // hands-free: on target for 1.5 s moves on by itself
    if (g.ok) { S.goodSince ||= t; if (t - S.goodSince > 1500) { beep(1560, 160); speak('Perfect. Press it on firmly. Checking the fit.', true); go('check'); return; } }
    else S.goodSince = 0;
  }
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
    } catch (e) {
      const m = { NotAllowedError: 'Camera permission was blocked. Allow the camera for this site (lock icon in the address bar), then try again.',
        NotFoundError: 'No camera found on this device.', NotReadableError: 'The camera is busy in another app. Close it and try again.',
        OverconstrainedError: 'This camera is not supported. Try the other camera.' }[e.name];
      toast(m || e.message || 'Could not start camera', 6000); ui.primary.textContent = 'Try again';
    }
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
$('#save').onclick = () => {
  if (window.ChehraAndroid) { toast(window.ChehraAndroid.saveImage(S.shot.split(',')[1]) ? 'Saved to Pictures/Chehra' : 'Could not save'); return; }
  const a = document.createElement('a'); a.href = S.shot; a.download = `chehra-fit-${Date.now()}.jpg`; a.click();
};
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
  document.querySelectorAll('#toneSeg button').forEach(b => b.classList.toggle('on', b.dataset.t === cfg.tone));
}
for (const [id, k, fmt] of sliders) $('#' + id).oninput = e => { cfg[k] = +e.target.value; $('#v' + id.slice(1)).textContent = fmt(cfg[k]); store.set(cfg); };
document.querySelectorAll('#toneSeg button').forEach(b => b.onclick = () => { cfg.tone = b.dataset.t; store.set(cfg); S.emaB = S.emaS = null; syncSettings(); });
$('#sVoice').onchange = e => { cfg.voice = e.target.checked; store.set(cfg); };
$('#sMesh').onchange = e => { cfg.mesh = e.target.checked; store.set(cfg); };
$('#setBtn').onclick = () => { syncSettings(); $('#settings').classList.add('open'); };
$('#closeSet').onclick = () => $('#settings').classList.remove('open');
$('#resetCfg').onclick = () => { cfg = { ...DEF }; store.set(cfg); syncSettings(); };
$('#fromDet').onclick = () => {
  const d = S.det, h = S.head;
  if (!d?.found || !h) { toast('Place the device, turn to the side until it is outlined, then try again.', 3500); return; }
  const k = 1 / (S.mmPx * h.W);
  cfg.back = clamp(cfg.back + d.du * k, 0, 0.45); cfg.down = clamp(cfg.down + d.dv * k, 0, 0.45);
  cfg.angle = Math.round(d.rot); cfg.size = Math.round(clamp(d.sizeMM, 10, 45));
  store.set(cfg); S.sd = null; syncSettings(); toast('Saved. This placement is now the target.');
};
$('#shareCal').onclick = async () => {
  const url = calLink();
  try { if (navigator.share) await navigator.share({ title: 'Chehra calibration', url }); else { await navigator.clipboard.writeText(url); toast('Calibration link copied.'); } }
  catch { prompt('Copy this calibration link', url); }
};
$('#tapSet').onclick = () => { $('#settings').classList.remove('open'); S.tapSet = true; toast('Turn your head to the side, then tap the exact spot behind the ear lobe.', 4000); };
cv.addEventListener('pointerdown', e => {
  if (!S.tapSet) return;
  const h = S.head; if (!h || !sideOk(h)) { toast('Turn to the side first so the ear is in view.'); return; }
  const [x, y] = toVideo(e.clientX, e.clientY), dx = x - h.ear[0], dy = y - h.ear[1];
  // solve [down back] * [a b] * W = d  in the image plane
  const a1 = h.down[0], a2 = h.down[1], b1 = h.back[0], b2 = h.back[1], det = a1 * b2 - b1 * a2;
  if (Math.abs(det) < 1e-3) { toast('Turn further to the side and tap again.'); return; }
  cfg.down = clamp(((dx * b2 - b1 * dy) / det) / h.W, 0, 0.4); cfg.back = clamp(((a1 * dy - dx * a2) / det) / h.W, 0, 0.4);
  store.set(cfg); S.tapSet = false; S.base = {}; S.emaB = S.emaS = null;
  toast('Target saved. Rescan the bare skin for best precision.'); if (S.step !== 'intro') go('scan');
});

window.__chehra = { S, cfg: () => cfg, seedTracker }; // test hook
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
readCalFromLink();
go('intro'); showHistory();
requestAnimationFrame(frame);
