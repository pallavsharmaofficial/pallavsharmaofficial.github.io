// Ear-anchored tracker: carries the head frame past the yaw where the face mesh drops out.
// Pyramidal Lucas-Kanade on a small grey frame, then a robust similarity fit (shift, turn, scale).
const GW = 320, LV = 3, R = 4;

export function greyPyramid(src, sw, sh, cnv, c2d) {
  const w = GW, h = Math.round(sh * GW / sw);
  cnv.width = w; cnv.height = h; c2d.drawImage(src, 0, 0, w, h);
  const d = c2d.getImageData(0, 0, w, h).data, g = new Float32Array(w * h);
  for (let i = 0, o = 0; i < g.length; i++, o += 4) g[i] = 0.299 * d[o] + 0.587 * d[o + 1] + 0.114 * d[o + 2];
  const P = [{ g, w, h, k: w / sw }];
  for (let l = 1; l < LV; l++) {
    const p = P[l - 1], nw = p.w >> 1, nh = p.h >> 1, ng = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      const i = 2 * y * p.w + 2 * x; ng[y * nw + x] = (p.g[i] + p.g[i + 1] + p.g[i + p.w] + p.g[i + p.w + 1]) / 4;
    }
    P.push({ g: ng, w: nw, h: nh });
  }
  return P;
}

const bil = (I, x, y) => {
  const x0 = x | 0, y0 = y | 0, ax = x - x0, ay = y - y0, i = y0 * I.w + x0;
  return (I.g[i] * (1 - ax) + I.g[i + 1] * ax) * (1 - ay) + (I.g[i + I.w] * (1 - ax) + I.g[i + I.w + 1] * ax) * ay;
};
const inside = (I, x, y, m) => x >= m && y >= m && x < I.w - m - 1 && y < I.h - m - 1;

// strongest corners (Shi-Tomasi) in a disc, skipping a disc where the device will go
export function corners(P, cx, cy, rad, skipX, skipY, skipR, max = 60) {
  const I = P[0], out = [];
  for (let y = Math.max(R + 1, cy - rad | 0); y < Math.min(I.h - R - 2, cy + rad); y += 3)
    for (let x = Math.max(R + 1, cx - rad | 0); x < Math.min(I.w - R - 2, cx + rad); x += 3) {
      if ((x - cx) ** 2 + (y - cy) ** 2 > rad * rad || (x - skipX) ** 2 + (y - skipY) ** 2 < skipR * skipR) continue;
      let a = 0, b = 0, c = 0;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const i = (y + dy) * I.w + x + dx, gx = (I.g[i + 1] - I.g[i - 1]) / 2, gy = (I.g[i + I.w] - I.g[i - I.w]) / 2;
        a += gx * gx; b += gx * gy; c += gy * gy;
      }
      const e = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
      if (e > 60) out.push([x, y, e]);
    }
  out.sort((p, q) => q[2] - p[2]);
  const keep = [];
  for (const p of out) { if (keep.every(q => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 36)) keep.push(p); if (keep.length >= max) break; }
  return keep.map(p => [p[0], p[1]]);
}

export function lk(A, B, pts) {
  const res = [];
  for (const [x, y] of pts) {
    let gx = 0, gy = 0, ok = true, err = 0;
    for (let l = LV - 1; l >= 0 && ok; l--) {
      const s = 1 / (1 << l), I = A[l], J = B[l], px = x * s, py = y * s;
      if (!inside(I, px, py, R + 2)) { ok = false; break; }
      let Gxx = 0, Gxy = 0, Gyy = 0; const ia = [], ix = [], iy = [];
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        const a = bil(I, px + dx, py + dy), X = (bil(I, px + dx + 1, py + dy) - bil(I, px + dx - 1, py + dy)) / 2,
          Y = (bil(I, px + dx, py + dy + 1) - bil(I, px + dx, py + dy - 1)) / 2;
        ia.push(a); ix.push(X); iy.push(Y); Gxx += X * X; Gxy += X * Y; Gyy += Y * Y;
      }
      const det = Gxx * Gyy - Gxy * Gxy; if (det < 1e-3) { ok = false; break; }
      let vx = 0, vy = 0;
      for (let it = 0; it < 10; it++) {
        if (!inside(J, px + gx + vx, py + gy + vy, R + 1)) { ok = false; break; }
        let bx = 0, by = 0, k = 0; err = 0;
        for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++, k++) {
          const df = ia[k] - bil(J, px + dx + gx + vx, py + dy + gy + vy); bx += df * ix[k]; by += df * iy[k]; err += Math.abs(df);
        }
        const ux = (Gyy * bx - Gxy * by) / det, uy = (Gxx * by - Gxy * bx) / det; vx += ux; vy += uy;
        if (ux * ux + uy * uy < 0.0009) break;
      }
      if (l > 0) { gx = 2 * (gx + vx); gy = 2 * (gy + vy); } else { gx += vx; gy += vy; }
    }
    res.push(ok && err / ((2 * R + 1) ** 2) < 14 ? [x + gx, y + gy] : null);
  }
  return res;
}

// least-squares similarity p -> q, refit once without outliers; returns {a,b,tx,ty,n}
export function similarity(p, q) {
  let idx = p.map((_, i) => i);
  let m = null;
  for (let pass = 0; pass < 2; pass++) {
    if (idx.length < 6) return null;
    let pcx = 0, pcy = 0, qcx = 0, qcy = 0;
    for (const i of idx) { pcx += p[i][0]; pcy += p[i][1]; qcx += q[i][0]; qcy += q[i][1]; }
    pcx /= idx.length; pcy /= idx.length; qcx /= idx.length; qcy /= idx.length;
    let sa = 0, sb = 0, sp = 0;
    for (const i of idx) {
      const x = p[i][0] - pcx, y = p[i][1] - pcy, u = q[i][0] - qcx, v = q[i][1] - qcy;
      sa += x * u + y * v; sb += x * v - y * u; sp += x * x + y * y;
    }
    const a = sa / sp, b = sb / sp;
    m = { a, b, tx: qcx - (a * pcx - b * pcy), ty: qcy - (b * pcx + a * pcy) };
    const r = idx.map(i => Math.hypot(a * p[i][0] - b * p[i][1] + m.tx - q[i][0], b * p[i][0] + a * p[i][1] + m.ty - q[i][1]));
    const med = [...r].sort((x, y) => x - y)[r.length >> 1], lim = Math.max(1, 2.5 * med);
    idx = idx.filter((_, k) => r[k] <= lim);
  }
  m.n = idx.length; return m;
}
