// Epoch 0: the 160-network survey of plan section 2, re-run in JavaScript (plan section 8, step 1),
// and a speed test of engine-equivalent code.
//
//   node scripts/survey.mjs                 survey: 20 draws x 8 layer counts, 30 s each
//   node scripts/survey.mjs --hold=4000     the same, with the input held for 4,000 samples (0.25 s)
//   node scripts/survey.mjs --draws=6 --seconds=30 --seed=1 --threshold=0.05
//   node scripts/survey.mjs bench           speed: one 8-layer well, all 48 wells
//
// Plain Node, no imports from src: the network step, output stage and mixer below are copies of
// src/audio/engine-core.ts (which the tests hold to the prototype). Keep them in step by hand.
//
// WEIGHTS. These are NOT TF.js weights. They come from a seeded generator here that draws from the
// same distributions as the TF.js default initialisers:
//   simpleRNN kernel      glorotUniform  U(-a, a), a = sqrt(6 / (inputSize + 16))
//   simpleRNN recurrent   orthogonal     Gram-Schmidt of a 16 x 16 standard normal matrix
//   dense 16 -> 16        glorotUniform  a = sqrt(6 / 32)
//   dense 16 -> 1         glorotUniform  a = sqrt(6 / 17)
//   biases                zeros
// Weights are shared along a draw, as on the plate: the L-layer network of a draw is the first L
// layers of its 8-layer network.
//
// DEFINITIONS (the plan says: "Evolves" means still sounding at 30 s with spectral change of 0.05
// or more between blocks one second apart. The concrete choices below are mine.)
//   signal     the network's raw output after the app's 20 Hz one-pole high-pass, so that a
//              network stuck on a constant counts as silent, as it would be heard.
//   sounding   RMS of the signal over the last second >= 1e-4 (-80 dB re 1.0; raw peaks are ~1-2).
//   block      4,096 samples (256 ms), Hann window, magnitude spectrum (2,049 bins), scaled to sum 1.
//   change     between two blocks: half the sum of absolute differences of their scaled spectra
//              (0 = same spectrum, 1 = no overlap).
//   spectral change of a network: the median change over the pairs of blocks starting at seconds
//              (s, s + 1) for s = 19 .. 28, i.e. the last ten seconds.
//   evolve     sounding and spectral change >= threshold (0.05)
//   steady     sounding and spectral change < threshold
//   silent     not sounding
//   fading     (reported, not a class) evolve, with RMS of the last second below a tenth of the
//              RMS of second 15.

const N = 16;
const CLOCKS = [1, 16, 256, 4096, 8192, 16384, 32768, 65536];
const RATE = 16000;
const PLAN = ["0/19/1", "1/16/3", "3/14/3", "11/7/2", "12/5/3", "10/7/3", "14/5/1", "12/7/1"];

// ───────────────────────────── random numbers ─────────────────────────────

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function glorotUniform(rnd, fanIn, fanOut, length) {
  const a = Math.sqrt(6 / (fanIn + fanOut));
  const w = new Float32Array(length);
  for (let k = 0; k < length; k++) w[k] = (2 * rnd() - 1) * a;
  return w;
}

/** Orthogonal n x n: Gram-Schmidt (modified) on the rows of a standard normal matrix, as TF.js does. */
function orthogonal(rnd, n) {
  const m = new Float64Array(n * n);
  for (let k = 0; k < n * n; k += 2) {
    const u = 1 - rnd(), v = rnd(); // Box-Muller
    const r = Math.sqrt(-2 * Math.log(u));
    m[k] = r * Math.cos(2 * Math.PI * v);
    if (k + 1 < n * n) m[k + 1] = r * Math.sin(2 * Math.PI * v);
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < n; k++) dot += m[i * n + k] * m[j * n + k];
      for (let k = 0; k < n; k++) m[i * n + k] -= dot * m[j * n + k];
    }
    let norm = 0;
    for (let k = 0; k < n; k++) norm += m[i * n + k] ** 2;
    norm = Math.sqrt(norm);
    for (let k = 0; k < n; k++) m[i * n + k] /= norm;
  }
  return Float32Array.from(m);
}

/** The eight layers of one draw, in the engine's layout (see LayerWeights in src/contracts.ts). */
function drawRow(seed) {
  const layers = [];
  for (let i = 0; i < 8; i++) {
    const rnd = mulberry32((seed * 0x9e3779b1 + i * 0x85ebca6b + 0x1234567) >>> 0);
    layers.push({
      layer: i,
      R: orthogonal(rnd, N), // the transpose of an orthogonal matrix has the same distribution
      K: glorotUniform(rnd, CLOCKS[i], N, CLOCKS[i] * N),
      C: i > 0 ? glorotUniform(rnd, N, N, N * N) : null,
      W: i === 0 ? glorotUniform(rnd, N, 1, N) : null,
    });
  }
  return layers;
}

// ───────────────────────────── network (copy of Network in engine-core.ts) ─────────────────────────────

class Network {
  constructor(layers) {
    const L = (this.L = layers.length);
    this.R = layers.map((w) => w.R);
    this.Kf = layers.map((w, i) => (i > 0 ? w.K : null));
    this.C = layers.map((w) => w.C);
    this.K0 = layers[0].K;
    this.W = layers[0].W;
    this.h = []; this.cond = []; this.acc = [];
    for (let i = 0; i < L; i++) { this.h.push(new Float64Array(N)); this.cond.push(new Float64Array(N)); this.acc.push(new Float64Array(N)); }
    this.tmp = new Float64Array(N);
    this.t = 0; this.y = 0;
  }
  step(drive) {
    const L = this.L, h = this.h, cond = this.cond, acc = this.acc, tmp = this.tmp;
    const t = this.t;
    if (t > 0) {
      for (let i = L - 1; i >= 1; i--) {
        if ((t & (CLOCKS[i] - 1)) === 0) {
          const R = this.R[i], hi = h[i], ci = cond[i], ai = acc[i];
          for (let u = 0; u < N; u++) {
            let s = ai[u] + ci[u];
            const b = u * N;
            for (let v = 0; v < N; v++) s += R[b + v] * hi[v];
            tmp[u] = Math.tanh(s);
          }
          hi.set(tmp); ai.fill(0);
          const C = this.C[i], cd = cond[i - 1];
          for (let u = 0; u < N; u++) {
            let s = 0; const b = u * N;
            for (let v = 0; v < N; v++) s += C[b + v] * hi[v];
            cd[u] = s;
          }
        }
      }
    }
    const x = this.y + drive;
    const R0 = this.R[0], h0 = h[0], c0 = cond[0], K0 = this.K0, W = this.W;
    for (let u = 0; u < N; u++) {
      let s = x * K0[u] + c0[u];
      const b = u * N;
      for (let v = 0; v < N; v++) s += R0[b + v] * h0[v];
      tmp[u] = Math.tanh(s);
    }
    let y = 0;
    for (let u = 0; u < N; u++) { h0[u] = tmp[u]; y += W[u] * tmp[u]; }
    for (let i = 1; i < L; i++) {
      const K = this.Kf[i], ai = acc[i], b = (t & (CLOCKS[i] - 1)) * N;
      for (let u = 0; u < N; u++) ai[u] += K[b + u] * y;
    }
    this.y = y; this.t = t + 1;
    return y;
  }
}

// ───────────────────────────── output stage (copy of OutputStage) ─────────────────────────────

class OutputStage {
  constructor(fsHz = RATE, ceiling = 0.89, lookMs = 5, releaseMs = 100) {
    this.c = ceiling;
    this.L = Math.round((fsHz * lookMs) / 1000);
    this.R = 1 - (2 * Math.PI * 20) / fsHz;
    this.k = 1 - Math.exp(-1 / ((releaseMs / 1000) * fsHz));
    this.delay = new Float64Array(this.L);
    this.req = new Float64Array(this.L + 1).fill(1);
    this.avg = new Float64Array(this.L).fill(1);
    this.sum = this.L; this.g = 1; this.i = 0; this.j = 0; this.x1 = 0; this.y1 = 0;
  }
  step(x) {
    const c = this.c, L = this.L, req = this.req;
    const hp = x - this.x1 + this.R * this.y1; this.x1 = x; this.y1 = hp;
    const a = hp < 0 ? -hp : hp;
    req[this.j] = a > c ? c / a : 1; this.j = this.j === L ? 0 : this.j + 1;
    let m = 1; for (let q = 0; q <= L; q++) if (req[q] < m) m = req[q];
    const rel = this.g + (1 - this.g) * this.k;
    const g = (this.g = m < rel ? m : rel);
    const i = this.i;
    this.sum += g - this.avg[i]; this.avg[i] = g;
    const smooth = this.sum / L;
    let out = this.delay[i] * (g < smooth ? g : smooth);
    this.delay[i] = hp; this.i = i + 1 === L ? 0 : i + 1;
    if (out > c) out = c; else if (out < -c) out = -c;
    return out;
  }
}

// ───────────────────────────── spectra ─────────────────────────────

const BLOCK = 4096;
const HANN = Float64Array.from({ length: BLOCK }, (_, k) => 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / BLOCK));
const COS = Float64Array.from({ length: BLOCK / 2 }, (_, k) => Math.cos((2 * Math.PI * k) / BLOCK));
const SIN = Float64Array.from({ length: BLOCK / 2 }, (_, k) => Math.sin((2 * Math.PI * k) / BLOCK));
const REV = (() => {
  const bits = Math.log2(BLOCK), rev = new Uint32Array(BLOCK);
  for (let k = 0; k < BLOCK; k++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((k >> b) & 1) << (bits - 1 - b); rev[k] = r; }
  return rev;
})();

/** Magnitude spectrum of BLOCK samples of `x` from `start`, Hann window, scaled to sum 1 (or all 0). */
function spectrum(x, start, scale = true) {
  const re = new Float64Array(BLOCK), im = new Float64Array(BLOCK);
  for (let k = 0; k < BLOCK; k++) re[REV[k]] = x[start + k] * HANN[k];
  for (let size = 2; size <= BLOCK; size *= 2) {
    const half = size / 2, stride = BLOCK / size;
    for (let o = 0; o < BLOCK; o += size) {
      for (let k = 0; k < half; k++) {
        const c = COS[k * stride], s = -SIN[k * stride];
        const a = o + k, b = a + half;
        const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
  const mag = new Float64Array(BLOCK / 2 + 1);
  let total = 0;
  for (let k = 0; k <= BLOCK / 2; k++) { mag[k] = Math.hypot(re[k], im[k]); total += mag[k]; }
  if (scale && total > 0) for (let k = 0; k < mag.length; k++) mag[k] /= total;
  return mag;
}

/**
 * Coarse spectrum of the second starting at `start`: power averaged over six half-overlapping
 * blocks (Welch), pooled into 40 bands spaced evenly in log frequency from 50 Hz to 8 kHz, scaled
 * to sum 1. Steady noise gives nearly the same coarse spectrum every second; raw blocks do not.
 */
const BANDS = 40;
const BAND_OF = Int32Array.from({ length: BLOCK / 2 + 1 }, (_, k) => {
  const f = (k * RATE) / BLOCK;
  return f < 50 ? -1 : Math.min(BANDS - 1, Math.floor((BANDS * Math.log(f / 50)) / Math.log(8000 / 50)));
});
function coarseSpectrum(x, start) {
  const bands = new Float64Array(BANDS);
  for (let o = 0; o + BLOCK <= RATE; o += BLOCK / 2) {
    const mag = spectrum(x, start + o, false);
    for (let k = 0; k < mag.length; k++) if (BAND_OF[k] >= 0) bands[BAND_OF[k]] += mag[k] * mag[k];
  }
  let total = 0;
  for (let b = 0; b < BANDS; b++) total += bands[b];
  if (total > 0) for (let b = 0; b < BANDS; b++) bands[b] /= total;
  return bands;
}

const change = (a, b) => { let d = 0; for (let k = 0; k < a.length; k++) d += Math.abs(a[k] - b[k]); return d / 2; };
const rms = (x, from, to) => { let s = 0; for (let t = from; t < to; t++) s += x[t] * x[t]; return Math.sqrt(s / (to - from)); };
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };

// ───────────────────────────── survey ─────────────────────────────

const SOUNDING = 1e-4;

/** Run one network for `seconds`; returns its measurements. */
function measure(layers, seconds, hold) {
  const n = seconds * RATE;
  const net = new Network(layers);
  const x = new Float32Array(n);
  const hpR = 1 - (2 * Math.PI * 20) / RATE;
  let x1 = 0, y1 = 0, peak = 0;
  for (let t = 0; t < n; t++) {
    const y = net.step(t < hold ? 1 : 0);
    if (y > peak) peak = y; else if (-y > peak) peak = -y;
    y1 = y - x1 + hpR * y1; x1 = y; // 20 Hz high-pass, as in the app
    x[t] = y1;
  }
  return { peak, ...analyse(x, seconds) };
}

/** Measurements of a high-passed signal of `seconds` whole seconds. */
function analyse(x, seconds) {
  const n = seconds * RATE;
  const endRms = rms(x, n - RATE, n);
  const midRms = rms(x, Math.floor(n / 2), Math.floor(n / 2) + RATE);
  const fine = [], coarse = [];
  const cache = [new Map(), new Map()];
  const get = (which, s) => { const c = cache[which]; if (!c.has(s)) c.set(s, which ? coarseSpectrum(x, s * RATE) : spectrum(x, s * RATE)); return c.get(s); };
  for (let s = Math.max(0, seconds - 11); s + 1 <= seconds - 1; s++) {
    fine.push(change(get(0, s), get(0, s + 1)));
    coarse.push(change(get(1, s), get(1, s + 1)));
  }
  const med = (v) => (v.length ? median(v) : 0);
  return { endRms, midRms, change: med(fine), changeMax: fine.length ? Math.max(...fine) : 0, coarse: med(coarse) };
}

const classify = (m, threshold, by = "change") => (m.endRms < SOUNDING ? "silent" : m[by] >= threshold ? "evolve" : "steady");
const MARK = { evolve: "E", steady: "s", silent: "-" };
const cell = (m, threshold) => `${MARK[classify(m, threshold)]}${MARK[classify(m, threshold, "coarse")]} ${m.endRms < SOUNDING ? "       " : m.change.toFixed(2).slice(1) + "/" + m.coarse.toFixed(2).slice(1)}`;
const LEGEND = "Per network: class by raw blocks, class by coarse spectra (E evolve, s steady, - silent), then the two change values.";

function table(results, threshold, by, title) {
  console.log(`\n${title}`);
  console.log("| Column | Layers | Evolve / steady / silent | Plan (Python, 20 draws) |");
  console.log("|---|---|---|---|");
  let total = 0;
  for (let L = 1; L <= 8; L++) {
    const col = results.map((line) => line[L - 1]);
    const count = (c) => col.filter((m) => classify(m, threshold, by) === c).length;
    total += count("evolve");
    console.log(`| ${L} | ${L} | ${count("evolve")} / ${count("steady")} / ${count("silent")} | ${PLAN[L - 1].replaceAll("/", " / ")} |`);
  }
  const all = results.flat();
  console.log(`Evolve: ${total} of ${all.length} (plan: 63 of 160). Draws with at least one evolving well: ${results.filter((line) => line.some((m) => classify(m, threshold, by) === "evolve")).length} of ${results.length} (plan: 15 of 20).`);
  const plates = [];
  for (let p = 0; (p + 1) * 6 <= results.length; p++) plates.push(results.slice(p * 6, p * 6 + 6).flat().filter((m) => classify(m, threshold, by) === "evolve").length);
  if (plates.length) console.log(`Evolving wells on plates cut from consecutive draws (6 rows each): ${plates.join(", ")} of 48 (plan: 5, 21, 27).`);
  console.log(`How the count depends on the threshold: ${[0.01, 0.02, 0.05, 0.1, 0.2].map((th) => `${th}: ${all.filter((m) => classify(m, th, by) === "evolve").length}`).join("; ")}.`);
}

function survey(opts) {
  const { draws, seconds, hold, seed, threshold } = opts;
  console.log(`Survey: ${draws} draws x 8 layer counts, ${seconds} s each at 16 kHz. Input 1.0 for the first ${hold} sample${hold > 1 ? "s" : ""}. Seed ${seed}.`);
  console.log(`Weights: own seeded generator with the TF.js default distributions (not TF.js itself). Threshold ${threshold}.`);
  const t0 = performance.now();
  const results = []; // [draw][L-1]
  for (let d = 0; d < draws; d++) {
    const row = drawRow(seed * 1000 + d);
    const line = [];
    for (let L = 1; L <= 8; L++) line.push(measure(row.slice(0, L), seconds, hold));
    results.push(line);
    console.log(`  draw ${String(d + 1).padStart(2)}  ${line.map((m) => cell(m, threshold)).join("  ")}`);
  }
  const elapsed = (performance.now() - t0) / 1000;
  console.log(LEGEND);
  table(results, threshold, "change", `A. Change between raw 256 ms blocks one second apart (steady noise counts as change), threshold ${threshold}:`);
  table(results, threshold, "coarse", `B. Change between coarse one-second spectra one second apart (steady noise does not count), threshold ${threshold}:`);
  const all = results.flat();
  const evolving = all.filter((m) => classify(m, threshold) === "evolve");
  console.log(`\nFading among the evolving wells of A: ${evolving.filter((m) => m.endRms < 0.1 * m.midRms).length} (plan: 4). Largest raw peak: ${Math.max(...all.map((m) => m.peak)).toFixed(2)} (plan: 2.44).`);
  console.log(`Time: ${elapsed.toFixed(1)} s for ${(all.length * seconds / 60).toFixed(0)} minutes of audio.`);
}

// ───────────────────────────── speed ─────────────────────────────

/**
 * What Engine.process does for one block at 16 kHz: every sounding well stepped, its output
 * scaled, summed, measured for the reports (level, waveform ring, means of the fast layers),
 * then the output stage. A copy of Engine.render / runVoice without the messages.
 */
function bench() {
  const rows = [];
  for (let r = 0; r < 6; r++) rows.push(drawRow(4242 + r));
  const run = (label, wells, seconds = 10) => {
    const voices = wells.map(([r, L]) => ({ net: new Network(rows[r].slice(0, L)), wave: new Float32Array(256), wavePos: 0, sumSq: 0, unitSum: new Float64Array(L * N), unitCount: new Int32Array(L) }));
    const stage = new OutputStage();
    const mix = new Float64Array(128), out = new Float32Array(128);
    let time = 0;
    const block = () => {
      mix.fill(0);
      for (const v of voices) {
        const net = v.net, h = net.h, h0 = h[0], L = net.L, wave = v.wave, unitSum = v.unitSum;
        let wavePos = v.wavePos, sumSq = v.sumSq;
        for (let k = 0; k < 128; k++) {
          const t = time + k;
          const y = net.step(t < 128 ? 1 : 0) * 0.25;
          mix[k] += y; sumSq += y * y;
          wave[wavePos] = y; if (++wavePos === 256) wavePos = 0;
          for (let u = 0; u < N; u++) unitSum[u] += h0[u];
          for (let i = 1; i < L && i < 3; i++) {
            if (t > 0 && (t & (CLOCKS[i] - 1)) === 0) { const hi = h[i], b = i * N; for (let u = 0; u < N; u++) unitSum[b + u] += hi[u]; v.unitCount[i]++; }
          }
        }
        v.wavePos = wavePos; v.sumSq = sumSq;
      }
      for (let k = 0; k < 128; k++) out[k] = stage.step(mix[k]);
      time += 128;
      if (time % 512 === 0) for (const v of voices) { v.sumSq = 0; v.unitSum.fill(0); v.unitCount.fill(0); } // about 30 times a second
    };
    for (let b = 0; b < RATE / 128; b++) block(); // warm-up: 1 s
    const blocks = Math.floor((seconds * RATE) / 128);
    let worst = 0, check = 0;
    const t0 = performance.now();
    for (let b = 0; b < blocks; b++) {
      const b0 = performance.now();
      block();
      const dt = performance.now() - b0; if (dt > worst) worst = dt;
      check += out[0];
    }
    const ms = performance.now() - t0;
    console.log(`${label.padEnd(30)} ${((100 * ms) / (seconds * 1000)).toFixed(2).padStart(6)}% of real time | slowest block of 128 samples ${worst.toFixed(3)} ms (budget 8 ms) | check ${check.toFixed(3)}`);
  };
  console.log(`Engine-equivalent code, ${process.version}, blocks of 128 samples at 16 kHz, 10 s each:`);
  run("no wells (output stage only)", []);
  run("one 8-layer well", [[0, 8]]);
  for (let L = 1; L <= 8; L++) run(`six wells of ${L} layer${L > 1 ? "s" : ""}`, [0, 1, 2, 3, 4, 5].map((r) => [r, L]));
  const all = [];
  for (let r = 0; r < 6; r++) for (let L = 1; L <= 8; L++) all.push([r, L]);
  run("all 48 wells", all);
  console.log("Plan (prototype, networks only): one eight-layer well 0.7%, all 48 wells 27%, slowest block 0.2 ms.");
}

// ───────────────────────────── main ─────────────────────────────

/** The same measurements on recordings: 16-bit mono 16 kHz WAV files named well_A1.wav .. well_F8.wav. */
async function wavs(dir, threshold) {
  const fs = await import("node:fs");
  console.log(`Recordings in ${dir}, classified with the definitions of this script, threshold ${threshold}.`);
  const results = [];
  for (const letter of "ABCDEF") {
    const line = [];
    for (let col = 1; col <= 8; col++) {
      const b = fs.readFileSync(`${dir}/well_${letter}${col}.wav`);
      const seconds = Math.floor((b.length - 44) / 2 / RATE);
      const x = new Float32Array(seconds * RATE);
      const hpR = 1 - (2 * Math.PI * 20) / RATE;
      let x1 = 0, y1 = 0;
      for (let t = 0; t < x.length; t++) { const y = b.readInt16LE(44 + 2 * t) / 32768; y1 = y - x1 + hpR * y1; x1 = y; x[t] = y1; }
      line.push({ peak: 0, ...analyse(x, seconds) });
    }
    results.push(line);
    console.log(`  row ${letter}  ${line.map((m) => cell(m, threshold)).join("  ")}`);
  }
  console.log(LEGEND);
  for (const [by, name] of [["change", "A (raw blocks)"], ["coarse", "B (coarse spectra)"]]) {
    const all = results.flat();
    const count = (c) => all.filter((m) => classify(m, threshold, by) === c).length;
    console.log(`${name}: ${count("evolve")} evolve, ${count("steady")} steady, ${count("silent")} silent. Per column: ${[0, 1, 2, 3, 4, 5, 6, 7].map((c) => ["evolve", "steady", "silent"].map((k) => results.filter((line) => classify(line[c], threshold, by) === k).length).join("/")).join("  ")}`);
  }
}

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const a = args.find((s) => s.startsWith(`--${name}=`));
  return a ? Number(a.split("=")[1]) : fallback;
};
if (args.includes("bench")) bench();
else if (args[0] === "wav") await wavs(args[1], option("threshold", 0.05));
else survey({ draws: option("draws", 20), seconds: option("seconds", 30), hold: option("hold", 1), seed: option("seed", 1), threshold: option("threshold", 0.05) });
