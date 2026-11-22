// The audio engine of Epoch 0, free of DOM and worklet APIs so that it runs unchanged in the
// AudioWorklet (engine.worklet.ts) and in Node (tests).
//
//   Network      one well: 1..8 clocked layers of 16 tanh units, one output sample per step()
//   OutputStage  20 Hz high-pass, then a look-ahead limiter
//   Engine       messages in, mono blocks and state reports out
//
// Nothing in the per-sample path allocates. Reports allocate (their buffers are transferred).

import {
  CEILING,
  CLOCKS,
  HIGHPASS_HZ,
  LOOKAHEAD_MS,
  MAX_LAYERS,
  MAX_VOICES,
  N as UNITS,
  OUTPUT_GAIN,
  RELEASE_MS,
  REPORT_HZ,
  ROWS,
  SAMPLE_RATE,
  TICK_FROM_LAYER,
  WAVE_LEN,
  WELLS,
  wellLayers,
  wellRow,
} from "../plate/constants";
import type { EngineReport, LayerWeights, ToEngine, WellReport } from "../contracts";

// Local copies of the constants read in the inner loops. Bundled (as in the worklet) this makes no
// difference; run unbundled (vitest), an imported binding is a property lookup on every use, which
// made the loops several times slower.
const N = UNITS;
const GAIN = OUTPUT_GAIN;
const WAVE = WAVE_LEN;

/** CLOCKS[i] - 1. The clocks are powers of two, so `t & MASK[i]` is the position inside the frame. */
const MASK = new Int32Array(CLOCKS.map((c) => c - 1));

// ───────────────────────────── Network ─────────────────────────────

/**
 * One well. A port of ClockNet in plan/prototype/engine_check.mjs: same maths, same order of
 * operations, Float64 state, Float32 weights, Math.tanh.
 *
 * Slow layers keep no history buffer. Each adds weight * sample to 16 running sums as samples
 * arrive and uses the sums when its clock ticks, so the cost per sample is constant.
 *
 * The weight arrays are those of the row: they are shared with every other well of the row and
 * never copied.
 */
export class Network {
  readonly L: number;
  /** State per layer, N values each. */
  readonly h: Float64Array[] = [];
  /** Bit i is set if layer i (i >= 1) updated during the latest step(). */
  ticked = 0;

  private readonly R: Float32Array[] = []; // [receiver * N + sender]
  private readonly Kf: (Float32Array | null)[] = [null]; // layer i >= 1: [position * N + unit]
  private readonly C: (Float32Array | null)[] = [null]; // C[i], i >= 1: layer i -> layer i - 1
  private readonly K0: Float32Array;
  private readonly W: Float32Array;
  private readonly cond: Float64Array[] = [];
  private readonly acc: Float64Array[] = [];
  private readonly tmp = new Float64Array(N);
  private t = 0;
  private y = 0;

  /** `layers` are layers 0 .. L-1 of one row, in order. */
  constructor(layers: LayerWeights[]) {
    const L = layers.length;
    if (L < 1 || L > MAX_LAYERS) throw new Error(`Network: ${L} layers`);
    const problem = layers.map((w, i) => layerProblem(w, i)).find((p) => p !== null);
    if (problem) throw new Error(`Network: ${problem}`);
    this.L = L;
    for (let i = 0; i < L; i++) {
      this.R.push(layers[i].R);
      if (i > 0) {
        this.Kf.push(layers[i].K);
        this.C.push(layers[i].C);
      }
      this.h.push(new Float64Array(N));
      this.cond.push(new Float64Array(N));
      this.acc.push(new Float64Array(N));
    }
    this.K0 = layers[0].K;
    this.W = layers[0].W!;
  }

  /** Advance one sample. `drive` is added to the network's own previous output to form its input. */
  step(drive: number): number {
    const L = this.L;
    const h = this.h;
    const cond = this.cond;
    const acc = this.acc;
    const tmp = this.tmp;
    const t = this.t;
    let ticked = 0;
    if (t > 0) {
      // Slowest first, so a layer that ticks sees the fresh state of the layer above it.
      for (let i = L - 1; i >= 1; i--) {
        if ((t & MASK[i]) === 0) {
          const R = this.R[i];
          const hi = h[i];
          const ci = cond[i];
          const ai = acc[i];
          for (let u = 0; u < N; u++) {
            let s = ai[u] + ci[u];
            const b = u * N;
            for (let v = 0; v < N; v++) s += R[b + v] * hi[v];
            tmp[u] = Math.tanh(s);
          }
          hi.set(tmp);
          ai.fill(0);
          const C = this.C[i]!;
          const cd = cond[i - 1];
          for (let u = 0; u < N; u++) {
            let s = 0;
            const b = u * N;
            for (let v = 0; v < N; v++) s += C[b + v] * hi[v];
            cd[u] = s;
          }
          ticked |= 1 << i;
        }
      }
    }
    const x = this.y + drive;
    const R0 = this.R[0];
    const h0 = h[0];
    const c0 = cond[0];
    const K0 = this.K0;
    const W = this.W;
    for (let u = 0; u < N; u++) {
      let s = x * K0[u] + c0[u];
      const b = u * N;
      for (let v = 0; v < N; v++) s += R0[b + v] * h0[v];
      tmp[u] = Math.tanh(s);
    }
    let y = 0;
    for (let u = 0; u < N; u++) {
      h0[u] = tmp[u];
      y += W[u] * tmp[u];
    }
    for (let i = 1; i < L; i++) {
      const K = this.Kf[i]!;
      const ai = acc[i];
      const b = (t & MASK[i]) * N;
      for (let u = 0; u < N; u++) ai[u] += K[b + u] * y;
    }
    this.y = y;
    this.t = t + 1;
    this.ticked = ticked;
    return y;
  }

  /** Back to rest: all state zero, clocks at zero. */
  reset(): void {
    for (let i = 0; i < this.L; i++) {
      this.h[i].fill(0);
      this.cond[i].fill(0);
      this.acc[i].fill(0);
    }
    this.t = 0;
    this.y = 0;
    this.ticked = 0;
  }
}

/** null if `w` is a well-formed layer `layer` (any row), otherwise what is wrong with it. */
function layerProblem(w: LayerWeights, layer: number): string | null {
  if (!w || w.layer !== layer) return `expected layer ${layer}`;
  if (!(w.R instanceof Float32Array) || w.R.length !== N * N) return `layer ${layer}: R`;
  if (!(w.K instanceof Float32Array) || w.K.length !== CLOCKS[layer] * N) return `layer ${layer}: K`;
  if (layer === 0) {
    if (!(w.W instanceof Float32Array) || w.W.length !== N) return "layer 0: W";
  } else if (!(w.C instanceof Float32Array) || w.C.length !== N * N) {
    return `layer ${layer}: C`;
  }
  return null;
}

// ───────────────────────────── OutputStage ─────────────────────────────

/**
 * 20 Hz one-pole high-pass, then a look-ahead limiter. A port of plan/prototype/limiter_check.mjs.
 *
 * The limiter delays the signal by the look-ahead time. For every sample it works out the gain
 * that sample needs to stay under the ceiling, holds the lowest such gain over the look-ahead
 * window, lets it recover over the release time, and smooths the result with a moving average as
 * long as the look-ahead. A delayed sample is always inside the window when it leaves, so its
 * gain is never more than it needs: the output cannot pass the ceiling. (A final clamp removes
 * the last rounding error of that multiplication; it never changes a value by more than that.)
 */
export class OutputStage {
  readonly ceiling: number;
  /** Look-ahead, and so the delay, in samples. */
  readonly look: number;
  /** Lowest gain applied since the caller last set this back to 1. */
  minGain = 1;

  private readonly hpR: number;
  private readonly k: number;
  private readonly delay: Float64Array;
  private readonly req: Float64Array;
  private readonly avg: Float64Array;
  private sum: number;
  private g = 1;
  private i = 0;
  private j = 0;
  private x1 = 0;
  private y1 = 0;

  constructor(
    fsHz: number = SAMPLE_RATE,
    ceiling: number = CEILING,
    lookMs: number = LOOKAHEAD_MS,
    releaseMs: number = RELEASE_MS,
    highpassHz: number = HIGHPASS_HZ,
  ) {
    this.ceiling = ceiling;
    this.look = Math.max(1, Math.round((fsHz * lookMs) / 1000));
    this.hpR = 1 - (2 * Math.PI * highpassHz) / fsHz;
    this.k = 1 - Math.exp(-1 / ((releaseMs / 1000) * fsHz));
    this.delay = new Float64Array(this.look);
    this.req = new Float64Array(this.look + 1).fill(1);
    this.avg = new Float64Array(this.look).fill(1);
    this.sum = this.look;
  }

  step(x: number): number {
    const c = this.ceiling;
    const look = this.look;
    const req = this.req;
    const hp = x - this.x1 + this.hpR * this.y1; // remove constant offset
    this.x1 = x;
    this.y1 = hp;
    const a = hp < 0 ? -hp : hp;
    req[this.j] = a > c ? c / a : 1;
    this.j = this.j === look ? 0 : this.j + 1;
    let m = 1; // lowest gain needed by any sample in the window
    for (let q = 0; q <= look; q++) if (req[q] < m) m = req[q];
    const rel = this.g + (1 - this.g) * this.k; // hold, then release slowly
    const g = m < rel ? m : rel;
    this.g = g;
    const i = this.i;
    this.sum += g - this.avg[i]; // ramp the gain over the look-ahead time
    this.avg[i] = g;
    const smooth = this.sum / look;
    const gain = g < smooth ? g : smooth;
    let out = this.delay[i] * gain;
    this.delay[i] = hp;
    this.i = i + 1 === look ? 0 : i + 1;
    if (gain < this.minGain) this.minGain = gain;
    if (out > c) out = c;
    else if (out < -c) out = -c;
    return out;
  }

  reset(): void {
    this.delay.fill(0);
    this.req.fill(1);
    this.avg.fill(1);
    this.sum = this.look;
    this.g = 1;
    this.i = 0;
    this.j = 0;
    this.x1 = 0;
    this.y1 = 0;
    this.minGain = 1;
  }
}

// ───────────────────────────── Engine ─────────────────────────────

/** A sounding well and what is gathered for its next report. */
interface Voice {
  well: number;
  net: Network;
  /** 1 while the button is down, otherwise 0. */
  drive: number;
  sumSq: number;
  count: number;
  /** Ring of the last WAVE_LEN output samples (after the fixed gain). */
  wave: Float32Array;
  wavePos: number;
  /** Sums of the states of the fast layers since the last report, [layer * N + unit]. */
  unitSum: Float64Array;
  /** How many states went into unitSum, per layer. */
  unitCount: Int32Array;
  ticks: number;
}

/** Samples mixed in one go. Longer blocks are cut into chunks of this size. */
const CHUNK = 1024;
/** Reports kept if nobody takes them. Older ones are dropped. */
const MAX_PENDING_REPORTS = 64;

export class Engine {
  /** Rate of the blocks filled by process(). The networks always run at SAMPLE_RATE. */
  readonly outputRate: number;

  private readonly rows: (LayerWeights | null)[][] = [];
  /** Sounding wells, oldest first. */
  private readonly voices: Voice[] = [];
  /** A well keeps its Voice between presses so that pressing does not rebuild it. */
  private readonly cache: (Voice | null)[] = new Array<Voice | null>(WELLS).fill(null);
  private readonly stage = new OutputStage(SAMPLE_RATE);
  private readonly mix = new Float64Array(CHUNK);
  private readonly reports: EngineReport[] = [];

  private maxVoices = MAX_VOICES;
  private reportHz = REPORT_HZ;
  /** Bit i: layer i updates more often than reports are sent, so its report value is a mean. */
  private fastMask = 0;
  /** Count of 16 kHz samples produced. */
  private time = 0;
  private reportAcc = 0;
  private untilReport = 0;
  private peak = 0;

  // Rate conversion, used only when outputRate !== SAMPLE_RATE.
  private readonly passthrough: boolean;
  private scratch = new Float32Array(0);
  private phase = 0;
  private prev = 0;
  private cur = 0;

  constructor(opts: { outputRate: number }) {
    const rate = Math.round(opts.outputRate);
    if (!(rate > 0) || !Number.isFinite(rate)) throw new Error(`Engine: output rate ${opts.outputRate}`);
    this.outputRate = rate;
    this.passthrough = rate === SAMPLE_RATE;
    if (!this.passthrough) this.scratch = new Float32Array(Math.ceil((2048 * SAMPLE_RATE) / rate) + 2);
    for (let r = 0; r < ROWS; r++) this.rows.push(new Array<LayerWeights | null>(MAX_LAYERS).fill(null));
    this.setReportHz(REPORT_HZ);
  }

  // ── messages ──

  handle(msg: ToEngine): void {
    switch (msg.type) {
      case "weights":
        this.setWeights(msg.weights);
        break;
      case "clear":
        this.voices.length = 0;
        this.cache.fill(null);
        for (const row of this.rows) row.fill(null);
        break;
      case "press":
        this.press(msg.well);
        break;
      case "release": {
        const v = this.find(msg.well);
        if (v) v.drive = 0;
        break;
      }
      case "stop": {
        const k = this.voices.findIndex((v) => v.well === msg.well);
        if (k >= 0) this.voices.splice(k, 1);
        break;
      }
      case "stopAll":
        this.voices.length = 0;
        break;
      case "config":
        if (typeof msg.maxVoices === "number" && msg.maxVoices >= 0) {
          this.maxVoices = Math.floor(msg.maxVoices);
          this.trimVoices();
        }
        if (typeof msg.reportHz === "number" && msg.reportHz > 0 && Number.isFinite(msg.reportHz)) {
          this.setReportHz(msg.reportHz);
        }
        break;
    }
  }

  private setWeights(w: LayerWeights): void {
    if (!w || !Number.isInteger(w.row) || w.row < 0 || w.row >= ROWS) return;
    if (!Number.isInteger(w.layer) || w.layer < 0 || w.layer >= MAX_LAYERS) return;
    if (layerProblem(w, w.layer) !== null) return;
    const row = this.rows[w.row];
    if (row[w.layer] !== null) {
      // A layer is being replaced: wells built on the old arrays must not go on using them.
      for (let well = 0; well < WELLS; well++) {
        if (wellRow(well) !== w.row || wellLayers(well) <= w.layer) continue;
        const k = this.voices.findIndex((v) => v.well === well);
        if (k >= 0) this.voices.splice(k, 1);
        this.cache[well] = null;
      }
    }
    row[w.layer] = w;
  }

  private press(well: number): void {
    if (!Number.isInteger(well) || well < 0 || well >= WELLS) return;
    const sounding = this.find(well);
    if (sounding) {
      sounding.drive = 1;
      return;
    }
    let v = this.cache[well];
    if (!v) {
      const L = wellLayers(well);
      const row = this.rows[wellRow(well)];
      const layers: LayerWeights[] = [];
      for (let i = 0; i < L; i++) {
        const w = row[i];
        if (!w) return; // not all loaded yet: ignore the press
        layers.push(w);
      }
      v = {
        well,
        net: new Network(layers),
        drive: 0,
        sumSq: 0,
        count: 0,
        wave: new Float32Array(WAVE_LEN),
        wavePos: 0,
        unitSum: new Float64Array(L * N),
        unitCount: new Int32Array(L),
        ticks: 0,
      };
      this.cache[well] = v;
    }
    // Start from rest.
    v.net.reset();
    v.drive = 1;
    v.sumSq = 0;
    v.count = 0;
    v.wave.fill(0);
    v.wavePos = 0;
    v.unitSum.fill(0);
    v.unitCount.fill(0);
    v.ticks = 0;
    this.voices.push(v);
    this.trimVoices();
  }

  private find(well: number): Voice | undefined {
    const voices = this.voices;
    for (let k = 0; k < voices.length; k++) if (voices[k].well === well) return voices[k];
    return undefined;
  }

  /** Over the cap: the wells that started first stop first. */
  private trimVoices(): void {
    const over = this.voices.length - this.maxVoices;
    if (over > 0) this.voices.splice(0, over);
  }

  private setReportHz(hz: number): void {
    this.reportHz = Math.min(hz, SAMPLE_RATE);
    this.fastMask = 0;
    for (let i = 0; i < MAX_LAYERS; i++) {
      if (SAMPLE_RATE / CLOCKS[i] > this.reportHz) this.fastMask |= 1 << i;
    }
    for (const v of this.voices) {
      v.unitSum.fill(0);
      v.unitCount.fill(0);
    }
    // Start counting afresh: the next report comes one interval from now.
    this.reportAcc = 0;
    this.scheduleReport();
  }

  // ── audio ──

  /** Fill one mono block at outputRate. */
  process(out: Float32Array): void {
    const M = out.length;
    if (this.passthrough) {
      this.render(out, M);
      return;
    }
    // Linear interpolation between consecutive 16 kHz samples. `phase / rate` is the position
    // between `prev` and `cur`; it is kept as an integer so the two rates never drift apart.
    const rate = this.outputRate;
    let phase = this.phase;
    let prev = this.prev;
    let cur = this.cur;
    const need = Math.floor((phase + M * SAMPLE_RATE) / rate);
    if (this.scratch.length < need) this.scratch = new Float32Array(need); // only for unusually long blocks
    const src = this.scratch;
    this.render(src, need);
    const inv = 1 / rate;
    let j = 0;
    for (let k = 0; k < M; k++) {
      out[k] = prev + (cur - prev) * (phase * inv);
      phase += SAMPLE_RATE;
      while (phase >= rate) {
        phase -= rate;
        prev = cur;
        cur = src[j++];
      }
    }
    this.phase = phase;
    this.prev = prev;
    this.cur = cur;
  }

  /** Produce `n` samples at 16 kHz into `buf`: wells, fixed gain, high-pass, limiter. */
  private render(buf: Float32Array, n: number): void {
    const mix = this.mix;
    const stage = this.stage;
    const voices = this.voices;
    let off = 0;
    while (off < n) {
      let seg = n - off;
      if (seg > this.untilReport) seg = this.untilReport;
      if (seg > CHUNK) seg = CHUNK;
      for (let k = 0; k < seg; k++) mix[k] = 0;
      for (let v = 0; v < voices.length; v++) this.runVoice(voices[v], seg);
      let peak = this.peak;
      for (let k = 0; k < seg; k++) {
        let x = mix[k];
        if (x - x !== 0) x = 0; // NaN or infinite (broken weights) must not get into the filter state
        const y = stage.step(x);
        buf[off + k] = y;
        if (y > peak) peak = y;
        else if (-y > peak) peak = -y;
      }
      this.peak = peak;
      off += seg;
      this.time += seg;
      this.untilReport -= seg;
      if (this.untilReport === 0) {
        this.pushReport();
        this.scheduleReport();
      }
    }
  }

  /** Step one well `n` times, add its output to the mix and gather its report values. */
  private runVoice(v: Voice, n: number): void {
    const mix = this.mix;
    const net = v.net;
    const L = net.L;
    const h = net.h;
    const h0 = h[0];
    const drive = v.drive;
    const wave = v.wave;
    const unitSum = v.unitSum;
    const unitCount = v.unitCount;
    const fastMask = this.fastMask;
    const fast0 = (fastMask & 1) !== 0;
    let wavePos = v.wavePos;
    let sumSq = v.sumSq;
    let ticks = v.ticks;
    for (let k = 0; k < n; k++) {
      const y = net.step(drive) * GAIN;
      mix[k] += y;
      sumSq += y * y;
      wave[wavePos] = y;
      if (++wavePos === WAVE) wavePos = 0;
      if (fast0) for (let u = 0; u < N; u++) unitSum[u] += h0[u];
      const ticked = net.ticked;
      if (ticked !== 0) {
        ticks |= ticked;
        const fast = ticked & fastMask;
        if (fast !== 0) {
          for (let i = 1; i < L; i++) {
            if ((fast & (1 << i)) === 0) continue;
            const hi = h[i];
            const b = i * N;
            for (let u = 0; u < N; u++) unitSum[b + u] += hi[u];
            unitCount[i]++;
          }
        }
      }
    }
    if (fast0) unitCount[0] += n;
    v.wavePos = wavePos;
    v.sumSq = sumSq;
    v.count += n;
    v.ticks = ticks;
  }

  // ── reports ──

  /** The next report is due after this many samples, so that there are exactly reportHz per second. */
  private scheduleReport(): void {
    const k = Math.max(1, Math.ceil((SAMPLE_RATE - this.reportAcc) / this.reportHz));
    this.reportAcc += k * this.reportHz - SAMPLE_RATE;
    if (this.reportAcc < 0) this.reportAcc = 0;
    this.untilReport = k;
  }

  private pushReport(): void {
    const tickMask = ~((1 << TICK_FROM_LAYER) - 1);
    const wells: WellReport[] = [];
    for (const v of this.voices) {
      const L = v.net.L;
      const wave = new Float32Array(WAVE_LEN);
      wave.set(v.wave.subarray(v.wavePos), 0); // oldest first
      wave.set(v.wave.subarray(0, v.wavePos), WAVE_LEN - v.wavePos);
      const units = new Float32Array(L * N);
      for (let i = 0; i < L; i++) {
        const hi = v.net.h[i];
        const count = v.unitCount[i];
        const b = i * N;
        if (count > 0) for (let u = 0; u < N; u++) units[b + u] = v.unitSum[b + u] / count;
        else for (let u = 0; u < N; u++) units[b + u] = hi[u];
      }
      wells.push({
        well: v.well,
        level: v.count > 0 ? Math.sqrt(v.sumSq / v.count) : 0,
        wave,
        units,
        ticks: v.ticks & tickMask,
        held: v.drive !== 0,
      });
      v.sumSq = 0;
      v.count = 0;
      v.unitSum.fill(0);
      v.unitCount.fill(0);
      v.ticks = 0;
    }
    this.reports.push({ type: "report", time: this.time, wells, gain: this.stage.minGain, peak: this.peak });
    if (this.reports.length > MAX_PENDING_REPORTS) this.reports.shift();
    this.stage.minGain = 1;
    this.peak = 0;
  }

  /**
   * The oldest report not yet taken, or null. Reports fall due reportHz times per second of engine
   * time; call this after every process() (repeatedly, if the blocks are long) to collect them.
   */
  takeReport(): EngineReport | null {
    return this.reports.length > 0 ? this.reports.shift()! : null;
  }
}
