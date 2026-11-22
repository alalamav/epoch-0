// The audio engine (src/audio/engine-core.ts) against:
//   (a) the prototype network of plan/prototype/engine_check.mjs, run side by side;
//   (b) stored reference output of the prototypes (tests/fixtures/engine-reference.json);
//   (c) the output-stage table of plan section 4;
//   (d) the message semantics of src/contracts.ts.
//
// Weights are deterministic (tests/helpers/engine-weights.ts): the prototype's generator and scales,
// not TF.js. tests/parity.test.ts covers TF.js.

import { describe, expect, it } from "vitest";

import { Engine, Network, OutputStage } from "../src/audio/engine-core";
import type { EngineReport, LayerWeights, ToEngine } from "../src/contracts";
import {
  CEILING,
  CLOCKS,
  N,
  OUTPUT_GAIN,
  REPORT_HZ,
  SAMPLE_RATE,
  TICK_FROM_LAYER,
  WAVE_LEN,
  wellIndex,
} from "../src/plate/constants";
import fixture from "./fixtures/engine-reference.json";
import { ClockNet, ProtoOutputStage } from "./helpers/engine-prototype";
import * as tw from "./helpers/engine-weights";
import { cloneLayer, flatOfPrefix, flatWeights, lcg, toLayers } from "./helpers/engine-weights";

// ───────────────────────────── helpers ─────────────────────────────

const LONG = 70000; // long enough for the slowest layer (65,536 samples) to update once

/** Rows of deterministic weights, built once (an eight-layer row has 2 million weights). */
const rowCache = new Map<number, LayerWeights[]>();
const ROW_SEEDS = [12345, 777, 4242];
function rowWeights(row: number): LayerWeights[] {
  let layers = rowCache.get(row);
  if (!layers) {
    layers = toLayers(8, flatWeights(8, ROW_SEEDS[row]), row);
    rowCache.set(row, layers);
  }
  return layers;
}

/** Send layers 0 .. count-1 of a row (copies, as a message would carry them). */
function load(engine: Engine, row: number, count = 8): void {
  for (let i = 0; i < count; i++) engine.handle({ type: "weights", weights: cloneLayer(rowWeights(row)[i]) });
}

/** The prototype network that is well (row, layers). */
const protoWell = (row: number, layers: number): ClockNet =>
  new ClockNet(layers, flatOfPrefix(rowWeights(row), layers), Float32Array);

/** Runs an engine block by block, keeping its output and reports. */
class Rig {
  readonly engine: Engine;
  readonly reports: EngineReport[] = [];
  private chunks: Float32Array[] = [];
  constructor(outputRate = SAMPLE_RATE, private readonly block = 128) {
    this.engine = new Engine({ outputRate });
  }
  send(...msgs: ToEngine[]): this {
    for (const m of msgs) this.engine.handle(m);
    return this;
  }
  /** Process `n` output samples in blocks. */
  run(n: number): this {
    for (let done = 0; done < n; ) {
      const len = Math.min(this.block, n - done);
      const out = new Float32Array(len).fill(123); // must be overwritten
      this.engine.process(out);
      this.chunks.push(out);
      done += len;
      for (let r = this.engine.takeReport(); r; r = this.engine.takeReport()) this.reports.push(r);
    }
    return this;
  }
  get out(): Float32Array {
    const total = this.chunks.reduce((a, c) => a + c.length, 0);
    const all = new Float32Array(total);
    let o = 0;
    for (const c of this.chunks) {
      all.set(c, o);
      o += c.length;
    }
    this.chunks = [all];
    return all;
  }
  get last(): EngineReport {
    return this.reports[this.reports.length - 1];
  }
}

/**
 * What the prototypes produce for the same performance: each well is a ClockNet driven by
 * `drive(t)` from sample `from` (rest before), summed, gain, prototype output stage.
 */
interface RefWell {
  net: ClockNet;
  drive: (t: number) => number;
  from?: number;
  to?: number;
}
function reference(wells: RefWell[], n: number): { out: Float64Array; dry: Float64Array[] } {
  const stage = new ProtoOutputStage();
  const out = new Float64Array(n);
  const dry = wells.map(() => new Float64Array(n));
  for (let t = 0; t < n; t++) {
    let x = 0;
    wells.forEach((w, k) => {
      const from = w.from ?? 0;
      if (t < from || t >= (w.to ?? Infinity)) return;
      const y = w.net.step(w.drive(t - from)) * OUTPUT_GAIN;
      dry[k][t] = y;
      x += y;
    });
    out[t] = stage.step(x);
  }
  return { out, dry };
}

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>, n = Math.min(a.length, b.length)): number {
  let m = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (!(d <= m)) m = d; // NaN counts
  }
  return m;
}
const peakOf = (a: ArrayLike<number>, from = 0, to = a.length): number => {
  let m = 0;
  for (let i = from; i < to; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]);
  return m;
};
const rmsOf = (a: ArrayLike<number>, from = 0, to = a.length): number => {
  let s = 0;
  for (let i = from; i < to; i++) s += a[i] * a[i];
  return Math.sqrt(s / (to - from));
};
const held = () => 1;
/** Float32 rounding of a signal of size about 1. */
const F32 = 2e-7;

describe("test helpers", () => {
  it("use the constants of the app", () => {
    expect(tw.N).toBe(N);
    expect(tw.CLOCKS).toEqual([...CLOCKS]);
    expect(SAMPLE_RATE).toBe(16000);
    expect(OUTPUT_GAIN).toBe(0.25);
    expect(CEILING).toBe(0.89);
  });
});

// ───────────────────────────── (a) Network against the prototype ─────────────────────────────

describe("Network", () => {
  for (let L = 1; L <= 8; L++) {
    it(`agrees with the prototype, ${L} layer${L > 1 ? "s" : ""}, ${LONG} samples`, () => {
      const flat = flatWeights(L, 1000 + L);
      const proto = new ClockNet(L, flat, Float32Array);
      const net = new Network(toLayers(L, flat));
      expect(net.L).toBe(L);
      expect(net.h.length).toBe(L);
      let worst = 0;
      let worstState = 0;
      let energy = 0;
      let ticks = 0;
      for (let t = 0; t < LONG; t++) {
        // a short press, then free running, then a second press: both kinds of input
        const drive = t < 2000 || (t >= 40000 && t < 41000) ? 1 : 0;
        const a = proto.step(drive);
        const b = net.step(drive);
        worst = Math.max(worst, Math.abs(a - b));
        energy += b * b;
        ticks |= net.ticked;
        let expectTicked = 0;
        for (let i = 1; i < L; i++) if (t > 0 && t % CLOCKS[i] === 0) expectTicked |= 1 << i;
        if (net.ticked !== expectTicked) throw new Error(`ticked ${net.ticked} at ${t}, expected ${expectTicked}`);
        if (t % 1009 === 0 || t === LONG - 1) {
          for (let i = 0; i < L; i++) worstState = Math.max(worstState, maxDiff(net.h[i], proto.h[i]));
        }
      }
      expect(worst).toBeLessThan(1e-12);
      expect(worstState).toBeLessThan(1e-12);
      expect(Math.sqrt(energy / LONG)).toBeGreaterThan(0.01); // the comparison is not of silence
      expect(ticks).toBe((1 << L) - 2); // every slow layer updated at least once
    });
  }

  it("shares the weight arrays of the row instead of copying them", () => {
    const flat = flatWeights(3, 1003);
    const layers = toLayers(3, flat);
    const untouched = toLayers(3, flat);
    const a = new Network(layers);
    const b = new Network(layers.slice(0, 1)); // the one-layer well of the same row
    const a0 = new Network(untouched);
    const b0 = new Network(untouched.slice(0, 1));
    for (let t = 0; t < 50; t++) {
      expect(a.step(1)).toBe(a0.step(1));
      expect(b.step(1)).toBe(b0.step(1));
    }
    // Changing the row's arrays changes both wells built on them.
    layers[0].W![0] += 0.5;
    expect(a.step(1)).not.toBe(a0.step(1));
    expect(b.step(1)).not.toBe(b0.step(1));
  });

  it("reset() returns it to rest", () => {
    const layers = toLayers(4, flatWeights(4, 1004));
    const net = new Network(layers);
    const first = Array.from({ length: 9000 }, (_, t) => net.step(t < 100 ? 1 : 0));
    net.reset();
    for (const h of net.h) expect(Array.from(h).every((v) => v === 0)).toBe(true);
    const again = Array.from({ length: 9000 }, (_, t) => net.step(t < 100 ? 1 : 0));
    expect(again).toEqual(first);
  });

  it("stays at rest without input", () => {
    const net = new Network(toLayers(3, flatWeights(3, 1003)));
    for (let t = 0; t < 1000; t++) expect(net.step(0)).toBe(0);
  });

  it("rejects layers that are missing, out of order or of the wrong size", () => {
    const layers = toLayers(3, flatWeights(3, 1003));
    expect(() => new Network([])).toThrow();
    expect(() => new Network([layers[0], layers[2]])).toThrow();
    expect(() => new Network([{ ...layers[0], W: null }])).toThrow();
    expect(() => new Network([layers[0], { ...layers[1], K: new Float32Array(8) }])).toThrow();
  });
});

// ───────────────────────────── (b) stored reference output ─────────────────────────────

describe("stored reference output", () => {
  // Same machine and same JavaScript engine: exact. The tolerance is there for a Math.tanh that
  // rounds differently; such a difference is amplified by wells that keep changing, so on another
  // JavaScript engine the later checkpoints of those wells may legitimately fail.
  const TOL = 1e-9;
  const sample = (y: ArrayLike<number>): { head: number[]; checkpoints: number[] } => {
    const head = Array.from({ length: fixture.head }, (_, t) => y[t]);
    const checkpoints: number[] = [];
    for (let t = fixture.head; t < fixture.samples; t += fixture.every) checkpoints.push(y[t]);
    checkpoints.push(y[fixture.samples - 1]);
    return { head, checkpoints };
  };

  it("covers 1 to 8 layers, pulse and hold", () => {
    expect(fixture.networks.length).toBe(16);
    expect(new Set(fixture.networks.map((c) => c.layers)).size).toBe(8);
    expect(fixture.samples).toBeGreaterThan(65536);
  });

  for (const c of fixture.networks) {
    it(`Network, ${c.layers} layers, input held for ${c.hold} sample${c.hold > 1 ? "s" : ""}`, () => {
      const net = new Network(toLayers(c.layers, flatWeights(c.layers, c.seed)));
      const y = new Float64Array(fixture.samples);
      for (let t = 0; t < y.length; t++) y[t] = net.step(t < c.hold ? 1 : 0);
      const got = sample(y);
      expect(got.checkpoints.length).toBe(c.checkpoints.length);
      expect(maxDiff(got.head, c.head)).toBeLessThan(TOL);
      expect(maxDiff(got.checkpoints, c.checkpoints)).toBeLessThan(TOL);
      expect(rmsOf(c.checkpoints)).toBeGreaterThan(0.01);
    });
  }

  it("Engine: three wells, summed, gain, high-pass, limiter", () => {
    const ch = fixture.chain;
    const engine = new Engine({ outputRate: SAMPLE_RATE });
    ch.rowSeeds.forEach((seed, r) => {
      for (const w of toLayers(8, flatWeights(8, seed), r)) engine.handle({ type: "weights", weights: w });
    });
    const out = new Float32Array(fixture.samples);
    for (const w of ch.wells) engine.handle({ type: "press", well: wellIndex(w.row, w.layers - 1) });
    engine.process(out.subarray(0, ch.hold));
    for (const w of ch.wells) engine.handle({ type: "release", well: wellIndex(w.row, w.layers - 1) });
    for (let o = ch.hold; o < out.length; o += 128) engine.process(out.subarray(o, Math.min(out.length, o + 128)));
    const got = sample(out);
    expect(maxDiff(got.head, ch.head)).toBeLessThan(F32);
    expect(maxDiff(got.checkpoints, ch.checkpoints)).toBeLessThan(F32);
    // the limiter was at work in the reference, so this covers it
    expect(ch.peakBeforeLimiter).toBeGreaterThan(CEILING);
    expect(ch.lowestGain).toBeLessThan(0.95);
  });
});

// ───────────────────────────── (c) OutputStage ─────────────────────────────

describe("OutputStage", () => {
  /** A stand-in for summed wells: eight partials with slow beats plus a little noise, peak 1. */
  function mixture(n: number): Float64Array {
    const rnd = lcg(99);
    const x = new Float64Array(n);
    const f = [110, 173, 262, 415, 660, 1049, 1667, 2650];
    for (let t = 0; t < n; t++) {
      let s = 0.1 * rnd();
      for (let k = 0; k < f.length; k++) {
        s += Math.sin((2 * Math.PI * f[k] * t) / 16000 + k) * (0.6 + 0.4 * Math.sin((2 * Math.PI * (0.3 + 0.17 * k) * t) / 16000));
      }
      x[t] = s;
    }
    let p = 0; // the largest sample becomes +1
    for (let t = 0; t < n; t++) if (Math.abs(x[t]) > Math.abs(p)) p = x[t];
    for (let t = 0; t < n; t++) x[t] /= p;
    return x;
  }
  const n = 3 * 16000;
  const mix = mixture(n);
  /** The high-pass alone, as in the prototype. */
  function highpass(x: Float64Array): Float64Array {
    const R = 1 - (2 * Math.PI * 20) / 16000;
    const y = new Float64Array(x.length);
    let x1 = 0;
    let y1 = 0;
    for (let t = 0; t < x.length; t++) {
      y1 = x[t] - x1 + R * y1;
      x1 = x[t];
      y[t] = y1;
    }
    return y;
  }
  function run(x: Float64Array, stage = new OutputStage()): { y: Float64Array; stage: OutputStage } {
    const y = new Float64Array(x.length);
    for (let t = 0; t < x.length; t++) y[t] = stage.step(x[t]);
    return { y, stage };
  }

  it("has a 5 ms look-ahead at 16 kHz", () => {
    expect(new OutputStage().look).toBe(80);
    expect(new OutputStage(48000).look).toBe(240);
  });

  it("row 1: a sum far too loud (peak 4.10) comes out at the ceiling, not above", () => {
    const x = mix.map((v) => v * 3.6 + 0.5);
    expect(peakOf(x)).toBeCloseTo(4.1, 2);
    const { y, stage } = run(x);
    expect(peakOf(y)).toBeLessThanOrEqual(CEILING);
    expect(peakOf(y)).toBeGreaterThan(CEILING - 0.01);
    expect(stage.minGain).toBeLessThan(0.3);
  });

  it("row 2: a sum slightly too loud (peak 0.96) comes out at the ceiling, not above", () => {
    const x = mix.map((v) => v * 0.96);
    const hp = highpass(x);
    expect(peakOf(hp)).toBeGreaterThan(CEILING);
    const { y, stage } = run(x);
    expect(peakOf(y)).toBeLessThanOrEqual(CEILING);
    expect(peakOf(y)).toBeGreaterThan(CEILING - 0.01);
    expect(stage.minGain).toBeGreaterThan(0.85);
    expect(stage.minGain).toBeLessThan(1);
  });

  it("row 3: one well (peak 0.25) passes with no gain change: only the high-pass and 5 ms of delay", () => {
    const x = mix.map((v) => v * 0.25);
    const hp = highpass(x);
    const { y, stage } = run(x);
    expect(stage.minGain).toBe(1);
    for (let t = 0; t < 80; t++) expect(y[t]).toBe(0);
    let worst = 0;
    for (let t = 80; t < n; t++) worst = Math.max(worst, Math.abs(y[t] - hp[t - 80]));
    expect(worst).toBe(0);
    // and the high-pass leaves the level of the audible band alone
    expect(rmsOf(y, 16000) / rmsOf(x, 16000)).toBeGreaterThan(0.99);
    expect(rmsOf(y, 16000) / rmsOf(x, 16000)).toBeLessThan(1.01);
  });

  it("row 4: a constant 2.0 is 0.89 at the start, then 0", () => {
    const x = new Float64Array(n).fill(2);
    const { y } = run(x);
    expect(peakOf(y, 0, 1600)).toBeLessThanOrEqual(CEILING);
    expect(peakOf(y, 0, 1600)).toBeGreaterThan(CEILING - 0.01);
    expect(peakOf(y, 16000)).toBeLessThan(1e-12);
    expect(Math.abs(y[n - 1])).toBeLessThan(1e-100);
  });

  it("is the prototype's output stage, sample for sample", () => {
    for (const [gain, dc] of [[3.6, 0.5], [0.96, 0], [0.25, 0], [0, 2]]) {
      const proto = new ProtoOutputStage();
      const stage = new OutputStage();
      let worst = 0;
      for (let t = 0; t < n; t++) {
        const x = mix[t] * gain + dc;
        worst = Math.max(worst, Math.abs(stage.step(x) - proto.step(x)));
      }
      // identical arithmetic; only the final clamp may remove a rounding error above the ceiling
      expect(worst).toBeLessThan(1e-15);
    }
  });

  it("never passes the ceiling, whatever comes in", () => {
    for (const fs of [16000, 44100, 48000]) {
      const rnd = lcg(7 + fs);
      const stage = new OutputStage(fs);
      let peak = 0;
      let amp = 1;
      for (let t = 0; t < 200000; t++) {
        if (t % 997 === 0) amp = Math.pow(10, 4 * rnd()); // 0.01 .. 100, jumping without warning
        const x = t % 20011 < 3 ? 1000 : amp * 2 * rnd() + (t % 50000 < 25000 ? 5 : -5);
        peak = Math.max(peak, Math.abs(stage.step(x)));
      }
      expect(peak).toBeLessThanOrEqual(CEILING);
      expect(peak).toBeGreaterThan(CEILING - 0.01);
    }
  });

  it("recovers its gain after a peak, over about 100 ms", () => {
    const stage = new OutputStage();
    for (let t = 0; t < 4000; t++) stage.step(t % 2 ? 4 : -4); // gain needed: about 0.11
    const quiet = (t: number): number => (t % 2 ? 0.1 : -0.1);
    const amplitude = 0.2 / (2 - (2 * Math.PI * 20) / 16000); // of the quiet square wave after the high-pass
    const gainAt = (samples: number): number => {
      let g = 0;
      for (let t = 0; t < samples; t++) g = Math.abs(stage.step(quiet(t))) / amplitude;
      return g;
    };
    const after20ms = gainAt(320 + 80);
    const after500ms = gainAt(8000);
    expect(after20ms).toBeLessThan(0.5);
    expect(after500ms).toBeGreaterThan(0.98);
    expect(after500ms).toBeLessThanOrEqual(1.0001);
  });

  it("reset() returns it to its first state", () => {
    const stage = new OutputStage();
    const a = Array.from({ length: 2000 }, (_, t) => stage.step(mix[t] * 3));
    stage.reset();
    const b = Array.from({ length: 2000 }, (_, t) => stage.step(mix[t] * 3));
    expect(b).toEqual(a);
  });
});

// ───────────────────────────── (d) Engine ─────────────────────────────

describe("Engine: messages", () => {
  const A1 = wellIndex(0, 0);
  const A2 = wellIndex(0, 1);
  const A3 = wellIndex(0, 2);
  const A4 = wellIndex(0, 3);
  const B2 = wellIndex(1, 1);
  const B3 = wellIndex(1, 2);

  it("is silent at rest and reports no wells", () => {
    const rig = new Rig();
    load(rig.engine, 0, 4);
    rig.run(3200);
    expect(peakOf(rig.out)).toBe(0);
    expect(rig.reports.length).toBe(6);
    for (const r of rig.reports) expect(r).toMatchObject({ type: "report", wells: [], gain: 1, peak: 0 });
  });

  it("press starts a well from rest with input 1.0 on every sample until release", () => {
    const rig = new Rig();
    load(rig.engine, 0, 3);
    rig.send({ type: "press", well: A3 }).run(1920);
    expect(rig.last.wells.map((w) => [w.well, w.held])).toEqual([[A3, true]]);
    rig.send({ type: "release", well: A3 }).run(8000 - 1920);
    expect(rig.last.wells.map((w) => [w.well, w.held])).toEqual([[A3, false]]);
    const ref = reference([{ net: protoWell(0, 3), drive: (t) => (t < 1920 ? 1 : 0) }], 8000);
    expect(maxDiff(rig.out, ref.out)).toBeLessThan(F32);
    expect(rmsOf(rig.out, 4000)).toBeGreaterThan(0.01); // it keeps sounding after release
  });

  it("a held input is not the same as a pulse (so 'every sample' is tested)", () => {
    const a = reference([{ net: protoWell(0, 3), drive: (t) => (t < 1920 ? 1 : 0) }], 3000);
    const b = reference([{ net: protoWell(0, 3), drive: (t) => (t < 1 ? 1 : 0) }], 3000);
    expect(maxDiff(a.out, b.out)).toBeGreaterThan(0.01);
  });

  it("press on a sounding well only turns the input on again; the well does not restart", () => {
    const rig = new Rig();
    load(rig.engine, 0, 3);
    rig.send({ type: "press", well: A3 }).run(1280);
    rig.send({ type: "release", well: A3 }).run(1280);
    rig.send({ type: "press", well: A3 }).run(1280);
    expect(rig.last.wells[0].held).toBe(true);
    rig.send({ type: "release", well: A3 }).run(1280);
    const ref = reference([{ net: protoWell(0, 3), drive: (t) => (t < 1280 || (t >= 2560 && t < 3840) ? 1 : 0) }], 5120);
    expect(maxDiff(rig.out, ref.out)).toBeLessThan(F32);
    const restarted = reference(
      [
        { net: protoWell(0, 3), drive: (t) => (t < 1280 ? 1 : 0), to: 2560 },
        { net: protoWell(0, 3), drive: (t) => (t < 1280 ? 1 : 0), from: 2560 },
      ],
      5120,
    );
    expect(maxDiff(rig.out, restarted.out)).toBeGreaterThan(0.01);
  });

  it("stop silences the well and returns it to rest; the next press starts from zero state", () => {
    const rig = new Rig();
    load(rig.engine, 0, 3);
    rig.send({ type: "press", well: A3 }).run(1280).send({ type: "release", well: A3 }).run(1280);
    rig.send({ type: "stop", well: A3 }).run(3840);
    expect(rig.last.wells).toEqual([]);
    // 5 ms of look-ahead still to come out, then only the tail of the 20 Hz high-pass
    const out = rig.out;
    expect(peakOf(out, 2560 + 80 + 1600)).toBeLessThan(1e-3);
    rig.send({ type: "press", well: A3 }).run(1280).send({ type: "release", well: A3 }).run(1280);
    const ref = reference(
      [
        { net: protoWell(0, 3), drive: (t) => (t < 1280 ? 1 : 0), to: 2560 },
        { net: protoWell(0, 3), drive: (t) => (t < 1280 ? 1 : 0), from: 6400 },
      ],
      8960,
    );
    expect(maxDiff(rig.out, ref.out)).toBeLessThan(F32);
    // release and stop of wells that are not sounding do nothing
    rig.send({ type: "release", well: A1 }, { type: "stop", well: A1 }, { type: "stop", well: 999 });
  });

  it("several wells sound at once and the output is their sum", () => {
    const rig = new Rig();
    load(rig.engine, 0, 4);
    load(rig.engine, 1, 3);
    rig.send({ type: "press", well: A2 }).run(640);
    rig.send({ type: "press", well: B3 }, { type: "release", well: A2 }).run(640);
    rig.send({ type: "press", well: A4 }).run(1280);
    rig.send({ type: "release", well: B3 }, { type: "release", well: A4 }).run(8000 - 2560);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2, B3, A4]); // in the order they started
    const ref = reference(
      [
        { net: protoWell(0, 2), drive: (t) => (t < 640 ? 1 : 0) },
        { net: protoWell(1, 3), drive: (t) => (t < 1920 ? 1 : 0), from: 640 },
        { net: protoWell(0, 4), drive: (t) => (t < 1280 ? 1 : 0), from: 1280 },
      ],
      8000,
    );
    expect(maxDiff(rig.out, ref.out)).toBeLessThan(F32);
    // each well's own waveform in the report is that well alone
    const end = rig.last.time;
    expect(end).toBeLessThanOrEqual(8000);
    rig.last.wells.forEach((w, k) => {
      expect(maxDiff(w.wave, ref.dry[k].subarray(end - WAVE_LEN, end))).toBeLessThan(F32);
    });
  });

  it("wells of a row share that row's weights: A1..A8 are prefixes of the same eight layers", () => {
    const rig = new Rig();
    load(rig.engine, 0, 8);
    const wells = [0, 1, 2, 3, 4, 5, 6, 7].map((col) => wellIndex(0, col));
    for (const well of wells) rig.send({ type: "press", well });
    rig.run(640);
    for (const well of wells) rig.send({ type: "release", well });
    rig.run(9600 - 640);
    const ref = reference(wells.map((_, col) => ({ net: protoWell(0, col + 1), drive: (t: number) => (t < 640 ? 1 : 0) })), 9600);
    expect(maxDiff(rig.out, ref.out)).toBeLessThan(F32);
    // and the engine keeps the arrays it was sent rather than copying them
    const engine = new Engine({ outputRate: SAMPLE_RATE });
    const mine = cloneLayer(rowWeights(0)[0]);
    engine.handle({ type: "weights", weights: mine });
    mine.W!.fill(0); // an output layer of zeros: silence
    engine.handle({ type: "press", well: A1 });
    const out = new Float32Array(2000);
    engine.process(out);
    expect(peakOf(out)).toBe(0);
    expect(engine.takeReport()!.wells[0].level).toBe(0);
  });

  it("ignores a press while the well's layers are not all loaded", () => {
    const rig = new Rig();
    load(rig.engine, 0, 2); // layers 0 and 1 only
    rig.send({ type: "press", well: A3 }, { type: "press", well: B2 }).run(1600);
    expect(peakOf(rig.out)).toBe(0);
    expect(rig.last.wells).toEqual([]);
    rig.send({ type: "press", well: A2 }).run(1600); // A2 needs only what is loaded
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2]);
    // the ignored press is not remembered: loading the layer later does not start the well
    rig.send({ type: "weights", weights: cloneLayer(rowWeights(0)[2]) }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2]);
    rig.send({ type: "press", well: A3 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2, A3]);
  });

  it("ignores layers that arrive out of order until the gap is filled", () => {
    const rig = new Rig();
    rig.send({ type: "weights", weights: cloneLayer(rowWeights(0)[1]) }); // layer 1 before layer 0
    rig.send({ type: "press", well: A1 }, { type: "press", well: A2 }).run(640);
    expect(rig.last.wells).toEqual([]);
    rig.send({ type: "weights", weights: cloneLayer(rowWeights(0)[0]) }, { type: "press", well: A2 }).run(640);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2]);
  });

  it("stopAll stops every well; the weights stay", () => {
    const rig = new Rig();
    load(rig.engine, 0, 3);
    load(rig.engine, 1, 3);
    rig.send({ type: "press", well: A3 }, { type: "press", well: B2 }).run(1600);
    expect(rig.last.wells.length).toBe(2);
    rig.send({ type: "stopAll" }).run(8000);
    expect(rig.last.wells).toEqual([]);
    expect(peakOf(rig.out, 1600 + 80 + 1600)).toBeLessThan(1e-3);
    rig.send({ type: "press", well: B2 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([B2]);
  });

  it("clear stops every well and drops all weights", () => {
    const rig = new Rig();
    load(rig.engine, 0, 3);
    rig.send({ type: "press", well: A3 }).run(1600);
    rig.send({ type: "clear" }).run(1600);
    expect(rig.last.wells).toEqual([]);
    rig.send({ type: "press", well: A3 }, { type: "press", well: A1 }).run(1600);
    expect(rig.last.wells).toEqual([]);
    // a new plate: other weights in the same row
    const other = toLayers(3, flatWeights(3, 31337), 0);
    for (const w of other) rig.send({ type: "weights", weights: w });
    const before = rig.out.length;
    rig.send({ type: "press", well: A3 }).run(3200);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A3]);
    const ref = reference([{ net: new ClockNet(3, flatWeights(3, 31337), Float32Array), drive: held }], 3200);
    // the high-pass still carries a faint tail of the plate before; the well itself is the new one
    expect(maxDiff(rig.last.wells[0].wave, ref.dry[0].subarray(rig.last.time - before - WAVE_LEN, rig.last.time - before))).toBeLessThan(F32);
    expect(maxDiff(rig.out.subarray(before), ref.out)).toBeLessThan(1e-3);
  });

  it("replacing a loaded layer stops the wells that were built on it", () => {
    const rig = new Rig();
    load(rig.engine, 0, 3);
    rig.send({ type: "press", well: A1 }, { type: "press", well: A3 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A1, A3]);
    rig.send({ type: "weights", weights: toLayers(3, flatWeights(3, 5), 0)[2] }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A1]); // A1 does not use layer 2
    rig.send({ type: "press", well: A3 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A1, A3]);
  });

  it("maxVoices: over the cap, the well that started first stops", () => {
    const rig = new Rig();
    load(rig.engine, 0, 4);
    rig.send({ type: "config", maxVoices: 2 });
    rig.send({ type: "press", well: A1 }, { type: "press", well: A2 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A1, A2]);
    rig.send({ type: "press", well: A3 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2, A3]);
    rig.send({ type: "press", well: A2 }).run(1600); // already sounding: nothing is stopped
    expect(rig.last.wells.map((w) => w.well)).toEqual([A2, A3]);
    rig.send({ type: "press", well: A1 }).run(1600);
    expect(rig.last.wells.map((w) => w.well)).toEqual([A3, A1]);
    rig.send({ type: "config", maxVoices: 1 }).run(1600); // lowering the cap applies at once
    expect(rig.last.wells.map((w) => w.well)).toEqual([A1]);
    rig.send({ type: "config", maxVoices: 48 }, { type: "press", well: A2 }, { type: "press", well: A3 }, { type: "press", well: A4 }).run(1600);
    expect(rig.last.wells.length).toBe(4);
  });

  it("ignores malformed messages without throwing", () => {
    const rig = new Rig();
    load(rig.engine, 0, 2);
    const bad = [
      { type: "press", well: -1 },
      { type: "press", well: 48 },
      { type: "press", well: 1.5 },
      { type: "press" },
      { type: "weights", weights: { ...cloneLayer(rowWeights(0)[2]), K: new Float32Array(5) } },
      { type: "weights", weights: { ...cloneLayer(rowWeights(0)[2]), row: 9 } },
      { type: "weights", weights: { ...cloneLayer(rowWeights(0)[2]), C: null } },
      { type: "weights" },
      { type: "config", maxVoices: -3, reportHz: 0 },
      { type: "config", reportHz: Number.NaN },
      { type: "nonsense" },
    ] as unknown as ToEngine[];
    for (const m of bad) expect(() => rig.engine.handle(m)).not.toThrow();
    rig.send({ type: "press", well: A3 }).run(1600); // layer 2 was never accepted
    expect(rig.last.wells).toEqual([]);
    expect(rig.reports.length).toBe(3); // the report rate was not changed either
  });
});

describe("Engine: reports", () => {
  it(`come ${REPORT_HZ} times per second of engine time, whatever the block size`, () => {
    for (const block of [128, 100, 16000, 7]) {
      const rig = new Rig(SAMPLE_RATE, block);
      rig.run(2 * SAMPLE_RATE);
      expect(rig.reports.length).toBe(2 * REPORT_HZ);
      const times = rig.reports.map((r) => r.time);
      expect(times[REPORT_HZ - 1]).toBe(SAMPLE_RATE);
      expect(times[2 * REPORT_HZ - 1]).toBe(2 * SAMPLE_RATE);
      for (let k = 1; k < times.length; k++) {
        expect([533, 534]).toContain(times[k] - times[k - 1]);
      }
    }
  });

  it("takeReport gives each report once, then null", () => {
    const engine = new Engine({ outputRate: SAMPLE_RATE });
    expect(engine.takeReport()).toBeNull();
    engine.process(new Float32Array(533));
    expect(engine.takeReport()).toBeNull();
    engine.process(new Float32Array(1));
    expect(engine.takeReport()!.time).toBe(534);
    expect(engine.takeReport()).toBeNull();
    // reports nobody takes are bounded: the oldest are dropped
    engine.process(new Float32Array(100 * 16000));
    let count = 0;
    let lastTime = 0;
    for (let r = engine.takeReport(); r; r = engine.takeReport()) {
      count++;
      lastTime = r.time;
    }
    expect(count).toBe(64);
    expect(lastTime).toBeGreaterThan(100 * 16000 - 534);
  });

  it("config reportHz changes the rate", () => {
    const rig = new Rig();
    rig.send({ type: "config", reportHz: 10 }).run(SAMPLE_RATE);
    expect(rig.reports.map((r) => r.time)).toEqual([1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 14400, 16000]);
    // with 10 reports a second, layer 2 (62.5 updates a second) is still averaged and layer 3 (3.9) is not;
    // with 2 a second, layer 3 is averaged too
    const slow = new Rig();
    load(slow.engine, 0, 4);
    slow.send({ type: "config", reportHz: 2 }, { type: "press", well: wellIndex(0, 3) }).run(16000);
    expect(slow.reports.map((r) => r.time)).toEqual([8000, 16000]);
    const proto = protoWell(0, 4);
    const sum = new Float64Array(N);
    let count = 0;
    for (let t = 0; t < 16000; t++) {
      proto.step(1);
      if (t >= 8000 && t % 4096 === 0) {
        for (let u = 0; u < N; u++) sum[u] += proto.h[3][u];
        count++;
      }
    }
    expect(count).toBe(2); // layer 3 updated at 8,192 and 12,288
    expect(maxDiff(slow.last.wells[0].units.subarray(3 * N), sum.map((v) => v / count))).toBeLessThan(F32);
  });

  it("carry level, waveform, unit values, ticks and held, in the shapes of the contract", () => {
    const L = 8;
    const well = wellIndex(0, L - 1);
    const rig = new Rig();
    load(rig.engine, 0, 8);
    const HOLD = 4096 + 640;
    rig.send({ type: "press", well }).run(HOLD).send({ type: "release", well }).run(LONG - HOLD);

    // The same performance on the prototype, gathering what the reports should say.
    const proto = protoWell(0, L);
    const fast = [0, 1, 2]; // 16,000, 1,000 and 62.5 updates a second: faster than 30 reports a second
    let r = 0;
    let sumSq = 0;
    let count = 0;
    let ticks = 0;
    let everTicked = 0;
    const sums = fast.map(() => new Float64Array(N));
    const counts = fast.map(() => 0);
    const dry = new Float32Array(LONG);
    let worstUnits = 0;
    let worstWave = 0;
    let worstLevel = 0;
    for (let t = 0; t < LONG; t++) {
      const y = proto.step(t < HOLD ? 1 : 0) * OUTPUT_GAIN;
      dry[t] = y;
      sumSq += y * y;
      count++;
      for (const i of fast) {
        if (t % CLOCKS[i] === 0 && (i === 0 || t > 0)) {
          for (let u = 0; u < N; u++) sums[i][u] += proto.h[i][u];
          counts[i]++;
        }
      }
      for (let i = TICK_FROM_LAYER; i < L; i++) if (t > 0 && t % CLOCKS[i] === 0) ticks |= 1 << i;
      const report = rig.reports[r];
      if (report && report.time === t + 1) {
        expect(report.type).toBe("report");
        expect(report.wells.length).toBe(1);
        const w = report.wells[0];
        expect(w.well).toBe(well);
        expect(w.held).toBe(t + 1 <= HOLD);
        expect(w.wave).toBeInstanceOf(Float32Array);
        expect(w.wave.length).toBe(WAVE_LEN);
        expect(w.units).toBeInstanceOf(Float32Array);
        expect(w.units.length).toBe(L * N);
        expect(w.ticks).toBe(ticks);
        everTicked |= w.ticks;
        worstLevel = Math.max(worstLevel, Math.abs(w.level - Math.sqrt(sumSq / count)));
        worstWave = Math.max(worstWave, maxDiff(w.wave, dry.subarray(t + 1 - WAVE_LEN, t + 1)));
        for (let i = 0; i < L; i++) {
          for (let u = 0; u < N; u++) {
            const k = fast.indexOf(i);
            const want = k >= 0 && counts[k] > 0 ? sums[k][u] / counts[k] : proto.h[i][u];
            worstUnits = Math.max(worstUnits, Math.abs(w.units[i * N + u] - want));
            expect(Math.abs(w.units[i * N + u])).toBeLessThanOrEqual(1);
          }
        }
        expect(report.peak).toBeLessThanOrEqual(CEILING);
        expect(report.gain).toBeGreaterThan(0);
        expect(report.gain).toBeLessThanOrEqual(1);
        sumSq = 0;
        count = 0;
        ticks = 0;
        sums.forEach((s) => s.fill(0));
        counts.fill(0);
        r++;
      }
    }
    expect(r).toBe(rig.reports.length);
    expect(r).toBe(Math.floor((LONG * REPORT_HZ) / SAMPLE_RATE));
    expect(worstLevel).toBeLessThan(1e-9);
    expect(worstWave).toBeLessThan(F32);
    expect(worstUnits).toBeLessThan(F32);
    // layers 4..7 all updated within the run, and nothing below TICK_FROM_LAYER is ever flagged
    expect(everTicked).toBe(0b11110000);
    // the mean of a fast layer is not just its latest state (the unit display is really averaged)
    const lastReport = rig.last.wells[0];
    let differs = 0;
    for (let u = 0; u < N; u++) differs = Math.max(differs, Math.abs(lastReport.units[u] - proto.h[0][u]));
    expect(differs).toBeGreaterThan(0.01);
  });

  it("tick bits follow the clocks: layer 4 every 8,192 samples, layer 7 at 65,536", () => {
    const rig = new Rig();
    load(rig.engine, 0, 8);
    rig.send({ type: "press", well: wellIndex(0, 7) }, { type: "press", well: wellIndex(0, 4) }).run(LONG);
    const seen: Record<number, number[]> = { 4: [], 5: [], 6: [], 7: [] };
    let prev = 0;
    for (const report of rig.reports) {
      const w8 = report.wells.find((w) => w.well === wellIndex(0, 7))!;
      const w5 = report.wells.find((w) => w.well === wellIndex(0, 4))!;
      for (let i = 4; i < 8; i++) {
        if (w8.ticks & (1 << i)) {
          // the update happened at a multiple of the clock inside this report's interval
          const at = Math.floor((report.time - 1) / CLOCKS[i]) * CLOCKS[i];
          expect(at).toBeGreaterThanOrEqual(prev);
          expect(at).toBeGreaterThan(0);
          seen[i].push(at);
        }
      }
      expect(w5.ticks).toBe(w8.ticks & 0b10000); // a five-layer well has only layer 4 to flag
      prev = report.time;
    }
    expect(seen[4]).toEqual([8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536]);
    expect(seen[5]).toEqual([16384, 32768, 49152, 65536]);
    expect(seen[6]).toEqual([32768, 65536]);
    expect(seen[7]).toEqual([65536]);
  });

  it("a well that has just started reports its waveform with silence in front", () => {
    const rig = new Rig();
    load(rig.engine, 0, 1);
    rig.run(400).send({ type: "press", well: 0 }).run(134);
    expect(rig.last.time).toBe(534);
    const w = rig.last.wells[0];
    const ref = reference([{ net: protoWell(0, 1), drive: held }], 134);
    expect(Array.from(w.wave.subarray(0, WAVE_LEN - 134)).every((v) => v === 0)).toBe(true);
    expect(maxDiff(w.wave.subarray(WAVE_LEN - 134), ref.dry[0])).toBeLessThan(F32);
    expect(w.level).toBeCloseTo(rmsOf(ref.dry[0]), 9);
  });

  it("gain and peak describe the limiter and the final output since the last report", () => {
    const rig = new Rig();
    load(rig.engine, 0, 8);
    load(rig.engine, 1, 8);
    for (let col = 0; col < 8; col++) rig.send({ type: "press", well: wellIndex(0, col) }, { type: "press", well: wellIndex(1, col) });
    rig.run(16000);
    const out = rig.out;
    let prev = 0;
    let lowest = 1;
    for (const report of rig.reports) {
      expect(report.peak).toBeCloseTo(peakOf(out, prev, report.time), 6);
      lowest = Math.min(lowest, report.gain);
      prev = report.time;
    }
    expect(lowest).toBeLessThan(0.7); // sixteen wells: the limiter is working
  });
});

describe("Engine: output", () => {
  it("never exceeds the ceiling, with every well of three rows sounding", () => {
    const rig = new Rig();
    for (let row = 0; row < 3; row++) {
      load(rig.engine, row, 8);
      for (let col = 0; col < 8; col++) rig.send({ type: "press", well: wellIndex(row, col) });
    }
    rig.run(8000);
    for (let row = 0; row < 3; row++) for (let col = 0; col < 8; col++) rig.send({ type: "release", well: wellIndex(row, col) });
    rig.run(24000);
    rig.send({ type: "stopAll" }).run(1600); // the cut at a stop is a step: it must be held too
    for (let col = 0; col < 8; col++) rig.send({ type: "press", well: wellIndex(0, col) });
    rig.run(8000);
    const out = rig.out;
    expect(peakOf(out)).toBeLessThanOrEqual(CEILING);
    expect(peakOf(out)).toBeGreaterThan(CEILING - 0.01);
    expect(Math.min(...rig.reports.map((r) => r.gain))).toBeLessThan(0.5);
    expect(Math.max(...rig.reports.map((r) => r.peak))).toBeLessThanOrEqual(CEILING);
    expect(out.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("survives weights that are not numbers: silence, not NaN, and other wells go on", () => {
    const rig = new Rig();
    load(rig.engine, 0, 2);
    const broken = toLayers(1, flatWeights(1, 3), 1);
    broken[0].W!.fill(Number.NaN);
    rig.send({ type: "weights", weights: broken[0] }, { type: "press", well: wellIndex(1, 0) }).run(1600);
    expect(peakOf(rig.out)).toBe(0);
    rig.send({ type: "stop", well: wellIndex(1, 0) }, { type: "press", well: wellIndex(0, 1) }).run(1600);
    expect(rig.out.every((v) => Number.isFinite(v))).toBe(true);
    expect(peakOf(rig.out, 1600)).toBeGreaterThan(0.01);
  });

  for (const [rate, block16, blockOut] of [[48000, 128, 384], [44100, 160, 441], [22050, 320, 441], [8000, 128, 64]]) {
    it(`at ${rate} Hz the networks still run at 16 kHz: the output is the 16 kHz signal, interpolated`, () => {
      const n16 = 16000;
      const blocks = n16 / block16;
      const native = new Rig(SAMPLE_RATE, block16);
      const other = new Rig(rate, blockOut);
      expect(other.engine.outputRate).toBe(rate);
      for (const rig of [native, other]) {
        load(rig.engine, 0, 4);
        load(rig.engine, 1, 3);
      }
      // the same messages at the same engine times (blocks of both cover the same stretch of time)
      const script: [number, ToEngine][] = [
        [0, { type: "press", well: wellIndex(0, 3) }],
        [10, { type: "release", well: wellIndex(0, 3) }],
        [10, { type: "press", well: wellIndex(1, 2) }],
        [30, { type: "release", well: wellIndex(1, 2) }],
        [70, { type: "stop", well: wellIndex(0, 3) }],
      ];
      for (let b = 0; b < blocks; b++) {
        for (const [at, msg] of script) {
          if (at !== b) continue;
          native.send(msg);
          other.send(msg);
        }
        native.run(block16);
        other.run(blockOut);
      }
      const e = native.out;
      const got = other.out;
      expect(got.length).toBe((n16 * rate) / SAMPLE_RATE);
      expect(rmsOf(e)).toBeGreaterThan(0.01);
      // straight lines between consecutive 16 kHz samples, two 16 kHz samples (0.125 ms) late
      const at = (i: number): number => (i >= 0 ? e[i] : 0);
      let worst = 0;
      for (let k = 0; k < got.length; k++) {
        const p = (k * SAMPLE_RATE) / rate - 2;
        const i = Math.floor(p);
        if (i + 1 >= e.length) break;
        const want = at(i) + (at(i + 1) - at(i)) * (p - i);
        worst = Math.max(worst, Math.abs(got[k] - want));
      }
      expect(worst).toBeLessThan(1e-5);
      expect(peakOf(got)).toBeLessThanOrEqual(CEILING);
      // engine time, and with it the reports, do not depend on the output rate
      expect(other.reports.map((r) => r.time)).toEqual(native.reports.map((r) => r.time));
      expect(other.reports.map((r) => r.wells.map((w) => w.well))).toEqual(native.reports.map((r) => r.wells.map((w) => w.well)));
      expect(other.last.wells[0].level).toBe(native.last.wells[0].level);
    });
  }

  it("at 16 kHz the block is the engine's own signal (no conversion), in any block size", () => {
    const a = new Rig(SAMPLE_RATE, 128);
    const b = new Rig(SAMPLE_RATE, 1000);
    for (const rig of [a, b]) {
      load(rig.engine, 0, 3);
      rig.send({ type: "press", well: wellIndex(0, 2) }).run(8000);
    }
    expect(maxDiff(a.out, b.out)).toBe(0);
    const ref = reference([{ net: protoWell(0, 3), drive: held }], 8000);
    expect(maxDiff(a.out, ref.out)).toBeLessThan(F32);
  });

  it("the output rate keeps exact step with 16 kHz over a long run (44.1 kHz, odd block sizes)", () => {
    const engine = new Engine({ outputRate: 44100 });
    let produced = 0;
    let reports = 0;
    let time = 0;
    const sizes = [128, 127, 441, 1, 300];
    for (let k = 0; produced < 44100 * 20; k++) {
      const block = new Float32Array(sizes[k % sizes.length]);
      engine.process(block);
      produced += block.length;
      for (let r = engine.takeReport(); r; r = engine.takeReport()) {
        reports++;
        time = r.time;
      }
    }
    const engineSamples = Math.floor((produced * SAMPLE_RATE) / 44100);
    expect(reports).toBe(Math.floor((engineSamples * REPORT_HZ) / SAMPLE_RATE));
    expect(engineSamples - time).toBeLessThan(535);
    expect(() => new Engine({ outputRate: 0 })).toThrow();
    expect(() => new Engine({ outputRate: Number.NaN })).toThrow();
  });
});

// ───────────────────────────── speed (not a test of correctness) ─────────────────────────────

// VITE_ENGINE_BENCH=1 npx vitest run tests/engine.test.ts -t speed
describe.runIf(import.meta.env.VITE_ENGINE_BENCH)("Engine: speed", () => {
  it("prints the share of real time used, in blocks of 128 samples", () => {
    const seconds = 10;
    const measure = (label: string, wells: number[], rate = SAMPLE_RATE): void => {
      const engine = new Engine({ outputRate: rate });
      for (let row = 0; row < 6; row++) {
        const layers = toLayers(8, flatWeights(8, 100 + row), row);
        for (const w of layers) engine.handle({ type: "weights", weights: w });
      }
      for (const well of wells) engine.handle({ type: "press", well });
      const block = new Float32Array(128);
      engine.process(block);
      for (const well of wells) engine.handle({ type: "release", well });
      for (let b = 0; b < rate / 128; b++) engine.process(block); // warm-up
      let worst = 0;
      const blocks = Math.floor((seconds * rate) / 128);
      const t0 = performance.now();
      for (let b = 0; b < blocks; b++) {
        const b0 = performance.now();
        engine.process(block);
        while (engine.takeReport());
        const dt = performance.now() - b0;
        if (dt > worst) worst = dt;
      }
      const ms = performance.now() - t0;
      const budget = (128 / rate) * 1000;
      console.log(
        `${label}: ${((100 * ms) / (seconds * 1000)).toFixed(2)}% of real time; slowest block ${worst.toFixed(3)} ms (budget ${budget.toFixed(2)} ms)`,
      );
    };
    const all = Array.from({ length: 48 }, (_, k) => k);
    measure("no wells (output stage only)", []);
    measure("one 8-layer well", [wellIndex(0, 7)]);
    measure("all 48 wells", all);
    measure("one 8-layer well, 48 kHz output", [wellIndex(0, 7)], 48000);
    measure("all 48 wells, 48 kHz output", all, 48000);
    expect(true).toBe(true);
  });
});
