// Engine against TensorFlow.js (plan section 4: "A test compares the engine with TF.js on a short
// run, so the claim 'these are TF.js networks' stays checkable").
//
// The reference (tests/helpers/tf-reference.ts) runs a well with the live TF.js layers themselves:
// SimpleRNNCell.call() and Dense.apply() on float32 tensors, a slow layer being fed its frame of
// CLOCKS[i] samples as one input vector, oldest first. The engine gets the arrays that
// generateLayer() read from those same layers.
//
// Two kinds of comparison, because many wells are chaotic: TF.js computes in float32 and the engine
// keeps float64 state, so the two start about 1e-7 apart and a chaotic well multiplies that
// difference as it runs (measured with these weights: 1e-6 to 5e-4 after 1,000 samples of a
// three-layer well, and of order 1 after 3,000 in the worst rows). That growth is the network, not
// an error, so:
//   1. Free run, short: both run on their own from rest. The tolerances are set from the measured
//      growth and are still far below the size of a real mistake (a wrong index or a missed tick
//      changes the output by 0.1 or more at once).
//   2. One step at a time, long: before every step the reference is given the engine's state, both
//      take one step, and the results are compared. Nothing accumulates, so the tolerance is that
//      of single float32 operations, and the run can be long enough for the slowest layer of an
//      eight-layer well to tick twice (131,072 samples).
// Everything is fixed (plate numbers, lengths), so the test does not depend on chance.

import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it } from "vitest";

import { Network } from "../src/audio/engine-core";
import { CLOCKS, MAX_LAYERS, N, ROWS } from "../src/plate/constants";
import { buildTfLayers, initWeights, readLayerWeights, type TfLayerSet } from "../src/plate/weights";
import { cellCall, cellWithConditioning, TfReference } from "./helpers/tf-reference";

const PLATE = 48213;
/** The button is held for the first samples, then released. */
const drive = (t: number): number => (t < 64 ? 1 : 0);

function well(plate: number, row: number, layers: number) {
  const sets: TfLayerSet[] = [];
  for (let i = 0; i < layers; i++) sets.push(buildTfLayers(plate, row, i));
  const ref = new TfReference(sets);
  const net = new Network(sets.map(readLayerWeights));
  return {
    ref,
    net,
    dispose() {
      ref.dispose();
      for (const s of sets) s.dispose();
    },
  };
}

function maxStateDiff(ref: TfReference, net: Network, layer: number): number {
  const a = ref.state(layer);
  let m = 0;
  for (let u = 0; u < N; u++) m = Math.max(m, Math.abs(a[u] - net.h[layer][u]));
  return m;
}

beforeAll(async () => {
  await initWeights();
});

describe("the reference is TF.js's own computation", () => {
  it("cellWithConditioning with zero conditioning equals SimpleRNNCell.call()", () => {
    for (let layer = 0; layer < 5; layer++) {
      const set = buildTfLayers(PLATE, 1, layer);
      const size = layer === 0 ? 1 : CLOCKS[layer];
      const input = tf.randomUniform([1, size], -1, 1, "float32", 11 + layer);
      const state = tf.randomUniform([1, N], -1, 1, "float32", 31 + layer);
      const zero = tf.zeros([1, N]);
      const a = cellCall(set, input, state);
      const b = cellWithConditioning(set, input, state, zero);
      expect(Array.from(b.dataSync())).toEqual(Array.from(a.dataSync()));
      expect(Math.max(...Array.from(a.abs().dataSync()))).toBeGreaterThan(0.1);
      tf.dispose([input, state, zero, a, b]);
      set.dispose();
    }
  });
});

describe("free run from rest", () => {
  it("one layer, 2,000 samples: SimpleRNNCell.call() and Dense.apply() only", () => {
    // Measured maximum difference over the six rows: 1.5e-6.
    for (let row = 0; row < ROWS; row++) {
      const w = well(PLATE, row, 1);
      expect(w.net.L).toBe(1);
      let worst = 0;
      let power = 0;
      for (let t = 0; t < 2000; t++) {
        const a = w.ref.step(drive(t));
        const b = w.net.step(drive(t));
        worst = Math.max(worst, Math.abs(a - b));
        power += b * b;
      }
      expect(Math.sqrt(power / 2000), `row ${row} is silent, nothing compared`).toBeGreaterThan(0.05);
      expect(worst, `row ${row}`).toBeLessThan(1e-5);
      expect(maxStateDiff(w.ref, w.net, 0), `row ${row} state`).toBeLessThan(1e-5);
      w.dispose();
    }
  });

  it("three layers, 1,000 samples: 62 ticks of layer 1 and 3 ticks of layer 2", () => {
    // Measured maximum difference over the six rows: 1.3e-5 in the first 300 samples (which hold
    // 18 ticks of layer 1 and the first tick of layer 2), 4.7e-4 by 1,000 samples.
    for (let row = 0; row < ROWS; row++) {
      const w = well(PLATE, row, 3);
      let early = 0;
      let worst = 0;
      let power = 0;
      for (let t = 0; t < 1000; t++) {
        const a = w.ref.step(drive(t));
        const b = w.net.step(drive(t));
        const d = Math.abs(a - b);
        if (t < 300) early = Math.max(early, d);
        worst = Math.max(worst, d);
        power += b * b;
      }
      expect(Math.sqrt(power / 1000), `row ${row} is silent, nothing compared`).toBeGreaterThan(0.05);
      expect(early, `row ${row}, first 300 samples`).toBeLessThan(5e-5);
      expect(worst, `row ${row}, 1,000 samples`).toBeLessThan(2e-3);
      for (let i = 0; i < 3; i++) expect(maxStateDiff(w.ref, w.net, i), `row ${row} layer ${i} state`).toBeLessThan(2e-3);
      // The slow layers did something: their states are neither zero nor all saturated.
      for (let i = 1; i < 3; i++) expect(Math.max(...Array.from(w.net.h[i], Math.abs)), `row ${row} layer ${i}`).toBeGreaterThan(0.05);
      w.dispose();
    }
  });

  it("reset() returns the engine to rest: the same run again gives the same samples", () => {
    const w = well(PLATE, 2, 3);
    const first: number[] = [];
    for (let t = 0; t < 600; t++) first.push(w.net.step(drive(t)));
    w.net.reset();
    for (let i = 0; i < 3; i++) expect(Array.from(w.net.h[i]).every((v) => v === 0)).toBe(true);
    const second: number[] = [];
    for (let t = 0; t < 600; t++) second.push(w.net.step(drive(t)));
    expect(second).toEqual(first);
    w.dispose();
  });
});

/**
 * One step from the same state differs only by float32 rounding in TF.js: about 6e-8 per operation
 * on numbers near 1, summed over 16 to 65,536 terms and passed through tanh (slope at most 1) and
 * the output layer. Measured over the runs below: at most 1.3e-6 in any state, 2.8e-7 in the
 * output. Mistakes tried against it (a kernel shifted by one position, a frame read backwards, a
 * matrix left untransposed, one slow-layer weight set to zero) gave differences of 1e-4 to 1.
 */
const ONE_STEP_TOLERANCE = 1e-5;

/**
 * One step at a time: the reference starts every step from the engine's state.
 * Returns the largest difference seen in the output and in each layer's state, and how often each layer ticked.
 */
function oneStepRun(row: number, layers: number, samples: number) {
  const w = well(PLATE, row, layers);
  let worstY = 0;
  let power = 0;
  const worstH = new Array<number>(layers).fill(0);
  const ticks = new Array<number>(layers).fill(0);
  const moved = new Array<number>(layers).fill(0);
  for (let t = 0; t < samples; t++) {
    const ticked: number[] = [0];
    for (let i = 1; i < layers; i++) if (t > 0 && t % CLOCKS[i] === 0) ticked.push(i);
    const a = w.ref.step(drive(t));
    const snapshot = ticked.map((i) => Float64Array.from(w.net.h[i]));
    const b = w.net.step(drive(t));
    worstY = Math.max(worstY, Math.abs(a - b));
    power += b * b;
    ticked.forEach((i, k) => {
      worstH[i] = Math.max(worstH[i], maxStateDiff(w.ref, w.net, i));
      ticks[i]++;
      {
        let change = 0;
        for (let u = 0; u < N; u++) change = Math.max(change, Math.abs(w.net.h[i][u] - snapshot[k][u]));
        moved[i] = Math.max(moved[i], change);
      }
    });
    w.ref.force(w.net.h, b);
  }
  w.dispose();
  return { worstY, worstH, ticks, moved, rms: Math.sqrt(power / samples) };
}

describe("one step at a time, from the engine's state", () => {
  it("four layers, every row, 8,400 samples: two ticks of layer 3", () => {
    for (let row = 0; row < ROWS; row++) {
      const r = oneStepRun(row, 4, 8400);
      expect(r.ticks).toEqual([8400, 524, 32, 2]);
      expect(r.rms, `row ${row} is silent, nothing compared`).toBeGreaterThan(0.05);
      expect(r.worstY, `row ${row} output`).toBeLessThan(ONE_STEP_TOLERANCE);
      r.worstH.forEach((d, i) => expect(d, `row ${row} layer ${i} state`).toBeLessThan(ONE_STEP_TOLERANCE));
      for (let i = 0; i < 4; i++) expect(r.moved[i], `row ${row} layer ${i} never changed`).toBeGreaterThan(0.01);
    }
  });

  it("eight layers, 131,336 samples: every layer ticks at least twice", () => {
    // Twice, because a layer's first tick starts from a state of zero and so does not use its
    // recurrent weights; the second tick does.
    const samples = 2 * CLOCKS[MAX_LAYERS - 1] + 264;
    const r = oneStepRun(0, MAX_LAYERS, samples);
    expect(r.ticks).toEqual([samples, 8208, 513, 32, 16, 8, 4, 2]);
    expect(r.rms).toBeGreaterThan(0.05);
    expect(r.worstY, "output").toBeLessThan(ONE_STEP_TOLERANCE);
    r.worstH.forEach((d, i) => expect(d, `layer ${i} state`).toBeLessThan(ONE_STEP_TOLERANCE));
    for (let i = 0; i < MAX_LAYERS; i++) expect(r.moved[i], `layer ${i} never changed`).toBeGreaterThan(0.01);
    console.log(
      `one-step parity, 8 layers, ${samples} samples: output differs by at most ${r.worstY.toExponential(2)}, ` +
        `states by at most ${r.worstH.map((d) => d.toExponential(1)).join(" ")}`,
    );
  });
});
