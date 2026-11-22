// The weights of a plate: same number -> same weights; shapes; zero biases; TF.js defaults.

import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type { FromWeightsWorker, LayerWeights, WeightRole } from "../src/contracts";
import { CLOCKS, MAX_LAYERS, N, PLATE_COUNT, ROWS, formatPlate, weightCount } from "../src/plate/constants";
import { parsePlate, randomPlate, seedFor } from "../src/plate/seed";
import {
  FAST_LAYERS,
  buildTfLayers,
  generateLayer,
  generatePlate,
  generationOrder,
  initWeights,
  readLayerWeights,
  transferList,
} from "../src/plate/weights";

const PLATE = 48213;
const ROLES: WeightRole[] = ["kernel", "recurrent", "down", "out"];

const same = (a: Float32Array | null, b: Float32Array | null): boolean => {
  if (a === null || b === null) return a === b;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};
const sameLayer = (a: LayerWeights, b: LayerWeights): boolean =>
  a.row === b.row && a.layer === b.layer && same(a.R, b.R) && same(a.K, b.K) && same(a.C, b.C) && same(a.W, b.W);

beforeAll(async () => {
  await initWeights();
});

describe("seedFor", () => {
  it("gives a non-zero 32-bit integer, the same every time", () => {
    for (const role of ROLES) {
      const s = seedFor(PLATE, 5, 3, role);
      expect(Number.isInteger(s)).toBe(true);
      expect(s).toBeGreaterThan(0);
      expect(s).toBeLessThan(2 ** 32);
      expect(seedFor(PLATE, 5, 3, role)).toBe(s);
    }
    // Stored values: the seeds must not change between versions, or every plate changes.
    expect(seedFor(0, 0, 0, "kernel")).toBe(1364076727);
    expect(ROLES.map((r) => seedFor(PLATE, 5, 3, r))).toEqual(STORED_SEEDS);
  });

  it("is different for every plate, row, layer and role", () => {
    const seen = new Set<number>();
    let count = 0;
    const plates = [0, 1, 2, 3, PLATE, PLATE + 1, PLATE_COUNT - 2, PLATE_COUNT - 1];
    for (let p = 100; p < 1100; p++) plates.push(p);
    for (const plate of plates) {
      for (let row = 0; row < ROWS; row++) {
        for (let layer = 0; layer < MAX_LAYERS; layer++) {
          for (const role of ROLES) {
            seen.add(seedFor(plate, row, layer, role));
            count++;
          }
        }
      }
    }
    expect(seen.size).toBe(count);
    expect(seen.has(0)).toBe(false);
  });

  it("is well mixed: neighbouring inputs give unrelated seeds", () => {
    // Each of the 32 bits should differ in about half of 2,000 neighbouring pairs.
    const flips = new Array<number>(32).fill(0);
    const pairs = 2000;
    for (let p = 0; p < pairs; p++) {
      const x = seedFor(p, 0, 0, "kernel") ^ seedFor(p + 1, 0, 0, "kernel");
      for (let b = 0; b < 32; b++) if ((x >>> b) & 1) flips[b]++;
    }
    for (let b = 0; b < 32; b++) {
      expect(flips[b] / pairs, `bit ${b}`).toBeGreaterThan(0.4);
      expect(flips[b] / pairs, `bit ${b}`).toBeLessThan(0.6);
    }
  });

  it("rejects values outside the plate", () => {
    expect(() => seedFor(-1, 0, 0, "kernel")).toThrow(RangeError);
    expect(() => seedFor(PLATE_COUNT, 0, 0, "kernel")).toThrow(RangeError);
    expect(() => seedFor(1.5, 0, 0, "kernel")).toThrow(RangeError);
    expect(() => seedFor(0, ROWS, 0, "kernel")).toThrow(RangeError);
    expect(() => seedFor(0, 0, MAX_LAYERS, "kernel")).toThrow(RangeError);
    expect(() => seedFor(0, 0, 0, "bias" as WeightRole)).toThrow(RangeError);
  });
});

describe("plate numbers", () => {
  it("parsePlate accepts six digits and nothing else", () => {
    expect(parsePlate("048213")).toBe(48213);
    expect(parsePlate("000000")).toBe(0);
    expect(parsePlate("999999")).toBe(999999);
    for (const bad of [null, "", "48213", "0482130", " 048213", "048213 ", "04821a", "-48213", "+48213", "048.13", "0x1234", "1e5000", "４８２１３０", "048213\n"]) {
      expect(parsePlate(bad), JSON.stringify(bad)).toBeNull();
    }
    expect(parsePlate(undefined as unknown as null)).toBeNull();
    expect(parsePlate(48213 as unknown as string)).toBeNull();
  });

  it("parsePlate reads back what formatPlate prints", () => {
    for (const p of [0, 7, 48213, 999999]) expect(parsePlate(formatPlate(p))).toBe(p);
  });

  it("randomPlate stays in range and varies", () => {
    const seen = new Set<number>();
    for (let i = 0; i < 500; i++) {
      const p = randomPlate();
      expect(Number.isInteger(p)).toBe(true);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThan(PLATE_COUNT);
      seen.add(p);
    }
    expect(seen.size).toBeGreaterThan(400);
    const spy = vi.spyOn(Math, "random");
    spy.mockReturnValue(0);
    expect(randomPlate()).toBe(0);
    spy.mockReturnValue(0.9999999999999999);
    expect(randomPlate()).toBe(PLATE_COUNT - 1);
    spy.mockRestore();
  });
});

describe("generateLayer", () => {
  it("uses the CPU backend", () => {
    expect(tf.getBackend()).toBe("cpu");
  });

  it("gives every layer the shapes the engine expects", () => {
    for (let layer = 0; layer < MAX_LAYERS; layer++) {
      const w = generateLayer(PLATE, 1, layer);
      expect(w.row).toBe(1);
      expect(w.layer).toBe(layer);
      expect(w.R).toBeInstanceOf(Float32Array);
      expect(w.K).toBeInstanceOf(Float32Array);
      expect(w.R.length).toBe(N * N);
      expect(w.K.length).toBe(layer === 0 ? N : CLOCKS[layer] * N);
      if (layer === 0) {
        expect(w.C).toBeNull();
        expect(w.W).toBeInstanceOf(Float32Array);
        expect(w.W!.length).toBe(N);
      } else {
        expect(w.W).toBeNull();
        expect(w.C).toBeInstanceOf(Float32Array);
        expect(w.C!.length).toBe(N * N);
      }
      for (const a of [w.R, w.K, w.C, w.W]) if (a) expect(a.every(Number.isFinite)).toBe(true);
    }
  });

  it("gives the same weights for the same plate number, however often and in whatever order", () => {
    const first: LayerWeights[] = [];
    for (let layer = 0; layer < MAX_LAYERS; layer++) first.push(generateLayer(PLATE, 2, layer));
    generateLayer(7, 0, 3); // other work in between
    for (let layer = MAX_LAYERS - 1; layer >= 0; layer--) {
      expect(sameLayer(generateLayer(PLATE, 2, layer), first[layer]), `layer ${layer}`).toBe(true);
    }
  });

  it("gives the same weights from a fresh copy of the module", async () => {
    const before = generateLayer(PLATE, 4, 4);
    vi.resetModules();
    const fresh = await import("../src/plate/weights");
    await fresh.initWeights();
    expect(fresh.generateLayer).not.toBe(generateLayer);
    expect(sameLayer(fresh.generateLayer(PLATE, 4, 4), before)).toBe(true);
  });

  it("gives the weights stored here from an earlier run (a new process each time)", () => {
    // If these change, plate numbers no longer mean what they meant: old links give new sounds.
    const a = generateLayer(PLATE, 5, 0);
    expect(Array.from(a.R.slice(0, 3))).toEqual([0.13599812984466553, -0.3233565390110016, 0.19432863593101501]);
    expect(Array.from(a.K.slice(0, 3))).toEqual([-0.26543670892715454, 0.38595518469810486, 0.17453156411647797]);
    expect(Array.from(a.W!.slice(0, 3))).toEqual([0.10850713402032852, 0.23111431300640106, -0.037572864443063736]);
    const b = generateLayer(PLATE, 2, 3);
    expect(Array.from(b.R.slice(0, 3))).toEqual([0.05985260009765625, 0.21335114538669586, -0.7230598330497742]);
    expect(Array.from(b.K.slice(0, 3))).toEqual([0.028365081176161766, -0.015820518136024475, 0.02214980497956276]);
    expect(Array.from(b.C!.slice(0, 3))).toEqual([0.3183037340641022, 0.1583220213651657, 0.39741185307502747]);
  });

  it("gives different weights for a different plate, row, layer or role", () => {
    const base = generateLayer(PLATE, 2, 1);
    for (const other of [generateLayer(PLATE + 1, 2, 1), generateLayer(PLATE, 3, 1), generateLayer(0, 2, 1)]) {
      expect(same(other.R, base.R)).toBe(false);
      expect(same(other.K, base.K)).toBe(false);
      expect(same(other.C, base.C)).toBe(false);
    }
    // Another layer: the arrays of equal size differ too.
    const up = generateLayer(PLATE, 2, 2);
    expect(same(up.R, base.R)).toBe(false);
    expect(same(up.C, base.C)).toBe(false);
    expect(same(up.K.subarray(0, base.K.length), base.K)).toBe(false);
    // Roles: arrays of equal size within one layer differ.
    expect(same(base.R, base.C)).toBe(false);
    const zero = generateLayer(PLATE, 2, 0);
    expect(same(zero.K, zero.W)).toBe(false);
    const other = generateLayer(PLATE + 1, 2, 0);
    expect(same(other.W, zero.W)).toBe(false);
    expect(same(other.K, zero.K)).toBe(false);
    // No two of the 48 layers of a plate share a recurrent matrix.
    const firsts = new Set<string>();
    for (const { row, layer } of generationOrder()) {
      if (layer >= FAST_LAYERS) continue;
      firsts.add(Array.from(generateLayer(PLATE, row, layer).R.slice(0, 4)).join());
    }
    expect(firsts.size).toBe(ROWS * FAST_LAYERS);
  });

  it("shares weights along a row: a well is the first layers of its row, and the label's count is their size", () => {
    // LayerWeights carries no column. Well (row, L layers) uses layers 0 .. L-1 of the row, so a
    // well with one more layer has exactly the same first layers.
    let carried = 0;
    for (let layers = 1; layers <= MAX_LAYERS; layers++) {
      const w = generateLayer(PLATE, 0, layers - 1);
      // Arrays carried, plus the zero biases that are not: 16 per simpleRNN, 16 per down layer... the
      // label counts the simpleRNN bias (16) and, for layer 0, the output bias (1).
      carried += w.R.length + w.K.length + (w.C ? w.C.length : 0) + (w.W ? w.W.length : 0) + N + (w.W ? 1 : 0);
      expect(carried, `${layers} layers`).toBe(weightCount(layers));
    }
  });

  it("recurrent weights are orthogonal (TF.js default), in every layer", () => {
    for (let row = 0; row < ROWS; row++) {
      for (let layer = 0; layer < MAX_LAYERS; layer += row === 0 ? 1 : 3) {
        const { R } = generateLayer(PLATE, row, layer);
        let worst = 0;
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            let rows = 0;
            let cols = 0;
            for (let k = 0; k < N; k++) {
              rows += R[i * N + k] * R[j * N + k];
              cols += R[k * N + i] * R[k * N + j];
            }
            const want = i === j ? 1 : 0;
            worst = Math.max(worst, Math.abs(rows - want), Math.abs(cols - want));
          }
        }
        expect(worst, `row ${row} layer ${layer}`).toBeLessThan(1e-5);
      }
    }
  });

  it("input, down and output weights are glorot-uniform (TF.js default)", () => {
    const check = (a: Float32Array, fanIn: number, fanOut: number, what: string): void => {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      let max = 0;
      let sum = 0;
      let sq = 0;
      for (const v of a) {
        max = Math.max(max, Math.abs(v));
        sum += v;
        sq += v * v;
      }
      expect(max, `${what} bound`).toBeLessThanOrEqual(limit * (1 + 1e-6));
      if (a.length >= 256) {
        const mean = sum / a.length;
        const sd = Math.sqrt(sq / a.length - mean * mean);
        const expected = limit / Math.sqrt(3);
        expect(max, `${what} reaches the bound`).toBeGreaterThan(0.9 * limit);
        expect(Math.abs(mean), `${what} mean`).toBeLessThan((5 * expected) / Math.sqrt(a.length));
        expect(sd / expected, `${what} spread`).toBeGreaterThan(0.9);
        expect(sd / expected, `${what} spread`).toBeLessThan(1.1);
      } else {
        expect(max, `${what} is not all small`).toBeGreaterThan(0.3 * limit);
      }
    };
    for (let layer = 0; layer < MAX_LAYERS; layer++) {
      const w = generateLayer(PLATE, 3, layer);
      check(w.K, layer === 0 ? 1 : CLOCKS[layer], N, `K of layer ${layer}`);
      if (w.C) check(w.C, N, N, `C of layer ${layer}`);
      if (w.W) check(w.W, N, 1, `W of layer ${layer}`);
    }
  });
});

describe("the TF.js layers behind the weights", () => {
  it("are a simpleRNN and a dense layer with default settings, and all biases are zero", () => {
    for (let layer = 0; layer < MAX_LAYERS; layer++) {
      const set = buildTfLayers(PLATE, 4, layer);
      const size = layer === 0 ? 1 : CLOCKS[layer];
      expect(set.rnn.getClassName()).toBe("SimpleRNN");
      const config = set.rnn.getConfig();
      expect(config.units).toBe(N);
      expect(config.activation).toBe("tanh");
      expect(config.useBias).toBe(true);
      const [kernel, recurrent, bias] = set.rnn.getWeights();
      expect(set.rnn.getWeights().length).toBe(3);
      expect(kernel.shape).toEqual([size, N]);
      expect(recurrent.shape).toEqual([N, N]);
      expect(bias.shape).toEqual([N]);
      expect(Array.from(bias.dataSync()).every((v) => v === 0)).toBe(true);

      const dense = layer === 0 ? set.out! : set.down!;
      expect(layer === 0 ? set.down : set.out).toBeNull();
      expect(dense.getClassName()).toBe("Dense");
      expect(dense.getConfig().activation).toBe("linear");
      expect(dense.getConfig().useBias).toBe(true);
      const [dk, db] = dense.getWeights();
      expect(dk.shape).toEqual(layer === 0 ? [N, 1] : [N, N]);
      expect(db.shape).toEqual(layer === 0 ? [1] : [N]);
      expect(Array.from(db.dataSync()).every((v) => v === 0)).toBe(true);
      set.dispose();
    }
  });

  it("are stored by TF.js as [sender, receiver]; the engine's arrays are the transposes, the kernel is as stored", () => {
    for (const layer of [0, 2]) {
      const set = buildTfLayers(PLATE, 0, layer);
      const w = readLayerWeights(set);
      const [kernel, recurrent] = set.rnn.getWeights();
      const k = kernel.dataSync();
      const r = recurrent.dataSync();
      expect(same(w.K, Float32Array.from(k))).toBe(true);
      const dense = (layer === 0 ? set.out! : set.down!).getWeights()[0].dataSync();
      for (let receiver = 0; receiver < N; receiver++) {
        for (let sender = 0; sender < N; sender++) {
          expect(w.R[receiver * N + sender]).toBe(r[sender * N + receiver]);
          if (w.C) expect(w.C[receiver * N + sender]).toBe(dense[sender * N + receiver]);
        }
      }
      if (w.W) expect(same(w.W, Float32Array.from(dense))).toBe(true);
      // The arrays are copies: they survive the layers and own their buffers.
      set.dispose();
      expect(w.K.byteOffset).toBe(0);
      expect(w.K.buffer.byteLength).toBe(w.K.byteLength);
      expect(same(w.K, generateLayer(PLATE, 0, layer).K)).toBe(true);
    }
  });

  it("the seeded orthogonal initialiser of TF.js honours its seed (plan section 9)", () => {
    const build = (seed: number): Float32Array => {
      const set = buildTfLayers(seed, 0, 0);
      const r = Float32Array.from(set.rnn.getWeights()[1].dataSync());
      set.dispose();
      return r;
    };
    expect(same(build(1), build(1))).toBe(true);
    expect(same(build(1), build(2))).toBe(false);
  });

  it("leave no tensors behind", () => {
    generateLayer(PLATE, 0, 0); // warm up
    const before = tf.memory();
    for (let i = 0; i < 3; i++) for (let layer = 0; layer < MAX_LAYERS; layer++) generateLayer(PLATE + i, i, layer);
    const set = buildTfLayers(PLATE, 0, 5);
    expect(tf.memory().numTensors).toBe(before.numTensors + 5);
    set.dispose();
    const after = tf.memory();
    expect(after.numTensors).toBe(before.numTensors);
    expect(after.numBytes).toBe(before.numBytes);
    expect(after.numDataBuffers).toBe(before.numDataBuffers);
  });
});

describe("the order of a plate (weights worker)", () => {
  it("is layers 0..3 of every row, then layers 4..7 layer by layer, row by row", () => {
    const order = [...generationOrder()].map(({ row, layer }) => `${"ABCDEF"[row]}${layer}`);
    expect(order.join(" ")).toBe(
      "A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 E0 E1 E2 E3 F0 F1 F2 F3 " +
        "A4 B4 C4 D4 E4 F4 A5 B5 C5 D5 E5 F5 A6 B6 C6 D6 E6 F6 A7 B7 C7 D7 E7 F7",
    );
  });

  it("generatePlate posts 48 layers in that order with their buffers, then done", async () => {
    const messages: FromWeightsWorker[] = [];
    const transfers: ArrayBuffer[][] = [];
    let pauses = 0;
    await generatePlate(
      PLATE,
      (msg, transfer) => {
        messages.push(msg);
        transfers.push(transfer);
      },
      { pause: async () => void pauses++ },
    );
    expect(messages.length).toBe(ROWS * MAX_LAYERS + 1);
    expect(pauses).toBe(ROWS * MAX_LAYERS);
    expect(messages[messages.length - 1]).toEqual({ type: "done", plate: PLATE });
    const order = [...generationOrder()];
    const buffers = new Set<ArrayBuffer>();
    order.forEach(({ row, layer }, i) => {
      const msg = messages[i];
      if (msg.type !== "layer") throw new Error(`message ${i} is ${msg.type}`);
      expect(msg.plate).toBe(PLATE);
      expect(msg.weights.row).toBe(row);
      expect(msg.weights.layer).toBe(layer);
      const w = msg.weights;
      const arrays = [w.R, w.K, w.C, w.W].filter((a): a is Float32Array => a !== null);
      expect(transfers[i]).toEqual(arrays.map((a) => a.buffer));
      expect(transferList(w)).toEqual(transfers[i]);
      for (const a of arrays) {
        expect(a.byteOffset).toBe(0);
        expect(a.buffer.byteLength).toBe(a.byteLength);
        expect(buffers.has(a.buffer as ArrayBuffer)).toBe(false);
        buffers.add(a.buffer as ArrayBuffer);
      }
    });
    // The messages survive a structured clone with transfer, as postMessage does.
    const first = messages[0];
    if (first.type !== "layer") throw new Error("unreachable");
    const expected = generateLayer(PLATE, 0, 0);
    const cloned = structuredClone(first, { transfer: transfers[0] });
    expect(first.weights.R.length).toBe(0); // transferred away
    expect(sameLayer(cloned.weights, expected)).toBe(true);
    // And they are the weights generateLayer gives.
    const last = messages[ROWS * MAX_LAYERS - 1];
    if (last.type !== "layer") throw new Error("unreachable");
    expect(sameLayer(last.weights, generateLayer(PLATE, ROWS - 1, MAX_LAYERS - 1))).toBe(true);
  });

  it("generatePlate posts an error instead of throwing", async () => {
    for (const bad of [-1, PLATE_COUNT, 1.5, Number.NaN, "7" as unknown as number]) {
      const messages: FromWeightsWorker[] = [];
      await generatePlate(bad, (msg) => messages.push(msg));
      expect(messages.length).toBe(1);
      expect(messages[0].type).toBe("error");
      expect(messages[0].plate).toBe(bad);
      if (messages[0].type === "error") expect(messages[0].message).toMatch(/plate out of range/);
    }
    // A failure part of the way through: layers so far, then one error, no done.
    const messages: FromWeightsWorker[] = [];
    await generatePlate(PLATE, (msg) => {
      messages.push(msg);
      if (messages.length === 3) throw new Error("port closed");
    });
    expect(messages.map((m) => m.type)).toEqual(["layer", "layer", "layer", "error"]);
    const last = messages[3];
    if (last.type === "error") expect(last.message).toBe("port closed");
  });

  it("generatePlate stops quietly when cancelled", async () => {
    const messages: FromWeightsWorker[] = [];
    await generatePlate(PLATE, (msg) => messages.push(msg), { cancelled: () => messages.length >= 5 });
    expect(messages.length).toBe(5);
    expect(messages.every((m) => m.type === "layer")).toBe(true);
  });
});

describe("load time (plan section 9)", () => {
  it("reports how long the layers take in Node", async () => {
    generateLayer(1, 0, 7); // warm up
    const per: string[] = [];
    for (let layer = 0; layer < MAX_LAYERS; layer++) {
      const t0 = performance.now();
      for (let row = 0; row < ROWS; row++) generateLayer(2, row, layer);
      per.push(`layer ${layer}: ${((performance.now() - t0) / ROWS).toFixed(1)} ms`);
    }
    const t0 = performance.now();
    let fast = 0;
    let count = 0;
    await generatePlate(3, (msg) => {
      if (msg.type === "layer" && ++count === ROWS * FAST_LAYERS) fast = performance.now() - t0;
    });
    const total = performance.now() - t0;
    console.log(
      `generateLayer, mean of ${ROWS} rows: ${per.join(", ")}\n` +
        `whole plate (48 layers): ${total.toFixed(0)} ms; layers 0-3 of all rows ready after ${fast.toFixed(0)} ms`,
    );
    expect(count).toBe(ROWS * MAX_LAYERS);
    expect(total).toBeLessThan(20_000);
  });
});

const STORED_SEEDS = [3926274408, 3299076083, 2811709438, 1135935353];
