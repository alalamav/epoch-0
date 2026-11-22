// Deterministic test weights: the generator and the scales of the speed test in
// plan/prototype/engine_check.mjs ("library-like scales", not TF.js weights).

// No runtime imports, so that tests/helpers/engine-make-fixture.mjs can load this file in plain Node.
// engine.test.ts checks that N and CLOCKS here equal those of src/plate/constants.ts.

import type { LayerWeights } from "../../src/contracts";

export const N = 16;
export const CLOCKS = [1, 16, 256, 4096, 8192, 16384, 32768, 65536];

/** The prototype's generator: uniform in [-0.5, 0.5). */
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

export const flatLength = (L: number): number =>
  L * N * N + CLOCKS.slice(1, L).reduce((a, c) => a + c * N, 0) + (L - 1) * N * N + 2 * N;

/**
 * Weights of an L-layer network in the flat layout ClockNet reads:
 * R[0..L-1], K[1..L-1], C (layer 1 -> 0, 2 -> 1, ...), K0, W.
 * `outScale` scales the output weights W (1.2 in the prototype).
 */
export function flatWeights(L: number, seed: number, outScale = 1.2): Float64Array {
  const rnd = lcg(seed);
  const w = new Float64Array(flatLength(L));
  let o = 0;
  for (let i = 0; i < L; i++) for (let k = 0; k < N * N; k++) w[o++] = rnd() * 0.85;
  for (let i = 1; i < L; i++) {
    const sd = Math.sqrt(2 / (CLOCKS[i] + N)) * 3.4;
    for (let k = 0; k < CLOCKS[i] * N; k++) w[o++] = rnd() * sd;
  }
  for (let i = 0; i < L - 1; i++) for (let k = 0; k < N * N; k++) w[o++] = rnd() * 0.85;
  for (let k = 0; k < N; k++) w[o++] = rnd() * 1.2;
  for (let k = 0; k < N; k++) w[o++] = rnd() * outScale;
  return w;
}

/** The same weights as LayerWeights, rounded to Float32 exactly as `new ClockNet(L, w, Float32Array)` does. */
export function toLayers(L: number, w: Float64Array, row = 0): LayerWeights[] {
  let o = 0;
  const take = (len: number): Float32Array => {
    const v = Float32Array.from(w.subarray(o, o + len));
    o += len;
    return v;
  };
  const R: Float32Array[] = [];
  for (let i = 0; i < L; i++) R.push(take(N * N));
  const K: Float32Array[] = [new Float32Array(0)];
  for (let i = 1; i < L; i++) K.push(take(CLOCKS[i] * N));
  const C: (Float32Array | null)[] = [null];
  for (let i = 1; i < L; i++) C.push(take(N * N));
  K[0] = take(N);
  const W = take(N);
  const layers: LayerWeights[] = [];
  for (let i = 0; i < L; i++) layers.push({ row, layer: i, R: R[i], K: K[i], C: C[i], W: i === 0 ? W : null });
  return layers;
}

/**
 * The flat weights of the first `L` layers of a row whose `total` layers are given, i.e. what
 * ClockNet needs to be the well (row, L). Inverse of toLayers for a prefix of the row.
 */
export function flatOfPrefix(layers: LayerWeights[], L: number): Float64Array {
  const w = new Float64Array(flatLength(L));
  let o = 0;
  const put = (a: Float32Array): void => {
    w.set(a, o);
    o += a.length;
  };
  for (let i = 0; i < L; i++) put(layers[i].R);
  for (let i = 1; i < L; i++) put(layers[i].K);
  for (let i = 1; i < L; i++) put(layers[i].C!);
  put(layers[0].K);
  put(layers[0].W!);
  return w;
}

/** Copies, as a message would carry them (the engine keeps the arrays it is sent). */
export const cloneLayer = (w: LayerWeights): LayerWeights => ({
  row: w.row,
  layer: w.layer,
  R: w.R.slice(),
  K: w.K.slice(),
  C: w.C ? w.C.slice() : null,
  W: w.W ? w.W.slice() : null,
});
