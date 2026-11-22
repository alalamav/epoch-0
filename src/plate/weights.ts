// TensorFlow.js supplies the weights and nothing else (plan section 4).
//
// Every layer of every row is built as a real TF.js layer with default settings, and its weights are
// read out with getWeights(). The only change from the defaults is that each initialiser is given a
// seed (seed.ts), so that a plate number always gives the same weights.
//
// Pure module: no DOM, no worker globals. Used by weights.worker.ts and by the Node tests.
//
// SEEDED ORTHOGONAL WEIGHTS (plan section 9). The plan warned that TF.js may ignore the seed of its
// orthogonal initialiser. Checked against the installed tfjs-layers 4.22.0: it does not. Its
// Orthogonal.apply passes the seed to randomNormal, and two layers built with the same seed have
// bit-identical recurrent weights (tests/weights.test.ts checks this, and checks stored values, on
// every run). So the plain path is used: `tf.initializers.orthogonal({seed})` on the layer itself, and
// no setWeights fallback. Two things the check turned up:
//   1. The initialiser calls chained tensor methods (rMat.flatten().stridedSlice(...), .sign(),
//      .transpose(), .reshape()). These exist only if tfjs-core's chained ops are registered, which
//      the umbrella "@tensorflow/tfjs" package does but "@tensorflow/tfjs-core" alone does not.
//      Without the import below, building a simpleRNN throws "rMat.flatten is not a function".
//   2. A seed of 0 is treated as "no seed" by the normal sampler. seedFor never returns 0.

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import * as tfl from "@tensorflow/tfjs-layers";

import type { FromWeightsWorker, LayerWeights } from "../contracts";
import { CLOCKS, MAX_LAYERS, N, PLATE_COUNT, ROWS } from "./constants";
import { seedFor } from "./seed";

/** Layers 0 .. FAST_LAYERS-1 of every row are generated first (plan section 9, load time). */
export const FAST_LAYERS = 4;

type SimpleRNNLayer = ReturnType<typeof tfl.layers.simpleRNN>;
type DenseLayer = ReturnType<typeof tfl.layers.dense>;

/** The live TF.js layers behind one LayerWeights. Call dispose() when done. */
export interface TfLayerSet {
  plate: number;
  row: number;
  layer: number;
  /** tf.layers.simpleRNN({units: 16}); input size 1 for layer 0, CLOCKS[layer] otherwise. */
  rnn: SimpleRNNLayer;
  /** Layer i > 0: tf.layers.dense({units: 16}) from this layer's state to layer i - 1. null for layer 0. */
  down: DenseLayer | null;
  /** Layer 0: tf.layers.dense({units: 1}), the output. null for other layers. */
  out: DenseLayer | null;
  dispose(): void;
}

let ready: Promise<void> | null = null;

/** Selects the TF.js CPU backend (the only one imported; no WebGL). Safe to call more than once. */
export function initWeights(): Promise<void> {
  if (!ready) {
    ready = (async () => {
      await tf.setBackend("cpu");
      await tf.ready();
    })();
  }
  return ready;
}

function checkIndex(plate: number, row: number, layer: number): void {
  if (!Number.isInteger(plate) || plate < 0 || plate >= PLATE_COUNT) throw new RangeError(`plate out of range: ${plate}`);
  if (!Number.isInteger(row) || row < 0 || row >= ROWS) throw new RangeError(`row out of range: ${row}`);
  if (!Number.isInteger(layer) || layer < 0 || layer >= MAX_LAYERS) throw new RangeError(`layer out of range: ${layer}`);
}

/**
 * Builds the real TF.js layers for one layer of one row. Everything is the TF.js default
 * (tanh, bias of zeros, glorotUniform kernel, orthogonal recurrent kernel, linear dense layers);
 * the initialisers named here are those defaults with a seed.
 * Call initWeights() first.
 */
export function buildTfLayers(plate: number, row: number, layer: number): TfLayerSet {
  checkIndex(plate, row, layer);
  const inputSize = layer === 0 ? 1 : CLOCKS[layer];

  const rnn = tfl.layers.simpleRNN({
    units: N,
    kernelInitializer: tfl.initializers.glorotUniform({ seed: seedFor(plate, row, layer, "kernel") }),
    recurrentInitializer: tfl.initializers.orthogonal({ seed: seedFor(plate, row, layer, "recurrent") }),
  });
  // Applying a layer to a symbolic input is how TF.js builds it: the weights are created here.
  // Input shape [time steps, input size]; no numbers flow.
  rnn.apply(tfl.input({ shape: [null, inputSize] }));

  let down: DenseLayer | null = null;
  let out: DenseLayer | null = null;
  if (layer > 0) {
    down = tfl.layers.dense({
      units: N,
      kernelInitializer: tfl.initializers.glorotUniform({ seed: seedFor(plate, row, layer, "down") }),
    });
    down.apply(tfl.input({ shape: [N] }));
  } else {
    out = tfl.layers.dense({
      units: 1,
      kernelInitializer: tfl.initializers.glorotUniform({ seed: seedFor(plate, row, layer, "out") }),
    });
    out.apply(tfl.input({ shape: [N] }));
  }

  return {
    plate,
    row,
    layer,
    rnn,
    down,
    out,
    dispose() {
      rnn.dispose();
      down?.dispose();
      out?.dispose();
    },
  };
}

function expectShape(t: tf.Tensor, shape: number[], what: string): void {
  if (t.shape.length !== shape.length || t.shape.some((d, i) => d !== shape[i])) {
    throw new Error(`${what}: expected shape [${shape}], TF.js gave [${t.shape}]`);
  }
}

function expectZero(t: tf.Tensor, what: string): void {
  const d = t.dataSync();
  for (let i = 0; i < d.length; i++) if (d[i] !== 0) throw new Error(`${what}: expected a bias of zeros`);
}

/** TF.js [sender, receiver] (row-major) -> engine [receiver * n + sender]. */
function transposed(t: tf.Tensor, n: number): Float32Array {
  const d = t.dataSync();
  const o = new Float32Array(n * n);
  for (let s = 0; s < n; s++) for (let r = 0; r < n; r++) o[r * n + s] = d[s * n + r];
  return o;
}

/** A copy: on the CPU backend dataSync() can return the backend's own array, which must not be transferred. */
const copied = (t: tf.Tensor): Float32Array => Float32Array.from(t.dataSync() as Float32Array);

/** Reads the weights of live TF.js layers into the order the engine uses (see LayerWeights). */
export function readLayerWeights(set: TfLayerSet): LayerWeights {
  const { row, layer } = set;
  const inputSize = layer === 0 ? 1 : CLOCKS[layer];

  // simpleRNN: [kernel, recurrent kernel, bias].
  const [kernel, recurrent, bias] = set.rnn.getWeights();
  expectShape(kernel, [inputSize, N], "kernel");
  expectShape(recurrent, [N, N], "recurrent kernel");
  expectShape(bias, [N], "bias");
  expectZero(bias, "simpleRNN");

  // Kernel: TF.js shape [input size, units], row-major, and the cell computes input · kernel, so
  // unit u receives sum over p of input[p] * kernel[p * N + u]. A frame is fed oldest sample first,
  // so p is the position in the frame and the flat array is already [position * N + unit].
  const K = copied(kernel);
  // Recurrent kernel: the cell computes state · recurrent, so TF.js holds [sender, receiver].
  const R = transposed(recurrent, N);

  let C: Float32Array | null = null;
  let W: Float32Array | null = null;
  if (set.down) {
    const [k, b] = set.down.getWeights();
    expectShape(k, [N, N], "down kernel");
    expectZero(b, "down dense");
    C = transposed(k, N); // dense computes input · kernel: [sender, receiver] in TF.js
  }
  if (set.out) {
    const [k, b] = set.out.getWeights();
    expectShape(k, [N, 1], "output kernel");
    expectZero(b, "output dense");
    W = copied(k);
  }
  return { row, layer, R, K, C, W };
}

/**
 * The weights of one layer of one row of a plate, from real TF.js layers that are built, read and
 * disposed. The same (plate, row, layer) always gives the same numbers. Call initWeights() first.
 */
export function generateLayer(plate: number, row: number, layer: number): LayerWeights {
  const set = buildTfLayers(plate, row, layer);
  try {
    return readLayerWeights(set);
  } finally {
    set.dispose();
  }
}

/**
 * The order in which a plate is generated: layers 0..3 of every row first (rows A..F), then
 * layers 4..7, layer by layer, row by row. So every well of the first four columns can sound
 * before the large slow layers are made.
 */
export function* generationOrder(): Generator<{ row: number; layer: number }> {
  for (let row = 0; row < ROWS; row++) {
    for (let layer = 0; layer < Math.min(FAST_LAYERS, MAX_LAYERS); layer++) yield { row, layer };
  }
  for (let layer = FAST_LAYERS; layer < MAX_LAYERS; layer++) {
    for (let row = 0; row < ROWS; row++) yield { row, layer };
  }
}

/** The buffers of a LayerWeights, for the transfer list of postMessage. */
export function transferList(w: LayerWeights): ArrayBuffer[] {
  const list = [w.R.buffer as ArrayBuffer, w.K.buffer as ArrayBuffer];
  if (w.C) list.push(w.C.buffer as ArrayBuffer);
  if (w.W) list.push(w.W.buffer as ArrayBuffer);
  return list;
}

export interface GeneratePlateOptions {
  /** Awaited after each layer so the caller's thread can breathe. Default: none. */
  pause?: () => Promise<void>;
  /** Checked before each layer; when true, generation stops without "done". */
  cancelled?: () => boolean;
}

/**
 * The whole job of the weights worker, without the worker: emits one "layer" message per layer in
 * generationOrder(), then "done"; or "error" if anything fails. Never throws.
 */
export async function generatePlate(
  plate: number,
  emit: (msg: FromWeightsWorker, transfer: ArrayBuffer[]) => void,
  options: GeneratePlateOptions = {},
): Promise<void> {
  try {
    if (!Number.isInteger(plate) || plate < 0 || plate >= PLATE_COUNT) throw new RangeError(`plate out of range: ${plate}`);
    await initWeights();
    for (const { row, layer } of generationOrder()) {
      if (options.cancelled?.()) return;
      const weights = generateLayer(plate, row, layer);
      emit({ type: "layer", plate, weights }, transferList(weights));
      if (options.pause) await options.pause();
    }
    if (options.cancelled?.()) return;
    emit({ type: "done", plate }, []);
  } catch (err) {
    emit({ type: "error", plate, message: err instanceof Error ? err.message : String(err) }, []);
  }
}
