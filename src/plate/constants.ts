// Shared constants for Epoch 0. Plain values and pure helpers only: this file is imported by the
// audio worklet, the weights worker, the main thread and the tests.

/** Units per layer. */
export const N = 16;
export const ROWS = 6;
export const COLS = 8;
export const WELLS = ROWS * COLS;
export const MAX_LAYERS = 8;

/** Clock of each layer in audio samples. Layer 0 is the fastest. All are powers of two. */
export const CLOCKS = [1, 16, 256, 4096, 8192, 16384, 32768, 65536] as const;

/** The networks always run at this rate; the clocks are counted in these samples. */
export const SAMPLE_RATE = 16000;

export const ROW_LETTERS = "ABCDEF";

// Output stage (plan section 4): network sum -> fixed gain -> 20 Hz high-pass -> limiter.
export const OUTPUT_GAIN = 0.25;
export const HIGHPASS_HZ = 20;
export const CEILING = 0.89;
export const LOOKAHEAD_MS = 5;
export const RELEASE_MS = 100;

/** State reports from the engine per second. */
export const REPORT_HZ = 30;
/** Output samples per well in each report (drives the rim). */
export const WAVE_LEN = 256;
/** Layers with index >= this pulse their ring when they update ("the four slowest"). */
export const TICK_FROM_LAYER = 4;

/** A press on a sounding well that is released within this time is a tap: the well stops. */
export const TAP_MS = 300;
/** Cap on wells sounding at once. To be lowered for phones after device tests. */
export const MAX_VOICES = 48;

/** Plate numbers are 0 .. 999999, shown zero-padded to six digits. */
export const PLATE_COUNT = 1_000_000;

// Drawing rule (plan section 6). Well radius is 1.
export const RING_R0 = 0.2;
export const RING_STEP = 0.1;
export const DOT_R0 = 0.01;
export const DOT_GAIN = 0.022;

/** Well index 0..47, row-major. `col` is 0-based; the well has col + 1 layers. */
export const wellIndex = (row: number, col: number): number => row * COLS + col;
export const wellRow = (well: number): number => Math.floor(well / COLS);
export const wellCol = (well: number): number => well % COLS;
export const wellLayers = (well: number): number => (well % COLS) + 1;
/** "F6" */
export const wellName = (well: number): string => ROW_LETTERS[wellRow(well)] + String(wellCol(well) + 1);

/**
 * Number of weights in a well with `layers` layers, as printed on the label.
 * 1 layer: 16 input + 256 recurrent + 16 bias + 16 output + 1 output bias = 305.
 * Each further layer i adds CLOCKS[i]*16 input + 256 recurrent + 16 bias + 256 passed down.
 * Gives 305, 1089, 5713, 71777, 203377, 466049, 990865, 2039969.
 */
export function weightCount(layers: number): number {
  let n = N + N * N + N + N + 1;
  for (let i = 1; i < layers; i++) n += CLOCKS[i] * N + N * N + N + N * N;
  return n;
}

/** Updates per second of layer i. */
export const updatesPerSecond = (layer: number): number => SAMPLE_RATE / CLOCKS[layer];

export const formatPlate = (plate: number): string => String(plate).padStart(6, "0");
