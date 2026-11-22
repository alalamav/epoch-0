// Contracts between the modules of Epoch 0. Types only (no runtime code).
// Every module is written against this file. Do not change it without telling the coordinator.

// ───────────────────────────── weights ─────────────────────────────

/** Role of a weight array, used when deriving its seed. */
export type WeightRole = "kernel" | "recurrent" | "down" | "out";

/**
 * The weights of one layer of one row (draw), in the order the engine reads them.
 * Weights are shared along a row: well (row, L layers) uses layers 0 .. L-1 of that row.
 * All biases are zero by TF.js default and are not carried.
 */
export interface LayerWeights {
  row: number;
  /** 0 is the fastest layer. */
  layer: number;
  /** Recurrent weights, N*N, indexed [receiver * N + sender] (transposed from TF.js). */
  R: Float32Array;
  /**
   * Input weights, in TF.js kernel order.
   * Layer 0: length N (input size 1), indexed [unit].
   * Layer i > 0: length CLOCKS[i] * N, indexed [position * N + unit]; position 0 is the oldest
   * sample of the frame of CLOCKS[i] output samples the layer reads when its clock ticks.
   */
  K: Float32Array;
  /**
   * Layer i > 0 only: the dense layer that passes this layer's state down to layer i - 1.
   * N*N, indexed [receiver in layer i-1 * N + sender in layer i] (transposed from TF.js).
   * null for layer 0.
   */
  C: Float32Array | null;
  /** Layer 0 only: output dense kernel, length N, indexed [unit]. null for other layers. */
  W: Float32Array | null;
}

/** Main thread -> weights worker. A new worker is created for each plate; the old one is terminated. */
export type ToWeightsWorker = { type: "generate"; plate: number };

/**
 * Weights worker -> main thread.
 * Order: layers 0..3 of every row first (rows A..F), then layers 4..7, layer by layer, row by row.
 * The typed arrays of a "layer" message are transferred.
 */
export type FromWeightsWorker =
  | { type: "layer"; plate: number; weights: LayerWeights }
  | { type: "done"; plate: number }
  | { type: "error"; plate: number; message: string };

// ───────────────────────────── engine ─────────────────────────────

/** Main thread -> engine (AudioWorkletNode.port). `well` is 0..47, row-major (see constants.ts). */
export type ToEngine =
  /** The typed arrays are transferred. */
  | { type: "weights"; weights: LayerWeights }
  /** New plate: stop everything and drop all weights. */
  | { type: "clear" }
  /** Button down. A resting well starts from zero state. Input is 1.0 until "release". Ignored if the well's layers are not all loaded. */
  | { type: "press"; well: number }
  /** Button up. Input returns to 0. The well keeps sounding. */
  | { type: "release"; well: number }
  /** The well stops and its state returns to zero. */
  | { type: "stop"; well: number }
  | { type: "stopAll" }
  | { type: "config"; maxVoices?: number; reportHz?: number };

export interface WellReport {
  well: number;
  /** RMS of this well's own output (after the fixed gain, before high-pass and limiter) since the last report. */
  level: number;
  /** Last WAVE_LEN samples of this well's own output (after the fixed gain), oldest first. */
  wave: Float32Array;
  /**
   * One display value per unit, indexed [layer * N + unit], each in -1..1.
   * Layers that update faster than the report rate give their mean since the last report;
   * slower layers give their current state.
   */
  units: Float32Array;
  /** Bit i is set if layer i updated since the last report. Only layers >= TICK_FROM_LAYER are flagged. */
  ticks: number;
  /** True while the button input is 1.0. */
  held: boolean;
}

/** Engine -> main thread, REPORT_HZ times a second while the context runs. Only sounding wells are listed. */
export interface EngineReport {
  type: "report";
  /** Count of 16 kHz samples the engine has produced. */
  time: number;
  wells: WellReport[];
  /** Lowest limiter gain since the last report (1 = no reduction). */
  gain: number;
  /** Peak of the final output since the last report. */
  peak: number;
}

export type FromEngine = EngineReport;

/** src/audio/graph.ts */
export interface AudioGraph {
  readonly context: AudioContext;
  /** On the final output; feeds the spectrogram in the dish view. */
  readonly analyser: AnalyserNode;
  /** True if the context runs at 16 kHz; false if the engine converts inside the worklet. */
  readonly native16k: boolean;
  post(msg: ToEngine, transfer?: Transferable[]): void;
  /** Set by the caller. */
  onReport: ((report: EngineReport) => void) | null;
  /** Call inside the first user gesture. */
  resume(): Promise<void>;
  close(): Promise<void>;
}

// ───────────────────────────── view ─────────────────────────────

/** The small matrices of a row kept on the main thread for drawing. Index = layer; grows as layers load. */
export interface RowMatrices {
  /** R[i]: recurrent weights of layer i, [receiver * N + sender]. */
  R: Float32Array[];
  /** C[i] for i >= 1: layer i -> layer i-1, [receiver * N + sender]. C[0] is null. */
  C: (Float32Array | null)[];
}

/**
 * Where a row's units sit, from the drawing rule (plan section 6, steps 1-4).
 * Computed from the inside out, so adding a layer never moves the rings already placed.
 */
export interface RowLayout {
  /** Number of layers laid out so far. */
  layers: number;
  /** slots[layer][unit] = slot 0..15 of that unit on its ring. Layer 0 is the identity. */
  slots: Uint8Array[];
  /**
   * links[i] (i >= 1) joins ring i-1 (inner) to ring i (outer): for each inner unit u,
   * sender[u] is its strongest sender in ring i and weight[u] the signed weight. links[0] is null.
   */
  links: ({ sender: Uint8Array; weight: Float32Array } | null)[];
}

export interface Theme {
  paper: string;
  ink: string;
  hairline: string;
  stain: string;
}

/** Circle of a well in CSS pixels, relative to the top-left of the plate canvas. */
export interface WellGeometry {
  cx: number;
  cy: number;
  r: number;
}

/** src/view/scene.ts — draws all 48 wells into one canvas that sits behind the button grid. */
export interface PlateScene {
  /** Canvas CSS size, device pixel ratio and the circle of each of the 48 wells (measured from the buttons). */
  resize(width: number, height: number, dpr: number, wells: WellGeometry[]): void;
  /** Call again whenever more layers of the row have loaded. A well is drawn once all its layers are laid out. */
  setRowLayout(row: number, layout: RowLayout): void;
  /** New plate: forget all layouts and states. */
  clear(): void;
  /** Stain the rim of a well while its button is down. */
  setPressed(well: number, down: boolean): void;
  /** Latest engine report. Wells absent from it are at rest (all states zero). */
  applyReport(report: EngineReport): void;
  setTheme(theme: Theme): void;
  /** Reduced motion: no tick pulses, dots update four times a second. */
  setReducedMotion(reduced: boolean): void;
  /** Draw one frame. `now` is performance.now(). Cheap when nothing changed. */
  render(now: number): void;
  dispose(): void;
}

/** src/view/dish.ts — the single-well view with all weights and a spectrogram. */
export interface DishView {
  /** Show `well`. The layout and matrices are those of the well's row. */
  open(well: number, layout: RowLayout, matrices: RowMatrices): void;
  close(): void;
  readonly openWell: number | null;
  applyReport(report: EngineReport): void;
  /** Source for the spectrogram (final output). null until audio exists. */
  setAnalyser(analyser: AnalyserNode | null): void;
  setTheme(theme: Theme): void;
  setReducedMotion(reduced: boolean): void;
  /** Call when the container changes size. */
  resize(): void;
  render(now: number): void;
  dispose(): void;
}
