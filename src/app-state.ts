// Pure state for the main thread: which wells are loaded, which are sounding, and the status line.
// No DOM, no audio; everything here is covered by tests/ui-state.test.ts.

import { COLS, MAX_LAYERS, ROWS, WELLS, wellLayers, wellRow } from "./plate/constants";

export type WellState = "loading" | "resting" | "sounding" | "held";

export function wellState(ready: boolean, held: boolean, sounding: boolean): WellState {
  if (!ready) return "loading";
  if (held) return "held";
  return sounding ? "sounding" : "resting";
}

// ───────────────────────────── readiness ─────────────────────────────

export interface Readiness {
  /** Record that a layer of a row has arrived. Returns the wells that became ready because of it. */
  addLayer(row: number, layer: number): number[];
  /** Number of layers of the row loaded without a gap, counted from layer 0. */
  layers(row: number): number;
  /** True once all of the well's layers have arrived. */
  isReady(well: number): boolean;
  /** Layers arrived so far, of ROWS * MAX_LAYERS. */
  count(): number;
  complete(): boolean;
  reset(): void;
}

export function createReadiness(): Readiness {
  const have = new Uint8Array(ROWS * MAX_LAYERS);
  const prefix = new Uint8Array(ROWS);
  let count = 0;
  return {
    addLayer(row, layer) {
      if (!(row >= 0 && row < ROWS && layer >= 0 && layer < MAX_LAYERS)) return [];
      if (have[row * MAX_LAYERS + layer]) return [];
      have[row * MAX_LAYERS + layer] = 1;
      count++;
      const before = prefix[row];
      let n = before;
      while (n < MAX_LAYERS && have[row * MAX_LAYERS + n]) n++;
      prefix[row] = n;
      const wells: number[] = [];
      // A well with L layers sits in column L - 1.
      for (let col = before; col < Math.min(n, COLS); col++) wells.push(row * COLS + col);
      return wells;
    },
    layers: (row) => prefix[row] ?? 0,
    isReady: (well) => well >= 0 && well < WELLS && prefix[wellRow(well)] >= wellLayers(well),
    count: () => count,
    complete: () => count === ROWS * MAX_LAYERS,
    reset() {
      have.fill(0);
      prefix.fill(0);
      count = 0;
    },
  };
}

// ───────────────────────────── sounding ─────────────────────────────

export interface SoundingTracker {
  /** Button down: the well counts as sounding from now, before any report lists it. */
  press(well: number, now: number): void;
  stop(well: number, now: number): void;
  stopAll(now: number): void;
  /** Wells listed in an engine report. Returns true if the sounding set changed. */
  report(wells: readonly number[], now: number): boolean;
  isSounding(well: number): boolean;
  sounding(): number[];
  /** New plate: forget everything. */
  clear(): void;
}

export interface SoundingOptions {
  /** A pressed well absent from reports for this long was refused by the engine. */
  pendingMs?: number;
  /** After a stop, reports already on their way may still list the well for this long. */
  stopGraceMs?: number;
}

/**
 * The "sounding" set: wells in the latest engine report, plus wells pressed but not yet reported,
 * minus wells just stopped whose stop the engine has not yet acknowledged.
 * `now` is any millisecond clock (performance.now()).
 */
export function createSoundingTracker(options: SoundingOptions = {}): SoundingTracker {
  const pendingMs = options.pendingMs ?? 500;
  const stopGraceMs = options.stopGraceMs ?? 250;
  let reported = new Set<number>();
  const pending = new Map<number, number>();
  const stopping = new Map<number, number>();

  const isSounding = (well: number) => reported.has(well) || pending.has(well);

  function stop(well: number, now: number) {
    pending.delete(well);
    if (reported.delete(well)) stopping.set(well, now);
    else stopping.set(well, now);
  }

  return {
    press(well, now) {
      stopping.delete(well);
      if (!reported.has(well)) pending.set(well, now);
    },
    stop,
    stopAll(now) {
      for (const well of [...reported, ...pending.keys()]) stop(well, now);
    },
    report(wells, now) {
      const next = new Set<number>();
      const listed = new Set(wells);
      for (const well of listed) {
        const stoppedAt = stopping.get(well);
        if (stoppedAt !== undefined && now - stoppedAt < stopGraceMs) continue;
        next.add(well);
      }
      for (const [well, at] of stopping) {
        if (!listed.has(well) || now - at >= stopGraceMs) stopping.delete(well);
      }
      let changed = next.size !== reported.size;
      if (!changed) for (const well of next) if (!reported.has(well)) changed = true;
      // A well moving from pending to reported stays sounding; a pending well that expires changes.
      for (const [well, at] of pending) {
        if (next.has(well)) pending.delete(well);
        else if (now - at > pendingMs) {
          pending.delete(well);
          changed = true;
        }
      }
      reported = next;
      return changed;
    },
    isSounding,
    sounding: () => [...new Set([...reported, ...pending.keys()])].sort((a, b) => a - b),
    clear() {
      reported = new Set();
      pending.clear();
      stopping.clear();
    },
  };
}

// ───────────────────────────── status line ─────────────────────────────

export interface StatusInput {
  /** Worker, audio or scene failure. Shown in preference to everything else. */
  error: string | null;
  /** The audio context stopped running after the first press (iOS interruption, tab policy). */
  audioPaused: boolean;
  /** Layers loaded so far and in total. */
  loaded: number;
  total: number;
}

export interface Status {
  kind: "idle" | "loading" | "hint" | "error";
  text: string;
}

export function statusLine(input: StatusInput): Status {
  if (input.error) return { kind: "error", text: input.error };
  if (input.audioPaused) return { kind: "hint", text: "audio paused · tap to resume" };
  if (input.loaded < input.total) {
    return { kind: "loading", text: `generating weights · ${input.loaded} of ${input.total} layers` };
  }
  return { kind: "idle", text: "" };
}
