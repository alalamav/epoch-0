// Press logic (plan section 3) as a pure state machine. No DOM: see press-dom.ts for the events.
//
// One rule: the input is 1.0 for as long as the button is down.
//  - down on a well            -> onPress(well)
//  - up / cancel               -> onRelease(well)
//  - up within tapMs on a well that was already sounding before the press -> also onStop(well)
// A press on a resting well always starts it and never stops it on release.
// A cancelled press (pointercancel, lost capture, blur) releases but never stops.
//
// A "source" is one finger, one mouse or one key. Several sources may hold different wells
// (multi-touch) or the same well; a well is pressed once and released when its last source ends.

import { TAP_MS } from "../plate/constants";

export interface PressOptions {
  /** False while the well's weights are not loaded: the press is ignored. */
  isReady(well: number): boolean;
  /** Asked just before onPress, to decide whether a short press is a tap that stops the well. */
  isSounding(well: number): boolean;
  onPress(well: number): void;
  onRelease(well: number): void;
  onStop(well: number): void;
  /** Every down on a well, ready or not (for "last well touched"). */
  onTouch?(well: number): void;
  tapMs?: number;
}

/** What the DOM layer needs. */
export interface PressSink {
  /** Returns true if the well is now held by this source. `time` is in milliseconds. */
  down(source: string, well: number, time: number): boolean;
  up(source: string, time: number): void;
  cancel(source: string): void;
}

export interface PressLogic extends PressSink {
  isHeld(well: number): boolean;
  heldWells(): number[];
  /** Cancel every source (window blur, page hidden): all inputs return to 0, nothing stops. */
  releaseAll(): void;
  /** Forget everything without calling back (new plate: the engine has dropped all wells). */
  reset(): void;
}

interface Hold {
  sources: number;
  startedAt: number;
  wasSounding: boolean;
}

export function createPressLogic(options: PressOptions): PressLogic {
  const tapMs = options.tapMs ?? TAP_MS;
  const sourceWell = new Map<string, number>();
  const holds = new Map<number, Hold>();

  function end(source: string, time: number | null) {
    const well = sourceWell.get(source);
    if (well === undefined) return;
    sourceWell.delete(source);
    const hold = holds.get(well);
    if (!hold) return;
    hold.sources--;
    if (hold.sources > 0) return;
    holds.delete(well);
    options.onRelease(well);
    if (time !== null && hold.wasSounding && time - hold.startedAt < tapMs) options.onStop(well);
  }

  return {
    down(source, well, time) {
      const current = sourceWell.get(source);
      if (current === well) return true; // repeated down from the same source (key repeat)
      if (current !== undefined) end(source, null);
      options.onTouch?.(well);
      if (!options.isReady(well)) return false;
      sourceWell.set(source, well);
      const hold = holds.get(well);
      if (hold) {
        hold.sources++;
        return true;
      }
      holds.set(well, { sources: 1, startedAt: time, wasSounding: options.isSounding(well) });
      options.onPress(well);
      return true;
    },
    up(source, time) {
      end(source, time);
    },
    cancel(source) {
      end(source, null);
    },
    isHeld: (well) => holds.has(well),
    heldWells: () => [...holds.keys()],
    releaseAll() {
      for (const source of [...sourceWell.keys()]) end(source, null);
    },
    reset() {
      sourceWell.clear();
      holds.clear();
    },
  };
}
