import { describe, expect, it } from "vitest";
import { createReadiness, createSoundingTracker, statusLine, wellState } from "../src/app-state";
import { COLS, MAX_LAYERS, ROWS, WELLS, wellIndex } from "../src/plate/constants";
import { gridPlacement, measureWells, moveWell } from "../src/ui/grid-nav";
import { needsRewrite, plateFromSearch, plateHref, resolvePlate } from "../src/ui/plate-url";

/** Stand-in for src/plate/seed.ts parsePlate: one to six digits. */
const parse = (text: string | null): number | null => (text !== null && /^\d{1,6}$/.test(text) ? Number(text) : null);

describe("plate number in the link", () => {
  it("reads ?plate=", () => {
    expect(plateFromSearch("?plate=048213", parse)).toBe(48213);
    expect(plateFromSearch("?a=1&plate=7", parse)).toBe(7);
    expect(plateFromSearch("", parse)).toBeNull();
    expect(plateFromSearch("?plate=", parse)).toBeNull();
    expect(plateFromSearch("?plate=abc", parse)).toBeNull();
    expect(plateFromSearch("?plate=1234567", parse)).toBeNull();
    expect(plateFromSearch("?plates=12", parse)).toBeNull();
  });

  it("writes the plate zero-padded and keeps the rest of the link", () => {
    expect(plateHref("https://example.org/epoch/", 48213)).toBe("https://example.org/epoch/?plate=048213");
    expect(plateHref("https://example.org/epoch/?plate=1&x=2#about", 999999)).toBe(
      "https://example.org/epoch/?plate=999999&x=2#about",
    );
    expect(plateHref("http://localhost:5173/index.html?x=2", 0)).toBe("http://localhost:5173/index.html?x=2&plate=000000");
  });

  it("round-trips", () => {
    for (const plate of [0, 5, 48213, 999999]) {
      expect(plateFromSearch(new URL(plateHref("https://e.org/", plate)).search, parse)).toBe(plate);
    }
  });

  it("draws a plate when the link has none or an invalid one", () => {
    expect(resolvePlate("?plate=000042", parse, () => 777)).toEqual({ plate: 42, fromLink: true });
    expect(resolvePlate("", parse, () => 777)).toEqual({ plate: 777, fromLink: false });
    expect(resolvePlate("?plate=nope", parse, () => 777)).toEqual({ plate: 777, fromLink: false });
  });

  it("knows when the link needs rewriting to the six-digit form", () => {
    expect(needsRewrite("?plate=048213", 48213)).toBe(false);
    expect(needsRewrite("?plate=48213", 48213)).toBe(true);
    expect(needsRewrite("", 48213)).toBe(true);
    expect(needsRewrite("?plate=000001", 2)).toBe(true);
  });
});

describe("grid transposition", () => {
  it("places wells row by row when wide", () => {
    expect(gridPlacement(0, 0, false)).toEqual({ gridRow: 2, gridColumn: 2 }); // A1
    expect(gridPlacement(5, 5, false)).toEqual({ gridRow: 7, gridColumn: 7 }); // F6
    expect(gridPlacement(0, 7, false)).toEqual({ gridRow: 2, gridColumn: 9 }); // A8
    expect(gridPlacement(-1, 3, false)).toEqual({ gridRow: 1, gridColumn: 5 }); // column number 4
    expect(gridPlacement(2, -1, false)).toEqual({ gridRow: 4, gridColumn: 1 }); // row letter C
  });

  it("turns rows into columns when transposed, with A1 still top left", () => {
    expect(gridPlacement(0, 0, true)).toEqual({ gridRow: 2, gridColumn: 2 }); // A1
    expect(gridPlacement(5, 5, true)).toEqual({ gridRow: 7, gridColumn: 7 }); // F6
    expect(gridPlacement(0, 7, true)).toEqual({ gridRow: 9, gridColumn: 2 }); // A8: bottom of first column
    expect(gridPlacement(5, 0, true)).toEqual({ gridRow: 2, gridColumn: 7 }); // F1: top right
    expect(gridPlacement(-1, 3, true)).toEqual({ gridRow: 5, gridColumn: 1 }); // layer number 4 down the side
    expect(gridPlacement(2, -1, true)).toEqual({ gridRow: 1, gridColumn: 4 }); // letter C along the top
  });

  it("fills 8 by 6 wide and 6 by 8 transposed, each cell once", () => {
    for (const transposed of [false, true]) {
      const seen = new Set<string>();
      let maxRow = 0;
      let maxCol = 0;
      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
          const p = gridPlacement(row, col, transposed);
          seen.add(`${p.gridRow}/${p.gridColumn}`);
          maxRow = Math.max(maxRow, p.gridRow);
          maxCol = Math.max(maxCol, p.gridColumn);
        }
      }
      expect(seen.size).toBe(WELLS);
      expect([maxRow - 1, maxCol - 1]).toEqual(transposed ? [8, 6] : [6, 8]);
    }
  });

  it("moves with the arrow keys as seen on screen", () => {
    const B3 = wellIndex(1, 2);
    expect(moveWell(B3, { key: "ArrowRight" }, false)).toBe(wellIndex(1, 3));
    expect(moveWell(B3, { key: "ArrowLeft" }, false)).toBe(wellIndex(1, 1));
    expect(moveWell(B3, { key: "ArrowDown" }, false)).toBe(wellIndex(2, 2));
    expect(moveWell(B3, { key: "ArrowUp" }, false)).toBe(wellIndex(0, 2));
    // transposed: right is the next draw, down is one more layer
    expect(moveWell(B3, { key: "ArrowRight" }, true)).toBe(wellIndex(2, 2));
    expect(moveWell(B3, { key: "ArrowLeft" }, true)).toBe(wellIndex(0, 2));
    expect(moveWell(B3, { key: "ArrowDown" }, true)).toBe(wellIndex(1, 3));
    expect(moveWell(B3, { key: "ArrowUp" }, true)).toBe(wellIndex(1, 1));
  });

  it("agrees with the placement: an arrow always lands on the neighbouring grid cell", () => {
    const deltas: Record<string, [number, number]> = {
      ArrowRight: [0, 1],
      ArrowLeft: [0, -1],
      ArrowDown: [1, 0],
      ArrowUp: [-1, 0],
    };
    for (const transposed of [false, true]) {
      for (let well = 0; well < WELLS; well++) {
        const from = gridPlacement(Math.floor(well / COLS), well % COLS, transposed);
        for (const [key, [dRow, dCol]] of Object.entries(deltas)) {
          const to = moveWell(well, { key }, transposed)!;
          const p = gridPlacement(Math.floor(to / COLS), to % COLS, transposed);
          if (to === well) continue; // edge
          expect([p.gridRow - from.gridRow, p.gridColumn - from.gridColumn]).toEqual([dRow, dCol]);
        }
      }
    }
  });

  it("stops at the edges and handles Home and End", () => {
    expect(moveWell(0, { key: "ArrowLeft" }, false)).toBe(0);
    expect(moveWell(0, { key: "ArrowUp" }, true)).toBe(0);
    expect(moveWell(WELLS - 1, { key: "ArrowRight" }, false)).toBe(WELLS - 1);
    expect(moveWell(WELLS - 1, { key: "ArrowDown" }, true)).toBe(WELLS - 1);
    const C4 = wellIndex(2, 3);
    expect(moveWell(C4, { key: "Home" }, false)).toBe(wellIndex(2, 0));
    expect(moveWell(C4, { key: "End" }, false)).toBe(wellIndex(2, 7));
    expect(moveWell(C4, { key: "Home" }, true)).toBe(wellIndex(0, 3));
    expect(moveWell(C4, { key: "End" }, true)).toBe(wellIndex(5, 3));
    expect(moveWell(C4, { key: "Home", ctrlKey: true }, false)).toBe(0);
    expect(moveWell(C4, { key: "End", ctrlKey: true }, true)).toBe(WELLS - 1);
    expect(moveWell(C4, { key: "a" }, false)).toBeNull();
    expect(moveWell(C4, { key: "Tab" }, false)).toBeNull();
  });

  it("measures button circles relative to the canvas", () => {
    const wells = measureWells({ left: 10, top: 20, width: 500, height: 400 }, [
      { left: 30, top: 50, width: 44, height: 44 },
      { left: 100, top: 50, width: 60, height: 50 },
    ]);
    expect(wells).toEqual([
      { cx: 42, cy: 52, r: 22 },
      { cx: 120, cy: 55, r: 25 },
    ]);
  });
});

describe("readiness", () => {
  it("makes a well ready when all its layers have arrived", () => {
    const r = createReadiness();
    expect(r.isReady(0)).toBe(false);
    expect(r.addLayer(0, 0)).toEqual([wellIndex(0, 0)]);
    expect(r.isReady(wellIndex(0, 0))).toBe(true);
    expect(r.isReady(wellIndex(0, 1))).toBe(false);
    expect(r.isReady(wellIndex(1, 0))).toBe(false);
    expect(r.addLayer(0, 1)).toEqual([wellIndex(0, 1)]);
    expect(r.layers(0)).toBe(2);
  });

  it("follows the worker's order: layers 0-3 of every row, then the slow layers", () => {
    const r = createReadiness();
    for (let row = 0; row < ROWS; row++) for (let layer = 0; layer < 4; layer++) r.addLayer(row, layer);
    for (let well = 0; well < WELLS; well++) expect(r.isReady(well)).toBe(well % COLS < 4);
    expect(r.count()).toBe(24);
    expect(r.complete()).toBe(false);
    for (let layer = 4; layer < MAX_LAYERS; layer++) for (let row = 0; row < ROWS; row++) r.addLayer(row, layer);
    for (let well = 0; well < WELLS; well++) expect(r.isReady(well)).toBe(true);
    expect(r.complete()).toBe(true);
  });

  it("waits for gaps, ignores repeats and nonsense, and resets", () => {
    const r = createReadiness();
    expect(r.addLayer(2, 1)).toEqual([]);
    expect(r.isReady(wellIndex(2, 1))).toBe(false);
    expect(r.addLayer(2, 0)).toEqual([wellIndex(2, 0), wellIndex(2, 1)]);
    expect(r.addLayer(2, 0)).toEqual([]);
    expect(r.addLayer(9, 0)).toEqual([]);
    expect(r.addLayer(0, 8)).toEqual([]);
    expect(r.count()).toBe(2);
    r.reset();
    expect(r.count()).toBe(0);
    expect(r.isReady(wellIndex(2, 0))).toBe(false);
  });
});

describe("sounding set", () => {
  it("counts a pressed well as sounding before any report lists it", () => {
    const t = createSoundingTracker();
    expect(t.isSounding(5)).toBe(false);
    t.press(5, 0);
    expect(t.isSounding(5)).toBe(true);
    t.report([], 20); // a report already on its way
    expect(t.isSounding(5)).toBe(true);
    t.report([5], 40);
    expect(t.isSounding(5)).toBe(true);
    expect(t.sounding()).toEqual([5]);
  });

  it("stays sounding while the audio context is suspended (no reports)", () => {
    const t = createSoundingTracker();
    t.press(5, 0);
    expect(t.isSounding(5)).toBe(true);
    t.report([5], 60_000);
    expect(t.isSounding(5)).toBe(true);
  });

  it("drops a pressed well the engine never reports", () => {
    const t = createSoundingTracker({ pendingMs: 500 });
    t.press(5, 0);
    expect(t.report([], 400)).toBe(false);
    expect(t.isSounding(5)).toBe(true);
    expect(t.report([], 600)).toBe(true);
    expect(t.isSounding(5)).toBe(false);
  });

  it("follows the reports", () => {
    const t = createSoundingTracker();
    expect(t.report([1, 2], 0)).toBe(true);
    expect(t.report([1, 2], 33)).toBe(false);
    expect(t.sounding()).toEqual([1, 2]);
    expect(t.report([2], 66)).toBe(true);
    expect(t.isSounding(1)).toBe(false);
  });

  it("stops at once and ignores a late report that still lists the well", () => {
    const t = createSoundingTracker({ stopGraceMs: 250 });
    t.report([3], 0);
    t.stop(3, 10);
    expect(t.isSounding(3)).toBe(false);
    t.report([3], 20); // posted before the engine saw the stop
    expect(t.isSounding(3)).toBe(false);
    t.report([], 50);
    expect(t.isSounding(3)).toBe(false);
    // pressed again: a report listing it counts again
    t.press(3, 60);
    expect(t.isSounding(3)).toBe(true);
    t.report([3], 90);
    expect(t.isSounding(3)).toBe(true);
  });

  it("believes the engine again after the grace time", () => {
    const t = createSoundingTracker({ stopGraceMs: 250 });
    t.report([3], 0);
    t.stop(3, 10);
    t.report([3], 300);
    expect(t.isSounding(3)).toBe(true);
  });

  it("stops all and clears", () => {
    const t = createSoundingTracker();
    t.report([1, 2], 0);
    t.press(7, 5);
    t.stopAll(10);
    expect(t.sounding()).toEqual([]);
    t.report([1, 2, 7], 20);
    expect(t.sounding()).toEqual([]);
    t.press(4, 30);
    t.clear();
    expect(t.sounding()).toEqual([]);
  });
});

describe("well state and status line", () => {
  it("derives the well state", () => {
    expect(wellState(false, false, false)).toBe("loading");
    expect(wellState(false, true, true)).toBe("loading");
    expect(wellState(true, false, false)).toBe("resting");
    expect(wellState(true, false, true)).toBe("sounding");
    expect(wellState(true, true, true)).toBe("held");
    expect(wellState(true, true, false)).toBe("held");
  });

  it("shows errors first, then the audio hint, then loading", () => {
    expect(statusLine({ error: "weights failed: x", audioPaused: true, loaded: 3, total: 48 })).toEqual({
      kind: "error",
      text: "weights failed: x",
    });
    expect(statusLine({ error: null, audioPaused: true, loaded: 3, total: 48 }).kind).toBe("hint");
    expect(statusLine({ error: null, audioPaused: false, loaded: 3, total: 48 })).toEqual({
      kind: "loading",
      text: "generating weights · 3 of 48 layers",
    });
    expect(statusLine({ error: null, audioPaused: false, loaded: 48, total: 48 })).toEqual({ kind: "idle", text: "" });
  });
});
