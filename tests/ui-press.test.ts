import { beforeEach, describe, expect, it } from "vitest";
import { TAP_MS } from "../src/plate/constants";
import { createPressLogic, type PressLogic } from "../src/ui/press";

/** A stand-in for main.ts: a pressed well counts as sounding at once; a stopped one rests. */
function harness() {
  const log: string[] = [];
  const sounding = new Set<number>();
  const notReady = new Set<number>();
  const touched: number[] = [];
  const logic: PressLogic = createPressLogic({
    isReady: (well) => !notReady.has(well),
    isSounding: (well) => sounding.has(well),
    onTouch: (well) => touched.push(well),
    onPress(well) {
      sounding.add(well);
      log.push(`press ${well}`);
    },
    onRelease(well) {
      log.push(`release ${well}`);
    },
    onStop(well) {
      sounding.delete(well);
      log.push(`stop ${well}`);
    },
  });
  return { log, sounding, notReady, touched, logic };
}

describe("press logic", () => {
  let h: ReturnType<typeof harness>;
  beforeEach(() => {
    h = harness();
  });

  it("starts a resting well on press and never stops it on release, however short", () => {
    expect(h.logic.down("p1", 5, 1000)).toBe(true);
    expect(h.log).toEqual(["press 5"]);
    expect(h.logic.isHeld(5)).toBe(true);
    h.logic.up("p1", 1010);
    expect(h.log).toEqual(["press 5", "release 5"]);
    expect(h.logic.isHeld(5)).toBe(false);
    expect(h.sounding.has(5)).toBe(true);
  });

  it("stops a sounding well on a tap", () => {
    h.logic.down("p1", 5, 0);
    h.logic.up("p1", 50);
    h.log.length = 0;
    h.logic.down("p1", 5, 2000);
    h.logic.up("p1", 2000 + TAP_MS - 1);
    expect(h.log).toEqual(["press 5", "release 5", "stop 5"]);
    expect(h.sounding.has(5)).toBe(false);
  });

  it("keeps a sounding well going when it is held, with the input on meanwhile", () => {
    h.sounding.add(5);
    h.logic.down("p1", 5, 2000);
    expect(h.log).toEqual(["press 5"]);
    h.logic.up("p1", 2000 + TAP_MS);
    expect(h.log).toEqual(["press 5", "release 5"]);
    expect(h.sounding.has(5)).toBe(true);
  });

  it("alternates start, stop, start on repeated taps", () => {
    for (let i = 0; i < 3; i++) {
      h.logic.down("p1", 9, i * 1000);
      h.logic.up("p1", i * 1000 + 40);
    }
    expect(h.log).toEqual([
      "press 9",
      "release 9",
      "press 9",
      "release 9",
      "stop 9",
      "press 9",
      "release 9",
    ]);
  });

  it("handles several fingers on different wells independently", () => {
    h.sounding.add(2);
    h.logic.down("p1", 1, 0);
    h.logic.down("p2", 2, 10);
    h.logic.down("p3", 3, 20);
    expect(h.logic.heldWells().sort()).toEqual([1, 2, 3]);
    h.logic.up("p2", 60); // tap on a sounding well
    h.logic.up("p1", 5000);
    h.logic.up("p3", 30);
    expect(h.log).toEqual(["press 1", "press 2", "press 3", "release 2", "stop 2", "release 1", "release 3"]);
    expect(h.logic.heldWells()).toEqual([]);
  });

  it("presses a well once for two sources and releases it with the last", () => {
    h.logic.down("p1", 4, 0);
    h.logic.down("kSpace", 4, 100);
    expect(h.log).toEqual(["press 4"]);
    h.logic.up("p1", 150);
    expect(h.log).toEqual(["press 4"]);
    expect(h.logic.isHeld(4)).toBe(true);
    h.logic.up("kSpace", 200);
    expect(h.log).toEqual(["press 4", "release 4"]);
  });

  it("releases but never stops on cancel", () => {
    h.sounding.add(7);
    h.logic.down("p1", 7, 0);
    h.logic.cancel("p1");
    expect(h.log).toEqual(["press 7", "release 7"]);
    expect(h.sounding.has(7)).toBe(true);
    // and the source is free again
    h.logic.up("p1", 10);
    expect(h.log).toEqual(["press 7", "release 7"]);
  });

  it("ignores wells whose weights are not loaded, but reports the touch", () => {
    h.notReady.add(40);
    expect(h.logic.down("p1", 40, 0)).toBe(false);
    h.logic.up("p1", 20);
    expect(h.log).toEqual([]);
    expect(h.logic.isHeld(40)).toBe(false);
    expect(h.touched).toEqual([40]);
    h.notReady.delete(40);
    expect(h.logic.down("p1", 40, 100)).toBe(true);
    expect(h.log).toEqual(["press 40"]);
  });

  it("ignores key repeat: a second down from the same source changes nothing", () => {
    h.sounding.add(3);
    h.logic.down("kSpace", 3, 0);
    h.logic.down("kSpace", 3, 500);
    h.logic.down("kSpace", 3, 530);
    expect(h.log).toEqual(["press 3"]);
    h.logic.up("kSpace", 560); // held 560 ms since the first down: not a tap
    expect(h.log).toEqual(["press 3", "release 3"]);
  });

  it("moves a source that goes down elsewhere without an up", () => {
    h.logic.down("p1", 1, 0);
    h.logic.down("p1", 2, 10);
    expect(h.log).toEqual(["press 1", "release 1", "press 2"]);
  });

  it("ignores an up or cancel from an unknown source", () => {
    h.logic.up("nobody", 0);
    h.logic.cancel("nobody");
    expect(h.log).toEqual([]);
  });

  it("releaseAll returns every input to zero and stops nothing", () => {
    h.sounding.add(1);
    h.logic.down("p1", 1, 0);
    h.logic.down("p2", 2, 0);
    h.logic.releaseAll();
    expect(h.log).toEqual(["press 1", "press 2", "release 1", "release 2"]);
    expect(h.logic.heldWells()).toEqual([]);
  });

  it("reset forgets holds without calling back", () => {
    h.logic.down("p1", 1, 0);
    h.log.length = 0;
    h.logic.reset();
    expect(h.logic.isHeld(1)).toBe(false);
    h.logic.up("p1", 10);
    expect(h.log).toEqual([]);
  });

  it("honours a custom tap time", () => {
    const log: string[] = [];
    const logic = createPressLogic({
      isReady: () => true,
      isSounding: () => true,
      onPress: () => {},
      onRelease: () => {},
      onStop: (w) => log.push(`stop ${w}`),
      tapMs: 100,
    });
    logic.down("a", 0, 0);
    logic.up("a", 100);
    logic.down("a", 0, 1000);
    logic.up("a", 1099);
    expect(log).toEqual(["stop 0"]);
  });
});
