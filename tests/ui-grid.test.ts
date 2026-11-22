// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COLS, ROWS, WELLS, wellIndex, wellName } from "../src/plate/constants";
import { createControls } from "../src/ui/controls";
import { createPlateGrid, type PlateGrid } from "../src/ui/grid";
import { createSpecimenLabel } from "../src/ui/label";
import { createPressLogic } from "../src/ui/press";

function setup() {
  const log: string[] = [];
  const ready = new Set<number>();
  const sounding = new Set<number>();
  let clock = 0;
  const logic = createPressLogic({
    isReady: (well) => ready.has(well),
    isSounding: (well) => sounding.has(well),
    onPress(well) {
      sounding.add(well);
      log.push(`press ${wellName(well)}`);
    },
    onRelease: (well) => log.push(`release ${wellName(well)}`),
    onStop(well) {
      sounding.delete(well);
      log.push(`stop ${wellName(well)}`);
    },
  });
  const grid = createPlateGrid(document.body, { press: logic, now: () => clock });
  const makeReady = (...wells: number[]) => {
    for (const well of wells) {
      ready.add(well);
      grid.setWellState(well, "resting");
    }
  };
  return { log, ready, sounding, logic, grid, makeReady, tick: (ms: number) => (clock += ms) };
}

const key = (target: Element, type: "keydown" | "keyup", init: KeyboardEventInit) =>
  target.dispatchEvent(new KeyboardEvent(type, { bubbles: true, cancelable: true, ...init }));

function pointer(target: Element, type: string, pointerId = 1, init: PointerEventInit = {}) {
  const Ctor = (window as unknown as { PointerEvent?: typeof PointerEvent }).PointerEvent;
  const pointerType = init.pointerType ?? "touch";
  // jsdom may lack PointerEvent: a MouseEvent carrying the pointer fields stands in.
  const event = Ctor
    ? new Ctor(type, { bubbles: true, cancelable: true, ...init, pointerId, pointerType })
    : Object.assign(new MouseEvent(type, { bubbles: true, cancelable: true, ...init }), { pointerId, pointerType });
  target.dispatchEvent(event);
  return event;
}

describe("plate grid", () => {
  let s: ReturnType<typeof setup>;
  let grid: PlateGrid;
  beforeEach(() => {
    s = setup();
    grid = s.grid;
  });
  afterEach(() => {
    grid.destroy();
    document.body.replaceChildren();
  });

  it("has 48 real buttons in a labelled grid of six rows", () => {
    expect(grid.buttons).toHaveLength(WELLS);
    for (const button of grid.buttons) {
      expect(button.tagName).toBe("BUTTON");
      expect(button.type).toBe("button");
    }
    const gridEl = grid.element.querySelector('[role="grid"]')!;
    expect(gridEl).not.toBeNull();
    const rows = gridEl.querySelectorAll('[role="row"]');
    expect(rows).toHaveLength(ROWS + 1);
    expect(rows[0].querySelectorAll('[role="columnheader"]')).toHaveLength(COLS);
    for (let row = 0; row < ROWS; row++) {
      const r = rows[row + 1];
      expect(r.querySelector('[role="rowheader"]')!.textContent).toBe("ABCDEF"[row]);
      const cells = r.querySelectorAll('[role="gridcell"] > button');
      expect(cells).toHaveLength(COLS);
      // DOM order is row-major whatever the orientation
      cells.forEach((b, col) => expect(b).toBe(grid.buttons[wellIndex(row, col)]));
    }
    grid.setPlate(48213);
    expect(gridEl.getAttribute("aria-label")).toContain("Plate 048213");
    expect(grid.canvas.tagName).toBe("CANVAS");
    expect(grid.canvas.getAttribute("aria-hidden")).toBe("true");
    expect(grid.element.contains(grid.canvas)).toBe(true);
  });

  it("starts with every well loading and disabled for assistive technology", () => {
    for (let well = 0; well < WELLS; well++) {
      const b = grid.buttons[well];
      expect(b.getAttribute("aria-disabled")).toBe("true");
      expect(b.hasAttribute("aria-pressed")).toBe(false);
      expect(b.dataset.state).toBe("loading");
      expect(b.getAttribute("aria-label")).toMatch(new RegExp(`^Well ${wellName(well)}, .+ layers?, loading$`));
    }
    expect(grid.buttons[wellIndex(5, 5)].getAttribute("aria-label")).toBe("Well F6, six layers, loading");
  });

  it("labels wells by state with aria-pressed", () => {
    const F6 = wellIndex(5, 5);
    const b = grid.buttons[F6];
    grid.setWellState(F6, "resting");
    expect(b.getAttribute("aria-label")).toBe("Well F6, six layers, resting");
    expect(b.getAttribute("aria-pressed")).toBe("false");
    expect(b.hasAttribute("aria-disabled")).toBe(false);
    grid.setWellState(F6, "sounding");
    expect(b.getAttribute("aria-label")).toBe("Well F6, six layers, sounding");
    expect(b.getAttribute("aria-pressed")).toBe("true");
    grid.setWellState(F6, "held");
    expect(b.getAttribute("aria-label")).toBe("Well F6, six layers, held");
    expect(b.getAttribute("aria-pressed")).toBe("true");
    expect(grid.wellState(F6)).toBe("held");
    grid.setWellState(F6, "loading");
    expect(b.getAttribute("aria-disabled")).toBe("true");
  });

  it("does not press a well that is still loading, by key, pointer or click", () => {
    const b = grid.buttons[3];
    b.focus();
    key(b, "keydown", { key: " ", code: "Space" });
    key(b, "keyup", { key: " ", code: "Space" });
    pointer(b, "pointerdown");
    pointer(b, "pointerup");
    s.tick(1000);
    b.click();
    expect(s.log).toEqual([]);
    expect(s.logic.heldWells()).toEqual([]);
  });

  it("has one tab stop and moves it with the arrow keys", () => {
    const tabbable = () => grid.buttons.filter((b) => b.tabIndex === 0);
    expect(tabbable()).toEqual([grid.buttons[0]]);
    grid.buttons[0].focus();
    key(document.activeElement!, "keydown", { key: "ArrowRight" });
    expect(document.activeElement).toBe(grid.buttons[wellIndex(0, 1)]);
    key(document.activeElement!, "keydown", { key: "ArrowDown" });
    expect(document.activeElement).toBe(grid.buttons[wellIndex(1, 1)]);
    key(document.activeElement!, "keydown", { key: "End" });
    expect(document.activeElement).toBe(grid.buttons[wellIndex(1, 7)]);
    key(document.activeElement!, "keydown", { key: "ArrowRight" }); // edge
    expect(document.activeElement).toBe(grid.buttons[wellIndex(1, 7)]);
    key(document.activeElement!, "keydown", { key: "ArrowUp" });
    key(document.activeElement!, "keydown", { key: "ArrowLeft" });
    expect(document.activeElement).toBe(grid.buttons[wellIndex(0, 6)]);
    expect(tabbable()).toEqual([grid.buttons[wellIndex(0, 6)]]);
    key(document.activeElement!, "keydown", { key: "End", ctrlKey: true });
    expect(document.activeElement).toBe(grid.buttons[WELLS - 1]);
  });

  it("crosses loading wells with the arrow keys", () => {
    s.makeReady(0);
    grid.buttons[0].focus();
    key(document.activeElement!, "keydown", { key: "ArrowRight" });
    expect(document.activeElement).toBe(grid.buttons[1]);
    expect(grid.buttons[1].getAttribute("aria-disabled")).toBe("true");
  });

  it("transposes placement and arrow keys on a narrow screen, keeping DOM order", () => {
    const A8 = grid.buttons[wellIndex(0, 7)];
    const F1 = grid.buttons[wellIndex(5, 0)];
    expect(grid.element.dataset.layout).toBe("wide");
    expect([A8.style.gridRow, A8.style.gridColumn]).toEqual(["2", "9"]);
    expect([F1.style.gridRow, F1.style.gridColumn]).toEqual(["7", "2"]);
    const order = grid.buttons.map((b) => b.dataset.name);

    grid.setTransposed(true);
    expect(grid.transposed).toBe(true);
    expect(grid.element.dataset.layout).toBe("tall");
    expect([A8.style.gridRow, A8.style.gridColumn]).toEqual(["9", "2"]);
    expect([F1.style.gridRow, F1.style.gridColumn]).toEqual(["2", "7"]);
    const letters = [...grid.element.querySelectorAll<HTMLElement>('[role="rowheader"]')];
    expect(letters.map((n) => n.style.gridRow)).toEqual(["1", "1", "1", "1", "1", "1"]);
    expect(letters.map((n) => n.style.gridColumn)).toEqual(["2", "3", "4", "5", "6", "7"]);
    expect([...grid.element.querySelectorAll("button")].map((b) => b.dataset.name)).toEqual(order);

    grid.buttons[0].focus();
    key(document.activeElement!, "keydown", { key: "ArrowRight" });
    expect(document.activeElement).toBe(grid.buttons[wellIndex(1, 0)]); // B1 is to the right of A1
    key(document.activeElement!, "keydown", { key: "ArrowDown" });
    expect(document.activeElement).toBe(grid.buttons[wellIndex(1, 1)]); // B2 is below B1
  });

  it("Space and Enter press on keydown and release on keyup, ignoring key repeat", () => {
    s.makeReady(0, 1);
    const b = grid.buttons[0];
    b.focus();
    key(b, "keydown", { key: " ", code: "Space" });
    expect(s.log).toEqual(["press A1"]);
    expect(s.logic.isHeld(0)).toBe(true);
    s.tick(600);
    key(b, "keydown", { key: " ", code: "Space", repeat: true });
    key(b, "keydown", { key: " ", code: "Space", repeat: true });
    expect(s.log).toEqual(["press A1"]);
    key(b, "keyup", { key: " ", code: "Space" });
    expect(s.log).toEqual(["press A1", "release A1"]);

    // now sounding: a short Enter is a tap and stops it
    s.tick(1000);
    key(b, "keydown", { key: "Enter", code: "Enter" });
    s.tick(80);
    key(b, "keyup", { key: "Enter", code: "Enter" });
    expect(s.log).toEqual(["press A1", "release A1", "press A1", "release A1", "stop A1"]);
  });

  it("holding a sounding well with a key keeps it going", () => {
    s.makeReady(0);
    s.sounding.add(0);
    const b = grid.buttons[0];
    b.focus();
    key(b, "keydown", { key: "Enter", code: "Enter" });
    s.tick(900);
    key(b, "keyup", { key: "Enter", code: "Enter" });
    expect(s.log).toEqual(["press A1", "release A1"]);
  });

  it("prevents the default action of press keys (no scroll, no double activation)", () => {
    s.makeReady(0);
    const b = grid.buttons[0];
    const down = new KeyboardEvent("keydown", { key: " ", code: "Space", bubbles: true, cancelable: true });
    b.dispatchEvent(down);
    expect(down.defaultPrevented).toBe(true);
    // the click a browser sends after a key press is swallowed
    b.click();
    key(b, "keyup", { key: " ", code: "Space" });
    b.click();
    expect(s.log).toEqual(["press A1", "release A1"]);
  });

  it("releases a key press when focus leaves the well", () => {
    s.makeReady(0);
    s.sounding.add(0);
    const b = grid.buttons[0];
    b.focus();
    key(b, "keydown", { key: " ", code: "Space" });
    key(b, "keydown", { key: "ArrowRight" });
    expect(document.activeElement).toBe(grid.buttons[1]);
    expect(s.log).toEqual(["press A1", "release A1"]); // cancelled: released, not stopped
    key(b, "keyup", { key: " ", code: "Space" });
    expect(s.log).toEqual(["press A1", "release A1"]);
  });

  it("pointer down presses and focuses, pointer up releases; a tap on a sounding well stops it", () => {
    s.makeReady(10);
    const b = grid.buttons[10];
    const down = pointer(b, "pointerdown");
    expect(down.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(b);
    expect(b.tabIndex).toBe(0);
    expect(s.log).toEqual(["press B3"]);
    s.tick(2000);
    pointer(b, "pointerup");
    expect(s.log).toEqual(["press B3", "release B3"]);
    s.tick(500);
    pointer(b, "pointerdown");
    s.tick(100);
    pointer(b, "pointerup");
    expect(s.log).toEqual(["press B3", "release B3", "press B3", "release B3", "stop B3"]);
  });

  it("handles two pointers on two wells and a cancelled pointer", () => {
    s.makeReady(0, 1);
    s.sounding.add(1);
    pointer(grid.buttons[0], "pointerdown", 1);
    pointer(grid.buttons[1], "pointerdown", 2);
    expect(s.logic.heldWells().sort()).toEqual([0, 1]);
    pointer(grid.buttons[1], "pointercancel", 2);
    pointer(grid.buttons[0], "pointerup", 1);
    expect(s.log).toEqual(["press A1", "press A2", "release A2", "release A1"]);
    expect(s.sounding.has(1)).toBe(true);
  });

  it("ignores other mouse buttons", () => {
    s.makeReady(0);
    pointer(grid.buttons[0], "pointerdown", 1, { button: 2, pointerType: "mouse" });
    expect(s.log).toEqual([]);
  });

  it("treats a bare click (screen reader activation) as a tap", () => {
    s.makeReady(0);
    s.tick(5000);
    grid.buttons[0].click();
    expect(s.log).toEqual(["press A1", "release A1"]);
    s.tick(5000);
    grid.buttons[0].click();
    expect(s.log).toEqual(["press A1", "release A1", "press A1", "release A1", "stop A1"]);
  });

  it("blocks the context menu on the plate", () => {
    const e = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    grid.buttons[5].dispatchEvent(e);
    expect(e.defaultPrevented).toBe(true);
  });

  it("measures 48 circles and reports them to a resize observer hook", () => {
    grid.canvas.getBoundingClientRect = () => ({ left: 10, top: 10, width: 800, height: 600 }) as DOMRect;
    grid.buttons.forEach((b, well) => {
      const row = Math.floor(well / COLS);
      const col = well % COLS;
      b.getBoundingClientRect = () => ({ left: 40 + col * 90, top: 40 + row * 90, width: 80, height: 80 }) as DOMRect;
    });
    const wells = grid.measure();
    expect(wells).toHaveLength(WELLS);
    expect(wells[0]).toEqual({ cx: 70, cy: 70, r: 40 });
    expect(wells[wellIndex(5, 7)]).toEqual({ cx: 70 + 7 * 90, cy: 70 + 5 * 90, r: 40 });

    const callback = vi.fn();
    const stop = grid.observeResize(callback);
    expect(callback).toHaveBeenCalledTimes(1);
    expect(callback.mock.calls[0][3]).toHaveLength(WELLS);
    grid.setTransposed(true); // a change of orientation re-measures
    expect(callback).toHaveBeenCalledTimes(2);
    window.dispatchEvent(new Event("resize"));
    expect(callback).toHaveBeenCalledTimes(3);
    stop();
    window.dispatchEvent(new Event("resize"));
    expect(callback).toHaveBeenCalledTimes(3);
  });
});

describe("controls and label", () => {
  afterEach(() => document.body.replaceChildren());

  function controls() {
    const calls: string[] = [];
    const copied: string[] = [];
    const c = createControls(document.body, {
      onRinse: () => calls.push("rinse"),
      onNewPlate: () => calls.push("new"),
      onMagnify: () => calls.push("magnify"),
      getLink: () => "https://example.org/?plate=048213",
      copyText: async (text) => void copied.push(text),
    });
    c.setPlate(48213);
    const button = (text: string) =>
      [...c.element.querySelectorAll("button")].find((b) => b.textContent!.startsWith(text))!;
    return { c, calls, copied, button };
  }

  it("forwards the buttons and copies the plate link when the plate number is pressed", async () => {
    const { c, calls, copied, button } = controls();
    button("rinse plate").click();
    button("new plate").click();
    button("plate 048213").click();
    await Promise.resolve();
    expect(calls).toEqual(["rinse", "new"]);
    expect(copied).toEqual(["https://example.org/?plate=048213"]);
    expect(button("about")).toBeUndefined();
    expect(button("lens")).toBeUndefined();
    expect(c.element.querySelector("input")).toBeNull();
    expect([...c.element.querySelectorAll("button")].map((b) => b.dataset.ctl)).toEqual([
      "rinse",
      "new",
      "magnify",
      "plate",
    ]);
  });

  it("calls the single-well view 'magnify', whatever well was touched, and enables it once a well can be shown", () => {
    const { c, calls, button } = controls();
    const magnify = button("magnify");
    expect(magnify.textContent).toBe("magnify");
    expect(magnify.disabled).toBe(true);
    c.setMagnifyEnabled(true);
    expect(magnify.disabled).toBe(false);
    magnify.click();
    expect(calls).toEqual(["magnify"]);
    expect(button("dish")).toBeUndefined();
  });

  it("dims rinse while nothing is sounding", () => {
    const { c, button } = controls();
    const rinse = button("rinse");
    expect(rinse.dataset.idle).toBe("true");
    c.setSounding(true);
    expect(rinse.dataset.idle).toBe("false");
    c.setSounding(false);
    expect(rinse.dataset.idle).toBe("true");
    expect(rinse.disabled).toBe(false); // still harmless to press
  });

  it("renders the specimen label line by line", () => {
    const label = createSpecimenLabel(document.body);
    label.set(48213, wellIndex(5, 5));
    const lines = [...label.element.children].map((n) => n.textContent);
    expect(lines).toEqual([
      "PLATE 048213 · WELL F6",
      "layers 6 · units 96 · weights 466,049",
      "clocks 16000 / 1000 / 62.5 / 3.9 / 2.0 / 1.0 per second",
      "training data 0 h · epochs 0",
      "signal: network > 20 Hz high-pass > limiter",
    ]);
    label.set(48213, null);
    expect(label.element.children[0].textContent).toBe("PLATE 048213 · 48 WELLS");
  });
});
