// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { wellIndex, wellName } from "../src/plate/constants";
import {
  directionLabel,
  directionOfKey,
  directionText,
  neighbour,
  swipeDirection,
  SWIPE_MAX_MS,
  SWIPE_MIN_PX,
} from "../src/ui/dish-nav";
import { createDishPanel } from "../src/ui/dish-panel";
import type { PressSink } from "../src/ui/press";

const well = (name: string) => wellIndex("ABCDEF".indexOf(name[0]), Number(name.slice(1)) - 1);

describe("neighbour", () => {
  it("walks along the row with left and right: same draw, one layer fewer or more", () => {
    expect(wellName(neighbour(well("F6"), "left")!)).toBe("F5");
    expect(wellName(neighbour(well("F6"), "right")!)).toBe("F7");
  });

  it("walks along the column with up and down: another draw, same layers", () => {
    expect(wellName(neighbour(well("C4"), "up")!)).toBe("B4");
    expect(wellName(neighbour(well("C4"), "down")!)).toBe("D4");
  });

  it("stops at the edges without wrapping", () => {
    expect(neighbour(well("A1"), "left")).toBeNull();
    expect(neighbour(well("A1"), "up")).toBeNull();
    expect(neighbour(well("F8"), "right")).toBeNull();
    expect(neighbour(well("F8"), "down")).toBeNull();
    expect(neighbour(well("B8"), "right")).toBeNull(); // not A-row-next: B8 -> C1 would be a wrap
    expect(neighbour(well("C1"), "left")).toBeNull();
  });

  it("is its own inverse inside the plate", () => {
    for (let w = 0; w < 48; w++) {
      for (const [there, back] of [
        ["left", "right"],
        ["up", "down"],
      ] as const) {
        const n = neighbour(w, there);
        if (n !== null) expect(neighbour(n, back)).toBe(w);
      }
    }
  });
});

describe("keys, swipes and button text", () => {
  it("maps only the four arrow keys", () => {
    expect(directionOfKey("ArrowLeft")).toBe("left");
    expect(directionOfKey("ArrowRight")).toBe("right");
    expect(directionOfKey("ArrowUp")).toBe("up");
    expect(directionOfKey("ArrowDown")).toBe("down");
    expect(directionOfKey("a")).toBeNull();
    expect(directionOfKey("Enter")).toBeNull();
  });

  it("turns pages: dragging left brings the well on the right, dragging up the one below", () => {
    expect(swipeDirection(-80, 5, 200)).toBe("right");
    expect(swipeDirection(80, -5, 200)).toBe("left");
    expect(swipeDirection(4, -90, 200)).toBe("down");
    expect(swipeDirection(-4, 90, 200)).toBe("up");
  });

  it("ignores short, slow and diagonal movements", () => {
    expect(swipeDirection(-(SWIPE_MIN_PX - 1), 0, 100)).toBeNull();
    expect(swipeDirection(-200, 0, SWIPE_MAX_MS + 1)).toBeNull();
    expect(swipeDirection(-60, 55, 100)).toBeNull();
    expect(swipeDirection(0, 0, 0)).toBeNull();
  });

  it("names the target well, or shows a bare arrow at the edge", () => {
    expect(directionText(well("F6"), "left")).toBe("← F5");
    expect(directionText(well("F6"), "right")).toBe("F7 →");
    expect(directionText(well("F6"), "up")).toBe("↑ E6");
    expect(directionText(well("F6"), "down")).toBe("↓");
    expect(directionLabel(well("F6"), "right")).toBe("well F7: same draw, one more layer");
    expect(directionLabel(well("F6"), "down")).toBe("last draw");
  });
});

describe("dish panel navigation", () => {
  afterEach(() => document.body.replaceChildren());

  const sink: PressSink = { down: () => true, up() {}, cancel() {} };

  function panel(canOpen: (w: number) => boolean = () => true) {
    const asked: string[] = [];
    const p = createDishPanel(document.body, {
      press: sink,
      onClose: () => asked.push("close"),
      onResize() {},
      canOpen,
      onNavigate: (w) => {
        asked.push(wellName(w));
        p.open(48213, w); // what main.ts does
      },
    });
    const arrow = (direction: string) =>
      p.dialog.querySelector<HTMLButtonElement>(`.dish-arrow[data-direction="${direction}"]`)!;
    const key = (k: string, init: KeyboardEventInit = {}) =>
      p.dialog.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...init }));
    return { p, asked, arrow, key };
  }

  it("labels the four buttons for the open well and disables the edge", () => {
    const { p, arrow } = panel();
    p.open(48213, well("F6"));
    expect(arrow("left").textContent).toBe("← F5");
    expect(arrow("right").textContent).toBe("F7 →");
    expect(arrow("up").textContent).toBe("↑ E6");
    expect(arrow("down").disabled).toBe(true);
    expect(arrow("right").getAttribute("aria-label")).toBe("well F7: same draw, one more layer");
    expect(p.dialog.querySelector("h2")!.textContent).toBe("dish · well F6");
  });

  it("moves with the buttons and the arrow keys, and stays put at the edge", () => {
    const { p, asked, arrow, key } = panel();
    p.open(48213, well("F6"));
    arrow("right").click();
    expect(p.well).toBe(well("F7"));
    key("ArrowRight");
    expect(p.well).toBe(well("F8"));
    key("ArrowRight"); // edge
    key("ArrowDown"); // edge
    key("ArrowUp");
    key("ArrowLeft");
    expect(asked).toEqual(["F7", "F8", "E8", "E7"]);
    expect(p.dialog.querySelector("h2")!.textContent).toBe("dish · well E7");
  });

  it("leaves arrow keys alone with a modifier or while typing in a field", () => {
    const { p, asked, key } = panel();
    p.open(48213, well("C4"));
    key("ArrowRight", { shiftKey: true });
    key("ArrowRight", { metaKey: true });
    const input = document.createElement("input");
    p.dialog.append(input);
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    expect(asked).toEqual([]);
  });

  it("does not move to a well whose weights are not loaded, until they are", () => {
    let loaded = false;
    const { p, asked, arrow, key } = panel((w) => loaded || w !== well("C5"));
    p.open(48213, well("C4"));
    expect(arrow("right").disabled).toBe(true);
    key("ArrowRight");
    expect(asked).toEqual([]);
    loaded = true;
    p.setMapState(well("C5"), "resting"); // loading -> resting refreshes the buttons
    expect(arrow("right").disabled).toBe(false);
    key("ArrowRight");
    expect(asked).toEqual(["C5"]);
  });

  it("shows every well's state on the map, marks the open one, and jumps on a click", () => {
    const { p, asked } = panel();
    const cell = (name: string) => p.dialog.querySelector<HTMLElement>(`.dish-map-cell[data-well="${well(name)}"]`)!;
    expect(p.dialog.querySelectorAll(".dish-map-cell")).toHaveLength(48);
    for (let w = 0; w < 48; w++) p.setMapState(w, "resting");
    p.setMapState(well("B2"), "sounding");
    p.open(48213, well("F6"));
    expect(cell("B2").dataset.state).toBe("sounding");
    expect(cell("F6").dataset.current).toBe("true");
    cell("B2").click();
    expect(asked).toEqual(["B2"]);
    expect(cell("B2").dataset.current).toBe("true");
    expect(cell("F6").dataset.current).toBeUndefined();
    cell("B2").click(); // already there
    expect(asked).toEqual(["B2"]);
  });

  it("has play / pause as the first control, labelled by what a tap will do", () => {
    const { p } = panel();
    for (let w = 0; w < 48; w++) p.setMapState(w, "resting");
    p.open(48213, well("F6"));
    const play = p.dialog.querySelector<HTMLElement>(".dish-play")!;
    expect(p.dialog.querySelector(".dish-nav")!.firstElementChild).toBe(play);
    expect(p.dialog.querySelector(".dish-foot .dish-play")).toBeNull();
    expect(play.textContent).toBe("play");
    expect(play.getAttribute("aria-pressed")).toBe("false");
    expect(play.getAttribute("aria-label")).toBe("Well F6, six layers, resting");
    p.setWellState("sounding");
    expect(play.textContent).toBe("pause");
    expect(play.getAttribute("aria-pressed")).toBe("true");
    p.setWellState("held");
    expect(play.textContent).toBe("pause");
    expect(play.dataset.state).toBe("held");
    p.setWellState("loading");
    expect(play.textContent).toBe("loading");
    expect(play.getAttribute("aria-disabled")).toBe("true");
  });

  it("starts play / pause from the well's known state when moving", () => {
    const { p } = panel();
    for (let w = 0; w < 48; w++) p.setMapState(w, "resting");
    p.setMapState(well("A3"), "sounding");
    p.open(48213, well("A3"));
    expect(p.dialog.querySelector(".dish-hold")!.textContent).toBe("pause");
    p.open(48213, well("A4"));
    expect(p.dialog.querySelector(".dish-hold")!.textContent).toBe("play");
  });
});
