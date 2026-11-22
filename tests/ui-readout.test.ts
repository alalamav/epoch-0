// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { WELLS, wellIndex } from "../src/plate/constants";
import { READOUT_IDLE, createPlateBadge, createReadout, magnifyTarget, readoutText } from "../src/ui/readout";

const F6 = wellIndex(5, 5);

describe("readout text", () => {
  it("is one line: well, layers, weights", () => {
    expect(readoutText(F6)).toBe("F6 · 6 layers · 466,049 weights");
    expect(readoutText(wellIndex(0, 0))).toBe("A1 · 1 layer · 305 weights");
    expect(readoutText(wellIndex(5, 7))).toBe("F8 · 8 layers · 2,039,969 weights");
  });

  it("is a hint before any well is touched", () => {
    expect(readoutText(null)).toBe(READOUT_IDLE);
  });

  it("stays short enough for a 320 px screen", () => {
    for (let well = 0; well < WELLS; well++) expect(readoutText(well).length).toBeLessThanOrEqual(34);
  });
});

describe("magnify target", () => {
  const all = () => true;

  it("is the last well touched when it can be shown", () => {
    expect(magnifyTarget(F6, WELLS, all, () => 0.99)).toBe(F6);
  });

  it("is a random well when none was touched", () => {
    expect(magnifyTarget(null, WELLS, all, () => 0)).toBe(0);
    expect(magnifyTarget(null, WELLS, all, () => 0.5)).toBe(24);
    expect(magnifyTarget(null, WELLS, all, () => 0.999999)).toBe(47);
    expect(magnifyTarget(null, WELLS, all, () => 1)).toBe(47); // out-of-range random stays on the plate
  });

  it("draws only from wells that can be shown", () => {
    const firstFour = (well: number) => well % 8 < 4; // slow layers still loading
    for (const r of [0, 0.3, 0.6, 0.99]) expect(magnifyTarget(null, WELLS, firstFour, () => r)! % 8).toBeLessThan(4);
    // the last well touched is not loaded yet: fall back to a random loaded one
    expect(magnifyTarget(wellIndex(2, 7), WELLS, firstFour, () => 0)).toBe(0);
  });

  it("is null while nothing can be shown", () => {
    expect(magnifyTarget(null, WELLS, () => false)).toBeNull();
    expect(magnifyTarget(F6, WELLS, () => false)).toBeNull();
  });
});

describe("readout button and plate badge", () => {
  afterEach(() => {
    document.body.replaceChildren();
    vi.useRealTimers();
  });

  it("shows the last well and opens the magnified view when pressed", () => {
    let opened = 0;
    const readout = createReadout(document.body, { onOpen: () => opened++ });
    expect(readout.element.textContent).toBe(READOUT_IDLE);
    readout.set(F6);
    expect(readout.element.textContent).toBe("F6 · 6 layers · 466,049 weights");
    expect(readout.element.getAttribute("aria-label")).toContain("Magnify well F6");
    readout.element.click();
    expect(opened).toBe(1);
    readout.set(null);
    expect(readout.element.textContent).toBe(READOUT_IDLE);
  });

  it("shows the plate number and copies the link when pressed", async () => {
    vi.useFakeTimers();
    const copied: string[] = [];
    const button = document.createElement("button");
    document.body.append(button);
    const badge = createPlateBadge(button, {
      getLink: () => "https://example.org/?plate=048213",
      copyText: async (text) => void copied.push(text),
    });
    badge.setPlate(48213);
    expect(button.textContent).toBe("plate 048213");
    button.click();
    await Promise.resolve();
    await Promise.resolve();
    expect(copied).toEqual(["https://example.org/?plate=048213"]);
    expect(button.textContent).toBe("link copied");
    vi.advanceTimersByTime(1700);
    expect(button.textContent).toBe("plate 048213");
  });

  it("says so when the link cannot be copied, and a new plate replaces the message", async () => {
    const button = document.createElement("button");
    const badge = createPlateBadge(button, { getLink: () => "x", copyText: () => Promise.reject(new Error("no")) });
    badge.setPlate(7);
    button.click();
    await Promise.resolve();
    await Promise.resolve();
    expect(button.textContent).toBe("copy failed");
    badge.setPlate(77);
    expect(button.textContent).toBe("plate 000077");
  });
});
