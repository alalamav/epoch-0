import { describe, expect, it } from "vitest";
import { DOT_GAIN, DOT_R0, RING_R0, WAVE_LEN } from "../src/plate/constants";
import {
  ARC_FILL,
  CENTRE_R0,
  DENSE_LIMIT,
  RIM_GAIN,
  RIM_TAPER,
  SLOT_ANGLE,
  TICK_MS,
  angularDistance,
  centreRadius,
  dotRadius,
  isNegative,
  isSmallWell,
  linkHalfWidth,
  linkOpacity,
  parseCssColor,
  rimRadii,
  ringRadius,
  slotArc,
  tickPulse,
  unitAngle,
  weightOpacity,
  weightStrength,
  wrapAngle,
} from "../src/view/layout";

describe("angles", () => {
  it("wraps into [0, 2π) and measures the short way round", () => {
    expect(wrapAngle(-Math.PI / 2)).toBeCloseTo(1.5 * Math.PI, 12);
    expect(wrapAngle(5 * Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(angularDistance(0.1, 2 * Math.PI - 0.1)).toBeCloseTo(0.2, 12);
    expect(angularDistance(0, Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(angularDistance(1, 1)).toBe(0);
  });
});

describe("live values (steps 5-8)", () => {
  it("dot radius is 0.010 + 0.022 |state|", () => {
    expect(dotRadius(0)).toBeCloseTo(DOT_R0, 12);
    expect(dotRadius(1)).toBeCloseTo(DOT_R0 + DOT_GAIN, 12);
    expect(dotRadius(-0.5)).toBeCloseTo(0.021, 12);
    expect(dotRadius(-3)).toBeCloseTo(0.032, 12); // clamped
    expect(dotRadius(NaN)).toBeCloseTo(DOT_R0, 12);
    expect(isNegative(-0.2)).toBe(true);
    expect(isNegative(0)).toBe(false);
    expect(isNegative(0.2)).toBe(false);
  });

  it("centre dot grows with level and stays inside the innermost ring", () => {
    expect(centreRadius(0)).toBe(CENTRE_R0);
    let last = centreRadius(0);
    for (let level = 0.05; level <= 0.5; level += 0.05) {
      expect(centreRadius(level)).toBeGreaterThan(last);
      last = centreRadius(level);
    }
    expect(centreRadius(10)).toBeCloseTo(centreRadius(0.5), 12);
    expect(centreRadius(10)).toBeLessThan(RING_R0 - dotRadius(1));
    expect(centreRadius(NaN)).toBe(CENTRE_R0);
  });

  it("a resting rim is a plain circle", () => {
    const rim = rimRadii(null);
    expect(rim.length).toBe(WAVE_LEN);
    expect([...rim].every((v) => v === 1)).toBe(true);
  });

  it("a sounding rim is displaced by the wave and closes on itself", () => {
    const wave = new Float32Array(WAVE_LEN).fill(0.5);
    const rim = rimRadii(wave);
    expect(rim[WAVE_LEN / 2]).toBeCloseTo(1 + RIM_GAIN * 0.5, 6);
    expect(rim[RIM_TAPER]).toBeCloseTo(1 + RIM_GAIN * 0.5, 6);
    // Faded at the seam, symmetric, and rising monotonically into the wave.
    expect(Math.abs(rim[0] - 1)).toBeLessThan(RIM_GAIN * 0.5 * 0.02);
    expect(rim[0]).toBeCloseTo(rim[WAVE_LEN - 1], 6);
    for (let k = 1; k <= RIM_TAPER; k++) expect(rim[k]).toBeGreaterThanOrEqual(rim[k - 1]);

    const negative = rimRadii(new Float32Array(WAVE_LEN).fill(-0.5));
    expect(negative[100]).toBeCloseTo(1 - RIM_GAIN * 0.5, 6);

    // Clamped to +-1, tolerant of bad or short input, and writes into the buffer it is given.
    const wild = new Float32Array(WAVE_LEN).fill(9);
    wild[101] = -9;
    wild[102] = NaN;
    const out = new Float32Array(WAVE_LEN);
    expect(rimRadii(wild, out)).toBe(out);
    expect(out[100]).toBeCloseTo(1 + RIM_GAIN, 6);
    expect(out[101]).toBeCloseTo(1 - RIM_GAIN, 6);
    expect(out[102]).toBe(1);
    expect(rimRadii(new Float32Array(10).fill(1))[200]).toBe(1);
  });

  it("a tick darkens the ring and fades out", () => {
    expect(tickPulse(0)).toBe(1);
    expect(tickPulse(TICK_MS / 2)).toBeCloseTo(0.25, 12);
    expect(tickPulse(TICK_MS)).toBe(0);
    expect(tickPulse(TICK_MS * 10)).toBe(0);
    expect(tickPulse(-5)).toBe(0);
    expect(tickPulse(Infinity)).toBe(0);
    expect(tickPulse(NaN)).toBe(0);
  });
});

describe("weights to strokes (steps 4 and 10)", () => {
  it("strength is |weight| against the Glorot limit of a 16 -> 16 dense layer", () => {
    expect(DENSE_LIMIT).toBeCloseTo(0.4330127, 6);
    expect(weightStrength(0)).toBe(0);
    expect(weightStrength(-DENSE_LIMIT)).toBe(1);
    expect(weightStrength(DENSE_LIMIT / 2)).toBeCloseTo(0.5, 12);
    expect(weightStrength(5)).toBe(1);
    expect(weightStrength(0.3, 0.6)).toBeCloseTo(0.5, 12);
  });

  it("link width and opacity grow with |weight| and ignore the sign", () => {
    let w = linkHalfWidth(0);
    let o = linkOpacity(0);
    expect(w).toBeGreaterThan(0);
    expect(o).toBeCloseTo(0.3, 12);
    for (let v = 0.05; v < DENSE_LIMIT; v += 0.05) {
      expect(linkHalfWidth(v)).toBeGreaterThan(w);
      expect(linkOpacity(v)).toBeGreaterThan(o);
      expect(linkHalfWidth(-v)).toBe(linkHalfWidth(v));
      expect(linkOpacity(-v)).toBe(linkOpacity(v));
      w = linkHalfWidth(v);
      o = linkOpacity(v);
    }
    expect(linkOpacity(1)).toBeCloseTo(0.9, 12);
    // Even the widest link is thinner than a resting dot.
    expect(linkHalfWidth(1)).toBeLessThan(DOT_R0);
  });

  it("dish opacity is the square of the strength", () => {
    expect(weightOpacity(0, 0.6)).toBe(0);
    expect(weightOpacity(-0.3, 0.6)).toBeCloseTo(0.25, 12);
    expect(weightOpacity(2, 0.6)).toBe(1);
  });
});

describe("small wells (step 9)", () => {
  it("switches below a radius of 40 px", () => {
    expect(isSmallWell(39.9)).toBe(true);
    expect(isSmallWell(40)).toBe(false);
  });

  it("an arc segment sits on its ring, centred on its slot, shorter than the slot", () => {
    for (const [layer, slot] of [
      [0, 0],
      [3, 5],
      [7, 15],
    ]) {
      const [x0, y0, x1, y1] = slotArc(layer, slot);
      const r = ringRadius(layer);
      expect(Math.hypot(x0, y0)).toBeCloseTo(r, 12);
      expect(Math.hypot(x1, y1)).toBeCloseTo(r, 12);
      const a0 = Math.atan2(x0, -y0);
      const a1 = Math.atan2(x1, -y1);
      expect(angularDistance(a0, a1)).toBeCloseTo(ARC_FILL * SLOT_ANGLE, 9);
      const mid = Math.atan2(x0 + x1, -(y0 + y1));
      expect(angularDistance(mid, unitAngle(layer, slot))).toBeLessThan(1e-9);
    }
  });
});

describe("parseCssColor", () => {
  it("reads the palette of the plan", () => {
    const paper = parseCssColor("#F3F1EA")!;
    expect(paper[0]).toBeCloseTo(0xf3 / 255, 12);
    expect(paper[1]).toBeCloseTo(0xf1 / 255, 12);
    expect(paper[2]).toBeCloseTo(0xea / 255, 12);
    expect(paper[3]).toBe(1);
    expect(parseCssColor(" #2e5e8c ")).toEqual([0x2e / 255, 0x5e / 255, 0x8c / 255, 1]);
  });

  it("reads short hex, alpha hex and rgb()", () => {
    expect(parseCssColor("#fff")).toEqual([1, 1, 1, 1]);
    expect(parseCssColor("#0008")).toEqual([0, 0, 0, 0x88 / 255]);
    expect(parseCssColor("#ff000080")).toEqual([1, 0, 0, 0x80 / 255]);
    expect(parseCssColor("rgb(255, 0, 51)")).toEqual([1, 0, 0.2, 1]);
    expect(parseCssColor("rgba(0, 255, 0, 0.5)")).toEqual([0, 1, 0, 0.5]);
    expect(parseCssColor("rgb(100% 0% 0% / 50%)")).toEqual([1, 0, 0, 0.5]);
  });

  it("leaves everything else to the browser", () => {
    expect(parseCssColor("oklch(0.5 0.1 250)")).toBeNull();
    expect(parseCssColor("rebeccapurple")).toBeNull();
    expect(parseCssColor("#12")).toBeNull();
    expect(parseCssColor("#ggg")).toBeNull();
    expect(parseCssColor("rgb(1, 2)")).toBeNull();
    expect(parseCssColor("rgb(a, b, c)")).toBeNull();
  });
});
