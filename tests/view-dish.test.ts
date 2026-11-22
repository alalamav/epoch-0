// @vitest-environment jsdom
// Smoke tests of the two views without a real canvas: every 2D context call is counted by a stand-in,
// and WebGL is absent (jsdom), so the plate scene must fall back to drawing nothing without throwing.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EngineReport, RowMatrices, Theme, WellReport } from "../src/contracts";
import { N, WAVE_LEN, wellIndex } from "../src/plate/constants";
import { createDishView } from "../src/view/dish";
import { computeRowLayout } from "../src/view/layout";
import { createPlateScene } from "../src/view/scene";

const theme: Theme = { paper: "#F3F1EA", ink: "#1C1B19", hairline: "#B9B5A8", stain: "#2E5E8C" };

type Log = Record<string, number>;
const logs = new Map<HTMLCanvasElement, Log>();
const realGetContext = HTMLCanvasElement.prototype.getContext;

function countingContext(log: Log): unknown {
  return new Proxy(
    {},
    {
      get: (_target, key) =>
        typeof key === "string"
          ? () => {
              log[key] = (log[key] ?? 0) + 1;
            }
          : undefined,
      set: () => true,
    },
  );
}

function matrices(layers: number): RowMatrices {
  let seed = 9;
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const m = (scale: number): Float32Array => Float32Array.from({ length: N * N }, () => (rnd() * 2 - 1) * scale);
  const R: Float32Array[] = [];
  const C: (Float32Array | null)[] = [];
  for (let i = 0; i < layers; i++) {
    R.push(m(0.5));
    C.push(i === 0 ? null : m(0.433));
  }
  return { R, C };
}

function report(well: number, layers: number, sign: number, ticks = 0): EngineReport {
  const r: WellReport = {
    well,
    level: 0.2,
    wave: new Float32Array(WAVE_LEN).fill(0.3),
    units: new Float32Array(layers * N).fill(0.5 * sign),
    ticks,
    held: false,
  };
  return { type: "report", time: 0, wells: [r], gain: 1, peak: 0.3 };
}

function sizedContainer(width: number, height: number): HTMLElement {
  const el = document.createElement("div");
  Object.defineProperty(el, "clientWidth", { value: width });
  Object.defineProperty(el, "clientHeight", { value: height });
  document.body.appendChild(el);
  return el;
}

beforeEach(() => {
  logs.clear();
  HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, kind: string) {
    if (kind !== "2d") return null; // no WebGL here
    let log = logs.get(this);
    if (!log) logs.set(this, (log = {}));
    return countingContext(log);
  } as typeof HTMLCanvasElement.prototype.getContext;
});

afterEach(() => {
  HTMLCanvasElement.prototype.getContext = realGetContext;
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("dish view", () => {
  it("fills the container with a square dish and a spectrogram strip", () => {
    const container = sizedContainer(400, 500);
    const dish = createDishView(container, theme);
    const canvases = [...container.querySelectorAll("canvas")];
    expect(canvases.length).toBe(3);
    const [weights, live, spectrum] = canvases;
    expect(weights.style.width).toBe(weights.style.height);
    expect(live.style.width).toBe(weights.style.width);
    expect(parseFloat(weights.style.width)).toBeGreaterThan(300);
    expect(parseFloat(weights.style.width) + parseFloat(spectrum.style.height)).toBeLessThanOrEqual(500);
    expect(spectrum.style.width).toBe(weights.style.width);
    for (const c of canvases) expect(c.getAttribute("aria-hidden")).toBe("true");
    dish.dispose();
    expect(container.querySelector("canvas")).toBeNull();
  });

  it("draws all weights once, and the dots again only when something changed", () => {
    const container = sizedContainer(400, 500);
    const dish = createDishView(container, theme);
    const [weights, live] = [...container.querySelectorAll("canvas")];
    expect(dish.openWell).toBeNull();

    const layers = 3;
    const well = wellIndex(2, layers - 1);
    const m = matrices(layers);
    dish.open(well, computeRowLayout(m.C), m);
    expect(dish.openWell).toBe(well);
    dish.render(1000);

    const w = logs.get(weights)!;
    // Up to 2 * 256 lines between rings and 3 * 256 chords (the faintest are skipped).
    expect(w.stroke).toBeGreaterThan(400);
    expect(w.stroke).toBeLessThanOrEqual(5 * N * N);
    expect(w.lineTo).toBeGreaterThan(200); // straight lines between rings
    expect(w.lineTo).toBeLessThanOrEqual(2 * N * N);
    expect(w.quadraticCurveTo).toBeGreaterThan(200); // chords inside rings
    expect(w.setLineDash).toBeGreaterThan(0);

    const l = logs.get(live)!;
    // At rest: 48 filled dots and the centre dot; rim and three ring guides stroked.
    expect(l.fill).toBe(layers * N + 1);
    expect(l.stroke).toBe(1 + layers);

    // Nothing changed: no drawing at all.
    const before = JSON.stringify([w, l]);
    dish.render(1016);
    dish.render(1033);
    expect(JSON.stringify([w, l])).toBe(before);

    // Negative states are open dots: a paper fill plus a stain stroke each.
    const strokes = l.stroke;
    const fills = l.fill;
    dish.applyReport(report(well, layers, -1));
    dish.render(1050);
    expect(l.stroke - strokes).toBe(1 + layers + layers * N);
    expect(l.fill - fills).toBe(layers * N + 1);
    expect(logs.get(weights)!.stroke).toBe(w.stroke); // the weights are not redrawn

    // A report without the well puts it back to rest.
    dish.applyReport({ type: "report", time: 1, wells: [], gain: 1, peak: 0 });
    const strokes2 = l.stroke;
    dish.render(1083);
    expect(l.stroke - strokes2).toBe(1 + layers);

    // Reports for other wells are ignored while this one rests.
    const after = JSON.stringify(l);
    dish.applyReport(report(well + 1, layers + 1, 1));
    dish.render(1100);
    dish.render(1116);
    expect(JSON.stringify(l)).toBe(after);

    dish.close();
    expect(dish.openWell).toBeNull();
    dish.render(1200);
    dish.dispose();
  });

  it("pulses a slow ring on a tick, but not with reduced motion", () => {
    const container = sizedContainer(400, 500);
    const dish = createDishView(container, theme);
    const live = container.querySelectorAll("canvas")[1];
    const layers = 6;
    const well = wellIndex(0, layers - 1);
    const m = matrices(layers);
    dish.open(well, computeRowLayout(m.C), m);
    dish.render(0);
    const l = logs.get(live)!;

    let strokes = l.stroke;
    dish.applyReport(report(well, layers, 1, 1 << 5));
    dish.render(100);
    // Rim + six guides + one extra stroke for the ticking ring.
    expect(l.stroke - strokes).toBe(1 + layers + 1);
    // The pulse keeps the view drawing until it has faded, then it goes quiet.
    strokes = l.stroke;
    dish.render(200);
    expect(l.stroke - strokes).toBe(1 + layers + 1);
    dish.render(2000);
    strokes = l.stroke;
    dish.render(2016);
    expect(l.stroke).toBe(strokes);

    // Reduced motion: no pulse, and updates at most four times a second.
    dish.setReducedMotion(true);
    dish.render(3000);
    dish.applyReport(report(well, layers, 1, 1 << 5));
    dish.render(3300);
    strokes = l.stroke;
    dish.applyReport(report(well, layers, 1, 1 << 4));
    dish.render(3333); // 33 ms after the last update: held back
    expect(l.stroke).toBe(strokes);
    dish.render(3560); // 260 ms: applied, without a pulse
    expect(l.stroke - strokes).toBe(1 + layers);
    dish.dispose();
  });

  it("scrolls the spectrogram only while an analyser is running", () => {
    const container = sizedContainer(400, 500);
    const dish = createDishView(container, theme);
    const spectrum = container.querySelectorAll("canvas")[2];
    const m = matrices(2);
    dish.open(wellIndex(0, 1), computeRowLayout(m.C), m);
    dish.render(0);
    dish.render(100);
    expect(logs.get(spectrum)?.drawImage).toBeUndefined();

    let reads = 0;
    const context = { state: "running", sampleRate: 16000 };
    const analyser = {
      frequencyBinCount: 1024,
      context,
      getByteFrequencyData(a: Uint8Array) {
        reads++;
        a.fill(200);
      },
    } as unknown as AnalyserNode;
    dish.setAnalyser(analyser);
    dish.render(200); // starts the clock
    expect(reads).toBe(0);
    dish.render(300); // 100 ms at 40 px/s: four columns
    expect(reads).toBe(1);
    const s = logs.get(spectrum)!;
    expect(s.drawImage).toBe(1);
    expect(s.fillRect).toBeGreaterThan(30); // one per pixel row of the strip

    context.state = "suspended";
    dish.render(400);
    dish.render(500);
    expect(reads).toBe(1);

    dish.setAnalyser(null);
    dish.render(600);
    dish.render(700);
    expect(reads).toBe(1);
    dish.dispose();
  });

  it("shows only the rim until all layers of the well are laid out", () => {
    const container = sizedContainer(400, 500);
    const dish = createDishView(container, theme);
    const [weights, live] = [...container.querySelectorAll("canvas")];
    const m = matrices(2); // two layers loaded, but the well has five
    dish.open(wellIndex(1, 4), computeRowLayout(m.C), m);
    dish.render(0);
    expect(logs.get(weights)!.stroke).toBeUndefined();
    expect(logs.get(live)!.stroke).toBe(1);
    expect(logs.get(live)!.fill).toBeUndefined();
    dish.dispose();
  });
});

describe("plate scene without WebGL", () => {
  it("falls back to a scene that draws nothing and never throws", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    const canvas = document.createElement("canvas");
    const scene = createPlateScene(canvas, theme);
    const m = matrices(8);
    scene.resize(800, 600, 2, [{ cx: 50, cy: 50, r: 40 }]);
    scene.setRowLayout(0, computeRowLayout(m.C));
    scene.setPressed(3, true);
    scene.applyReport(report(3, 4, 1));
    scene.setReducedMotion(true);
    scene.setTheme(theme);
    scene.render(0);
    scene.clear();
    scene.dispose();
    expect(console.warn).toHaveBeenCalled();
  });
});
