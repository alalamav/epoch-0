// The dish: one well, large (plan section 6, step 10).
//
// Three stacked 2D canvases inside the container:
//   weights  - every weight between neighbouring rings (straight lines) and every recurrent weight
//              (chords inside its ring), opacity by |weight|; redrawn only when the well, the size or
//              the theme changes
//   live     - rim, ring guides with tick pulses, centre dot and unit dots; redrawn when a report arrives
//   spectrum - a scrolling spectrogram of the final output, 0-8 kHz, below the dish
// Sign is shown twice: ink / solid / filled for positive, stain / dashed / open for negative.
// The dish is drawn with Canvas 2D on purpose: about 3,800 lines drawn once need no second WebGL context.

import type { DishView, EngineReport, RowLayout, RowMatrices, Theme, WellReport } from "../contracts";
import { MAX_LAYERS, N, TICK_FROM_LAYER, WAVE_LEN, WELLS, wellLayers } from "../plate/constants";
import {
  DENSE_LIMIT,
  REDUCED_HZ,
  RIM_GAIN,
  TAU,
  buildDotPositions,
  centreRadius,
  dotRadius,
  layoutCovers,
  rimRadii,
  ringRadius,
  tickPulse,
  weightOpacity,
} from "./layout";

/** Reference size of a recurrent weight (entries of a random orthogonal 16 x 16 matrix are about 0.25, rarely over 0.6). */
const RECURRENT_REF = 0.6;
/** Weights fainter than this are not drawn. */
const MIN_ALPHA = 0.012;
const DENSE_ALPHA = 0.8;
const RECURRENT_ALPHA = 0.7;
/** A chord is bowed sideways by this share of its length, so a -> b and b -> a do not hide each other. */
const CHORD_BOW = 0.08;
/** Radius (well units) of the loop that stands for a unit's weight onto itself. */
const SELF_LOOP_R = 0.024;
const MIN_DOT_PX = 1.5;

const SPECTRUM_HZ = 8000;
/** Scroll speed of the spectrogram in CSS pixels per second. */
const SPECTRUM_PX_PER_S = 40;
const SPECTRUM_GAP = 8;

function makeCanvas(root: HTMLElement): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.setAttribute("aria-hidden", "true");
  c.style.position = "absolute";
  c.style.display = "block";
  c.style.pointerEvents = "none";
  root.appendChild(c);
  return c;
}

export function createDishView(container: HTMLElement, theme: Theme): DishView {
  const root = document.createElement("div");
  root.style.position = "relative";
  root.style.width = "100%";
  root.style.height = "100%";
  root.style.overflow = "hidden";
  container.appendChild(root);

  const weightsCanvas = makeCanvas(root);
  const liveCanvas = makeCanvas(root);
  const spectrumCanvas = makeCanvas(root);
  const weightsCtx = weightsCanvas.getContext("2d");
  const liveCtx = liveCanvas.getContext("2d");
  const spectrumCtx = spectrumCanvas.getContext("2d");

  let colours: Theme = { ...theme };
  let disposed = false;
  let reduced = false;

  // Layout of the canvases, CSS pixels.
  let dpr = 1;
  let side = 0; // the dish is side x side
  let radius = 0; // pixels per well unit
  let spectrumW = 0; // device pixels
  let spectrumH = 0;

  // What is shown.
  let openWell: number | null = null;
  let layers = 0;
  let layout: RowLayout | null = null;
  let matrices: RowMatrices | null = null;
  let positions: Float32Array | null = null; // [x, y] per unit, well units

  // Live state of the open well.
  const units = new Float32Array(MAX_LAYERS * N);
  const wave = new Float32Array(WAVE_LEN);
  const rim = new Float32Array(WAVE_LEN);
  let active = false;
  let held = false;
  let level = 0;
  let pending: WellReport | null | undefined; // undefined: nothing new; null: the well is at rest
  let pendingTicks = 0;
  let lastApplied = -Infinity;
  const tickStart = new Float64Array(MAX_LAYERS).fill(-Infinity);
  let pulsing = false;

  let weightsDirty = true;
  let liveDirty = true;

  // Spectrogram.
  let analyser: AnalyserNode | null = null;
  let bins: Uint8Array<ArrayBuffer> = new Uint8Array(0);
  let spectrumTime = -Infinity;

  const ready = (): boolean => openWell !== null && layoutCovers(layout, layers) && positions !== null;

  function rest(): void {
    active = false;
    held = false;
    level = 0;
    units.fill(0);
    wave.fill(0);
  }

  // ───────────────────────────── drawing ─────────────────────────────

  function begin(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Origin at the centre of the dish, one unit = one CSS pixel.
    ctx.setTransform(dpr, 0, 0, dpr, (side / 2) * dpr, (side / 2) * dpr);
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.lineCap = "butt";
  }

  function drawWeights(): void {
    weightsDirty = false;
    const ctx = weightsCtx;
    if (!ctx) return;
    begin(ctx, weightsCanvas);
    const pos = positions;
    if (!ready() || !pos || !matrices || side <= 0) return;
    const R = radius;
    const dash = [Math.max(2, R * 0.012), Math.max(2, R * 0.012)];

    const style = (weight: number, alpha: number, width: number): void => {
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = weight < 0 ? colours.stain : colours.ink;
      ctx.lineWidth = width;
      ctx.setLineDash(weight < 0 ? dash : []);
    };

    // Recurrent weights: chords inside each ring, from sender to receiver.
    for (let l = 0; l < layers; l++) {
      const m = matrices.R[l];
      if (!m || m.length < N * N) continue;
      const loop = Math.max(2.5, R * SELF_LOOP_R);
      for (let recv = 0; recv < N; recv++) {
        for (let send = 0; send < N; send++) {
          const weight = m[recv * N + send];
          const alpha = RECURRENT_ALPHA * weightOpacity(weight, RECURRENT_REF);
          if (!(alpha >= MIN_ALPHA)) continue;
          const x0 = R * pos[(l * N + send) * 2];
          const y0 = R * pos[(l * N + send) * 2 + 1];
          style(weight, alpha, 1);
          ctx.beginPath();
          if (recv === send) {
            ctx.arc(x0, y0, loop, 0, TAU);
          } else {
            const x1 = R * pos[(l * N + recv) * 2];
            const y1 = R * pos[(l * N + recv) * 2 + 1];
            const mx = (x0 + x1) / 2 - (y1 - y0) * CHORD_BOW;
            const my = (y0 + y1) / 2 + (x1 - x0) * CHORD_BOW;
            ctx.moveTo(x0, y0);
            ctx.quadraticCurveTo(mx, my, x1, y1);
          }
          ctx.stroke();
        }
      }
    }

    // Weights between neighbouring rings: every sender of ring i to every receiver of ring i - 1.
    for (let i = 1; i < layers; i++) {
      const m = matrices.C[i];
      if (!m || m.length < N * N) continue;
      const link = layout ? layout.links[i] : null;
      for (let recv = 0; recv < N; recv++) {
        for (let send = 0; send < N; send++) {
          const weight = m[recv * N + send];
          const alpha = DENSE_ALPHA * weightOpacity(weight, DENSE_LIMIT);
          if (!(alpha >= MIN_ALPHA)) continue;
          // The strongest sender of each unit (the link drawn on the plate) is a little wider.
          const strongest = !!link && link.sender[recv] === send;
          style(weight, strongest ? Math.max(alpha, 0.85) : alpha, strongest ? 1.75 : 1);
          ctx.beginPath();
          ctx.moveTo(R * pos[((i - 1) * N + recv) * 2], R * pos[((i - 1) * N + recv) * 2 + 1]);
          ctx.lineTo(R * pos[(i * N + send) * 2], R * pos[(i * N + send) * 2 + 1]);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }

  function drawLive(now: number): void {
    liveDirty = false;
    pulsing = false;
    const ctx = liveCtx;
    if (!ctx) return;
    begin(ctx, liveCanvas);
    if (openWell === null || side <= 0) return;
    const R = radius;

    // Rim: displaced by the last output samples while the well sounds, stained while the button is down.
    rimRadii(active ? wave : null, rim);
    ctx.beginPath();
    for (let k = 0; k < WAVE_LEN; k++) {
      const a = (k / WAVE_LEN) * TAU;
      const r = R * rim[k];
      const x = r * Math.sin(a);
      const y = -r * Math.cos(a);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.lineJoin = "round";
    ctx.strokeStyle = held ? colours.stain : active ? colours.ink : colours.hairline;
    ctx.lineWidth = held ? 2.5 : 1.25;
    ctx.stroke();

    const pos = positions;
    if (!ready() || !pos) return;

    // Ring guides; a ring darkens briefly when its layer ticks.
    pulsing = false;
    for (let l = 0; l < layers; l++) {
      const pulse = reduced ? 0 : tickPulse(now - tickStart[l]);
      ctx.beginPath();
      ctx.arc(0, 0, R * ringRadius(l), 0, TAU);
      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = colours.hairline;
      ctx.lineWidth = 1;
      ctx.stroke();
      if (pulse > 0) {
        pulsing = true;
        ctx.globalAlpha = pulse;
        ctx.strokeStyle = colours.ink;
        ctx.lineWidth = 1 + 1.5 * pulse;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Centre dot.
    ctx.fillStyle = colours.ink;
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(MIN_DOT_PX, R * centreRadius(active ? level : 0)), 0, TAU);
    ctx.fill();

    // Unit dots: filled ink for positive (and zero), open stain for negative.
    const count = layers * N;
    for (let k = 0; k < count; k++) {
      const state = active ? units[k] : 0;
      const r = Math.max(MIN_DOT_PX, R * dotRadius(state));
      const x = R * pos[k * 2];
      const y = R * pos[k * 2 + 1];
      ctx.beginPath();
      if (state < 0) {
        const stroke = Math.max(1.25, r * 0.35);
        ctx.arc(x, y, Math.max(0.5, r - stroke / 2), 0, TAU);
        ctx.fillStyle = colours.paper;
        ctx.fill();
        ctx.strokeStyle = colours.stain;
        ctx.lineWidth = stroke;
        ctx.stroke();
      } else {
        ctx.arc(x, y, r, 0, TAU);
        ctx.fillStyle = colours.ink;
        ctx.fill();
      }
    }
  }

  function clearSpectrum(): void {
    if (!spectrumCtx) return;
    spectrumCtx.setTransform(1, 0, 0, 1, 0, 0);
    spectrumCtx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
    spectrumTime = -Infinity;
  }

  /** Scroll the spectrogram left and add the newest spectrum at its right edge. Idle without a running analyser. */
  function stepSpectrum(now: number): void {
    const ctx = spectrumCtx;
    const node = analyser;
    if (!ctx || !node || openWell === null || spectrumW <= 0 || spectrumH <= 0) return;
    if (node.context.state !== "running") {
      spectrumTime = -Infinity;
      return;
    }
    const elapsed = now - spectrumTime;
    if (!(elapsed >= 0 && elapsed < 1000)) {
      spectrumTime = now; // first frame, or the page was in the background: start again from here
      return;
    }
    if (reduced && elapsed < 1000 / REDUCED_HZ) return;
    const pxPerMs = (SPECTRUM_PX_PER_S * dpr) / 1000;
    const columns = Math.min(spectrumW, Math.floor(elapsed * pxPerMs));
    if (columns < 1) return;
    spectrumTime += columns / pxPerMs;

    if (bins.length !== node.frequencyBinCount) bins = new Uint8Array(node.frequencyBinCount);
    node.getByteFrequencyData(bins);
    const nyquist = node.context.sampleRate / 2;
    const top = Math.max(1, Math.min(bins.length, Math.round((SPECTRUM_HZ / nyquist) * bins.length)));

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "copy";
    ctx.drawImage(spectrumCanvas, -columns, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(spectrumW - columns, 0, columns, spectrumH);
    ctx.fillStyle = colours.ink;
    for (let y = 0; y < spectrumH; y++) {
      // Row 0 is the top of the strip: 8 kHz. The loudest bin of the row's band is shown.
      const b1 = Math.min(top, Math.ceil(((spectrumH - y) / spectrumH) * top));
      const b0 = Math.max(0, Math.min(b1 - 1, Math.floor(((spectrumH - y - 1) / spectrumH) * top)));
      let v = 0;
      for (let b = b0; b < b1; b++) if (bins[b] > v) v = bins[b];
      const alpha = (v / 255 - 0.12) / 0.88;
      if (alpha <= 0.01) continue;
      ctx.globalAlpha = alpha > 1 ? 1 : alpha;
      ctx.fillRect(spectrumW - columns, y, columns, 1);
    }
    ctx.globalAlpha = 1;
  }

  // ───────────────────────────── reports ─────────────────────────────

  function flushReport(now: number): void {
    if (pending === undefined) return;
    const report = pending;
    const sounding = report !== null;
    if (reduced && sounding === active && now - lastApplied < 1000 / REDUCED_HZ) return;
    pending = undefined;
    lastApplied = now;
    if (report) {
      active = true;
      held = report.held;
      level = report.level;
      const n = Math.min(report.units.length, units.length);
      for (let k = 0; k < n; k++) units[k] = report.units[k];
      const m = Math.min(report.wave.length, WAVE_LEN);
      for (let k = 0; k < m; k++) wave[k] = report.wave[k];
      for (let k = m; k < WAVE_LEN; k++) wave[k] = 0;
      if (!reduced) {
        for (let l = TICK_FROM_LAYER; l < layers; l++) {
          if (pendingTicks & (1 << l)) {
            tickStart[l] = now;
            pulsing = true;
          }
        }
      }
      liveDirty = true;
    } else if (active) {
      rest();
      liveDirty = true;
    }
    pendingTicks = 0;
  }

  // ───────────────────────────── interface ─────────────────────────────

  function place(canvas: HTMLCanvasElement, left: number, top: number, w: number, h: number): void {
    canvas.style.left = `${left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const pw = Math.max(1, Math.round(w * dpr));
    const ph = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== pw) canvas.width = pw;
    if (canvas.height !== ph) canvas.height = ph;
  }

  function resize(): void {
    if (disposed) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    dpr = Math.min(3, window.devicePixelRatio > 0 ? window.devicePixelRatio : 1);
    // The spectrogram takes a strip under the dish when there is room for it.
    const strip = h >= 160 ? Math.max(36, Math.min(96, Math.round(h * 0.16))) : 0;
    side = Math.max(0, Math.floor(Math.min(w, h - (strip > 0 ? strip + SPECTRUM_GAP : 0))));
    // Leave room for the rim to move outwards.
    radius = Math.max(0, side / 2 / (1 + RIM_GAIN) - 2);
    const left = Math.floor((w - side) / 2);
    const top = Math.max(0, Math.floor((h - side - (strip > 0 ? strip + SPECTRUM_GAP : 0)) / 2));
    place(weightsCanvas, left, top, side, side);
    place(liveCanvas, left, top, side, side);
    place(spectrumCanvas, left, top + side + SPECTRUM_GAP, side, strip);
    spectrumCanvas.style.display = strip > 0 && side > 0 ? "block" : "none";
    spectrumW = strip > 0 && side > 0 ? spectrumCanvas.width : 0;
    spectrumH = strip > 0 && side > 0 ? spectrumCanvas.height : 0;
    spectrumTime = -Infinity; // resizing a canvas clears it
    weightsDirty = true;
    liveDirty = true;
  }

  resize();

  return {
    get openWell() {
      return openWell;
    },

    open(well, rowLayout, rowMatrices) {
      if (disposed || !(well >= 0 && well < WELLS)) return;
      if (well !== openWell) {
        rest();
        pending = undefined;
        pendingTicks = 0;
        tickStart.fill(-Infinity);
        clearSpectrum();
      }
      openWell = well;
      layers = wellLayers(well);
      layout = rowLayout;
      matrices = rowMatrices;
      positions = layoutCovers(rowLayout, layers) ? buildDotPositions(rowLayout, layers) : null;
      if (side <= 0) resize(); // the container may have been hidden when the view was created
      weightsDirty = true;
      liveDirty = true;
    },

    close() {
      if (openWell === null) return;
      openWell = null;
      layout = null;
      matrices = null;
      positions = null;
      rest();
      pending = undefined;
      pendingTicks = 0;
      tickStart.fill(-Infinity);
      clearSpectrum();
      weightsDirty = true;
      liveDirty = true;
    },

    applyReport(report: EngineReport) {
      if (disposed || openWell === null) return;
      let found: WellReport | null = null;
      for (const r of report.wells) {
        if (r.well === openWell) {
          found = r;
          break;
        }
      }
      pending = found;
      if (found) pendingTicks |= found.ticks;
    },

    setAnalyser(node) {
      analyser = node;
      spectrumTime = -Infinity;
    },

    setTheme(t) {
      colours = { ...t };
      clearSpectrum();
      weightsDirty = true;
      liveDirty = true;
    },

    setReducedMotion(value) {
      reduced = value;
      if (value) tickStart.fill(-Infinity);
      liveDirty = true;
    },

    resize,

    render(now) {
      if (disposed) return;
      flushReport(now);
      if (weightsDirty) drawWeights();
      if (liveDirty || pulsing) drawLive(now);
      stepSpectrum(now);
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      openWell = null;
      analyser = null;
      layout = null;
      matrices = null;
      positions = null;
      root.remove();
    },
  };
}
