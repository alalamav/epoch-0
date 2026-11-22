// The plate: all 48 wells in one WebGL canvas (plan section 6, steps 1-9).
//
// Flat, orthographic camera in CSS-pixel space (y down), no lighting, transparent clear colour.
// Three draw calls for the whole plate, in this order:
//   1. ring guides  - 216 instanced annuli (hairlines that darken when their layer ticks)
//   2. strips       - instanced thin quads: 48 rims of WAVE_LEN segments, 2,688 links, and (only for
//                     small wells) 3,456 arc segments
//   3. dots         - 3,456 unit dots + 48 centre dots, instanced discs
// Every instance has a fixed place in its buffer, so a report only rewrites the wells that sound.
// Lines are real quads: link width and opacity both follow |weight|; negative links are dashed in the
// fragment shader. Colours are written to the canvas as given (sRGB), without colour management.
//
// All geometry maths lives in layout.ts and is tested there; this file only fills buffers.

import {
  DoubleSide,
  DynamicDrawUsage,
  Float32BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  OrthographicCamera,
  Scene,
  ShaderMaterial,
  Vector3,
  WebGLRenderer,
} from "three";
import type { EngineReport, PlateScene, RowLayout, Theme, WellGeometry } from "../contracts";
import { MAX_LAYERS, N, ROWS, TICK_FROM_LAYER, WAVE_LEN, WELLS, wellLayers, wellRow } from "../plate/constants";
import {
  LINK_STRIDE,
  PLATE_DOTS,
  PLATE_LINKS,
  PLATE_RINGS,
  REDUCED_HZ,
  TAU,
  TICK_MS,
  buildDotPositions,
  buildLinks,
  centreRadius,
  dotRadius,
  isSmallWell,
  layoutCovers,
  linkHalfWidth,
  linkOpacity,
  parseCssColor,
  rimRadii,
  ringRadius,
  slotArc,
  tickPulse,
  wellDotOffset,
  wellLinkOffset,
  wellRingOffset,
} from "./layout";

// Tones understood by the shaders: 0 hairline, 1 ink, 2 stain (values in between are mixed).
const TONE_HAIR = 0;
const TONE_INK = 1;
const TONE_STAIN = 2;

// Sizes in CSS pixels.
const MIN_DOT_PX = 0.7; // a resting dot is 0.01 well radii, under a pixel on most plates
const MIN_ARC_PX = 0.5;
const MIN_LINK_PX = 0.25;
const RIM_HALF_PX = 0.6;
const RIM_HALF_PRESSED_PX = 1.1;
const GUIDE_HALF_PX = 0.5;
const GUIDE_ALPHA = 0.55;
/** Dash period of a negative link, in well radii (never under MIN_DASH_PX). */
const DASH_PERIOD = 0.09;
const MIN_DASH_PX = 3;
/** Style flag: extend the strip by its half width at both ends (joins the rim segments). */
const CAPPED = -1;

// Buffer layout.
const RIM_SEGS = WAVE_LEN;
const STRIP_RIMS = 0;
const STRIP_LINKS = STRIP_RIMS + WELLS * RIM_SEGS;
const STRIP_ARCS = STRIP_LINKS + PLATE_LINKS;
const STRIP_COUNT = STRIP_ARCS + PLATE_DOTS;
const DISC_CENTRES = 0;
const DISC_UNITS = DISC_CENTRES + WELLS;
const DISC_COUNT = DISC_UNITS + PLATE_DOTS;

const TONE_GLSL = /* glsl */ `
uniform vec3 uHair;
uniform vec3 uInk;
uniform vec3 uStain;
vec3 toneColour(float tone) {
  return tone < 1.0 ? mix(uHair, uInk, clamp(tone, 0.0, 1.0)) : mix(uInk, uStain, clamp(tone - 1.0, 0.0, 1.0));
}
`;

// Coverage of a pixel centred `x` away from the middle of a band of half width `hw`.
const BAND_GLSL = /* glsl */ `
uniform float uPx;
float band(float x, float hw) {
  return clamp((min(hw, x + 0.5 * uPx) - max(-hw, x - 0.5 * uPx)) / uPx, 0.0, 1.0);
}
`;

const DISC_VERT = /* glsl */ `
attribute vec4 aCircle; // centre x, y, radius, stroke half width (0 = filled); CSS px
attribute vec2 aStyle;  // tone, alpha
uniform float uPx;
varying vec2 vLocal;
varying vec2 vShape;
varying vec2 vStyle;
void main() {
  float ext = (aStyle.y > 0.0 && aCircle.z > 0.0) ? aCircle.z + aCircle.w + uPx : 0.0;
  vLocal = position.xy * ext;
  vShape = aCircle.zw;
  vStyle = aStyle;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(aCircle.xy + vLocal, 0.0, 1.0);
}
`;

const DISC_FRAG = /* glsl */ `
${TONE_GLSL}
${BAND_GLSL}
varying vec2 vLocal;
varying vec2 vShape;
varying vec2 vStyle;
void main() {
  float d = length(vLocal);
  float cov = vShape.y > 0.0 ? band(d - vShape.x, vShape.y) : clamp((vShape.x - d) / uPx + 0.5, 0.0, 1.0);
  gl_FragColor = vec4(toneColour(vStyle.x), vStyle.y * cov);
}
`;

const STRIP_VERT = /* glsl */ `
attribute vec4 aSeg;   // x0, y0, x1, y1; CSS px
attribute vec4 aStyle; // half width, tone, alpha, dash period (0 solid, < 0 solid with caps)
uniform float uPx;
varying float vAcross;
varying float vAlong;
varying vec4 vStyle;
void main() {
  vec2 d = aSeg.zw - aSeg.xy;
  float len = length(d);
  bool shown = len > 1e-5 && aStyle.z > 0.0;
  vec2 t = len > 1e-5 ? d / len : vec2(1.0, 0.0);
  vec2 n = vec2(-t.y, t.x);
  float ext = shown ? aStyle.x + uPx : 0.0;
  float cap = (shown && aStyle.w < 0.0) ? aStyle.x : 0.0;
  float along = shown ? position.x * (len + 2.0 * cap) - cap : 0.0;
  vAcross = position.y * ext;
  vAlong = along;
  vStyle = aStyle;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(aSeg.xy + t * along + n * vAcross, 0.0, 1.0);
}
`;

const STRIP_FRAG = /* glsl */ `
${TONE_GLSL}
${BAND_GLSL}
varying float vAcross;
varying float vAlong;
varying vec4 vStyle;
void main() {
  float cov = band(vAcross, vStyle.x);
  if (vStyle.w > 0.0) {
    float m = mod(vAlong, vStyle.w);
    float on = 0.62 * vStyle.w;
    cov *= smoothstep(0.0, uPx, m) * (1.0 - smoothstep(on - uPx, on, m));
  }
  gl_FragColor = vec4(toneColour(vStyle.y), vStyle.z * cov);
}
`;

/** An instanced attribute that uploads only the instances that changed. */
class Dyn {
  readonly array: Float32Array;
  readonly attr: InstancedBufferAttribute;
  private full = true;
  constructor(count: number, private readonly size: number) {
    this.array = new Float32Array(count * size);
    this.attr = new InstancedBufferAttribute(this.array, size);
    this.attr.setUsage(DynamicDrawUsage);
  }
  mark(start: number, count: number): void {
    if (count <= 0) return;
    if (!this.full) this.attr.addUpdateRange(start * this.size, count * this.size);
    this.attr.needsUpdate = true;
  }
  markAll(): void {
    this.full = true;
    this.attr.clearUpdateRanges();
    this.attr.needsUpdate = true;
  }
  /** Call after a frame was drawn: from now on partial uploads are enough. */
  uploaded(): void {
    this.full = false;
  }
}

function quadGeometry(x0: number): InstancedBufferGeometry {
  const g = new InstancedBufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute([x0, -1, 0, 1, -1, 0, 1, 1, 0, x0, 1, 0], 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  return g;
}

/** Any CSS colour -> [r, g, b] in 0..1. Formats layout.ts cannot parse are resolved by a 2D canvas. */
function resolveColour(css: string, fallback: [number, number, number]): [number, number, number] {
  const parsed = parseCssColor(css);
  if (parsed) return [parsed[0], parsed[1], parsed[2]];
  try {
    const c = document.createElement("canvas");
    c.width = 1;
    c.height = 1;
    const ctx = c.getContext("2d");
    if (!ctx) return fallback;
    ctx.fillStyle = css;
    ctx.fillRect(0, 0, 1, 1);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return d[3] === 0 ? fallback : [d[0] / 255, d[1] / 255, d[2] / 255];
  } catch {
    return fallback;
  }
}

/** Used when WebGL is not available: the plate still works, the wells are just not drawn. */
function createNullScene(): PlateScene {
  return {
    resize() {},
    setRowLayout() {},
    clear() {},
    setPressed() {},
    applyReport() {},
    setTheme() {},
    setReducedMotion() {},
    render() {},
    dispose() {},
  };
}

export function createPlateScene(canvas: HTMLCanvasElement, theme: Theme): PlateScene {
  let renderer: WebGLRenderer;
  try {
    // Edges are anti-aliased in the shaders, so multisampling is not needed.
    renderer = new WebGLRenderer({ canvas, alpha: true, antialias: false, premultipliedAlpha: true });
  } catch (err) {
    console.warn("Epoch 0: WebGL is not available, the wells are not drawn.", err);
    return createNullScene();
  }
  renderer.setClearColor(0x000000, 0);
  renderer.sortObjects = false;

  const camera = new OrthographicCamera(0, 1, 0, 1, -1, 1);
  const scene = new Scene();
  scene.matrixAutoUpdate = false;

  const uniforms = {
    uHair: { value: new Vector3() },
    uInk: { value: new Vector3() },
    uStain: { value: new Vector3() },
    uPx: { value: 1 },
  };
  const material = (vertexShader: string, fragmentShader: string): ShaderMaterial =>
    new ShaderMaterial({
      uniforms,
      vertexShader,
      fragmentShader,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
      side: DoubleSide, // the y-down camera flips the winding
    });
  const discMaterial = material(DISC_VERT, DISC_FRAG);
  const stripMaterial = material(STRIP_VERT, STRIP_FRAG);

  const guideCircle = new Dyn(PLATE_RINGS, 4);
  const guideStyle = new Dyn(PLATE_RINGS, 2);
  const stripSeg = new Dyn(STRIP_COUNT, 4);
  const stripStyle = new Dyn(STRIP_COUNT, 4);
  const discCircle = new Dyn(DISC_COUNT, 4);
  const discStyle = new Dyn(DISC_COUNT, 2);
  const dyns = [guideCircle, guideStyle, stripSeg, stripStyle, discCircle, discStyle];

  const guideGeometry = quadGeometry(-1);
  guideGeometry.setAttribute("aCircle", guideCircle.attr);
  guideGeometry.setAttribute("aStyle", guideStyle.attr);
  guideGeometry.instanceCount = PLATE_RINGS;
  const stripGeometry = quadGeometry(0);
  stripGeometry.setAttribute("aSeg", stripSeg.attr);
  stripGeometry.setAttribute("aStyle", stripStyle.attr);
  stripGeometry.instanceCount = STRIP_ARCS;
  const discGeometry = quadGeometry(-1);
  discGeometry.setAttribute("aCircle", discCircle.attr);
  discGeometry.setAttribute("aStyle", discStyle.attr);
  discGeometry.instanceCount = DISC_COUNT;

  const meshes = [
    new Mesh(guideGeometry, discMaterial),
    new Mesh(stripGeometry, stripMaterial),
    new Mesh(discGeometry, discMaterial),
  ];
  meshes.forEach((m, i) => {
    m.frustumCulled = false;
    m.matrixAutoUpdate = false;
    m.renderOrder = i;
    scene.add(m);
  });

  // ── state ──
  let geometry: WellGeometry[] = [];
  let sized = false;
  let dirty = true;
  let lost = false;
  let disposed = false;
  let reduced = false;

  const layouts: (RowLayout | null)[] = new Array<RowLayout | null>(ROWS).fill(null);
  const rowDots: (Float32Array | null)[] = new Array<Float32Array | null>(ROWS).fill(null);
  const rowLinks: (Float32Array | null)[] = new Array<Float32Array | null>(ROWS).fill(null);

  const UNITS = MAX_LAYERS * N;
  const active = new Uint8Array(WELLS);
  const held = new Uint8Array(WELLS);
  const pressed = new Uint8Array(WELLS);
  const level = new Float32Array(WELLS);
  const units = new Float32Array(WELLS * UNITS);
  const waves = new Float32Array(WELLS * WAVE_LEN);

  let pending: EngineReport | null = null;
  const pendingTicks = new Uint16Array(WELLS);
  let lastApplied = -Infinity;
  const tickStart = new Float64Array(WELLS * MAX_LAYERS).fill(-Infinity);
  const pulsing = new Set<number>(); // well * MAX_LAYERS + layer

  const rimSin = new Float32Array(RIM_SEGS);
  const rimCos = new Float32Array(RIM_SEGS);
  for (let k = 0; k < RIM_SEGS; k++) {
    rimSin[k] = Math.sin((k / RIM_SEGS) * TAU);
    rimCos[k] = Math.cos((k / RIM_SEGS) * TAU);
  }
  const rimTmp = new Float32Array(RIM_SEGS);

  // ── buffer writers ──
  function setStrip(
    i: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    half: number,
    tone: number,
    alpha: number,
    dash: number,
  ): void {
    const a = stripSeg.array;
    const s = stripStyle.array;
    const o = i * 4;
    a[o] = x0;
    a[o + 1] = y0;
    a[o + 2] = x1;
    a[o + 3] = y1;
    s[o] = half;
    s[o + 1] = tone;
    s[o + 2] = alpha;
    s[o + 3] = dash;
  }

  function setDisc(
    circle: Float32Array,
    style: Float32Array,
    i: number,
    x: number,
    y: number,
    radius: number,
    stroke: number,
    tone: number,
    alpha: number,
  ): void {
    const o = i * 4;
    circle[o] = x;
    circle[o + 1] = y;
    circle[o + 2] = radius;
    circle[o + 3] = stroke;
    style[i * 2] = tone;
    style[i * 2 + 1] = alpha;
  }

  /** True if the well's rings can be drawn (its circle is known and all its layers are laid out). */
  function drawable(w: number): boolean {
    const g = geometry[w];
    return !!g && g.r > 0 && layoutCovers(layouts[wellRow(w)], wellLayers(w));
  }

  function writeRim(w: number): void {
    const g = geometry[w];
    const start = STRIP_RIMS + w * RIM_SEGS;
    if (!g || !(g.r > 0)) {
      for (let k = 0; k < RIM_SEGS; k++) setStrip(start + k, 0, 0, 0, 0, 0, 0, 0, 0);
    } else {
      const down = pressed[w] === 1 || held[w] === 1;
      const half = down ? RIM_HALF_PRESSED_PX : RIM_HALF_PX;
      const tone = down ? TONE_STAIN : active[w] ? TONE_INK : TONE_HAIR;
      const base = Math.max(0, g.r - half);
      rimRadii(active[w] ? waves.subarray(w * WAVE_LEN, (w + 1) * WAVE_LEN) : null, rimTmp);
      let px = g.cx + base * rimTmp[0] * rimSin[0];
      let py = g.cy - base * rimTmp[0] * rimCos[0];
      for (let k = 0; k < RIM_SEGS; k++) {
        const j = k + 1 === RIM_SEGS ? 0 : k + 1;
        const qx = g.cx + base * rimTmp[j] * rimSin[j];
        const qy = g.cy - base * rimTmp[j] * rimCos[j];
        setStrip(start + k, px, py, qx, qy, half, tone, 1, CAPPED);
        px = qx;
        py = qy;
      }
    }
    stripSeg.mark(start, RIM_SEGS);
    stripStyle.mark(start, RIM_SEGS);
  }

  function writeGuide(w: number, layer: number, pulse: number): void {
    const i = wellRingOffset(w) + layer;
    const g = geometry[w];
    if (drawable(w)) {
      setDisc(
        guideCircle.array,
        guideStyle.array,
        i,
        g.cx,
        g.cy,
        g.r * ringRadius(layer),
        GUIDE_HALF_PX * (1 + pulse),
        TONE_HAIR + pulse * (TONE_INK - TONE_HAIR),
        GUIDE_ALPHA + (1 - GUIDE_ALPHA) * pulse,
      );
    } else {
      setDisc(guideCircle.array, guideStyle.array, i, 0, 0, 0, 0, 0, 0);
    }
    guideCircle.mark(i, 1);
    guideStyle.mark(i, 1);
  }

  /** Pulse of a ring at time `now` (0 when it is not ticking). */
  function pulseOf(w: number, layer: number, now: number): number {
    return reduced ? 0 : tickPulse(now - tickStart[w * MAX_LAYERS + layer]);
  }

  /** Links and ring guides of a well: they change only with the layout and the geometry. */
  function writeStatic(w: number, now: number): void {
    const layers = wellLayers(w);
    const g = geometry[w];
    const ok = drawable(w);
    for (let l = 0; l < layers; l++) writeGuide(w, l, ok ? pulseOf(w, l, now) : 0);

    const count = N * (layers - 1);
    const start = STRIP_LINKS + wellLinkOffset(w);
    const links = rowLinks[wellRow(w)];
    if (ok && links && !isSmallWell(g.r)) {
      const dash = Math.max(MIN_DASH_PX, g.r * DASH_PERIOD);
      for (let k = 0; k < count; k++) {
        const o = k * LINK_STRIDE;
        const weight = links[o + 4];
        setStrip(
          start + k,
          g.cx + g.r * links[o],
          g.cy + g.r * links[o + 1],
          g.cx + g.r * links[o + 2],
          g.cy + g.r * links[o + 3],
          Math.max(MIN_LINK_PX, g.r * linkHalfWidth(weight)),
          TONE_INK,
          linkOpacity(weight),
          weight < 0 ? dash : 0,
        );
      }
    } else {
      for (let k = 0; k < count; k++) setStrip(start + k, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    stripSeg.mark(start, count);
    stripStyle.mark(start, count);
  }

  /** Dots (or arc segments), centre dot and rim of a well: everything a report can change. */
  function writeLive(w: number): void {
    writeRim(w);
    const layers = wellLayers(w);
    const row = wellRow(w);
    const g = geometry[w];
    const ok = drawable(w);
    const small = ok && isSmallWell(g.r);
    const layout = layouts[row];
    const pos = rowDots[row];
    const count = N * layers;
    const dotStart = DISC_UNITS + wellDotOffset(w);
    const arcStart = STRIP_ARCS + wellDotOffset(w);
    const on = active[w] === 1;

    for (let k = 0; k < count; k++) {
      const state = on ? units[w * UNITS + k] : 0;
      const tone = state < 0 ? TONE_STAIN : TONE_INK;
      if (ok && pos && !small) {
        setDisc(
          discCircle.array,
          discStyle.array,
          dotStart + k,
          g.cx + g.r * pos[k * 2],
          g.cy + g.r * pos[k * 2 + 1],
          Math.max(MIN_DOT_PX, g.r * dotRadius(state)),
          0,
          tone,
          1,
        );
      } else {
        setDisc(discCircle.array, discStyle.array, dotStart + k, 0, 0, 0, 0, 0, 0);
      }
      if (small && layout) {
        const layer = (k / N) | 0;
        const arc = slotArc(layer, layout.slots[layer][k - layer * N]);
        setStrip(
          arcStart + k,
          g.cx + g.r * arc[0],
          g.cy + g.r * arc[1],
          g.cx + g.r * arc[2],
          g.cy + g.r * arc[3],
          Math.max(MIN_ARC_PX, g.r * dotRadius(state)),
          tone,
          1,
          0,
        );
      } else {
        setStrip(arcStart + k, 0, 0, 0, 0, 0, 0, 0, 0);
      }
    }
    discCircle.mark(dotStart, count);
    discStyle.mark(dotStart, count);
    stripSeg.mark(arcStart, count);
    stripStyle.mark(arcStart, count);

    if (ok) {
      const r = Math.max(MIN_DOT_PX, g.r * centreRadius(on ? level[w] : 0));
      setDisc(discCircle.array, discStyle.array, DISC_CENTRES + w, g.cx, g.cy, r, 0, TONE_INK, 1);
    } else {
      setDisc(discCircle.array, discStyle.array, DISC_CENTRES + w, 0, 0, 0, 0, 0, 0);
    }
    discCircle.mark(DISC_CENTRES + w, 1);
    discStyle.mark(DISC_CENTRES + w, 1);
  }

  function rebuildAll(): void {
    const now = typeof performance !== "undefined" ? performance.now() : 0;
    let anySmall = false;
    for (let w = 0; w < WELLS; w++) {
      writeStatic(w, now);
      writeLive(w);
      const g = geometry[w];
      if (g && isSmallWell(g.r)) anySmall = true;
    }
    // The arc segments sit at the end of the strip buffer and are only drawn when a well is small.
    stripGeometry.instanceCount = anySmall ? STRIP_COUNT : STRIP_ARCS;
    for (const d of dyns) d.markAll();
    dirty = true;
  }

  function restWell(w: number): void {
    active[w] = 0;
    held[w] = 0;
    level[w] = 0;
  }

  function sameWells(report: EngineReport): boolean {
    let n = 0;
    for (let w = 0; w < WELLS; w++) n += active[w];
    if (n !== report.wells.length) return false;
    for (const r of report.wells) if (r.well < 0 || r.well >= WELLS || !active[r.well]) return false;
    return true;
  }

  /** Move the latest report into the buffers. With reduced motion this happens REDUCED_HZ times a second. */
  function flushReport(now: number): void {
    const report = pending;
    if (!report) return;
    if (reduced && now - lastApplied < 1000 / REDUCED_HZ && sameWells(report)) return;
    pending = null;
    lastApplied = now;

    const seen = new Uint8Array(WELLS);
    for (const r of report.wells) {
      const w = r.well;
      if (!(w >= 0 && w < WELLS)) continue;
      seen[w] = 1;
      active[w] = 1;
      held[w] = r.held ? 1 : 0;
      level[w] = r.level;
      const n = Math.min(r.units.length, UNITS);
      for (let k = 0; k < n; k++) units[w * UNITS + k] = r.units[k];
      const m = Math.min(r.wave.length, WAVE_LEN);
      const off = w * WAVE_LEN;
      for (let k = 0; k < m; k++) waves[off + k] = r.wave[k];
      for (let k = m; k < WAVE_LEN; k++) waves[off + k] = 0;
      writeLive(w);
    }
    for (let w = 0; w < WELLS; w++) {
      if (seen[w]) {
        const ticks = pendingTicks[w];
        if (ticks && !reduced) {
          const layers = wellLayers(w);
          for (let l = TICK_FROM_LAYER; l < layers; l++) {
            if (ticks & (1 << l)) {
              tickStart[w * MAX_LAYERS + l] = now;
              pulsing.add(w * MAX_LAYERS + l);
            }
          }
        }
      } else if (active[w]) {
        restWell(w);
        writeLive(w);
      }
      pendingTicks[w] = 0;
    }
    dirty = true;
  }

  function stepPulses(now: number): void {
    if (pulsing.size === 0) return;
    for (const id of pulsing) {
      const w = (id / MAX_LAYERS) | 0;
      const layer = id - w * MAX_LAYERS;
      const age = now - tickStart[id];
      const done = reduced || !(age < TICK_MS);
      writeGuide(w, layer, done ? 0 : tickPulse(age));
      if (done) {
        tickStart[id] = -Infinity;
        pulsing.delete(id);
      }
    }
    dirty = true;
  }

  function applyTheme(t: Theme): void {
    uniforms.uHair.value.fromArray(resolveColour(t.hairline, [0.73, 0.71, 0.66]));
    uniforms.uInk.value.fromArray(resolveColour(t.ink, [0.11, 0.11, 0.1]));
    uniforms.uStain.value.fromArray(resolveColour(t.stain, [0.18, 0.37, 0.55]));
    dirty = true;
  }

  const onLost = (event: Event): void => {
    event.preventDefault(); // allow the browser to restore the context
    lost = true;
  };
  const onRestored = (): void => {
    // three.js rebuilds its programs and buffers from our arrays; one full frame is enough.
    lost = false;
    for (const d of dyns) d.markAll();
    dirty = true;
  };
  canvas.addEventListener("webglcontextlost", onLost);
  canvas.addEventListener("webglcontextrestored", onRestored);

  applyTheme(theme);
  rebuildAll();

  return {
    resize(width, height, dpr, wells) {
      if (disposed) return;
      const w = Math.max(0, Math.floor(width));
      const h = Math.max(0, Math.floor(height));
      sized = w > 0 && h > 0;
      geometry = wells.map((g) => ({ cx: g.cx, cy: g.cy, r: g.r }));
      if (!sized) return;
      const ratio = dpr > 0 ? dpr : 1;
      renderer.setPixelRatio(ratio);
      renderer.setSize(w, h, false); // the canvas is sized by CSS; only the drawing buffer changes
      camera.right = w;
      camera.bottom = h;
      camera.updateProjectionMatrix();
      uniforms.uPx.value = 1 / ratio;
      rebuildAll();
    },

    setRowLayout(row, layout) {
      if (disposed || !(row >= 0 && row < ROWS)) return;
      layouts[row] = layout;
      rowDots[row] = buildDotPositions(layout, layout.layers);
      rowLinks[row] = buildLinks(layout, layout.layers);
      const now = performance.now();
      for (let w = row * (WELLS / ROWS); w < (row + 1) * (WELLS / ROWS); w++) {
        writeStatic(w, now);
        writeLive(w);
      }
      dirty = true;
    },

    clear() {
      if (disposed) return;
      layouts.fill(null);
      rowDots.fill(null);
      rowLinks.fill(null);
      pending = null;
      pendingTicks.fill(0);
      tickStart.fill(-Infinity);
      pulsing.clear();
      pressed.fill(0);
      for (let w = 0; w < WELLS; w++) restWell(w);
      rebuildAll();
    },

    setPressed(well, down) {
      if (disposed || !(well >= 0 && well < WELLS)) return;
      const v = down ? 1 : 0;
      if (pressed[well] === v) return;
      pressed[well] = v;
      writeRim(well);
      dirty = true;
    },

    applyReport(report) {
      if (disposed) return;
      pending = report;
      for (const r of report.wells) if (r.well >= 0 && r.well < WELLS) pendingTicks[r.well] |= r.ticks;
    },

    setTheme(t) {
      if (disposed) return;
      applyTheme(t);
    },

    setReducedMotion(value) {
      reduced = value;
      // Pulses in flight are ended by stepPulses on the next frame.
    },

    render(now) {
      if (disposed || lost || !sized) return;
      flushReport(now);
      stepPulses(now);
      if (!dirty) return;
      renderer.render(scene, camera);
      for (const d of dyns) d.uploaded();
      dirty = false;
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      guideGeometry.dispose();
      stripGeometry.dispose();
      discGeometry.dispose();
      discMaterial.dispose();
      stripMaterial.dispose();
      renderer.dispose();
    },
  };
}
