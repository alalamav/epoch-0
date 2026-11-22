// Drawing rule, plan section 6: weights -> ring order and links, plus the pure geometry that the
// plate scene (scene.ts) and the dish view (dish.ts) share. No DOM and no three.js in this file,
// so everything here runs in node and is covered by tests/layout.test.ts and tests/view-geometry.test.ts.
//
// Conventions
// - Well units: the well has radius 1 and its centre is the origin.
// - Angles: 0 is twelve o'clock and angles grow clockwise on screen.
// - Positions are screen-oriented: x to the right, y DOWN. So angle a at radius r is (r sin a, -r cos a).

import {
  COLS,
  DOT_GAIN,
  DOT_R0,
  MAX_LAYERS,
  N,
  RING_R0,
  RING_STEP,
  ROWS,
  WAVE_LEN,
} from "../plate/constants";
import type { RowLayout } from "../contracts";

export const TAU = Math.PI * 2;
/** Angle between neighbouring slots of a ring. */
export const SLOT_ANGLE = TAU / N;

// ───────────────────────────── steps 1-2: rings and slots ─────────────────────────────

/** Step 1. Radius of the ring of layer `layer` (0 is the fastest, innermost). */
export function ringRadius(layer: number): number {
  return RING_R0 + RING_STEP * layer;
}

/** Wrap an angle into [0, 2π). */
export function wrapAngle(a: number): number {
  const w = a % TAU;
  return w < 0 ? w + TAU : w;
}

/** Shortest distance between two angles, 0..π. */
export function angularDistance(a: number, b: number): number {
  const d = wrapAngle(a - b);
  return d > Math.PI ? TAU - d : d;
}

/**
 * Step 2. Angle of slot `slot` on the ring of `layer`, in [0, 2π).
 * Each ring is turned half a slot (clockwise) against the one inside it.
 */
export function unitAngle(layer: number, slot: number): number {
  return wrapAngle((slot + 0.5 * layer) * SLOT_ANGLE);
}

/** Point at `angle` and `radius`, in well units (y down). */
export function polar(radius: number, angle: number): [number, number] {
  return [radius * Math.sin(angle), -radius * Math.cos(angle)];
}

/** Position of a unit in well units (well radius 1, centre at the origin, y down). */
export function unitPosition(layout: RowLayout, layer: number, unit: number): [number, number] {
  return polar(ringRadius(layer), unitAngle(layer, layout.slots[layer][unit]));
}

// ───────────────────────────── steps 3-4: ring order and links ─────────────────────────────

/** Aimed angles are rounded to this grid so that rounding noise cannot reorder equal aims. */
const AIM_GRID = 1e-9;
const FULL_TURN = Math.round(TAU / AIM_GRID);
/**
 * A rotation has to beat the best one so far by more than this to replace it. Far above the rounding
 * of sixteen aims on the grid, so rotations that fit equally well on paper also tie here.
 */
const FIT_EPS = 1e-6;
/** Below this resultant length a circular mean has no direction (e.g. two opposite units). */
const MEAN_EPS = 1e-9;

/** How many layers of a row can be laid out from `C`: C[0] (null) plus the unbroken run of matrices after it. */
export function layoutableLayers(C: readonly (Float32Array | null)[]): number {
  if (C.length === 0) return 0;
  let layers = 1;
  while (layers < C.length && layers < MAX_LAYERS) {
    const m = C[layers];
    if (!m || m.length < N * N) break;
    layers++;
  }
  return layers;
}

/** Result of placing one ring: steps 3 and 4 for a single pair of rings. */
export interface RingPlacement {
  /** slot[s] of each outer unit. */
  slots: Uint8Array;
  /** sender[u]: strongest sender (outer unit) of each inner unit. */
  sender: Uint8Array;
  /** weight[u]: the signed weight of that link. */
  weight: Float32Array;
  /** aim[s]: the angle each outer unit aimed for, in [0, 2π). */
  aim: Float64Array;
  /** The rotation (in slots) chosen for the sorted order. */
  rotation: number;
}

/**
 * Place the ring of layer `layer` (outer) around the ring of layer `layer - 1` (inner).
 * `c` is indexed [receiver in the inner ring * N + sender in the outer ring].
 *
 * Tie-breaks, all deterministic:
 * - equal absolute weights: the lower unit index wins;
 * - equal aims: the lower unit index gets the earlier slot;
 * - a mean angle without a direction (members cancel out): the unit aims for the member it drives most;
 * - equally good rotations: the smallest rotation wins.
 */
export function placeRing(c: Float32Array, innerSlots: Uint8Array, layer: number): RingPlacement {
  const inner = layer - 1;
  const innerAngle = new Float64Array(N);
  for (let u = 0; u < N; u++) innerAngle[u] = unitAngle(inner, innerSlots[u]);

  // Strongest sender of each inner unit.
  const sender = new Uint8Array(N);
  const weight = new Float32Array(N);
  for (let u = 0; u < N; u++) {
    let best = 0;
    let bestAbs = -1;
    for (let s = 0; s < N; s++) {
      const a = Math.abs(c[u * N + s]);
      if (a > bestAbs) {
        bestAbs = a;
        best = s;
      }
    }
    sender[u] = best;
    weight[u] = c[u * N + best];
  }

  // Where each outer unit wants to sit.
  const aim = new Float64Array(N);
  for (let s = 0; s < N; s++) {
    let count = 0;
    let sumSin = 0;
    let sumCos = 0;
    let only = -1;
    let strongestMember = -1;
    let strongestMemberAbs = -1;
    let strongestAny = 0;
    let strongestAnyAbs = -1;
    for (let u = 0; u < N; u++) {
      const a = Math.abs(c[u * N + s]);
      if (a > strongestAnyAbs) {
        strongestAnyAbs = a;
        strongestAny = u;
      }
      if (sender[u] !== s) continue;
      count++;
      only = u;
      sumSin += Math.sin(innerAngle[u]);
      sumCos += Math.cos(innerAngle[u]);
      if (a > strongestMemberAbs) {
        strongestMemberAbs = a;
        strongestMember = u;
      }
    }
    let target: number;
    if (count === 0) {
      target = innerAngle[strongestAny];
    } else if (count === 1) {
      target = innerAngle[only];
    } else if (Math.hypot(sumSin, sumCos) < MEAN_EPS) {
      target = innerAngle[strongestMember];
    } else {
      target = Math.atan2(sumSin, sumCos);
    }
    // Round to the grid; an aim that rounds up to a full turn is angle 0.
    const q = Math.round(wrapAngle(target) / AIM_GRID);
    aim[s] = q >= FULL_TURN ? 0 : q * AIM_GRID;
  }

  // Sort by aimed angle; ties keep unit order.
  const order: number[] = [];
  for (let s = 0; s < N; s++) order.push(s);
  order.sort((a, b) => aim[a] - aim[b] || a - b);

  // Consecutive slots, rotated to the best fit: least total angular distance to the aims.
  let rotation = 0;
  let bestCost = Infinity;
  for (let rot = 0; rot < N; rot++) {
    let cost = 0;
    for (let k = 0; k < N; k++) cost += angularDistance(unitAngle(layer, (k + rot) % N), aim[order[k]]);
    if (cost < bestCost - FIT_EPS) {
      bestCost = cost;
      rotation = rot;
    }
  }

  const slots = new Uint8Array(N);
  for (let k = 0; k < N; k++) slots[order[k]] = (k + rotation) % N;
  return { slots, sender, weight, aim, rotation };
}

/**
 * Steps 1-4 for a row. `C[i]` (i >= 1) is the dense layer from layer i down to layer i - 1,
 * indexed [receiver in the inner ring * N + sender in the outer ring]; `C[0]` is null.
 *
 * - An empty `C` gives a layout with 0 layers; `[null]` gives 1 layer; the layout stops at the first missing matrix.
 * - Ring 0 keeps library order. Ring i depends only on C[1..i], so more layers never move inner rings.
 */
export function computeRowLayout(C: (Float32Array | null)[]): RowLayout {
  const layers = layoutableLayers(C);
  const slots: Uint8Array[] = [];
  const links: RowLayout["links"] = [];
  if (layers === 0) return { layers, slots, links };

  const identity = new Uint8Array(N);
  for (let u = 0; u < N; u++) identity[u] = u;
  slots.push(identity);
  links.push(null);

  for (let i = 1; i < layers; i++) {
    const placed = placeRing(C[i] as Float32Array, slots[i - 1], i);
    slots.push(placed.slots);
    links.push({ sender: placed.sender, weight: placed.weight });
  }
  return { layers, slots, links };
}

/** True if every layer of a well with `layers` layers has been laid out. */
export function layoutCovers(layout: RowLayout | null | undefined, layers: number): boolean {
  return !!layout && layout.layers >= layers;
}

// ───────────────────────────── counts ─────────────────────────────

/** Unit dots in a well with `layers` layers (the centre dot is not counted). */
export const wellDotCount = (layers: number): number => N * layers;
/** Strongest-sender links in a well with `layers` layers. */
export const wellLinkCount = (layers: number): number => N * Math.max(0, layers - 1);
/** Rings in all wells of one row: 1 + 2 + ... + COLS. */
export const ROW_RINGS = (COLS * (COLS + 1)) / 2;
/** Unit dots on the whole plate: 3,456. */
export const PLATE_DOTS = ROWS * N * ROW_RINGS;
/** Links on the whole plate: 2,688. */
export const PLATE_LINKS = ROWS * N * ((COLS * (COLS - 1)) / 2);
/** Rings on the whole plate: 216. */
export const PLATE_RINGS = ROWS * ROW_RINGS;

/**
 * Offsets of a well inside plate-wide buffers that hold every well's rings / dots / links in well order
 * (row-major). A well with L layers owns L rings, 16 L dots and 16 (L - 1) links from its offset on.
 */
export function wellRingOffset(well: number): number {
  const col = well % COLS;
  return Math.floor(well / COLS) * ROW_RINGS + (col * (col + 1)) / 2;
}
export const wellDotOffset = (well: number): number => N * wellRingOffset(well);
export function wellLinkOffset(well: number): number {
  const col = well % COLS;
  return N * (Math.floor(well / COLS) * ((COLS * (COLS - 1)) / 2) + (col * (col - 1)) / 2);
}

// ───────────────────────────── geometry builders (well units) ─────────────────────────────

/** Positions of the first `layers` rings of a row: [x, y] per unit, indexed [(layer * N + unit) * 2]. */
export function buildDotPositions(layout: RowLayout, layers: number): Float32Array {
  const out = new Float32Array(wellDotCount(layers) * 2);
  for (let l = 0; l < layers; l++) {
    for (let u = 0; u < N; u++) {
      const [x, y] = unitPosition(layout, l, u);
      const o = (l * N + u) * 2;
      out[o] = x;
      out[o + 1] = y;
    }
  }
  return out;
}

/** Floats per link in `buildLinks`: x0, y0 (inner unit), x1, y1 (its strongest sender), signed weight. */
export const LINK_STRIDE = 5;

/**
 * The links of a well with `layers` layers, LINK_STRIDE floats each.
 * Link index is (pair - 1) * N + inner unit, where `pair` is the outer layer 1 .. layers-1.
 */
export function buildLinks(layout: RowLayout, layers: number): Float32Array {
  const out = new Float32Array(wellLinkCount(layers) * LINK_STRIDE);
  for (let i = 1; i < layers; i++) {
    const link = layout.links[i];
    if (!link) continue;
    for (let u = 0; u < N; u++) {
      const [x0, y0] = unitPosition(layout, i - 1, u);
      const [x1, y1] = unitPosition(layout, i, link.sender[u]);
      const o = ((i - 1) * N + u) * LINK_STRIDE;
      out[o] = x0;
      out[o + 1] = y0;
      out[o + 2] = x1;
      out[o + 3] = y1;
      out[o + 4] = link.weight[u];
    }
  }
  return out;
}

/** Share of a slot that an arc segment covers (step 9). */
export const ARC_FILL = 0.7;

/** Step 9. End points [x0, y0, x1, y1] of the arc segment that stands for a slot on a small well. */
export function slotArc(layer: number, slot: number): [number, number, number, number] {
  const a = unitAngle(layer, slot);
  const half = 0.5 * ARC_FILL * SLOT_ANGLE;
  const r = ringRadius(layer);
  const [x0, y0] = polar(r, a - half);
  const [x1, y1] = polar(r, a + half);
  return [x0, y0, x1, y1];
}

// ───────────────────────────── live values ─────────────────────────────

const clamp01 = (v: number): number => (v > 0 ? (v < 1 ? v : 1) : 0); // NaN -> 0

/** Step 5. Radius of a unit dot in well units for a display value in -1..1. */
export function dotRadius(state: number): number {
  return DOT_R0 + DOT_GAIN * clamp01(Math.abs(state));
}

/** Step 5. Negative states are drawn in the stain colour (and open in the dish view). */
export const isNegative = (state: number): boolean => state < 0;

/**
 * Largest absolute weight TF.js gives a 16 -> 16 dense layer (Glorot uniform: sqrt(6 / (16 + 16)) = 0.433).
 * Reference for the strength of a down-going weight.
 */
export const DENSE_LIMIT = Math.sqrt(6 / (N + N));

/** |weight| relative to `ref`, clamped to 0..1. */
export function weightStrength(weight: number, ref: number = DENSE_LIMIT): number {
  return clamp01(Math.abs(weight) / ref);
}

/**
 * Step 4. Half width of a link in well units. The strongest sender of a unit is the largest of 16
 * weights, so these all sit in the top of the range; the cube spreads them so the width can be read.
 */
export function linkHalfWidth(weight: number): number {
  const s = weightStrength(weight);
  return 0.0015 + 0.0065 * s * s * s;
}

/** Step 4. Opacity of a link, 0.3..0.9. */
export function linkOpacity(weight: number): number {
  const s = weightStrength(weight);
  return 0.3 + 0.6 * s * s * s;
}

/** Step 10. Opacity of one weight among all weights in the dish view, 0..1 (square of its strength). */
export function weightOpacity(weight: number, ref: number): number {
  const s = weightStrength(weight, ref);
  return s * s;
}

/** Radius of the centre dot at rest, in well units. */
export const CENTRE_R0 = 0.012;
/** How much the centre dot can grow. Stays well inside the innermost ring (0.2). */
export const CENTRE_GAIN = 0.075;
/** Level (RMS after the fixed gain) at which the centre dot is full size. */
export const CENTRE_LEVEL_REF = 0.5;

/** Step 6. Radius of the centre dot in well units for an output level (RMS). */
export function centreRadius(level: number): number {
  return CENTRE_R0 + CENTRE_GAIN * Math.sqrt(clamp01(level / CENTRE_LEVEL_REF));
}

/** Rim displacement in well radii per unit of output. */
export const RIM_GAIN = 0.06;
/** Samples at each end of the wave over which the displacement fades, so the rim closes on itself. */
export const RIM_TAPER = 16;

/**
 * Step 7. Radius of the rim (in well radii) at each of WAVE_LEN points around the well; point k is at
 * angle k / WAVE_LEN * 2π. `wave` null (a resting well) gives a plain circle of radius 1.
 */
export function rimRadii(wave: ArrayLike<number> | null, out: Float32Array = new Float32Array(WAVE_LEN)): Float32Array {
  const n = out.length;
  if (!wave) {
    out.fill(1);
    return out;
  }
  const len = wave.length;
  for (let k = 0; k < n; k++) {
    let v = k < len ? wave[k] : 0;
    if (!(v > -1)) v = v < -1 ? -1 : 0; // clamp below; NaN -> 0
    if (v > 1) v = 1;
    const edge = Math.min(k, n - 1 - k);
    const taper = edge >= RIM_TAPER ? 1 : 0.5 - 0.5 * Math.cos((Math.PI * (edge + 0.5)) / RIM_TAPER);
    out[k] = 1 + RIM_GAIN * v * taper;
  }
  return out;
}

/** How long a ring stays darkened after its layer ticks. */
export const TICK_MS = 450;

/** Step 8. Darkening of a ring `age` milliseconds after its layer ticked: 1 at the tick, 0 from TICK_MS on. */
export function tickPulse(age: number): number {
  if (!(age >= 0) || age >= TICK_MS) return 0;
  const t = 1 - age / TICK_MS;
  return t * t;
}

/** Step 9. Below this well radius (CSS pixels) links are dropped and rings become arc segments. */
export const SMALL_WELL_PX = 40;
export const isSmallWell = (radiusPx: number): boolean => radiusPx < SMALL_WELL_PX;

/** Dot updates per second when reduced motion is set (plan section 7). */
export const REDUCED_HZ = 4;

// ───────────────────────────── colours ─────────────────────────────

/**
 * Parse "#rgb", "#rgba", "#rrggbb", "#rrggbbaa", "rgb(...)" or "rgba(...)" into [r, g, b, a], each 0..1.
 * Returns null for anything else (the scene then asks the browser).
 */
export function parseCssColor(css: string): [number, number, number, number] | null {
  const s = css.trim().toLowerCase();
  if (s.startsWith("#")) {
    const hex = s.slice(1);
    if (!/^[0-9a-f]+$/.test(hex)) return null;
    if (hex.length === 3 || hex.length === 4) {
      const v = [...hex].map((ch) => parseInt(ch + ch, 16) / 255);
      return [v[0], v[1], v[2], v.length === 4 ? v[3] : 1];
    }
    if (hex.length === 6 || hex.length === 8) {
      const v: number[] = [];
      for (let i = 0; i < hex.length; i += 2) v.push(parseInt(hex.slice(i, i + 2), 16) / 255);
      return [v[0], v[1], v[2], v.length === 4 ? v[3] : 1];
    }
    return null;
  }
  const m = /^rgba?\(([^)]*)\)$/.exec(s);
  if (!m) return null;
  const parts = m[1].split(/[\s,/]+/).filter((p) => p.length > 0);
  if (parts.length !== 3 && parts.length !== 4) return null;
  const channel = (p: string, scale: number): number =>
    p.endsWith("%") ? parseFloat(p) / 100 : parseFloat(p) / scale;
  const out: [number, number, number, number] = [
    channel(parts[0], 255),
    channel(parts[1], 255),
    channel(parts[2], 255),
    parts.length === 4 ? channel(parts[3], 1) : 1,
  ];
  if (out.some((v) => Number.isNaN(v))) return null;
  return [clamp01(out[0]), clamp01(out[1]), clamp01(out[2]), clamp01(out[3])];
}
