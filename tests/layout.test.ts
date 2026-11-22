import { describe, expect, it } from "vitest";
import { COLS, MAX_LAYERS, N, ROWS, WELLS, wellLayers } from "../src/plate/constants";
import {
  LINK_STRIDE,
  PLATE_DOTS,
  PLATE_LINKS,
  PLATE_RINGS,
  SLOT_ANGLE,
  TAU,
  angularDistance,
  buildDotPositions,
  buildLinks,
  computeRowLayout,
  layoutCovers,
  layoutableLayers,
  placeRing,
  ringRadius,
  unitAngle,
  unitPosition,
  wellDotCount,
  wellDotOffset,
  wellLinkCount,
  wellLinkOffset,
  wellRingOffset,
} from "../src/view/layout";

/** C matrix [receiver (inner) * N + sender (outer)] with a small background and chosen entries. */
function matrix(entries: [recv: number, send: number, w: number][], background = 0): Float32Array {
  const c = new Float32Array(N * N).fill(background);
  for (const [recv, send, w] of entries) c[recv * N + send] = w;
  return c;
}

const identitySlots = (): Uint8Array => Uint8Array.from({ length: N }, (_, i) => i);

/** Small seeded generator (mulberry32), uniform in -0.433..0.433 like a Glorot 16 -> 16 layer. */
function randomMatrices(seed: number, layers: number): (Float32Array | null)[] {
  let a = seed >>> 0;
  const rnd = (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const C: (Float32Array | null)[] = [null];
  for (let i = 1; i < layers; i++) C.push(Float32Array.from({ length: N * N }, () => (rnd() * 2 - 1) * 0.433));
  return C;
}

const isPermutation = (slots: Uint8Array): boolean =>
  slots.length === N && [...slots].sort((x, y) => x - y).every((v, i) => v === i);

describe("rings and slots (steps 1-2)", () => {
  it("puts layer i at radius 0.20 + 0.10 i", () => {
    expect(ringRadius(0)).toBeCloseTo(0.2, 12);
    expect(ringRadius(3)).toBeCloseTo(0.5, 12);
    expect(ringRadius(7)).toBeCloseTo(0.9, 12);
  });

  it("has sixteen even slots and turns each ring half a slot against the one inside", () => {
    expect(SLOT_ANGLE).toBeCloseTo(TAU / 16, 12);
    expect(unitAngle(0, 0)).toBe(0);
    expect(unitAngle(0, 4)).toBeCloseTo(Math.PI / 2, 12);
    for (let layer = 1; layer < MAX_LAYERS; layer++) {
      for (let slot = 0; slot < N; slot++) {
        const turn = unitAngle(layer, slot) - unitAngle(layer - 1, slot);
        expect(angularDistance(turn, SLOT_ANGLE / 2)).toBeLessThan(1e-12);
        expect(unitAngle(layer, slot)).toBeGreaterThanOrEqual(0);
        expect(unitAngle(layer, slot)).toBeLessThan(TAU);
      }
    }
  });

  it("measures positions clockwise from twelve o'clock with y down", () => {
    const layout = computeRowLayout([null]);
    const [x0, y0] = unitPosition(layout, 0, 0);
    expect(x0).toBeCloseTo(0, 12);
    expect(y0).toBeCloseTo(-0.2, 12);
    const [x4, y4] = unitPosition(layout, 0, 4); // a quarter turn: three o'clock
    expect(x4).toBeCloseTo(0.2, 12);
    expect(y4).toBeCloseTo(0, 12);
  });
});

describe("computeRowLayout (steps 3-4)", () => {
  it("lays out nothing for an empty row and keeps library order in ring 0", () => {
    const empty = computeRowLayout([]);
    expect(empty.layers).toBe(0);
    expect(empty.slots).toEqual([]);
    const one = computeRowLayout([null]);
    expect(one.layers).toBe(1);
    expect([...one.slots[0]]).toEqual([...identitySlots()]);
    expect(one.links).toEqual([null]);
  });

  it("stops at the first missing matrix", () => {
    const C = randomMatrices(1, 5);
    C[3] = null;
    expect(layoutableLayers(C)).toBe(3);
    const layout = computeRowLayout(C);
    expect(layout.layers).toBe(3);
    expect(layout.slots.length).toBe(3);
    expect(layout.links.length).toBe(3);
    expect(layoutCovers(layout, 3)).toBe(true);
    expect(layoutCovers(layout, 4)).toBe(false);
    expect(layoutCovers(null, 1)).toBe(false);
  });

  it("finds each inner unit's strongest sender by absolute weight and keeps the sign", () => {
    // Inner unit 3 hears +0.30 from outer 2 and -0.40 from outer 9: the sender is 9, the link is negative.
    // Inner unit 4 hears -0.20 from outer 7 and +0.25 from outer 11: the sender is 11, positive.
    const c = matrix(
      [
        [3, 2, 0.3],
        [3, 9, -0.4],
        [4, 7, -0.2],
        [4, 11, 0.25],
      ],
      0.01,
    );
    const { links } = computeRowLayout([null, c]);
    expect(links[0]).toBeNull();
    expect(links[1]!.sender[3]).toBe(9);
    expect(links[1]!.weight[3]).toBeCloseTo(-0.4, 6);
    expect(links[1]!.sender[4]).toBe(11);
    expect(links[1]!.weight[4]).toBeCloseTo(0.25, 6);
  });

  it("breaks ties in absolute weight towards the lower unit index", () => {
    const c = matrix(
      [
        [0, 6, -0.5],
        [0, 12, 0.5],
      ],
      0,
    );
    const { links } = computeRowLayout([null, c]);
    expect(links[1]!.sender[0]).toBe(6);
    expect(links[1]!.weight[0]).toBe(-0.5);
    // A row of zeros: every sender ties, unit 0 wins.
    expect(links[1]!.sender[1]).toBe(0);
    expect(links[1]!.weight[1]).toBe(0);
  });

  it("places each outer unit at the inner unit it is strongest sender to", () => {
    // Inner unit u's strongest sender is outer unit p[u]. Each outer unit aims exactly at one inner unit,
    // so sorted by angle outer unit p[u] takes slot u. (Rotations 0 and -1 fit equally well: half a slot
    // to either side. The smaller rotation, 0, wins.)
    const p = [5, 12, 0, 9, 3, 15, 7, 1, 14, 2, 11, 6, 13, 4, 10, 8];
    const c = matrix(
      p.map((send, recv) => [recv, send, recv % 2 ? -0.4 : 0.4]),
      0.01,
    );
    const layout = computeRowLayout([null, c]);
    for (let u = 0; u < N; u++) {
      expect(layout.slots[1][p[u]]).toBe(u);
      expect(layout.links[1]!.sender[u]).toBe(p[u]);
      expect(layout.links[1]!.weight[u]).toBeCloseTo(u % 2 ? -0.4 : 0.4, 6);
    }
    // The same matrix again one ring further out: ring 2 follows the order of ring 1.
    const three = computeRowLayout([null, c, c]);
    for (let u = 0; u < N; u++) expect(three.slots[2][p[u]]).toBe(three.slots[1][u]);
  });

  it("aims for the circular mean of the inner units, across the 0 / 2π seam", () => {
    // Outer unit 5 is strongest sender to inner units 15 (337.5°) and 1 (22.5°).
    // Their circular mean is 0°, not the arithmetic mean 180°.
    const entries: [number, number, number][] = [
      [15, 5, 0.4],
      [1, 5, 0.4],
    ];
    // Every other inner unit u listens to outer unit u (inner 5 listens to outer 1, so 5 keeps only its two).
    for (let u = 0; u < N; u++) if (u !== 15 && u !== 1) entries.push([u, u === 5 ? 1 : u, 0.4]);
    const placed = placeRing(matrix(entries, 0.01), identitySlots(), 1);
    expect(angularDistance(placed.aim[5], 0)).toBeLessThan(1e-8);
    expect(placed.aim[5]).toBeLessThan(1e-8); // stored as 0, not as just under 2π

    // Three units on one side of the seam: 315°, 337.5° and 22.5° -> about 344.5°.
    const more = placeRing(
      matrix(
        [
          [14, 5, 0.4],
          [15, 5, 0.4],
          [1, 5, 0.4],
        ],
        0.01,
      ),
      identitySlots(),
      1,
    );
    const deg = (more.aim[5] * 360) / TAU;
    expect(deg).toBeGreaterThan(337.5);
    expect(deg).toBeLessThan(360);
    expect(deg).toBeCloseTo(344.53, 1);
  });

  it("uses the plain mean away from the seam", () => {
    // Outer unit 2 is strongest sender to inner units 4 (90°) and 6 (135°): it aims for 112.5°.
    const placed = placeRing(
      matrix(
        [
          [4, 2, 0.4],
          [6, 2, -0.4],
        ],
        0.01,
      ),
      identitySlots(),
      1,
    );
    expect(placed.aim[2]).toBeCloseTo((112.5 / 360) * TAU, 8);
  });

  it("sends a unit that is nobody's strongest sender to the unit it drives most", () => {
    // Outer unit 15 never wins a row (every inner unit u prefers outer unit u mod 15),
    // but its largest weight, -0.3, goes to inner unit 9 at 202.5°.
    const entries: [number, number, number][] = [];
    for (let u = 0; u < N; u++) entries.push([u, u % 15, 0.4]);
    for (let u = 0; u < N; u++) entries.push([u, 15, 0.05]);
    entries.push([9, 15, -0.3]);
    const placed = placeRing(matrix(entries, 0.01), identitySlots(), 1);
    expect([...placed.sender]).not.toContain(15);
    expect(placed.aim[15]).toBeCloseTo(9 * SLOT_ANGLE, 8);
    // It lands next to outer unit 9, which aims at the same inner unit; the lower index comes first.
    expect((placed.slots[15] - placed.slots[9] + N) % N).toBe(1);
  });

  it("falls back to the most driven member when the mean has no direction", () => {
    // Outer unit 3 is strongest sender to inner units 0 (0°) and 8 (180°), which cancel out.
    // It drives unit 8 harder, so it aims for 180°.
    const placed = placeRing(
      matrix(
        [
          [0, 3, 0.3],
          [8, 3, -0.4],
        ],
        0.01,
      ),
      identitySlots(),
      1,
    );
    expect(placed.aim[3]).toBeCloseTo(Math.PI, 8);
  });

  it("gives consecutive slots in order of aim, rotated to the least total angular distance", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const c = randomMatrices(seed, 2)[1] as Float32Array;
      const inner = identitySlots();
      const placed = placeRing(c, inner, 1);
      // Walking the slots from the rotation onwards visits the units in non-decreasing aim.
      const bySlot = new Array<number>(N);
      for (let s = 0; s < N; s++) bySlot[(placed.slots[s] - placed.rotation + N) % N] = s;
      for (let k = 1; k < N; k++) expect(placed.aim[bySlot[k]]).toBeGreaterThanOrEqual(placed.aim[bySlot[k - 1]]);
      // No other rotation of that order fits better (ties within 1e-6 go to the smaller rotation).
      const cost = (rot: number): number =>
        bySlot.reduce((sum, s, k) => sum + angularDistance(unitAngle(1, (k + rot) % N), placed.aim[s]), 0);
      for (let rot = 0; rot < N; rot++) expect(cost(placed.rotation)).toBeLessThanOrEqual(cost(rot) + 1e-6);
    }
  });

  it("makes every ring a permutation of the sixteen slots", () => {
    for (let seed = 1; seed <= 50; seed++) {
      const layout = computeRowLayout(randomMatrices(seed, MAX_LAYERS));
      expect(layout.layers).toBe(MAX_LAYERS);
      expect(layout.slots.length).toBe(MAX_LAYERS);
      for (const slots of layout.slots) expect(isPermutation(slots)).toBe(true);
      for (let i = 1; i < MAX_LAYERS; i++) {
        const link = layout.links[i]!;
        expect(link.sender.length).toBe(N);
        expect(link.weight.length).toBe(N);
        for (let u = 0; u < N; u++) expect(link.sender[u]).toBeLessThan(N);
      }
    }
    // Degenerate input still gives permutations.
    const zeros = computeRowLayout([null, new Float32Array(N * N), new Float32Array(N * N)]);
    for (const slots of zeros.slots) expect(isPermutation(slots)).toBe(true);
  });

  it("links carry the true strongest weight of the matrix", () => {
    const C = randomMatrices(7, MAX_LAYERS);
    const layout = computeRowLayout(C);
    for (let i = 1; i < MAX_LAYERS; i++) {
      const c = C[i] as Float32Array;
      const link = layout.links[i]!;
      for (let u = 0; u < N; u++) {
        const row = [...c.subarray(u * N, (u + 1) * N)];
        const strongest = Math.max(...row.map(Math.abs));
        expect(Math.abs(link.weight[u])).toBe(strongest);
        expect(link.weight[u]).toBe(c[u * N + link.sender[u]]);
      }
    }
  });

  it("is deterministic", () => {
    const a = computeRowLayout(randomMatrices(42, MAX_LAYERS));
    const b = computeRowLayout(randomMatrices(42, MAX_LAYERS));
    expect(a).toEqual(b);
  });

  it("never moves inner rings when layers are added", () => {
    for (let seed = 1; seed <= 10; seed++) {
      const C = randomMatrices(seed, MAX_LAYERS);
      const full = computeRowLayout(C);
      for (let layers = 1; layers < MAX_LAYERS; layers++) {
        const part = computeRowLayout(C.slice(0, layers));
        expect(part.layers).toBe(layers);
        for (let l = 0; l < layers; l++) {
          expect([...part.slots[l]]).toEqual([...full.slots[l]]);
          expect(part.links[l]).toEqual(full.links[l]);
        }
      }
    }
  });
});

describe("counts and buffers", () => {
  it("has 16 dots per ring and 16 links per pair of rings", () => {
    expect(wellDotCount(1)).toBe(16);
    expect(wellLinkCount(1)).toBe(0);
    expect(wellLinkCount(2)).toBe(16);
    expect(wellDotCount(8)).toBe(128);
    expect(wellLinkCount(8)).toBe(112);
  });

  it("has 3,456 dots, 2,688 links and 216 rings on the plate", () => {
    let dots = 0;
    let links = 0;
    let rings = 0;
    for (let w = 0; w < WELLS; w++) {
      dots += wellDotCount(wellLayers(w));
      links += wellLinkCount(wellLayers(w));
      rings += wellLayers(w);
    }
    expect(dots).toBe(3456);
    expect(links).toBe(2688);
    expect(rings).toBe(216);
    expect(PLATE_DOTS).toBe(3456);
    expect(PLATE_LINKS).toBe(2688);
    expect(PLATE_RINGS).toBe(216);
    expect(ROWS * COLS).toBe(48);
  });

  it("packs the wells back to back in plate-wide buffers", () => {
    let dots = 0;
    let links = 0;
    let rings = 0;
    for (let w = 0; w < WELLS; w++) {
      expect(wellDotOffset(w)).toBe(dots);
      expect(wellLinkOffset(w)).toBe(links);
      expect(wellRingOffset(w)).toBe(rings);
      dots += wellDotCount(wellLayers(w));
      links += wellLinkCount(wellLayers(w));
      rings += wellLayers(w);
    }
  });

  it("builds dot positions on the rings, one per slot", () => {
    const layout = computeRowLayout(randomMatrices(3, MAX_LAYERS));
    const pos = buildDotPositions(layout, MAX_LAYERS);
    expect(pos.length).toBe(128 * 2);
    for (let l = 0; l < MAX_LAYERS; l++) {
      const angles: number[] = [];
      for (let u = 0; u < N; u++) {
        const x = pos[(l * N + u) * 2];
        const y = pos[(l * N + u) * 2 + 1];
        expect(Math.hypot(x, y)).toBeCloseTo(ringRadius(l), 6);
        const [ux, uy] = unitPosition(layout, l, u);
        expect(x).toBeCloseTo(ux, 6);
        expect(y).toBeCloseTo(uy, 6);
        angles.push(Math.round(((Math.atan2(x, -y) + TAU) % TAU) / (SLOT_ANGLE / 2)));
      }
      // Sixteen different half-slot positions, all of the parity of the layer (the half-slot turn).
      expect(new Set(angles).size).toBe(N);
      for (const a of angles) expect(a % 2).toBe(l % 2);
    }
    // A well with fewer layers is a prefix of the same buffer.
    const three = buildDotPositions(layout, 3);
    expect([...three]).toEqual([...pos.subarray(0, three.length)]);
  });

  it("builds one link from each inner unit to its strongest sender", () => {
    const layout = computeRowLayout(randomMatrices(5, MAX_LAYERS));
    const links = buildLinks(layout, MAX_LAYERS);
    expect(links.length).toBe(112 * LINK_STRIDE);
    for (let i = 1; i < MAX_LAYERS; i++) {
      for (let u = 0; u < N; u++) {
        const o = ((i - 1) * N + u) * LINK_STRIDE;
        const [x0, y0] = unitPosition(layout, i - 1, u);
        const [x1, y1] = unitPosition(layout, i, layout.links[i]!.sender[u]);
        expect(links[o]).toBeCloseTo(x0, 6);
        expect(links[o + 1]).toBeCloseTo(y0, 6);
        expect(links[o + 2]).toBeCloseTo(x1, 6);
        expect(links[o + 3]).toBeCloseTo(y1, 6);
        expect(links[o + 4]).toBe(layout.links[i]!.weight[u]);
        // Inner end on ring i - 1, outer end on ring i.
        expect(Math.hypot(links[o], links[o + 1])).toBeCloseTo(ringRadius(i - 1), 6);
        expect(Math.hypot(links[o + 2], links[o + 3])).toBeCloseTo(ringRadius(i), 6);
      }
    }
    expect(buildLinks(layout, 1).length).toBe(0);
    expect([...buildLinks(layout, 4)]).toEqual([...links.subarray(0, 48 * LINK_STRIDE)]);
  });
});
