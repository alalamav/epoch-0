// Plate number -> one seed per weight array (plan section 4: "Each weight array gets a seed derived
// from plate number, row, layer and role").
//
// The seed is injective: (plate, row, layer, role) is packed into one integer below 2^32 and passed
// through a 32-bit mixer that is a bijection, so two different weight arrays can never share a seed.
// The seed is never 0, because TF.js treats a seed of 0 as "no seed" in its normal sampler
// (`seed ? seed : Math.random()` in tfjs-core rand_util.js), which the orthogonal initialiser uses.

import type { WeightRole } from "../contracts";
import { MAX_LAYERS, PLATE_COUNT, ROWS } from "./constants";

const ROLE_INDEX: Record<WeightRole, number> = { kernel: 0, recurrent: 1, down: 2, out: 3 };
const ROLE_COUNT = 4;

/** murmur3 finaliser: a bijection on 32-bit integers that maps only 0 to 0. */
function mix32(x: number): number {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}

/**
 * Seed of one weight array: an integer in 1 .. 2^32 - 1, the same for the same inputs on every
 * machine, and different for any two different (plate, row, layer, role).
 */
export function seedFor(plate: number, row: number, layer: number, role: WeightRole): number {
  if (!Number.isInteger(plate) || plate < 0 || plate >= PLATE_COUNT) throw new RangeError(`plate out of range: ${plate}`);
  if (!Number.isInteger(row) || row < 0 || row >= ROWS) throw new RangeError(`row out of range: ${row}`);
  if (!Number.isInteger(layer) || layer < 0 || layer >= MAX_LAYERS) throw new RangeError(`layer out of range: ${layer}`);
  const r = ROLE_INDEX[role];
  if (r === undefined) throw new RangeError(`unknown role: ${String(role)}`);
  // 1 .. 192,000,000: below 2^32, and never 0.
  const index = ((plate * ROWS + row) * MAX_LAYERS + layer) * ROLE_COUNT + r + 1;
  return mix32(index);
}

/** A plate number drawn at random, 0 .. PLATE_COUNT - 1. */
export function randomPlate(): number {
  return Math.min(PLATE_COUNT - 1, Math.floor(Math.random() * PLATE_COUNT));
}

/**
 * Reads a plate number as it is shown and linked: exactly six digits ("048213").
 * Anything else (null, spaces, signs, fewer or more digits) gives null.
 */
export function parsePlate(text: string | null): number | null {
  if (typeof text !== "string" || !/^[0-9]{6}$/.test(text)) return null;
  return Number(text);
}
