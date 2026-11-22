// The plate number is part of the link: ?plate=048213. Pure functions; the parser and the random
// draw belong to src/plate/seed.ts and are passed in.

import { formatPlate } from "../plate/constants";

export const PLATE_PARAM = "plate";

export type ParsePlate = (text: string | null) => number | null;

/** The plate named by a query string ("?plate=048213"), or null if absent or invalid. */
export function plateFromSearch(search: string, parse: ParsePlate): number | null {
  return parse(new URLSearchParams(search).get(PLATE_PARAM));
}

/** `href` with its plate parameter set, zero-padded. Other parameters and the hash are kept. */
export function plateHref(href: string, plate: number): string {
  const url = new URL(href);
  url.searchParams.set(PLATE_PARAM, formatPlate(plate));
  return url.toString();
}

export interface InitialPlate {
  plate: number;
  /** False if the link named no valid plate and one was drawn: the URL should be rewritten. */
  fromLink: boolean;
}

export function resolvePlate(search: string, parse: ParsePlate, random: () => number): InitialPlate {
  const plate = plateFromSearch(search, parse);
  return plate === null ? { plate: random(), fromLink: false } : { plate, fromLink: true };
}

/** True if the link does not already show `plate` in its canonical six-digit form. */
export function needsRewrite(search: string, plate: number): boolean {
  return new URLSearchParams(search).get(PLATE_PARAM) !== formatPlate(plate);
}
