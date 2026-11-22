// Specimen label and accessible well names. Pure text functions plus one small DOM factory.

import {
  COLS,
  HIGHPASS_HZ,
  N,
  ROWS,
  ROW_LETTERS,
  WELLS,
  formatPlate,
  updatesPerSecond,
  weightCount,
  wellLayers,
  wellName,
} from "../plate/constants";
import type { WellState } from "../app-state";

const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"];

/** 466049 -> "466,049" (independent of the runtime's locale data). */
export function groupThousands(n: number): string {
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Updates per second as printed on the label and in the plan's table:
 * 16000, 1000, 62.5, 3.9, 2.0, 1.0, 0.5, 0.24.
 */
export function formatRate(perSecond: number): string {
  if (Number.isInteger(perSecond)) return String(perSecond);
  return perSecond < 0.3 ? perSecond.toFixed(2) : perSecond.toFixed(1);
}

/** "16000 / 1000 / 62.5" for a well with `layers` layers. */
export function clocksText(layers: number): string {
  const rates: string[] = [];
  for (let i = 0; i < layers; i++) rates.push(formatRate(updatesPerSecond(i)));
  return rates.join(" / ");
}

export const SIGNAL_LINE = `signal: network > ${HIGHPASS_HZ} Hz high-pass > limiter`;
export const TRAINING_LINE = "training data 0 h · epochs 0";

/** The label's lines for a well, or for the untouched plate when `well` is null. */
export function specimenLabel(plate: number, well: number | null): string[] {
  if (well === null) {
    return [
      `PLATE ${formatPlate(plate)} · ${WELLS} WELLS`,
      `rows ${ROW_LETTERS[0]}–${ROW_LETTERS[ROWS - 1]}: ${NUMBER_WORDS[ROWS]} random draws`,
      `columns 1–${COLS}: layers, each added layer slower`,
      TRAINING_LINE,
      SIGNAL_LINE,
    ];
  }
  const layers = wellLayers(well);
  return [
    `PLATE ${formatPlate(plate)} · WELL ${wellName(well)}`,
    `layers ${layers} · units ${layers * N} · weights ${groupThousands(weightCount(layers))}`,
    `clocks ${clocksText(layers)} per second`,
    TRAINING_LINE,
    SIGNAL_LINE,
  ];
}

/** "six layers", "one layer" */
export function layersPhrase(layers: number): string {
  return `${NUMBER_WORDS[layers] ?? String(layers)} ${layers === 1 ? "layer" : "layers"}`;
}

/** "Well F6, six layers, sounding" */
export function wellAriaLabel(well: number, state: WellState): string {
  return `Well ${wellName(well)}, ${layersPhrase(wellLayers(well))}, ${state}`;
}

export interface SpecimenLabel {
  readonly element: HTMLElement;
  set(plate: number, well: number | null): void;
}

export function createSpecimenLabel(parent: HTMLElement): SpecimenLabel {
  const element = document.createElement("div");
  element.className = "specimen";
  element.setAttribute("role", "group");
  element.setAttribute("aria-label", "Specimen label");
  parent.append(element);
  let shown = "";
  return {
    element,
    set(plate, well) {
      const lines = specimenLabel(plate, well);
      const key = lines.join("\n");
      if (key === shown) return;
      shown = key;
      element.replaceChildren(
        ...lines.map((text) => {
          const line = document.createElement("div");
          line.className = "specimen-line";
          line.textContent = text;
          return line;
        }),
      );
    },
  };
}
