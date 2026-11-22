// Compact layouts (phones): the one-line readout that stands in for the specimen label, and the
// plate number control, which copies the link to the plate.

import { formatPlate, wellLayers, wellName, weightCount } from "../plate/constants";
import { groupThousands } from "./label";

export const clipboardCopy = (text: string): Promise<void> =>
  typeof navigator !== "undefined" && navigator.clipboard
    ? navigator.clipboard.writeText(text)
    : Promise.reject(new Error("no clipboard"));

export const READOUT_IDLE = "press a well";

/** "F6 · 6 layers · 466,049 weights"; a hint while no well has been touched. */
export function readoutText(well: number | null): string {
  if (well === null) return READOUT_IDLE;
  const layers = wellLayers(well);
  return `${wellName(well)} · ${layers} ${layers === 1 ? "layer" : "layers"} · ${groupThousands(weightCount(layers))} weights`;
}

/**
 * The well "magnify" opens: the last one touched if it can be shown, otherwise a random one of
 * those that can. null if none can. `random` returns a number in [0, 1).
 */
export function magnifyTarget(
  last: number | null,
  wells: number,
  canOpen: (well: number) => boolean,
  random: () => number = Math.random,
): number | null {
  if (last !== null && canOpen(last)) return last;
  const open: number[] = [];
  for (let well = 0; well < wells; well++) if (canOpen(well)) open.push(well);
  if (open.length === 0) return null;
  return open[Math.min(open.length - 1, Math.floor(random() * open.length))];
}

export interface Readout {
  readonly element: HTMLButtonElement;
  set(well: number | null): void;
}

/** A button showing the last well touched; pressing it magnifies that well. */
export function createReadout(parent: HTMLElement, options: { onOpen(): void }): Readout {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "readout";
  element.setAttribute("aria-haspopup", "dialog");
  const text = document.createElement("span");
  element.append(text);
  element.addEventListener("click", options.onOpen);
  parent.append(element);

  function set(well: number | null) {
    text.textContent = readoutText(well);
    element.setAttribute(
      "aria-label",
      well === null ? "No well touched yet. Magnify a random well" : `${readoutText(well)}. Magnify well ${wellName(well)}`,
    );
  }
  set(null);
  return { element, set };
}

export interface PlateBadge {
  setPlate(plate: number): void;
}

/** Turns `button` into "plate 048213"; pressing it copies the link. */
export function createPlateBadge(
  button: HTMLButtonElement,
  options: { getLink(): string; copyText?(text: string): Promise<void> },
): PlateBadge {
  const text = document.createElement("span");
  text.setAttribute("aria-live", "polite");
  button.replaceChildren(text);
  button.title = "copy the link to this plate";
  const copyText = options.copyText ?? clipboardCopy;
  let plate = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const show = () => (text.textContent = `plate ${formatPlate(plate)}`);
  button.addEventListener("click", () => {
    const say = (message: string) => {
      text.textContent = message;
      clearTimeout(timer);
      timer = setTimeout(show, 1600);
    };
    copyText(options.getLink()).then(
      () => say("link copied"),
      () => say("copy failed"),
    );
  });
  return {
    setPlate(next) {
      plate = next;
      clearTimeout(timer);
      show();
      button.setAttribute("aria-label", `Plate ${formatPlate(next)}: copy the link to this plate`);
    },
  };
}
