// The dialog that hosts the dish view (single well). It owns only the frame: a container for
// src/view/dish.ts, the specimen label of the well, and a button that presses the well, because
// a modal dialog covers the plate.

import { wellName } from "../plate/constants";
import type { WellState } from "../app-state";
import { createSheet } from "./about";
import { createSpecimenLabel, wellAriaLabel } from "./label";
import { bindPressSurface, type PressSurfaceOptions } from "./press-dom";
import type { PressSink } from "./press";

export interface DishPanelOptions extends PressSurfaceOptions {
  press: PressSink;
  /** The dialog was closed (button, Escape or backdrop). */
  onClose(): void;
  /** The container changed size. */
  onResize(): void;
}

export interface DishPanel {
  /** Give this to createDishView. */
  readonly container: HTMLElement;
  readonly well: number | null;
  open(plate: number, well: number): void;
  close(): void;
  setWellState(state: WellState): void;
}

export function createDishPanel(parent: HTMLElement, options: DishPanelOptions): DishPanel {
  const sheet = createSheet(parent, "dish", "sheet-dish");
  const container = document.createElement("div");
  container.className = "dish-container";

  const foot = document.createElement("div");
  foot.className = "dish-foot";
  const labelSlot = document.createElement("div");
  const label = createSpecimenLabel(labelSlot);
  label.element.style.border = "0";
  label.element.style.paddingLeft = "0";
  const hold = document.createElement("button");
  hold.type = "button";
  hold.className = "ctl dish-hold";
  const holdText = document.createElement("span");
  hold.append(holdText);
  foot.append(labelSlot, hold);
  sheet.body.append(container, foot);

  let well: number | null = null;
  bindPressSurface(hold, () => well, options.press, options);

  sheet.dialog.addEventListener("close", () => {
    if (well === null) return;
    well = null;
    options.onClose();
  });

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      if (well !== null) options.onResize();
    }).observe(container);
  }

  function setWellState(state: WellState) {
    if (well === null) return;
    const name = wellName(well);
    holdText.textContent =
      state === "loading" ? `${name} loading` : state === "resting" ? `press ${name}` : `${name} ${state}`;
    hold.setAttribute("aria-label", wellAriaLabel(well, state));
    if (state === "loading") {
      hold.setAttribute("aria-disabled", "true");
      hold.removeAttribute("aria-pressed");
    } else {
      hold.removeAttribute("aria-disabled");
      hold.setAttribute("aria-pressed", state === "resting" ? "false" : "true");
    }
  }

  return {
    container,
    get well() {
      return well;
    },
    open(plate, next) {
      well = next;
      sheet.title.textContent = `dish · well ${wellName(next)}`;
      label.set(plate, next);
      setWellState("resting");
      sheet.open();
    },
    close() {
      if (well === null) return;
      well = null;
      sheet.close();
      options.onClose();
    },
    setWellState,
  };
}
