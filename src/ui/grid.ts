// The plate: an outline with a notched corner at A1, row letters, column numbers, a canvas for
// the scene and 48 real buttons above it.
//
// DOM and ARIA order never change: a grid of six rows (draws A-F) of eight wells (1-8 layers).
// On a narrow screen only the CSS placement is transposed, so draws run across and layers down.

import { COLS, ROWS, ROW_LETTERS, WELLS, formatPlate, wellIndex, wellName } from "../plate/constants";
import type { WellGeometry } from "../contracts";
import type { WellState } from "../app-state";
import { gridPlacement, measureWells, moveWell } from "./grid-nav";
import { layersPhrase, wellAriaLabel } from "./label";
import { bindPressSurface, type PressSurfaceOptions } from "./press-dom";
import type { PressSink } from "./press";

export interface PlateGridOptions extends PressSurfaceOptions {
  press: PressSink;
}

export type ResizeCallback = (width: number, height: number, dpr: number, wells: WellGeometry[]) => void;

export interface PlateGrid {
  readonly element: HTMLElement;
  /** Behind the buttons; give it to createPlateScene. */
  readonly canvas: HTMLCanvasElement;
  readonly buttons: readonly HTMLButtonElement[];
  readonly transposed: boolean;
  setPlate(plate: number): void;
  setWellState(well: number, state: WellState): void;
  wellState(well: number): WellState;
  /** True: six columns by eight rows (phones). */
  setTransposed(transposed: boolean): void;
  /** "none" lets the buttons draw the wells themselves when there is no canvas scene. */
  setSceneAvailable(available: boolean): void;
  /** The 48 button circles relative to the canvas, for PlateScene.resize. */
  measure(): WellGeometry[];
  /** CSS size of the canvas. */
  canvasSize(): { width: number; height: number };
  /** Calls back now and whenever the plate changes size, orientation or pixel ratio. Returns a stop function. */
  observeResize(callback: ResizeCallback): () => void;
  focusWell(well: number): void;
  destroy(): void;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, role?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (role) node.setAttribute("role", role);
  return node;
}

export function createPlateGrid(parent: HTMLElement, options: PlateGridOptions): PlateGrid {
  const plate = el("div", "plate");
  plate.dataset.layout = "wide";
  plate.dataset.scene = "canvas";
  const face = el("div", "plate-face");
  const canvas = el("canvas", "plate-canvas");
  canvas.setAttribute("aria-hidden", "true");
  const grid = el("div", "plate-grid", "grid");
  grid.setAttribute("aria-rowcount", String(ROWS + 1));
  grid.setAttribute("aria-colcount", String(COLS + 1));

  const placed: { node: HTMLElement; row: number; col: number }[] = [];
  const place = (node: HTMLElement, row: number, col: number) => {
    placed.push({ node, row, col });
    return node;
  };

  // Header row: an empty corner, then the column numbers.
  const headRow = el("div", "plate-row", "row");
  headRow.setAttribute("aria-rowindex", "1");
  const corner = el("span", "plate-cell", "presentation");
  headRow.append(corner);
  for (let col = 0; col < COLS; col++) {
    const mark = el("span", "plate-mark", "columnheader");
    mark.textContent = String(col + 1);
    mark.setAttribute("aria-label", layersPhrase(col + 1));
    mark.setAttribute("aria-colindex", String(col + 2));
    headRow.append(place(mark, -1, col));
  }
  grid.append(headRow);

  const buttons: HTMLButtonElement[] = [];
  const states: WellState[] = [];
  const unbind: (() => void)[] = [];
  let transposed = false;
  let current = 0; // the well that takes Tab focus (roving tabindex)

  for (let row = 0; row < ROWS; row++) {
    const rowEl = el("div", "plate-row", "row");
    rowEl.setAttribute("aria-rowindex", String(row + 2));
    const mark = el("span", "plate-mark", "rowheader");
    mark.textContent = ROW_LETTERS[row];
    mark.setAttribute("aria-label", `draw ${ROW_LETTERS[row]}`);
    mark.setAttribute("aria-colindex", "1");
    rowEl.append(place(mark, row, -1));
    for (let col = 0; col < COLS; col++) {
      const well = wellIndex(row, col);
      const cell = el("div", "plate-cell", "gridcell");
      cell.setAttribute("aria-colindex", String(col + 2));
      const button = el("button", "well");
      button.type = "button";
      button.dataset.well = String(well);
      button.dataset.name = wellName(well);
      button.tabIndex = well === current ? 0 : -1;
      place(button, row, col);
      cell.append(button);
      rowEl.append(cell);
      buttons.push(button);
      states.push("resting");
      unbind.push(bindPressSurface(button, () => well, options.press, options));
      button.addEventListener("focus", () => setCurrent(well));
    }
    grid.append(rowEl);
  }

  face.append(canvas, grid);
  plate.append(face);
  parent.append(plate);

  function applyState(well: number, state: WellState) {
    const button = buttons[well];
    states[well] = state;
    button.dataset.state = state;
    button.setAttribute("aria-label", wellAriaLabel(well, state));
    // aria-disabled rather than disabled: a loading well stays focusable, so arrow keys and
    // screen readers can still cross it. The press logic ignores it.
    if (state === "loading") {
      button.setAttribute("aria-disabled", "true");
      button.removeAttribute("aria-pressed");
    } else {
      button.removeAttribute("aria-disabled");
      button.setAttribute("aria-pressed", state === "resting" ? "false" : "true");
    }
  }

  function setCurrent(well: number) {
    if (well === current) return;
    buttons[current].tabIndex = -1;
    current = well;
    buttons[current].tabIndex = 0;
  }

  function applyPlacement() {
    plate.dataset.layout = transposed ? "tall" : "wide";
    for (const { node, row, col } of placed) {
      const { gridRow, gridColumn } = gridPlacement(row, col, transposed);
      node.style.gridRow = String(gridRow);
      node.style.gridColumn = String(gridColumn);
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.altKey) return;
    const target = e.target as HTMLElement | null;
    const from = target?.dataset?.well;
    if (from === undefined) return;
    const to = moveWell(Number(from), e, transposed);
    if (to === null) return;
    e.preventDefault();
    if (to !== Number(from)) focusWell(to);
  }

  function focusWell(well: number) {
    if (!(well >= 0 && well < WELLS)) return;
    setCurrent(well);
    buttons[well].focus();
  }

  const preventMenu = (e: Event) => e.preventDefault();
  grid.addEventListener("keydown", onKeyDown);
  plate.addEventListener("contextmenu", preventMenu);

  for (let well = 0; well < WELLS; well++) applyState(well, "loading");
  applyPlacement();

  function measure(): WellGeometry[] {
    return measureWells(
      canvas.getBoundingClientRect(),
      buttons.map((b) => b.getBoundingClientRect()),
    );
  }

  const canvasSize = () => ({ width: face.clientWidth, height: face.clientHeight });

  const resizeCallbacks = new Set<ResizeCallback>();
  const notifyResize = () => {
    if (resizeCallbacks.size === 0) return;
    const { width, height } = canvasSize();
    const wells = measure();
    const dpr = globalThis.devicePixelRatio || 1;
    for (const callback of resizeCallbacks) callback(width, height, dpr, wells);
  };

  let observer: ResizeObserver | null = null;
  let observing = false;
  function startObserving() {
    if (observing) return;
    observing = true;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(notifyResize);
      observer.observe(face);
    }
    // Also covers zoom and moving between screens (pixel ratio changes without a size change).
    globalThis.addEventListener?.("resize", notifyResize);
  }
  function stopObserving() {
    if (!observing) return;
    observing = false;
    observer?.disconnect();
    observer = null;
    globalThis.removeEventListener?.("resize", notifyResize);
  }

  return {
    element: plate,
    canvas,
    buttons,
    get transposed() {
      return transposed;
    },
    setPlate(plate) {
      grid.setAttribute(
        "aria-label",
        `Plate ${formatPlate(plate)}: six rows of random draws, A to F; eight columns, one to eight layers`,
      );
    },
    setWellState(well, state) {
      if (states[well] !== state) applyState(well, state);
    },
    wellState: (well) => states[well],
    setTransposed(next) {
      if (next === transposed) return;
      transposed = next;
      applyPlacement();
      notifyResize();
    },
    setSceneAvailable(available) {
      plate.dataset.scene = available ? "canvas" : "none";
    },
    measure,
    canvasSize,
    observeResize(callback) {
      resizeCallbacks.add(callback);
      startObserving();
      const { width, height } = canvasSize();
      callback(width, height, globalThis.devicePixelRatio || 1, measure());
      return () => {
        resizeCallbacks.delete(callback);
        if (resizeCallbacks.size === 0) stopObserving();
      };
    },
    focusWell,
    destroy() {
      stopObserving();
      resizeCallbacks.clear();
      for (const off of unbind) off();
      grid.removeEventListener("keydown", onKeyDown);
      plate.removeEventListener("contextmenu", preventMenu);
      plate.remove();
    },
  };
}
