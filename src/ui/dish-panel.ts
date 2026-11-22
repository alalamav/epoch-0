// The dialog that hosts the dish view (single well). It owns only the frame: a container for
// src/view/dish.ts, the play / pause button of the well (a modal dialog covers the plate), the
// navigation between wells (four buttons, hidden on phones; the arrow keys; swipes on the dish; a
// small map of the plate) and the specimen label of the well.

import { COLS, WELLS, wellName } from "../plate/constants";
import type { WellState } from "../app-state";
import { createSheet } from "./about";
import {
  DISH_DIRECTIONS,
  directionLabel,
  directionOfKey,
  directionText,
  neighbour,
  swipeDirection,
  type DishDirection,
} from "./dish-nav";
import { createSpecimenLabel, wellAriaLabel } from "./label";
import { bindPressSurface, type PressSurfaceOptions } from "./press-dom";
import type { PressSink } from "./press";

export interface DishPanelOptions extends PressSurfaceOptions {
  press: PressSink;
  /** The dialog was closed (button, Escape or backdrop). */
  onClose(): void;
  /** The container changed size. */
  onResize(): void;
  /** Whether a well can be shown (its weights are loaded). */
  canOpen(well: number): boolean;
  /** The visitor asked for another well. The owner answers by calling open(plate, well). */
  onNavigate(well: number): void;
}

export interface DishPanel {
  /** Give this to createDishView. */
  readonly container: HTMLElement;
  readonly dialog: HTMLDialogElement;
  readonly well: number | null;
  /** Show `well`. May be called while open, to move to another well. */
  open(plate: number, well: number): void;
  close(): void;
  /** State of the open well, for the play / pause button. */
  setWellState(state: WellState): void;
  /** State of any well, for the map. */
  setMapState(well: number, state: WellState): void;
}

export function createDishPanel(parent: HTMLElement, options: DishPanelOptions): DishPanel {
  const sheet = createSheet(parent, "dish", "sheet-dish");

  // ── play / pause: the main control, at the top ─────────────
  // It is a press surface like a well: press starts, a tap on a sounding well stops it, and the
  // input stays on for as long as it is held.
  const nav = document.createElement("div");
  nav.className = "dish-nav";
  const hold = document.createElement("button");
  hold.type = "button";
  hold.className = "ctl dish-hold dish-play";
  hold.title = "tap to play or pause; hold to keep the input on";
  const holdText = document.createElement("span");
  hold.append(holdText);

  // ── navigation ─────────────────────────────────────────────
  const arrows = document.createElement("div");
  arrows.className = "dish-arrows";
  arrows.setAttribute("role", "group");
  arrows.setAttribute("aria-label", "Move to a neighbouring well");
  const arrowButtons = new Map<DishDirection, { button: HTMLButtonElement; text: HTMLSpanElement }>();
  for (const direction of DISH_DIRECTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ctl dish-arrow";
    button.dataset.direction = direction;
    const text = document.createElement("span");
    button.append(text);
    button.addEventListener("click", () => go(direction));
    arrowButtons.set(direction, { button, text });
    arrows.append(button);
  }

  // The map is a pointer shortcut; the buttons and arrow keys reach every well without it.
  const map = document.createElement("div");
  map.className = "dish-map";
  map.setAttribute("aria-hidden", "true");
  map.style.gridTemplateColumns = `repeat(${COLS}, auto)`;
  const cells: HTMLElement[] = [];
  const mapStates: WellState[] = [];
  for (let w = 0; w < WELLS; w++) {
    const cell = document.createElement("span");
    cell.className = "dish-map-cell";
    cell.dataset.well = String(w);
    cell.dataset.state = "loading";
    cell.title = wellName(w);
    cells.push(cell);
    mapStates.push("loading");
    map.append(cell);
  }
  map.addEventListener("click", (e) => {
    const from = (e.target as HTMLElement | null)?.dataset?.well;
    if (from === undefined) return;
    request(Number(from));
  });
  nav.append(hold, arrows, map);

  // ── dish and label ─────────────────────────────────────────
  const container = document.createElement("div");
  container.className = "dish-container";

  const foot = document.createElement("div");
  foot.className = "dish-foot";
  const labelSlot = document.createElement("div");
  const label = createSpecimenLabel(labelSlot);
  label.element.style.border = "0";
  label.element.style.paddingLeft = "0";
  foot.append(labelSlot);
  sheet.body.append(nav, container, foot);

  let well: number | null = null;
  bindPressSurface(hold, () => well, options.press, options);

  function request(next: number) {
    if (well === null || next === well || !options.canOpen(next)) return;
    options.onNavigate(next);
  }

  function go(direction: DishDirection) {
    if (well === null) return;
    const next = neighbour(well, direction);
    if (next !== null) request(next);
  }

  function refreshNav() {
    if (well === null) return;
    for (const [direction, { button, text }] of arrowButtons) {
      const next = neighbour(well, direction);
      text.textContent = directionText(well, direction);
      const describe = directionLabel(well, direction);
      button.setAttribute("aria-label", describe);
      button.title = describe;
      button.disabled = next === null || !options.canOpen(next);
    }
    for (let w = 0; w < WELLS; w++) {
      if (w === well) cells[w].dataset.current = "true";
      else delete cells[w].dataset.current;
    }
  }

  // Arrow keys anywhere in the dialog, except while typing.
  sheet.dialog.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    const direction = directionOfKey(e.key);
    if (direction === null) return;
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
    e.preventDefault();
    go(direction);
  });

  // Swipes on the dish.
  let swipe: { id: number; x: number; y: number; t: number } | null = null;
  container.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    swipe = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp };
  });
  container.addEventListener("pointerup", (e) => {
    if (!swipe || swipe.id !== e.pointerId) return;
    const direction = swipeDirection(e.clientX - swipe.x, e.clientY - swipe.y, e.timeStamp - swipe.t);
    swipe = null;
    if (direction !== null) go(direction);
  });
  container.addEventListener("pointercancel", () => (swipe = null));

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
    holdText.textContent = state === "loading" ? "loading" : state === "resting" ? "play" : "pause";
    hold.dataset.state = state;
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
    dialog: sheet.dialog,
    get well() {
      return well;
    },
    open(plate, next) {
      well = next;
      sheet.title.textContent = `dish · well ${wellName(next)}`;
      label.set(plate, next);
      setWellState(mapStates[next] === "loading" ? "resting" : mapStates[next]);
      refreshNav();
      sheet.open();
    },
    close() {
      if (well === null) return;
      well = null;
      sheet.close();
      options.onClose();
    },
    setWellState,
    setMapState(w, state) {
      if (mapStates[w] === state) return;
      const wasLoading = mapStates[w] === "loading";
      mapStates[w] = state;
      cells[w].dataset.state = state;
      // A neighbour that has just loaded becomes reachable.
      if (wasLoading && well !== null) refreshNav();
    },
  };
}
