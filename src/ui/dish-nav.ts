// Moving between wells inside the dish view. Pure functions.
//
// Left and right walk along a row: the same draw with one layer fewer or one more.
// Up and down walk along a column: the same number of layers in another draw.

import { COLS, ROWS, wellCol, wellIndex, wellName, wellRow } from "../plate/constants";

export type DishDirection = "left" | "right" | "up" | "down";

export const DISH_DIRECTIONS: readonly DishDirection[] = ["left", "right", "up", "down"];

/** The well next to `well` in a direction, or null at the edge of the plate. No wrap-around. */
export function neighbour(well: number, direction: DishDirection): number | null {
  let row = wellRow(well);
  let col = wellCol(well);
  if (direction === "left") col--;
  else if (direction === "right") col++;
  else if (direction === "up") row--;
  else row++;
  return row >= 0 && row < ROWS && col >= 0 && col < COLS ? wellIndex(row, col) : null;
}

/** "ArrowLeft" -> "left"; null for any other key. */
export function directionOfKey(key: string): DishDirection | null {
  switch (key) {
    case "ArrowLeft":
      return "left";
    case "ArrowRight":
      return "right";
    case "ArrowUp":
      return "up";
    case "ArrowDown":
      return "down";
    default:
      return null;
  }
}

export const SWIPE_MIN_PX = 40;
export const SWIPE_MAX_MS = 700;

/**
 * The direction a swipe moves to, as when turning pages: dragging the dish to the left brings in
 * the well on its right. null if the movement is too short, too slow or too diagonal.
 */
export function swipeDirection(dx: number, dy: number, ms: number): DishDirection | null {
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  if (!(ms >= 0 && ms <= SWIPE_MAX_MS) || Math.max(ax, ay) < SWIPE_MIN_PX) return null;
  if (ax >= 1.5 * ay) return dx < 0 ? "right" : "left";
  if (ay >= 1.5 * ax) return dy < 0 ? "down" : "up";
  return null;
}

/** Text of a navigation button: "← F5", "F7 →", "↑ E6", "↓ A6"; an arrow alone at the edge. */
export function directionText(well: number, direction: DishDirection): string {
  const to = neighbour(well, direction);
  const arrow = { left: "←", right: "→", up: "↑", down: "↓" }[direction];
  if (to === null) return arrow;
  return direction === "right" ? `${wellName(to)} ${arrow}` : `${arrow} ${wellName(to)}`;
}

/** What a navigation button does, for its aria-label and title. */
export function directionLabel(well: number, direction: DishDirection): string {
  const to = neighbour(well, direction);
  if (to === null) return { left: "no fewer layers", right: "no more layers", up: "first draw", down: "last draw" }[direction];
  const what = {
    left: "same draw, one layer fewer",
    right: "same draw, one more layer",
    up: "previous draw, same layers",
    down: "next draw, same layers",
  }[direction];
  return `well ${wellName(to)}: ${what}`;
}
