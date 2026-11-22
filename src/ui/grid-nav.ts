// Pure geometry of the plate grid: where a well sits on screen in either orientation,
// which well an arrow key leads to, and the circles handed to the scene.

import { COLS, ROWS, WELLS, wellCol, wellIndex, wellRow } from "../plate/constants";
import type { WellGeometry } from "../contracts";

/** The plate turns to six columns by eight rows below this width. */
export const NARROW_QUERY = "(max-width: 599.98px)";

/**
 * CSS grid position (1-based) of plate cell (row, col). Line 1 holds the headers, so pass
 * row = -1 for a column number and col = -1 for a row letter.
 * Wide: draws A-F run down, layers 1-8 run across. Tall (transposed): draws run across,
 * layers run down. A1 and the notch stay top left in both.
 */
export function gridPlacement(row: number, col: number, transposed: boolean): { gridRow: number; gridColumn: number } {
  return transposed ? { gridRow: col + 2, gridColumn: row + 2 } : { gridRow: row + 2, gridColumn: col + 2 };
}

export interface NavKey {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
}

/**
 * The well that a navigation key leads to from `well`, following what is seen on screen.
 * Returns `well` itself at an edge, and null for keys that do not navigate.
 */
export function moveWell(well: number, e: NavKey, transposed: boolean): number | null {
  const row = wellRow(well);
  const col = wellCol(well);
  // On screen: x runs along columns when wide and along rows when transposed.
  const step = (dx: number, dy: number): number => {
    const dRow = transposed ? dx : dy;
    const dCol = transposed ? dy : dx;
    const r = Math.min(ROWS - 1, Math.max(0, row + dRow));
    const c = Math.min(COLS - 1, Math.max(0, col + dCol));
    return wellIndex(r, c);
  };
  switch (e.key) {
    case "ArrowRight":
      return step(1, 0);
    case "ArrowLeft":
      return step(-1, 0);
    case "ArrowDown":
      return step(0, 1);
    case "ArrowUp":
      return step(0, -1);
    case "Home":
      if (e.ctrlKey || e.metaKey) return 0;
      return transposed ? wellIndex(0, col) : wellIndex(row, 0);
    case "End":
      if (e.ctrlKey || e.metaKey) return WELLS - 1;
      return transposed ? wellIndex(ROWS - 1, col) : wellIndex(row, COLS - 1);
    case "PageUp":
      return transposed ? wellIndex(row, 0) : wellIndex(0, col);
    case "PageDown":
      return transposed ? wellIndex(row, COLS - 1) : wellIndex(ROWS - 1, col);
    default:
      return null;
  }
}

export interface RectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Circles of the well buttons relative to the top-left of the canvas, in CSS pixels. */
export function measureWells(canvas: RectLike, buttons: readonly RectLike[]): WellGeometry[] {
  return buttons.map((b) => ({
    cx: b.left + b.width / 2 - canvas.left,
    cy: b.top + b.height / 2 - canvas.top,
    r: Math.min(b.width, b.height) / 2,
  }));
}
