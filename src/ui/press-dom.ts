// Turns pointer, keyboard and assistive-technology events on one element into PressSink calls.

import type { PressSink } from "./press";

export interface PressSurfaceOptions {
  /** Millisecond time of an event. Default: event.timeStamp. */
  now?: (event: Event) => number;
}

const isPressKey = (key: string) => key === " " || key === "Spacebar" || key === "Enter";

let surfaceCount = 0;

/**
 * Bind press-and-hold behaviour to `el` (a button). `getWell` gives the well it stands for,
 * or null if it currently stands for none. Returns a function that removes the listeners.
 *
 *  - pointer: down captures the pointer; up ends the press; cancel / lost capture / leaving
 *    without capture cancel it (release, never stop)
 *  - keyboard: Space or Enter down and up, key repeat ignored; blur cancels
 *  - a click that comes from neither (screen-reader activation, voice control) is a tap
 */
export function bindPressSurface(
  el: HTMLElement,
  getWell: () => number | null,
  sink: PressSink,
  options: PressSurfaceOptions = {},
): () => void {
  const now = options.now ?? ((event: Event) => event.timeStamp);
  const id = surfaceCount++;
  const pointers = new Set<number>();
  const keys = new Set<string>();
  let lastKeyEnd = -Infinity;

  const pointerSource = (pointerId: number) => `s${id}:p${pointerId}`;
  const keySource = (code: string) => `s${id}:k${code}`;

  function onPointerDown(e: PointerEvent) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const well = getWell();
    if (well === null) return;
    // No text selection, no drag, no compatibility mouse events. Focus by hand instead.
    e.preventDefault();
    el.focus({ preventScroll: true });
    const pointerId = e.pointerId ?? 0;
    if (pointers.has(pointerId)) return;
    pointers.add(pointerId);
    try {
      el.setPointerCapture(pointerId);
    } catch {
      // Not capturable (synthetic event, old browser): pointerleave ends the press instead.
    }
    sink.down(pointerSource(pointerId), well, now(e));
  }

  function onPointerUp(e: PointerEvent) {
    const pointerId = e.pointerId ?? 0;
    if (!pointers.delete(pointerId)) return;
    sink.up(pointerSource(pointerId), now(e));
  }

  function cancelPointer(e: PointerEvent) {
    const pointerId = e.pointerId ?? 0;
    if (!pointers.delete(pointerId)) return;
    sink.cancel(pointerSource(pointerId));
  }

  function onPointerLeave(e: PointerEvent) {
    const pointerId = e.pointerId ?? 0;
    let captured = false;
    try {
      captured = el.hasPointerCapture(pointerId);
    } catch {
      captured = false;
    }
    if (!captured) cancelPointer(e);
  }

  function onKeyDown(e: KeyboardEvent) {
    if (!isPressKey(e.key) || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault(); // no page scroll on Space, no synthetic click handling needed
    if (e.repeat) return;
    const code = e.code || e.key;
    if (keys.has(code)) return;
    const well = getWell();
    if (well === null) return;
    keys.add(code);
    sink.down(keySource(code), well, now(e));
  }

  function onKeyUp(e: KeyboardEvent) {
    const code = e.code || e.key;
    if (!keys.delete(code)) return;
    e.preventDefault();
    lastKeyEnd = now(e);
    sink.up(keySource(code), lastKeyEnd);
  }

  function onBlur() {
    for (const code of keys) sink.cancel(keySource(code));
    keys.clear();
  }

  function onClick(e: MouseEvent) {
    // Pointer clicks have detail >= 1 and were handled above. Keyboard clicks follow a key event.
    if (e.detail !== 0 || keys.size > 0 || pointers.size > 0) return;
    const t = now(e);
    if (t - lastKeyEnd < 100) return;
    const well = getWell();
    if (well === null) return;
    const source = `s${id}:click`;
    sink.down(source, well, t);
    sink.up(source, t);
  }

  const prevent = (e: Event) => e.preventDefault();

  el.addEventListener("pointerdown", onPointerDown);
  el.addEventListener("pointerup", onPointerUp);
  el.addEventListener("pointercancel", cancelPointer);
  el.addEventListener("lostpointercapture", cancelPointer);
  el.addEventListener("pointerleave", onPointerLeave);
  el.addEventListener("keydown", onKeyDown);
  el.addEventListener("keyup", onKeyUp);
  el.addEventListener("blur", onBlur);
  el.addEventListener("click", onClick);
  el.addEventListener("contextmenu", prevent);
  el.addEventListener("dragstart", prevent);

  return () => {
    for (const pointerId of pointers) sink.cancel(pointerSource(pointerId));
    pointers.clear();
    onBlur();
    el.removeEventListener("pointerdown", onPointerDown);
    el.removeEventListener("pointerup", onPointerUp);
    el.removeEventListener("pointercancel", cancelPointer);
    el.removeEventListener("lostpointercapture", cancelPointer);
    el.removeEventListener("pointerleave", onPointerLeave);
    el.removeEventListener("keydown", onKeyDown);
    el.removeEventListener("keyup", onKeyUp);
    el.removeEventListener("blur", onBlur);
    el.removeEventListener("click", onClick);
    el.removeEventListener("contextmenu", prevent);
    el.removeEventListener("dragstart", prevent);
  };
}
