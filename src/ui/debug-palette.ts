// Debug tool for choosing the accent ("stain") colour and the colour scheme.
// Shown with ?debug in the link, or toggled with Shift+D. The choice is kept in localStorage and
// applied on every load until "reset" is pressed.
//
// It works by setting two things on <html>, which styles.css and readDocumentTheme pick up:
//   data-scheme="light" | "dark"   (absent: follow the system)
//   style --color-stain: #RRGGBB   (absent: the value in styles.css)

import { readDocumentTheme } from "./theme";

export type Scheme = "auto" | "light" | "dark";
export type EffectiveScheme = "light" | "dark";

export interface StainPreset {
  id: string;
  name: string;
  light: string;
  dark: string;
}

/** The first preset is the one written in styles.css. */
export const STAIN_PRESETS: readonly StainPreset[] = [
  { id: "methylene", name: "methylene blue", light: "#2E5E8C", dark: "#6FA8DC" },
  { id: "safranin", name: "safranin", light: "#C8472B", dark: "#FF7A59" },
  { id: "eosin", name: "eosin", light: "#C2376E", dark: "#FF6FA8" },
  { id: "malachite", name: "malachite / GFP", light: "#0F7B62", dark: "#4FE0B0" },
  { id: "violet", name: "crystal violet", light: "#5B3A8C", dark: "#B79CFF" },
  { id: "iodine", name: "iodine", light: "#A8611A", dark: "#F2B24C" },
];

export const CUSTOM = "custom";

export interface PaletteState {
  scheme: Scheme;
  /** A preset id, or "custom". */
  preset: string;
  /** The free-form colours, one per scheme. They follow the last preset chosen, so editing starts from it. */
  custom: { light: string; dark: string };
}

export const defaultPaletteState = (): PaletteState => ({
  scheme: "auto",
  preset: STAIN_PRESETS[0].id,
  custom: { light: STAIN_PRESETS[0].light, dark: STAIN_PRESETS[0].dark },
});

/** "#abc", "abc", "#aabbcc", "AABBCC" -> "#AABBCC"; null for anything else. */
export function parseHex(text: string): string | null {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text.trim());
  if (!m) return null;
  const h = m[1].length === 3 ? [...m[1]].map((c) => c + c).join("") : m[1];
  return "#" + h.toUpperCase();
}

/** WCAG relative luminance of a "#RRGGBB" colour. */
export function luminance(hex: string): number {
  const channel = (i: number) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

/** WCAG contrast ratio of two "#RRGGBB" colours, 1 .. 21. */
export function contrastRatio(a: string, b: string): number {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

export const effectiveScheme = (scheme: Scheme, systemDark: boolean): EffectiveScheme =>
  scheme === "auto" ? (systemDark ? "dark" : "light") : scheme;

/** The stain colour a state asks for in a scheme. */
export function stainFor(state: PaletteState, scheme: EffectiveScheme): string {
  const preset = STAIN_PRESETS.find((p) => p.id === state.preset);
  return (preset ?? state.custom)[scheme];
}

/** True if the state changes nothing: system scheme and the stylesheet's own stain. */
export const isDefaultState = (state: PaletteState): boolean =>
  state.scheme === "auto" && state.preset === STAIN_PRESETS[0].id;

export const serializeState = (state: PaletteState): string => JSON.stringify(state);

/** Reads what serializeState wrote. Anything missing or malformed falls back to the default. */
export function parseState(text: string | null): PaletteState {
  const state = defaultPaletteState();
  if (!text) return state;
  try {
    const raw = JSON.parse(text) as Partial<PaletteState> | null;
    if (!raw || typeof raw !== "object") return state;
    if (raw.scheme === "light" || raw.scheme === "dark" || raw.scheme === "auto") state.scheme = raw.scheme;
    const light = typeof raw.custom?.light === "string" ? parseHex(raw.custom.light) : null;
    const dark = typeof raw.custom?.dark === "string" ? parseHex(raw.custom.dark) : null;
    if (light) state.custom.light = light;
    if (dark) state.custom.dark = dark;
    if (raw.preset === CUSTOM || STAIN_PRESETS.some((p) => p.id === raw.preset)) state.preset = raw.preset as string;
  } catch {
    // not JSON: default
  }
  return state;
}

/** ?debug, ?debug=1 -> true; ?debug=0 -> false; otherwise what was stored. */
export function debugRequested(search: string, stored: string | null): boolean {
  const param = new URLSearchParams(search).get("debug");
  if (param !== null) return param !== "0" && param !== "false";
  return stored === "1";
}

export const PALETTE_KEY = "epoch0.palette";
export const VISIBLE_KEY = "epoch0.debug";

type Store = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export interface DebugPaletteOptions {
  /** Whether the system asks for a dark scheme right now. */
  systemDark(): boolean;
  /** The document's colours changed: re-read the theme. */
  onChange(): void;
  /** Default: localStorage, if it can be used. */
  storage?: Store | null;
  /** Default: location.search. */
  search?: string;
}

export interface DebugPalette {
  readonly element: HTMLElement;
  readonly state: PaletteState;
  readonly visible: boolean;
  setVisible(visible: boolean): void;
  /** Write the state to <html> again (the system scheme changed) and call onChange. */
  apply(): void;
  /** Move the panel into another parent. A modal dialog makes everything outside it inert. */
  host(parent: HTMLElement): void;
}

function safeStorage(): Store | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function make<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export function createDebugPalette(parent: HTMLElement, options: DebugPaletteOptions): DebugPalette {
  const storage = options.storage === undefined ? safeStorage() : options.storage;
  const read = (key: string) => {
    try {
      return storage?.getItem(key) ?? null;
    } catch {
      return null;
    }
  };
  const write = (key: string, value: string | null) => {
    try {
      if (value === null) storage?.removeItem(key);
      else storage?.setItem(key, value);
    } catch {
      // private mode, quota: the choice just does not persist
    }
  };

  let state = parseState(read(PALETTE_KEY));
  let visible = debugRequested(options.search ?? location.search, read(VISIBLE_KEY));
  const root = document.documentElement;
  const scheme = () => effectiveScheme(state.scheme, options.systemDark());

  // ── panel ──────────────────────────────────────────────────
  const element = make("section", "debug-palette");
  element.setAttribute("aria-label", "Debug: palette");
  element.hidden = !visible;

  const head = make("div", "debug-head");
  const hide = make("button", "debug-btn", "hide");
  hide.type = "button";
  hide.title = "Shift+D shows it again";
  hide.addEventListener("click", () => setVisible(false));
  // The panel covers whatever is in its corner; this moves it to the next one.
  const corners = ["bottom-left", "top-left", "top-right", "bottom-right"] as const;
  let corner = 0;
  element.dataset.corner = corners[corner];
  const move = make("button", "debug-btn", "move");
  move.type = "button";
  move.title = "move the panel to the next corner";
  move.addEventListener("click", () => {
    corner = (corner + 1) % corners.length;
    element.dataset.corner = corners[corner];
  });
  const headButtons = make("span", "debug-row");
  headButtons.append(move, hide);
  head.append(make("span", "debug-title", "debug · palette"), headButtons);

  const schemeRow = make("div", "debug-row");
  schemeRow.setAttribute("role", "group");
  schemeRow.setAttribute("aria-label", "Colour scheme");
  schemeRow.append(make("span", "debug-key", "scheme"));
  const schemeButtons = new Map<Scheme, HTMLButtonElement>();
  for (const s of ["auto", "light", "dark"] as const) {
    const b = make("button", "debug-btn", s);
    b.type = "button";
    b.addEventListener("click", () => update({ scheme: s }));
    schemeButtons.set(s, b);
    schemeRow.append(b);
  }

  const stainRow = make("div", "debug-swatches");
  stainRow.setAttribute("role", "group");
  stainRow.setAttribute("aria-label", "Stain colour");
  const presetButtons = new Map<string, HTMLButtonElement>();
  for (const preset of STAIN_PRESETS) {
    const b = make("button", "debug-swatch");
    b.type = "button";
    b.title = `${preset.name}: ${preset.light} light, ${preset.dark} dark`;
    const chip = make("span", "debug-chip");
    chip.style.background = `linear-gradient(90deg, ${preset.light} 50%, ${preset.dark} 50%)`;
    b.append(chip, make("span", "", preset.name));
    b.addEventListener("click", () =>
      update({ preset: preset.id, custom: { light: preset.light, dark: preset.dark } }),
    );
    presetButtons.set(preset.id, b);
    stainRow.append(b);
  }

  const customRow = make("div", "debug-row");
  const customKey = make("label", "debug-key", "free");
  const color = make("input", "debug-color");
  color.type = "color";
  color.setAttribute("aria-label", "Stain colour, picker");
  const hex = make("input", "debug-hex");
  hex.type = "text";
  hex.size = 8;
  hex.maxLength = 7;
  hex.spellcheck = false;
  hex.autocomplete = "off";
  hex.setAttribute("aria-label", "Stain colour, hex");
  const customNote = make("span", "debug-note");
  customRow.append(customKey, color, hex, customNote);

  const setCustom = (value: string) => update({ preset: CUSTOM, custom: { ...state.custom, [scheme()]: value } });
  color.addEventListener("input", () => {
    const value = parseHex(color.value);
    if (value) setCustom(value);
  });
  const commitHex = () => {
    const value = parseHex(hex.value);
    if (value) {
      hex.removeAttribute("aria-invalid");
      if (value !== stainFor(state, scheme())) setCustom(value);
    } else {
      hex.setAttribute("aria-invalid", "true");
    }
  };
  hex.addEventListener("change", commitHex);
  hex.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commitHex();
    e.stopPropagation(); // typing here is not a shortcut or a dish arrow key
  });

  const readout = make("p", "debug-readout");
  readout.setAttribute("aria-live", "polite");

  const foot = make("div", "debug-row");
  const reset = make("button", "debug-btn", "reset");
  reset.type = "button";
  reset.addEventListener("click", () => update(defaultPaletteState()));
  foot.append(reset);

  element.append(head, schemeRow, stainRow, customRow, readout, foot);
  parent.append(element);

  // ── behaviour ──────────────────────────────────────────────
  function render() {
    const now = scheme();
    const stain = stainFor(state, now);
    for (const [s, b] of schemeButtons) b.setAttribute("aria-pressed", String(s === state.scheme));
    for (const [id, b] of presetButtons) b.setAttribute("aria-pressed", String(id === state.preset));
    color.value = stain.toLowerCase();
    if (document.activeElement !== hex) hex.value = stain;
    hex.removeAttribute("aria-invalid");
    customNote.textContent = `${state.preset === CUSTOM ? "in use · " : ""}sets the ${now} colour`;

    const theme = readDocumentTheme();
    const paper = parseHex(theme.paper);
    const ink = parseHex(theme.ink);
    const ratio = (other: string | null) => (other ? contrastRatio(stain, other).toFixed(2) + " : 1" : "–");
    readout.textContent = `${stain} (${now}) · on paper ${ratio(paper)} · against ink ${ratio(ink)}`;
  }

  function apply() {
    if (state.scheme === "auto") delete root.dataset.scheme;
    else root.dataset.scheme = state.scheme;
    if (isDefaultState(state) || state.preset === STAIN_PRESETS[0].id) root.style.removeProperty("--color-stain");
    else root.style.setProperty("--color-stain", stainFor(state, scheme()));
    render();
    options.onChange();
  }

  function update(patch: Partial<PaletteState>) {
    state = { ...state, ...patch };
    write(PALETTE_KEY, isDefaultState(state) ? null : serializeState(state));
    apply();
  }

  function setVisible(next: boolean) {
    visible = next;
    element.hidden = !next;
    write(VISIBLE_KEY, next ? "1" : null);
    if (next) render();
  }

  apply();

  return {
    element,
    get state() {
      return state;
    },
    get visible() {
      return visible;
    },
    setVisible,
    apply,
    host(next) {
      if (element.parentElement !== next) next.append(element);
    },
  };
}
