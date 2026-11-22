// Reads the palette from the CSS custom properties defined in styles.css.

import type { Theme } from "../contracts";

export const THEME_PROPERTIES: Record<keyof Theme, string> = {
  paper: "--color-paper",
  ink: "--color-ink",
  hairline: "--color-hairline",
  stain: "--color-stain",
};

/** The light palette of plan section 7, used if a property cannot be read. */
export const FALLBACK_THEME: Theme = {
  paper: "#F3F1EA",
  ink: "#1C1B19",
  hairline: "#B9B5A8",
  stain: "#2E5E8C",
};

/** `get` returns the value of a custom property ("" if unset), as getPropertyValue does. */
export function readTheme(get: (property: string) => string): Theme {
  const pick = (key: keyof Theme) => get(THEME_PROPERTIES[key]).trim() || FALLBACK_THEME[key];
  return { paper: pick("paper"), ink: pick("ink"), hairline: pick("hairline"), stain: pick("stain") };
}

export function readDocumentTheme(): Theme {
  const style = getComputedStyle(document.documentElement);
  return readTheme((property) => style.getPropertyValue(property));
}

export const sameTheme = (a: Theme, b: Theme): boolean =>
  a.paper === b.paper && a.ink === b.ink && a.hairline === b.hairline && a.stain === b.stain;
