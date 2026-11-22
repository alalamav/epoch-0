// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import {
  CUSTOM,
  PALETTE_KEY,
  STAIN_PRESETS,
  VISIBLE_KEY,
  contrastRatio,
  createDebugPalette,
  debugRequested,
  defaultPaletteState,
  effectiveScheme,
  isDefaultState,
  parseHex,
  parseState,
  serializeState,
  stainFor,
} from "../src/ui/debug-palette";

describe("palette maths", () => {
  it("normalises hex colours and rejects the rest", () => {
    expect(parseHex("#c8472b")).toBe("#C8472B");
    expect(parseHex("C8472B")).toBe("#C8472B");
    expect(parseHex(" #abc ")).toBe("#AABBCC");
    expect(parseHex("#abcd")).toBeNull();
    expect(parseHex("red")).toBeNull();
    expect(parseHex("")).toBeNull();
  });

  it("computes WCAG contrast", () => {
    expect(contrastRatio("#000000", "#FFFFFF")).toBeCloseTo(21, 5);
    expect(contrastRatio("#FFFFFF", "#000000")).toBeCloseTo(21, 5);
    expect(contrastRatio("#777777", "#777777")).toBe(1);
    expect(contrastRatio("#767676", "#FFFFFF")).toBeCloseTo(4.54, 2); // the well-known AA grey
  });

  it("lists the six stains, each a valid light and dark colour, the stylesheet's blue first", () => {
    expect(STAIN_PRESETS.map((p) => p.id)).toEqual(["methylene", "safranin", "eosin", "malachite", "violet", "iodine"]);
    for (const p of STAIN_PRESETS) {
      expect(parseHex(p.light)).toBe(p.light);
      expect(parseHex(p.dark)).toBe(p.dark);
    }
    expect(STAIN_PRESETS[0]).toMatchObject({ light: "#2E5E8C", dark: "#6FA8DC" });
  });

  it("resolves the scheme and the stain", () => {
    expect(effectiveScheme("auto", true)).toBe("dark");
    expect(effectiveScheme("auto", false)).toBe("light");
    expect(effectiveScheme("light", true)).toBe("light");
    expect(effectiveScheme("dark", false)).toBe("dark");
    const state = { ...defaultPaletteState(), preset: "safranin" };
    expect(stainFor(state, "light")).toBe("#C8472B");
    expect(stainFor(state, "dark")).toBe("#FF7A59");
    expect(stainFor({ ...state, preset: CUSTOM, custom: { light: "#111111", dark: "#EEEEEE" } }, "dark")).toBe("#EEEEEE");
  });

  it("round-trips a state and falls back on bad input", () => {
    const state = { scheme: "dark" as const, preset: CUSTOM, custom: { light: "#123456", dark: "#ABCDEF" } };
    expect(parseState(serializeState(state))).toEqual(state);
    expect(parseState(null)).toEqual(defaultPaletteState());
    expect(parseState("not json")).toEqual(defaultPaletteState());
    expect(parseState('{"scheme":"sepia","preset":"nope","custom":{"light":"zzz"}}')).toEqual(defaultPaletteState());
    expect(isDefaultState(defaultPaletteState())).toBe(true);
    expect(isDefaultState(state)).toBe(false);
  });

  it("is shown by ?debug, hidden by ?debug=0, otherwise as stored", () => {
    expect(debugRequested("?debug", null)).toBe(true);
    expect(debugRequested("?plate=048213&debug=1", null)).toBe(true);
    expect(debugRequested("?debug=0", "1")).toBe(false);
    expect(debugRequested("?plate=048213", "1")).toBe(true);
    expect(debugRequested("?plate=048213", null)).toBe(false);
  });
});

describe("debug palette panel", () => {
  afterEach(() => {
    document.body.replaceChildren();
    delete document.documentElement.dataset.scheme;
    document.documentElement.style.removeProperty("--color-stain");
  });

  function memory(initial: Record<string, string> = {}) {
    const data = new Map(Object.entries(initial));
    return {
      data,
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
      removeItem: (k: string) => void data.delete(k),
    };
  }

  function panel(opts: { stored?: Record<string, string>; search?: string; systemDark?: boolean } = {}) {
    const storage = memory(opts.stored);
    let changes = 0;
    let systemDark = opts.systemDark ?? false;
    const p = createDebugPalette(document.body, {
      systemDark: () => systemDark,
      onChange: () => changes++,
      storage,
      search: opts.search ?? "?debug",
    });
    const root = document.documentElement;
    const button = (text: string) =>
      [...p.element.querySelectorAll("button")].find((b) => b.textContent!.trim().startsWith(text))!;
    return {
      p,
      storage,
      root,
      button,
      changes: () => changes,
      stain: () => root.style.getPropertyValue("--color-stain"),
      setSystemDark: (v: boolean) => (systemDark = v),
      hex: p.element.querySelector<HTMLInputElement>(".debug-hex")!,
      color: p.element.querySelector<HTMLInputElement>(".debug-color")!,
      readout: () => p.element.querySelector(".debug-readout")!.textContent!,
    };
  }

  it("offers every listed stain, a free-form field and a light/dark switch", () => {
    const { p, button, hex, color } = panel();
    expect(p.element.hidden).toBe(false);
    expect(p.element.querySelectorAll(".debug-swatch")).toHaveLength(STAIN_PRESETS.length);
    for (const preset of STAIN_PRESETS) expect(button(preset.name)).toBeTruthy();
    for (const s of ["auto", "light", "dark"]) expect(button(s)).toBeTruthy();
    expect(hex.value).toBe("#2E5E8C");
    expect(color.type).toBe("color");
  });

  it("changes nothing on the document by default", () => {
    const { root, stain, changes } = panel();
    expect(root.dataset.scheme).toBeUndefined();
    expect(stain()).toBe("");
    expect(changes()).toBe(1); // the first apply
  });

  it("applies a preset in the colour of the current scheme and stores it", () => {
    const { button, stain, storage, changes, readout } = panel();
    button("safranin").click();
    expect(stain()).toBe("#C8472B");
    expect(button("safranin").getAttribute("aria-pressed")).toBe("true");
    expect(button("methylene").getAttribute("aria-pressed")).toBe("false");
    expect(changes()).toBe(2);
    expect(parseState(storage.data.get(PALETTE_KEY)!).preset).toBe("safranin");
    expect(readout()).toContain("#C8472B (light)");
    expect(readout()).toMatch(/on paper \d+\.\d\d : 1 · against ink \d+\.\d\d : 1/);
  });

  it("switches light and dark, and the stain follows", () => {
    const { button, root, stain } = panel();
    button("eosin").click();
    button("dark").click();
    expect(root.dataset.scheme).toBe("dark");
    expect(stain()).toBe("#FF6FA8");
    button("light").click();
    expect(root.dataset.scheme).toBe("light");
    expect(stain()).toBe("#C2376E");
    button("auto").click();
    expect(root.dataset.scheme).toBeUndefined();
  });

  it("follows the system scheme while on auto when apply() is called", () => {
    const { p, button, stain, setSystemDark } = panel();
    button("malachite").click();
    expect(stain()).toBe("#0F7B62");
    setSystemDark(true);
    p.apply();
    expect(stain()).toBe("#4FE0B0");
  });

  it("takes a free-form colour for the current scheme only, starting from the last preset", () => {
    const { p, button, hex, color, stain } = panel();
    button("iodine").click();
    hex.value = "0a8";
    hex.dispatchEvent(new Event("change"));
    expect(stain()).toBe("#00AA88");
    expect(p.state.preset).toBe(CUSTOM);
    expect(p.state.custom).toEqual({ light: "#00AA88", dark: "#F2B24C" });
    expect(button("iodine").getAttribute("aria-pressed")).toBe("false");
    button("dark").click();
    expect(stain()).toBe("#F2B24C"); // the dark side still follows iodine
    color.value = "#ff00ff";
    color.dispatchEvent(new Event("input"));
    expect(stain()).toBe("#FF00FF");
    expect(p.state.custom).toEqual({ light: "#00AA88", dark: "#FF00FF" });
  });

  it("marks an invalid hex and keeps the colour", () => {
    const { button, hex, stain } = panel();
    button("safranin").click();
    hex.value = "#12";
    hex.dispatchEvent(new Event("change"));
    expect(hex.getAttribute("aria-invalid")).toBe("true");
    expect(stain()).toBe("#C8472B");
  });

  it("resets to the stylesheet and clears the stored choice", () => {
    const { button, root, stain, storage } = panel();
    button("crystal violet").click();
    button("dark").click();
    button("reset").click();
    expect(root.dataset.scheme).toBeUndefined();
    expect(stain()).toBe("");
    expect(storage.data.has(PALETTE_KEY)).toBe(false);
  });

  it("applies a stored choice on load even while hidden", () => {
    const stored = serializeState({ scheme: "dark", preset: "safranin", custom: { light: "#C8472B", dark: "#FF7A59" } });
    const { p, root, stain } = panel({ stored: { [PALETTE_KEY]: stored }, search: "" });
    expect(p.visible).toBe(false);
    expect(p.element.hidden).toBe(true);
    expect(root.dataset.scheme).toBe("dark");
    expect(stain()).toBe("#FF7A59");
  });

  it("remembers whether it is shown, and can move into a dialog", () => {
    const { p, button, storage } = panel({ search: "" });
    expect(p.visible).toBe(false);
    p.setVisible(true);
    expect(storage.data.get(VISIBLE_KEY)).toBe("1");
    button("hide").click();
    expect(p.element.hidden).toBe(true);
    expect(storage.data.has(VISIBLE_KEY)).toBe(false);
    const dialog = document.createElement("dialog");
    document.body.append(dialog);
    p.host(dialog);
    expect(p.element.parentElement).toBe(dialog);
    p.host(document.body);
    expect(p.element.parentElement).toBe(document.body);
  });
});
