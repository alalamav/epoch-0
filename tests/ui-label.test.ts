import { describe, expect, it } from "vitest";
import { CLOCKS, WELLS, updatesPerSecond, wellIndex, wellName } from "../src/plate/constants";
import {
  clocksText,
  formatRate,
  groupThousands,
  layersPhrase,
  specimenLabel,
  wellAriaLabel,
} from "../src/ui/label";
import { aboutHtml } from "../src/ui/about";
import { FALLBACK_THEME, readTheme, sameTheme } from "../src/ui/theme";

const F6 = wellIndex(5, 5);

describe("specimen label", () => {
  it("matches the plan's example for F6 on plate 48213", () => {
    expect(specimenLabel(48213, F6)).toEqual([
      "PLATE 048213 · WELL F6",
      "layers 6 · units 96 · weights 466,049",
      "clocks 16000 / 1000 / 62.5 / 3.9 / 2.0 / 1.0 per second",
      "training data 0 h · epochs 0",
      "signal: network > 20 Hz high-pass > limiter",
    ]);
  });

  it("prints the weights of the plan's table for every column", () => {
    const table = ["305", "1,089", "5,713", "71,777", "203,377", "466,049", "990,865", "2,039,969"];
    table.forEach((weights, col) => {
      for (const row of [0, 3]) {
        const lines = specimenLabel(7, wellIndex(row, col));
        expect(lines[0]).toBe(`PLATE 000007 · WELL ${"AD"[row === 0 ? 0 : 1]}${col + 1}`);
        expect(lines[1]).toBe(`layers ${col + 1} · units ${(col + 1) * 16} · weights ${weights}`);
      }
    });
  });

  it("prints the clocks of the plan's table", () => {
    const table = ["16000", "1000", "62.5", "3.9", "2.0", "1.0", "0.5", "0.24"];
    expect(CLOCKS.map((_, i) => formatRate(updatesPerSecond(i)))).toEqual(table);
    expect(specimenLabel(0, wellIndex(0, 0))[2]).toBe("clocks 16000 per second");
    expect(specimenLabel(999999, wellIndex(2, 7))).toEqual([
      "PLATE 999999 · WELL C8",
      "layers 8 · units 128 · weights 2,039,969",
      "clocks 16000 / 1000 / 62.5 / 3.9 / 2.0 / 1.0 / 0.5 / 0.24 per second",
      "training data 0 h · epochs 0",
      "signal: network > 20 Hz high-pass > limiter",
    ]);
    expect(clocksText(3)).toBe("16000 / 1000 / 62.5");
  });

  it("has a label for the untouched plate", () => {
    const lines = specimenLabel(48213, null);
    expect(lines[0]).toBe("PLATE 048213 · 48 WELLS");
    expect(lines).toContain("training data 0 h · epochs 0");
    expect(lines).toContain("signal: network > 20 Hz high-pass > limiter");
  });

  it("groups thousands", () => {
    expect(groupThousands(0)).toBe("0");
    expect(groupThousands(999)).toBe("999");
    expect(groupThousands(1000)).toBe("1,000");
    expect(groupThousands(2039969)).toBe("2,039,969");
  });
});

describe("aria labels", () => {
  it("reads like the plan's example", () => {
    expect(wellAriaLabel(F6, "sounding")).toBe("Well F6, six layers, sounding");
    expect(wellAriaLabel(0, "resting")).toBe("Well A1, one layer, resting");
    expect(wellAriaLabel(wellIndex(1, 7), "held")).toBe("Well B8, eight layers, held");
    expect(wellAriaLabel(wellIndex(4, 3), "loading")).toBe("Well E4, four layers, loading");
  });

  it("names every well once", () => {
    const labels = new Set<string>();
    for (let well = 0; well < WELLS; well++) {
      const text = wellAriaLabel(well, "resting");
      expect(text).toContain(`Well ${wellName(well)},`);
      labels.add(text);
    }
    expect(labels.size).toBe(WELLS);
    expect(layersPhrase(1)).toBe("one layer");
    expect(layersPhrase(2)).toBe("two layers");
  });
});

describe("about text", () => {
  it("states the premise, the signal chain and related work", () => {
    const html = aboutHtml();
    expect(html).toContain('this "nature" is designed');
    expect(html).toContain("fixed gain 0.25 &gt; 20 Hz high-pass &gt; limiter");
    for (const name of ["deep image prior", "reservoir computing", "lottery ticket", "david tudor", "samplernn"]) {
      expect(html).toContain(name);
    }
  });

  it("starts with the first section open and makes every section collapsible, with related-work links", () => {
    const html = aboutHtml();
    expect(html).toContain("<details open>\n  <summary>nature without nurture</summary>");
    expect((html.match(/<details open>/g) ?? []).length).toBe(1);
    expect((html.match(/<details>/g) ?? []).length).toBe(2);
    // Only the playing section is a list.
    expect((html.match(/<ul>/g) ?? []).length).toBe(1);
    expect(html).toContain("<summary>playing</summary>\n  <ul>");
    for (const href of [
      "https://arxiv.org/abs/1711.10925",
      "https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf",
      "https://arxiv.org/abs/1803.03635",
      "https://www.lovely.com/titles/cd1602.html",
      "https://arxiv.org/abs/1612.07837",
    ]) {
      expect(html).toContain(`href="${href}"`);
    }
  });
});

describe("theme", () => {
  it("reads the four custom properties, trimmed", () => {
    const values: Record<string, string> = {
      "--color-paper": " #0d0e0f ",
      "--color-ink": "#e9e6dc",
      "--color-hairline": "#4b4a45",
      "--color-stain": "#6fa8dc",
    };
    const theme = readTheme((p) => values[p] ?? "");
    expect(theme).toEqual({ paper: "#0d0e0f", ink: "#e9e6dc", hairline: "#4b4a45", stain: "#6fa8dc" });
    expect(sameTheme(theme, { ...theme })).toBe(true);
    expect(sameTheme(theme, FALLBACK_THEME)).toBe(false);
  });

  it("falls back to the plan's light palette", () => {
    expect(readTheme(() => "")).toEqual({ paper: "#F3F1EA", ink: "#1C1B19", hairline: "#B9B5A8", stain: "#2E5E8C" });
  });
});
