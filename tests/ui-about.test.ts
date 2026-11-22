// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { createAbout } from "../src/ui/about";

describe("about sections", () => {
  it("opening one section closes the others", () => {
    const about = createAbout(document.body);
    const sections = Array.from(about.body.querySelectorAll("details"));
    expect(sections.length).toBe(3);
    const openOnes = () => sections.filter((s) => s.open);

    // Only the first section, "nature without nurture", starts open.
    expect(openOnes()).toEqual([sections[0]]);
    expect(sections[0].querySelector("summary")?.textContent).toBe("nature without nurture");

    // The toggle event is dispatched by hand so the test does not depend on jsdom firing it.
    const open = (section: HTMLDetailsElement) => {
      section.open = true;
      section.dispatchEvent(new Event("toggle"));
    };

    open(sections[1]);
    expect(openOnes()).toEqual([sections[1]]);

    open(sections[2]);
    expect(openOnes()).toEqual([sections[2]]);

    // Closing the open section leaves everything closed.
    sections[2].open = false;
    sections[2].dispatchEvent(new Event("toggle"));
    expect(openOnes()).toEqual([]);
  });
});
