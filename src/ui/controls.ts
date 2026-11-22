// Controls under the plate: rinse plate, new plate, the plate number, dish view, about.

import { formatPlate, wellName } from "../plate/constants";
import type { ParsePlate } from "./plate-url";

export interface ControlsOptions {
  parsePlate: ParsePlate;
  /** Stop all wells. */
  onRinse(): void;
  onNewPlate(): void;
  /** A valid plate number was typed. */
  onPlate(plate: number): void;
  /** Open the dish view for the last well touched. */
  onDish(): void;
  onAbout(): void;
  /** The link to copy. */
  getLink(): string;
  /** Default: navigator.clipboard.writeText. */
  copyText?(text: string): Promise<void>;
}

export interface Controls {
  readonly element: HTMLElement;
  setPlate(plate: number): void;
  /** The well the dish button opens (null: none touched yet) and whether it can be opened. */
  setDishWell(well: number | null, enabled: boolean): void;
}

function textButton(text: string, onClick: () => void): { button: HTMLButtonElement; label: HTMLSpanElement } {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ctl";
  const label = document.createElement("span");
  label.textContent = text;
  button.append(label);
  button.addEventListener("click", onClick);
  return { button, label };
}

export function createControls(parent: HTMLElement, options: ControlsOptions): Controls {
  const element = document.createElement("div");
  element.className = "controls";
  element.setAttribute("role", "group");
  element.setAttribute("aria-label", "Plate controls");

  const rinse = textButton("rinse plate", options.onRinse);
  rinse.button.title = "stop all wells";
  rinse.button.setAttribute("aria-label", "rinse plate: stop all wells");

  const fresh = textButton("new plate", options.onNewPlate);
  fresh.button.title = "draw new random weights";

  // The plate number: editable, six digits.
  const field = document.createElement("label");
  field.className = "plate-field";
  const fieldText = document.createElement("span");
  fieldText.textContent = "plate";
  const input = document.createElement("input");
  input.className = "plate-input";
  input.type = "text";
  input.inputMode = "numeric";
  input.pattern = "[0-9]*";
  input.maxLength = 6;
  input.size = 6;
  input.autocomplete = "off";
  input.spellcheck = false;
  input.setAttribute("aria-label", "Plate number, six digits");
  field.append(fieldText, input);

  let plate = 0;
  /** What was typed, with one to five digits padded to the six the link uses. */
  const typed = () => {
    const text = input.value.trim();
    return /^[0-9]{1,6}$/.test(text) ? text.padStart(6, "0") : text;
  };
  function commit() {
    const parsed = options.parsePlate(typed());
    if (parsed === null) {
      input.setAttribute("aria-invalid", "true");
      return;
    }
    input.removeAttribute("aria-invalid");
    input.value = formatPlate(parsed);
    if (parsed !== plate) options.onPlate(parsed);
  }
  function revert() {
    input.removeAttribute("aria-invalid");
    input.value = formatPlate(plate);
  }
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      revert();
    }
  });
  input.addEventListener("blur", () => {
    const parsed = options.parsePlate(typed());
    if (parsed === null) revert();
    else commit();
  });
  input.addEventListener("focus", () => input.select());

  const copyText =
    options.copyText ??
    ((text: string) =>
      navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject(new Error("no clipboard")));
  let copyTimer: ReturnType<typeof setTimeout> | undefined;
  const copy = textButton("copy link", () => {
    const say = (text: string) => {
      copy.label.textContent = text;
      clearTimeout(copyTimer);
      copyTimer = setTimeout(() => (copy.label.textContent = "copy link"), 1600);
    };
    copyText(options.getLink()).then(
      () => say("copied"),
      () => say("copy failed"),
    );
  });
  copy.button.title = "copy the link to this plate";
  copy.label.setAttribute("aria-live", "polite");

  const dish = textButton("dish", options.onDish);
  dish.button.disabled = true;
  dish.button.title = "single-well view of the last well touched";

  const about = textButton("about", options.onAbout);
  about.button.setAttribute("aria-haspopup", "dialog");

  element.append(rinse.button, fresh.button, field, copy.button, dish.button, about.button);
  parent.append(element);

  return {
    element,
    setPlate(next) {
      plate = next;
      revert();
    },
    setDishWell(well, enabled) {
      dish.label.textContent = well === null ? "dish" : `dish ${wellName(well)}`;
      dish.button.disabled = well === null || !enabled;
      dish.button.setAttribute(
        "aria-label",
        well === null ? "dish view: touch a well first" : `open dish view of well ${wellName(well)}`,
      );
    },
  };
}
