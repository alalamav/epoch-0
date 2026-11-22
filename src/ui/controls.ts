// Controls: rinse plate, new plate, magnify, plate number (copies the link). (About is in the header.)

import { clipboardCopy, createPlateBadge } from "./readout";

export { clipboardCopy };

export interface ControlsOptions {
  /** Stop all wells. */
  onRinse(): void;
  onNewPlate(): void;
  /** Open the single-well view on the last well touched, or on a random one. */
  onMagnify(): void;
  /** Current plate URL, copied when the plate number is pressed. */
  getLink(): string;
  copyText?(text: string): Promise<void>;
}

export interface Controls {
  readonly element: HTMLElement;
  /** False until a well can be shown (weights loaded, dish view available). */
  setMagnifyEnabled(enabled: boolean): void;
  /** Whether any well is sounding: "rinse" is dimmed when there is nothing to stop. */
  setSounding(any: boolean): void;
  /** Update the displayed plate number on the copy-link control. */
  setPlate(plate: number): void;
}

/** `more` is the part of the text that narrow layouts drop: "rinse" + " plate". */
function textButton(
  name: string,
  text: string,
  onClick: () => void,
  more = "",
): { button: HTMLButtonElement; label: HTMLSpanElement } {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ctl";
  button.dataset.ctl = name;
  const label = document.createElement("span");
  label.textContent = text;
  if (more) {
    const rest = document.createElement("span");
    rest.className = "ctl-more";
    rest.textContent = more;
    label.append(rest);
  }
  button.append(label);
  button.addEventListener("click", onClick);
  return { button, label };
}

export function createControls(parent: HTMLElement, options: ControlsOptions): Controls {
  const element = document.createElement("div");
  element.className = "controls";
  element.setAttribute("role", "group");
  element.setAttribute("aria-label", "Plate controls");

  const rinse = textButton("rinse", "rinse", options.onRinse, " plate");
  rinse.button.title = "stop all wells";
  rinse.button.setAttribute("aria-label", "rinse plate: stop all wells");
  rinse.button.dataset.idle = "true";

  const fresh = textButton("new", "new plate", options.onNewPlate);
  fresh.button.title = "draw new random weights";

  const magnify = textButton("magnify", "magnify", options.onMagnify);
  magnify.button.disabled = true;
  magnify.button.title = "single-well view of the last well touched, or of a random well";
  magnify.button.setAttribute("aria-label", "magnify: open the single-well view");
  magnify.button.setAttribute("aria-haspopup", "dialog");

  const plateButton = document.createElement("button");
  plateButton.type = "button";
  plateButton.className = "ctl";
  plateButton.dataset.ctl = "plate";
  const badge = createPlateBadge(plateButton, {
    getLink: options.getLink,
    copyText: options.copyText,
  });

  element.append(rinse.button, fresh.button, magnify.button, plateButton);
  parent.append(element);

  return {
    element,
    setMagnifyEnabled(enabled) {
      magnify.button.disabled = !enabled;
    },
    setSounding(any) {
      rinse.button.dataset.idle = String(!any);
    },
    setPlate(plate) {
      badge.setPlate(plate);
    },
  };
}
