// The "about" dialog in three sections: the concept (plan section 1, with the scale comparison and
// the limits of the metaphor), how to play the plate, and the signal chain with related work.

import {
  CEILING,
  HIGHPASS_HZ,
  LOOKAHEAD_MS,
  MAX_LAYERS,
  OUTPUT_GAIN,
  SAMPLE_RATE,
  WELLS,
  weightCount,
} from "../plate/constants";
import { groupThousands } from "./label";

/** The dialog body as HTML. Static text only; numbers come from constants.ts. */
export function aboutHtml(): string {
  const largest = groupThousands(weightCount(MAX_LAYERS));
  const smallest = groupThousands(weightCount(1));
  return `
<p>one lab plate of ${WELLS} small neural networks that have never been trained. press a well and hear what the network does.</p>

<details open>
  <summary>nature without nurture</summary>
  <p><b>nature</b> is what a network is before training: its architecture and the random weights it starts with. <b>nurture</b> is what training produces. for music models, nurture is recordings made by other people.</p>
  <p>nurture in generated music needs to answer: whose recordings, how many hours. EPOCH 0 sets nurture to zero and plays what is left.</p>
  <p>nature alone makes sound, not music. this "nature" is designed.</p>
</details>

<details>
  <summary>playing</summary>
  <ul>
    <li>rows A to F are six random draws of weights. each column adds one slower layer of 16 units, from one to eight, and weights are shared along a row: F6 is F5 with one more layer.</li>
    <li>press a well to play it. the network's input is 1.0 for as long as the button is down. tap a sounding well and it stops and its state returns to zero. several wells can sound at once; rinse plate stops them all.</li>
    <li>a well cannot show in advance whether it will be silent. you have to run it to find out.</li>
    <li>new plate draws new weights. the plate number is part of the link, so you can come back to a plate or share it. the same number gives the same kind of sound, but the waveform can differ from one browser to another.</li>
    <li>each well is drawn from its own weights and its live state: one ring per layer, sixteen units per ring, a line from each unit to its strongest sender.</li>
  </ul>
</details>

<details>
  <summary>signal and sources</summary>
  <p>the layers are tensorflow.js layers with their default random starting weights. they run sample by sample at ${groupThousands(SAMPLE_RATE)} samples a second. sum of the sounding networks &gt; fixed gain ${OUTPUT_GAIN} &gt; ${HIGHPASS_HZ} Hz high-pass &gt; limiter (${LOOKAHEAD_MS} ms look-ahead, ceiling ${CEILING}) &gt; speakers. nothing else is added.</p>
  <p>untrained image networks already carry structure (<a href="https://arxiv.org/abs/1711.10925" target="_blank" rel="noopener noreferrer">deep image prior</a>, ulyanov, vedaldi, lempitsky, 2018). random recurrent networks can be used unchanged (<a href="https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf" target="_blank" rel="noopener noreferrer">reservoir computing</a>, jaeger, 2001). some draws are better starts than others (<a href="https://arxiv.org/abs/1803.03635" target="_blank" rel="noopener noreferrer">the lottery ticket hypothesis</a>, frankle and carbin, 2019). david tudor played an analogue neural chip in feedback, live (<a href="https://www.lovely.com/titles/cd1602.html" target="_blank" rel="noopener noreferrer"><i>neural synthesis</i></a>, 1995). <a href="https://arxiv.org/abs/1612.07837" target="_blank" rel="noopener noreferrer">samplernn</a> (mehri et al., 2017) runs its layers at different clocks.</p>
</details>`;
}

export interface Sheet {
  readonly dialog: HTMLDialogElement;
  readonly body: HTMLElement;
  readonly title: HTMLElement;
  open(): void;
  close(): void;
  readonly isOpen: boolean;
}

let sheetCount = 0;

/** A modal <dialog> with a title, a close button and a body. Closes on Escape and on a backdrop click. */
export function createSheet(parent: HTMLElement, titleText: string, className = ""): Sheet {
  const dialog = document.createElement("dialog");
  dialog.className = `sheet ${className}`.trim();
  const titleId = `sheet-title-${sheetCount++}`;
  dialog.setAttribute("aria-labelledby", titleId);

  const head = document.createElement("div");
  head.className = "sheet-head";
  const title = document.createElement("h2");
  title.id = titleId;
  title.textContent = titleText;
  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "ctl";
  const closeLabel = document.createElement("span");
  closeLabel.textContent = "close";
  closeButton.append(closeLabel);
  head.append(title, closeButton);

  const body = document.createElement("div");
  body.className = "sheet-body";
  dialog.append(head, body);
  parent.append(dialog);

  const close = () => {
    if (!dialog.open) return;
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
  };
  closeButton.addEventListener("click", close);
  // A click that lands on the dialog element itself is on the backdrop (the content has padding inside).
  dialog.addEventListener("click", (e) => {
    if (e.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (!inside) close();
  });

  return {
    dialog,
    body,
    title,
    open() {
      if (dialog.open) return;
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
    },
    close,
    get isOpen() {
      return dialog.open;
    },
  };
}

export function createAbout(parent: HTMLElement): Sheet {
  const sheet = createSheet(parent, "Epoch 0 · about");
  sheet.body.innerHTML = aboutHtml();
  // Opening one section closes the others.
  const sections = Array.from(sheet.body.querySelectorAll("details"));
  for (const section of sections) {
    section.addEventListener("toggle", () => {
      if (!section.open) return;
      for (const other of sections) if (other !== section) other.open = false;
    });
  }
  return sheet;
}
