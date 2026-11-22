// The "about" dialog: the short text of plan section 1, with the limits of the metaphor,
// the scale comparison, the signal chain and related work.

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
<p>One lab plate of ${WELLS} small neural networks that have never been trained. Press a well and hear what the network does.</p>

<h3>Nature without nurture</h3>
<p><b>Nature</b> is what a network has before training: its wiring and the random numbers it starts with. <b>Nurture</b> is what training adds. For music models, nurture is recordings made by other people.</p>
<p>Most argument about generated music is about nurture: whose recordings, how many hours. Epoch 0 sets nurture to zero and plays what is left.</p>

<h3>The plate</h3>
<ul>
  <li>Rows are individuals: six random draws of weights, A to F.</li>
  <li>Columns are body plans: one to eight layers of 16 units. Each column adds one slower layer, so complexity grows from left to right. Along a row the weights are shared: F6 is F5 with one more slow layer attached.</li>
  <li>Nothing is taught. Every well reads: training data 0 h, epochs 0.</li>
</ul>

<h3>Playing</h3>
<ul>
  <li>Press a resting well: it starts.</li>
  <li>Hold: the network's input is 1.0 for as long as the button is down.</li>
  <li>Tap a sounding well: it stops and its state returns to zero.</li>
  <li>Several wells can sound at once. Rinse plate stops them all. New plate draws new weights; the plate number is part of the link, so a plate can be revisited and shared.</li>
  <li>A well cannot show in advance whether it will be silent. That is only known by running it.</li>
</ul>

<h3>What the tests behind the piece support</h3>
<ul>
  <li>Nature alone makes sound, not music. One layer holds a tone. With four or more layers, half or more of the draws tested were still changing after 30 seconds.</li>
  <li>Nature is uneven. Some draws change in almost every column and some in none, and simple statistics of the weights did not predict which.</li>
  <li>Without nurture, experience leaves little trace. A press disturbs a network while it lasts; afterwards the network mostly returns to what it was doing. The weights never change.</li>
</ul>

<h3>Limits of the metaphor</h3>
<ul>
  <li>This "nature" is designed. People chose the wiring and the rule for the random numbers.</li>
  <li>The draws are not inherited or selected.</li>
  <li>The piece does not show that training is unnecessary. It shows how much of a generated song is nurture.</li>
</ul>

<h3>Scale</h3>
<p>The smallest well has ${smallest} weights and the largest ${largest}, about 2 million, and no recordings. The smallest MusicGen has 300 million weights and was trained on tens of thousands of hours of music.</p>

<h3>Signal</h3>
<p>The layers are TensorFlow.js layers with their default random starting weights, seeded by the plate number. They run sample by sample at ${groupThousands(SAMPLE_RATE)} samples a second. Sum of the sounding networks &gt; fixed gain ${OUTPUT_GAIN} &gt; ${HIGHPASS_HZ} Hz high-pass &gt; limiter (${LOOKAHEAD_MS} ms look-ahead, ceiling ${CEILING}) &gt; speakers. Nothing else is added. The same plate number gives the same weights and the same kind of sound, not the same waveform on every browser.</p>
<p>Each well is drawn from its own weights and its live state: one ring per layer, sixteen units per ring, a line from each unit to its strongest sender.</p>

<h3>Related work</h3>
<ul>
  <li>Deep Image Prior (Ulyanov, Vedaldi, Lempitsky, 2018): untrained image networks already carry structure.</li>
  <li>Reservoir computing (Jaeger, 2001): random recurrent networks used unchanged.</li>
  <li>The lottery ticket hypothesis (Frankle and Carbin, 2019): some draws are better starts than others.</li>
  <li>David Tudor, <i>Neural Synthesis</i> (1995): an analogue neural chip in feedback, played live.</li>
  <li>SampleRNN (Mehri et al., 2017): layers running at different clocks; MusicGen (Copet et al., 2023): the scale comparison.</li>
</ul>`;
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
  return sheet;
}
