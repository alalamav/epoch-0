// Web Worker: builds the TF.js layers of a plate and posts their weights (plan section 4).
// Protocol: ToWeightsWorker in, FromWeightsWorker out (src/contracts.ts).
// All the work is in weights.ts (generatePlate), which the Node tests exercise; this file only
// connects it to the worker's message port.

import type { FromWeightsWorker, ToWeightsWorker } from "../contracts";
import { generatePlate } from "./weights";

const scope = self as unknown as DedicatedWorkerGlobalScope;

/** Lets queued messages in between layers. */
const pause = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

const post = (msg: FromWeightsWorker, transfer: ArrayBuffer[]): void => {
  scope.postMessage(msg, transfer);
};

// A new worker is made for each plate, so one "generate" is expected. If another arrives anyway,
// the job in progress stops at its next layer and the new one starts.
let job = 0;

scope.onmessage = (event: MessageEvent<ToWeightsWorker>): void => {
  const msg = event.data;
  if (!msg || msg.type !== "generate") return;
  const mine = ++job;
  void generatePlate(msg.plate, post, { pause, cancelled: () => mine !== job });
};
