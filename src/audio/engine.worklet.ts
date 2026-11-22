// AudioWorklet wrapper around the engine. Everything that makes sound is in engine-core.ts.
// Loaded by graph.ts with `?worker&url`, so Vite bundles this file and its imports into one module.

import type { ToEngine } from "../contracts";
import { Engine } from "./engine-core";

// The few names of AudioWorkletGlobalScope this file uses, declared here because the project's
// tsconfig describes the main thread (DOM), not the worklet scope.
declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor();
}
declare function registerProcessor(name: string, ctor: new () => AudioWorkletProcessor): void;

class Epoch0Processor extends AudioWorkletProcessor {
  private readonly engine = new Engine({ outputRate: sampleRate });

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent<ToEngine>) => this.engine.handle(event.data);
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const channels = outputs[0];
    if (channels && channels.length > 0) {
      const mono = channels[0];
      this.engine.process(mono);
      for (let c = 1; c < channels.length; c++) channels[c].set(mono);
    }
    for (let report = this.engine.takeReport(); report !== null; report = this.engine.takeReport()) {
      const transfer: Transferable[] = [];
      for (const well of report.wells) transfer.push(well.wave.buffer, well.units.buffer);
      this.port.postMessage(report, transfer);
    }
    return true; // keep running while silent: wells are started by message, not by input
  }
}

registerProcessor("epoch0-engine", Epoch0Processor);
