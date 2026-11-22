// The Web Audio graph: engine worklet -> analyser -> speakers. No audio library.

import type { AudioGraph, EngineReport, FromEngine, ToEngine } from "../contracts";
import { SAMPLE_RATE } from "../plate/constants";
import workletUrl from "./engine.worklet.ts?worker&url";

/**
 * Build the graph. May be called before any user gesture: the context then stays suspended, and
 * silent, until resume() is called inside one.
 *
 * The context is asked to run at 16 kHz, the rate the networks' clocks are counted in. If the
 * device refuses, the context runs at its own rate and the engine converts inside the worklet
 * (`native16k` is then false).
 */
export async function createAudioGraph(): Promise<AudioGraph> {
  let context: AudioContext;
  try {
    context = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: "interactive" });
  } catch {
    context = new AudioContext({ latencyHint: "interactive" });
  }

  let node: AudioWorkletNode;
  try {
    await context.audioWorklet.addModule(workletUrl);
    node = new AudioWorkletNode(context, "epoch0-engine", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
  } catch (error) {
    void context.close().catch(() => undefined);
    throw error;
  }

  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  node.connect(analyser);
  analyser.connect(context.destination);

  const graph: AudioGraph = {
    context,
    analyser,
    native16k: context.sampleRate === SAMPLE_RATE,
    onReport: null as ((report: EngineReport) => void) | null,
    post(msg: ToEngine, transfer?: Transferable[]): void {
      if (transfer && transfer.length > 0) node.port.postMessage(msg, transfer);
      else node.port.postMessage(msg);
    },
    async resume(): Promise<void> {
      if (context.state !== "running" && context.state !== "closed") await context.resume();
    },
    async close(): Promise<void> {
      node.port.onmessage = null;
      node.disconnect();
      analyser.disconnect();
      if (context.state !== "closed") await context.close();
    },
  };

  node.port.onmessage = (event: MessageEvent<FromEngine>) => {
    if (event.data.type === "report") graph.onReport?.(event.data);
  };
  node.onprocessorerror = (event) => {
    console.error("epoch0-engine: the audio processor stopped with an error", event);
  };

  return graph;
}
