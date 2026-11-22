// Writes tests/fixtures/engine-reference.json: the stored reference output that tests/engine.test.ts
// holds the engine to. Run once, from the project root:
//
//   node tests/helpers/engine-make-fixture.mjs
//
// The reference is computed by the PROTOTYPES (plan/prototype, copied in engine-prototype.ts), not
// by the engine, with Float32 weights, on the machine that runs this script. Needs a Node that
// loads .ts files directly (22.18+ / 23.6+).
//
// Do not regenerate to make a failing test pass unless the change in output is intended.

import fs from "node:fs";
import { ClockNet, ProtoOutputStage } from "./engine-prototype.ts";
import { flatOfPrefix, flatWeights, toLayers } from "./engine-weights.ts";

const SAMPLES = 70000; // 4.4 s: long enough for the slowest layer (65,536 samples) to update once
const HEAD = 64; // first samples, stored in full
const EVERY = 997; // then one sample in 997 (a prime, so checkpoints do not line up with the clocks)

const pick = (y) => {
  const head = Array.from(y.subarray(0, HEAD));
  const checkpoints = [];
  for (let t = HEAD; t < SAMPLES; t += EVERY) checkpoints.push(y[t]);
  checkpoints.push(y[SAMPLES - 1]);
  return { head, checkpoints };
};

// 1. Single networks, 1..8 layers. "pulse": input 1.0 on the first sample only.
//    "hold": input 1.0 for the first 4,000 samples (a quarter-second press).
const networks = [];
for (let L = 1; L <= 8; L++) {
  for (const hold of [1, 4000]) {
    const seed = 1000 + L;
    const net = new ClockNet(L, flatWeights(L, seed), Float32Array);
    const y = new Float64Array(SAMPLES);
    for (let t = 0; t < SAMPLES; t++) y[t] = net.step(t < hold ? 1 : 0);
    networks.push({ layers: L, seed, hold, ...pick(y) });
  }
}

// 2. The whole chain: three wells pressed together and held for 3,000 samples,
//    summed -> gain 0.25 -> 20 Hz high-pass -> limiter.
const chainSpec = { rowSeeds: [12345, 777], wells: [{ row: 0, layers: 3 }, { row: 0, layers: 8 }, { row: 1, layers: 5 }], hold: 3000 };
const rows = chainSpec.rowSeeds.map((seed, r) => toLayers(8, flatWeights(8, seed), r));
const nets = chainSpec.wells.map((w) => new ClockNet(w.layers, flatOfPrefix(rows[w.row], w.layers), Float32Array));
const stage = new ProtoOutputStage();
const out = new Float64Array(SAMPLES);
let peakIn = 0;
for (let t = 0; t < SAMPLES; t++) {
  let x = 0;
  for (const net of nets) x += net.step(t < chainSpec.hold ? 1 : 0) * 0.25;
  if (Math.abs(x) > peakIn) peakIn = Math.abs(x);
  out[t] = stage.step(x);
}

const fixture = {
  note: "Reference output of the prototypes in plan/prototype. Written by tests/helpers/engine-make-fixture.mjs; do not edit.",
  node: process.version,
  samples: SAMPLES,
  head: HEAD,
  every: EVERY,
  networks,
  chain: { ...chainSpec, peakBeforeLimiter: peakIn, lowestGain: stage.minGain, ...pick(out) },
};
fs.mkdirSync(new URL("../fixtures/", import.meta.url), { recursive: true });
fs.writeFileSync(new URL("../fixtures/engine-reference.json", import.meta.url), JSON.stringify(fixture) + "\n");
console.log(`wrote ${networks.length} network cases and the chain (sum peak ${peakIn.toFixed(3)}, lowest limiter gain ${stage.minGain.toFixed(3)})`);
for (const c of networks) {
  const rms = Math.sqrt(c.checkpoints.reduce((a, v) => a + v * v, 0) / c.checkpoints.length);
  console.log(`  ${c.layers} layers, hold ${c.hold}: checkpoint rms ${rms.toFixed(4)}, last ${c.checkpoints[c.checkpoints.length - 1].toFixed(6)}`);
}
