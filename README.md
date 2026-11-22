# Epoch 0

A browser piece: one lab plate of 48 small neural networks that have never been trained. Press a well and hear what the network does.

Nature is what a network has before training: its wiring and the random numbers it starts with. Nurture is what training adds, which for music models means recordings made by other people. Epoch 0 sets nurture to zero and plays what is left.

## The plate

The plate has 6 rows and 8 columns. Rows A to F are six random draws of weights. Columns run from one to eight layers of 16 units, and each added layer is slower than the last. Weights are shared along a row, so F6 is F5 with one more slow layer attached. The smallest well has 305 weights and the largest has 2,039,969.

Press a resting well to start it. While you hold the button, the network's input is 1.0. Tap a sounding well to stop it and return its state to zero. Several wells can sound at once; "rinse plate" stops them all and "new plate" draws new weights.

The plate number is part of the link (`?plate=048213`), so you can come back to a plate or share it. The same number gives the same weights and the same kind of sound, but the waveform can differ from one browser to another.

## Run it

```sh
npm install
npm run dev
```

Vite 7 needs Node 20.19 or newer.

| Command | What it does |
|---|---|
| `npm run dev` | Starts the Vite dev server |
| `npm run build` | Type-checks, then builds to `dist/` |
| `npm run preview` | Serves the build |
| `npm test` | Runs the Vitest suite once |
| `npm run typecheck` | Runs `tsc` without emitting |
| `npm run survey` | Runs the network survey described below |

Add `?debug` to the link, or press Shift+D, to open a palette for the accent colour and the colour scheme.

## How it works

The work is split across three threads.

A web worker builds every layer as a TensorFlow.js layer with default settings on the CPU backend, seeds it from the plate number, and reads the weights out. The weights are transferred to the audio thread once. TensorFlow.js does not run the networks.

The engine is an `AudioWorkletProcessor`. It steps every sounding network once per sample at 16,000 samples a second, then sends the sum through a fixed gain of 0.25, a 20 Hz high-pass and a limiter with 5 ms of look-ahead and a ceiling of 0.89. `tests/parity.test.ts` compares the engine with TensorFlow.js on a short run.

The main thread has the buttons, labels and controls. A three.js scene draws all 48 wells in one WebGL canvas from the state reports the engine sends 30 times a second. Each well shows one ring per layer and sixteen units per ring, with a line from each unit to its strongest sender. If WebGL is missing, the buttons draw the wells instead.

## Where things are

| Path | Contents |
|---|---|
| `src/plate/` | Constants, seeding, and the weights worker |
| `src/audio/` | The engine, its worklet wrapper, and the audio graph |
| `src/view/` | The three.js scene and the well layout |
| `src/ui/` | Grid, press logic, controls, specimen label, about dialog |
| `tests/` | Vitest tests, with engine fixtures and a TensorFlow.js reference |
| `scripts/survey.mjs` | The survey and a speed test |
| `plan/` | The build plan, test results, and 48 audio renders of a test plate |

## The survey

`npm run survey` runs 20 draws at each of the 8 layer counts for 30 seconds each, 160 networks in all, and counts how many are still changing at the end. It is plain Node with its own copy of the network step, and its weights come from a seeded generator that draws from the same distributions as the TensorFlow.js defaults.

```sh
npm run survey -- --hold=4000                      # hold the input for 4,000 samples
npm run survey -- --draws=6 --seconds=30 --seed=1  # a smaller run
npm run survey -- bench                            # speed of one 8-layer well and of all 48
```

The design and the test results behind it are in [plan/epoch0_plan.md](plan/epoch0_plan.md).
