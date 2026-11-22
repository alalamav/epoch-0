// Reference for the parity test: one well, run sample by sample by TensorFlow.js itself.
//
// Nothing here uses the engine's arrays. The state lives in TF.js tensors (float32) and every
// number is produced by the TF.js layers that generateLayer() reads its weights from:
//   - the output and the "pass state down" layers run through Dense.apply();
//   - the slowest layer of the well (which has no layer above it) runs through SimpleRNNCell.call();
//   - a layer that has a layer above it must add that layer's conditioning inside the tanh, which
//     SimpleRNNCell.call() has no argument for. Those layers run through cellWithConditioning(),
//     which is the cell's own expression written out with the cell's own kernel, recurrent kernel,
//     bias and activation, plus the conditioning. The parity test checks that with zero
//     conditioning it returns exactly what SimpleRNNCell.call() returns.
// Unlike the engine, a slow layer here keeps the frame of its last CLOCKS[i] samples and feeds it
// as one input vector, oldest sample first, as TF.js would be used.

import * as tf from "@tensorflow/tfjs-core";

import { CLOCKS, N } from "../../src/plate/constants";
import type { TfLayerSet } from "../../src/plate/weights";

/** The parts of TF.js's SimpleRNNCell that are used. */
interface SimpleRNNCellLike {
  call(inputs: tf.Tensor[], kwargs: Record<string, unknown>): tf.Tensor | tf.Tensor[];
  kernel: { read(): tf.Tensor };
  recurrentKernel: { read(): tf.Tensor };
  bias: { read(): tf.Tensor };
  activation: { apply(x: tf.Tensor): tf.Tensor };
}

export const cellOf = (set: TfLayerSet): SimpleRNNCellLike => (set.rnn as unknown as { cell: SimpleRNNCellLike }).cell;

/** SimpleRNNCell.call(): input [1, input size], state [1, N] -> new state [1, N]. */
export function cellCall(set: TfLayerSet, input: tf.Tensor, state: tf.Tensor): tf.Tensor {
  const result = cellOf(set).call([input, state], {});
  return Array.isArray(result) ? result[0] : result;
}

/**
 * activation(input · kernel + bias + conditioning + state · recurrentKernel), in the order
 * SimpleRNNCell.call() computes it, with the conditioning added before the recurrent term.
 */
export function cellWithConditioning(set: TfLayerSet, input: tf.Tensor, state: tf.Tensor, conditioning: tf.Tensor): tf.Tensor {
  const cell = cellOf(set);
  return tf.tidy(() => {
    let h = tf.matMul(input, cell.kernel.read());
    h = tf.add(h, cell.bias.read());
    h = tf.add(h, conditioning);
    return cell.activation.apply(tf.add(h, tf.matMul(state, cell.recurrentKernel.read())));
  });
}

export class TfReference {
  readonly L: number;
  /** State of each layer, [1, N]. */
  h: tf.Tensor[] = [];
  /** Conditioning each layer receives from the layer above, [1, N]. */
  private cond: tf.Tensor[] = [];
  /** Last CLOCKS[i] output samples of the network, oldest first once full. */
  private frames: (Float32Array | null)[] = [];
  private y: tf.Tensor;
  private t = 0;

  /** `sets[i]` holds the live TF.js layers of layer i of one row, i = 0 .. L-1. */
  constructor(private readonly sets: TfLayerSet[]) {
    this.L = sets.length;
    for (let i = 0; i < this.L; i++) {
      if (sets[i].layer !== i) throw new Error("layers must be given in order from layer 0");
      this.h.push(tf.zeros([1, N]));
      this.cond.push(tf.zeros([1, N]));
      this.frames.push(i === 0 ? null : new Float32Array(CLOCKS[i]));
    }
    this.y = tf.zeros([1, 1]);
  }

  private replace(list: tf.Tensor[], i: number, value: tf.Tensor): void {
    list[i].dispose();
    list[i] = tf.keep(value);
  }

  /** One sample. `drive` is the button input. Returns the output sample. */
  step(drive: number): number {
    const { L, sets, t } = this;
    return tf.tidy(() => {
      // Slow layers, slowest first: a layer updates when its clock ticks, from the frame of the
      // last CLOCKS[i] output samples, the conditioning from the layer above, and its own state.
      if (t > 0) {
        for (let i = L - 1; i >= 1; i--) {
          if (t % CLOCKS[i] !== 0) continue;
          const frame = tf.tensor2d(this.frames[i]!, [1, CLOCKS[i]]);
          const next =
            i === L - 1 ? cellCall(sets[i], frame, this.h[i]) : cellWithConditioning(sets[i], frame, this.h[i], this.cond[i]);
          this.replace(this.h, i, next);
          this.replace(this.cond, i - 1, sets[i].down!.apply(next) as tf.Tensor);
        }
      }
      // Fast layer: reads the previous output sample plus the button.
      const x = tf.add(this.y, tf.scalar(drive));
      const h0 = L === 1 ? cellCall(sets[0], x, this.h[0]) : cellWithConditioning(sets[0], x, this.h[0], this.cond[0]);
      this.replace(this.h, 0, h0);
      const y = sets[0].out!.apply(h0) as tf.Tensor;
      this.y.dispose();
      this.y = tf.keep(y);
      const value = y.dataSync()[0];
      for (let i = 1; i < L; i++) this.frames[i]![t % CLOCKS[i]] = value;
      this.t = t + 1;
      return value;
    });
  }

  /**
   * One-step mode: after a step, overwrite this reference's state with another implementation's,
   * so that the next step starts from exactly the same point in both. `h` is that implementation's
   * state per layer and `y` the output sample it just returned.
   * The conditioning needs no copy: it is always the "down" dense layer applied to the current
   * state of the layer above (zero state gives zero), so it is recomputed here by Dense.apply().
   */
  force(h: ArrayLike<number>[], y: number): void {
    const done = this.t - 1; // index of the step just taken
    if (done < 0) throw new Error("force() follows step()");
    tf.tidy(() => {
      this.y.dispose();
      this.y = tf.keep(tf.tensor2d([y], [1, 1]));
      for (let i = 1; i < this.L; i++) this.frames[i]![done % CLOCKS[i]] = y;
      for (let i = 0; i < this.L; i++) {
        // Layer i > 0 changed in that step only if its clock ticked.
        if (i > 0 && !(done > 0 && done % CLOCKS[i] === 0)) continue;
        const hi = tf.tensor2d(Float32Array.from(h[i]), [1, N]);
        this.replace(this.h, i, hi);
        if (i > 0) this.replace(this.cond, i - 1, this.sets[i].down!.apply(hi) as tf.Tensor);
      }
    });
  }

  /** State of layer i as plain numbers. */
  state(i: number): Float32Array {
    return Float32Array.from(this.h[i].dataSync());
  }

  dispose(): void {
    for (const t of this.h) t.dispose();
    for (const t of this.cond) t.dispose();
    this.y.dispose();
  }
}
