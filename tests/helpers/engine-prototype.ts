// The two prototypes from plan/prototype, copied here as the reference the engine is tested against.
// Only types were added. Do not "improve" these: they are the yardstick.
//   ClockNet     plan/prototype/engine_check.mjs
//   ProtoOutputStage  plan/prototype/limiter_check.mjs (class OutputStage there)

const N = 16;
const CLOCKS = [1, 16, 256, 4096, 8192, 16384, 32768, 65536];

type Arr = Float64Array | Float32Array;
type ArrCtor = Float64ArrayConstructor | Float32ArrayConstructor;

export class ClockNet {
  L: number;
  R: Arr[];
  Kf: (Arr | null)[];
  C: Arr[];
  K0: Arr;
  W: Arr;
  h: Float64Array[];
  cond: Float64Array[];
  acc: Float64Array[];
  tmp: Float64Array;
  t: number;
  y: number;

  constructor(L: number, w: Float64Array, A: ArrCtor = Float64Array) {
    this.L = L;
    let o = 0;
    const take = (len: number): Arr => { const v = A.from(w.subarray(o, o + len)); o += len; return v; };
    this.R = []; for (let i = 0; i < L; i++) this.R.push(take(N * N));          // [receiver * N + sender]
    this.Kf = [null]; for (let i = 1; i < L; i++) this.Kf.push(take(CLOCKS[i] * N)); // [position * N + unit]
    this.C = []; for (let i = 0; i < L - 1; i++) this.C.push(take(N * N));       // C[i]: layer i+1 -> layer i
    this.K0 = take(N); this.W = take(N);
    this.h = []; this.cond = []; this.acc = [];
    for (let i = 0; i < L; i++) { this.h.push(new Float64Array(N)); this.cond.push(new Float64Array(N)); this.acc.push(new Float64Array(N)); }
    this.tmp = new Float64Array(N);
    this.t = 0; this.y = 0;
  }
  step(drive: number): number {
    const { L, h, cond, acc, tmp } = this;
    const t = this.t;
    if (t > 0) {
      for (let i = L - 1; i >= 1; i--) {
        if ((t & (CLOCKS[i] - 1)) === 0) {
          const R = this.R[i], hi = h[i], ci = cond[i], ai = acc[i];
          for (let u = 0; u < N; u++) {
            let s = ai[u] + ci[u];
            const b = u * N;
            for (let v = 0; v < N; v++) s += R[b + v] * hi[v];
            tmp[u] = Math.tanh(s);
          }
          hi.set(tmp); ai.fill(0);
          const C = this.C[i - 1], cd = cond[i - 1];
          for (let u = 0; u < N; u++) {
            let s = 0; const b = u * N;
            for (let v = 0; v < N; v++) s += C[b + v] * hi[v];
            cd[u] = s;
          }
        }
      }
    }
    const x = this.y + drive;
    const R0 = this.R[0], h0 = h[0], c0 = cond[0], K0 = this.K0, W = this.W;
    for (let u = 0; u < N; u++) {
      let s = x * K0[u] + c0[u];
      const b = u * N;
      for (let v = 0; v < N; v++) s += R0[b + v] * h0[v];
      tmp[u] = Math.tanh(s);
    }
    let y = 0;
    for (let u = 0; u < N; u++) { h0[u] = tmp[u]; y += W[u] * tmp[u]; }
    for (let i = 1; i < L; i++) {
      const K = this.Kf[i]!, ai = acc[i], b = (t & (CLOCKS[i] - 1)) * N;
      for (let u = 0; u < N; u++) ai[u] += K[b + u] * y;
    }
    this.y = y; this.t = t + 1;
    return y;
  }
}

export class ProtoOutputStage {
  c: number; L: number; R: number; k: number;
  delay: Float64Array; req: Float64Array; avg: Float64Array;
  sum: number; g: number; i: number; j: number; x1: number; y1: number; minGain: number;

  constructor(fsHz = 16000, ceiling = 0.89, lookMs = 5, releaseMs = 100) {
    this.c = ceiling;
    this.L = Math.round(fsHz * lookMs / 1000);
    this.R = 1 - 2 * Math.PI * 20 / fsHz;          // one-pole DC blocker
    this.k = 1 - Math.exp(-1 / (releaseMs / 1000 * fsHz));
    this.delay = new Float64Array(this.L);         // x[n-L]
    this.req = new Float64Array(this.L + 1).fill(1);
    this.avg = new Float64Array(this.L).fill(1);
    this.sum = this.L; this.g = 1; this.i = 0; this.j = 0; this.x1 = 0; this.y1 = 0;
    this.minGain = 1;
  }
  step(x: number): number {
    const hp = x - this.x1 + this.R * this.y1; this.x1 = x; this.y1 = hp;      // remove constant offset
    const a = Math.abs(hp);
    this.req[this.j] = a > this.c ? this.c / a : 1; this.j = (this.j + 1) % (this.L + 1);
    let m = 1; for (let q = 0; q <= this.L; q++) if (this.req[q] < m) m = this.req[q];   // lowest gain needed in the window
    this.g = Math.min(m, this.g + (1 - this.g) * this.k);                       // hold, then release slowly
    this.sum += this.g - this.avg[this.i]; this.avg[this.i] = this.g;           // ramp the gain over the look-ahead time
    const out = this.delay[this.i] * Math.min(this.g, this.sum / this.L);
    this.delay[this.i] = hp; this.i = (this.i + 1) % this.L;
    const gs = this.sum / this.L; if (gs < this.minGain) this.minGain = gs;
    return out;
  }
}
