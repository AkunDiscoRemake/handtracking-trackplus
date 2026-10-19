// Rolling statistics for real measurements (no estimates): rates, means,
// standard deviation (jitter) and percentiles over a sliding window.

export class Rolling {
  constructor(size = 120) {
    this.buf = new Float64Array(size);
    this.n = 0;
    this.i = 0;
  }
  push(v) {
    if (!Number.isFinite(v)) return;
    this.buf[this.i] = v;
    this.i = (this.i + 1) % this.buf.length;
    if (this.n < this.buf.length) this.n++;
  }
  values() { return Array.from(this.buf.subarray(0, this.n)); }
  mean() {
    if (!this.n) return NaN;
    let s = 0;
    for (let k = 0; k < this.n; k++) s += this.buf[k];
    return s / this.n;
  }
  std() {
    if (this.n < 2) return NaN;
    const m = this.mean();
    let s = 0;
    for (let k = 0; k < this.n; k++) { const d = this.buf[k] - m; s += d * d; }
    return Math.sqrt(s / (this.n - 1));
  }
  pct(p) {
    if (!this.n) return NaN;
    const v = this.values().sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.floor(p * v.length))];
  }
  reset() { this.n = 0; this.i = 0; }
}

// Event-rate meter: FPS from inter-event intervals and their jitter.
export class RateMeter {
  constructor(size = 90) { this.iv = new Rolling(size); this.last = -1; }
  tick(tMs) {
    if (this.last >= 0) {
      const d = tMs - this.last;
      if (d > 0 && d < 2000) this.iv.push(d);
      else if (d >= 2000) this.iv.reset();
    }
    this.last = tMs;
  }
  fps() { const m = this.iv.mean(); return m > 0 ? 1000 / m : 0; }
  jitter() { return this.iv.std(); }
  reset() { this.iv.reset(); this.last = -1; }
}

export class Metrics {
  constructor() {
    this.render = new RateMeter();
    this.camera = new RateMeter();
    this.inference = new RateMeter();
    this.infMs = new Rolling(60);
    this.detMs = new Rolling(60);
    this.poseMs = new Rolling(60);
    this.preMs = new Rolling(60);
    this.captureToResult = new Rolling(60);
    this.captureToDisplay = new Rolling(60);
    this.residual = new Rolling(60);
    this.noisePx = new Rolling(60);
    this.dropped = 0;
    this.sent = 0;
    this.log = [];
    this.logging = false;
  }

  record(row) {
    if (!this.logging) return;
    this.log.push(row);
    if (this.log.length > 20000) this.log.shift();
  }

  csv() {
    if (!this.log.length) return '';
    const keys = Object.keys(this.log[0]);
    return [keys.join(','), ...this.log.map((r) => keys.map((k) => r[k]).join(','))].join('\n');
  }

  snapshot() {
    const r = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : '–');
    return {
      renderFps: r(this.render.fps(), 0),
      renderJitter: r(this.render.jitter()),
      cameraFps: r(this.camera.fps(), 0),
      cameraJitter: r(this.camera.jitter()),
      infFps: r(this.inference.fps(), 1),
      infJitter: r(this.inference.jitter()),
      infMs: r(this.infMs.mean()),
      infP95: r(this.infMs.pct(0.95)),
      detMs: r(this.detMs.mean()),
      poseMs: r(this.poseMs.mean()),
      preMs: r(this.preMs.mean()),
      c2r: r(this.captureToResult.mean()),
      c2rP95: r(this.captureToResult.pct(0.95)),
      c2d: r(this.captureToDisplay.mean()),
      residual: r(this.residual.mean()),
      noisePx: r(this.noisePx.mean(), 2),
      dropped: this.dropped,
      sent: this.sent,
    };
  }
}
