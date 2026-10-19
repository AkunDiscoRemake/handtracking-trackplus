// Temporal filters. All timestamps are in seconds on a monotonic clock
// (performance.now() / 1000 on the producing thread).

// One Euro Filter (Casiez, Roussel & Vogel, CHI 2012): an adaptive low-pass
// filter whose cutoff grows with speed, trading jitter at rest for lag in
// motion. Operates on a fixed-size vector; no allocation after construction.
export class OneEuroFilter {
  constructor(size, { minCutoff = 1.0, beta = 0.0, dCutoff = 1.0 } = {}) {
    this.size = size;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = new Float32Array(size);   // filtered value
    this.dx = new Float32Array(size);  // filtered derivative (units / s)
    this.t = -1;
  }

  reset() { this.t = -1; this.dx.fill(0); }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  // Filters `value` in place into this.x and returns this.x.
  // `scaleRef` lets the caller normalise speed (e.g. by hand size in px) so the
  // same beta works at any distance from the camera.
  filter(value, t, scaleRef = 1) {
    if (this.t < 0) {
      this.x.set(value);
      this.dx.fill(0);
      this.t = t;
      return this.x;
    }
    let dt = t - this.t;
    if (!(dt > 1e-4)) dt = 1e-4;
    if (dt > 0.5) {
      // Too old to be meaningful: restart rather than integrate a huge gap.
      this.x.set(value);
      this.dx.fill(0);
      this.t = t;
      return this.x;
    }
    this.t = t;
    const aD = OneEuroFilter.alpha(this.dCutoff, dt);
    const inv = 1 / Math.max(scaleRef, 1e-6);
    for (let i = 0; i < this.size; i++) {
      const rawD = (value[i] - this.x[i]) / dt;
      const d = this.dx[i] + aD * (rawD - this.dx[i]);
      this.dx[i] = d;
      const cutoff = this.minCutoff + this.beta * Math.abs(d) * inv;
      const a = OneEuroFilter.alpha(cutoff, dt);
      this.x[i] += a * (value[i] - this.x[i]);
    }
    return this.x;
  }
}

// Constant-velocity Kalman filter, independent per dimension (state [p, v]).
// Used for ROI centre/size and wrist tracking: it gives a principled velocity
// estimate, short-horizon prediction and coasting through occlusions.
export class KalmanCV {
  constructor(dims, { q = 50, r = 4, p0 = 100 } = {}) {
    this.n = dims;
    this.q = q;   // white-acceleration spectral density (units²/s³)
    this.r = r;   // measurement variance (units²)
    this.p0 = p0;
    this.pos = new Float64Array(dims);
    this.vel = new Float64Array(dims);
    // Covariance per dim: [p00, p01, p11]
    this.P = new Float64Array(dims * 3);
    this.t = -1;
    this.initialized = false;
  }

  reset() { this.initialized = false; this.t = -1; }

  init(z, t) {
    for (let i = 0; i < this.n; i++) {
      this.pos[i] = z[i];
      this.vel[i] = 0;
      this.P[i * 3] = this.r;
      this.P[i * 3 + 1] = 0;
      this.P[i * 3 + 2] = this.p0;
    }
    this.t = t;
    this.initialized = true;
  }

  // Advances the state to time t (no measurement).
  predict(t) {
    if (!this.initialized) return;
    const dt = Math.max(0, t - this.t);
    if (dt === 0) return;
    const q = this.q;
    const dt2 = dt * dt, dt3 = dt2 * dt;
    for (let i = 0; i < this.n; i++) {
      const k = i * 3;
      const p00 = this.P[k], p01 = this.P[k + 1], p11 = this.P[k + 2];
      this.pos[i] += this.vel[i] * dt;
      this.P[k] = p00 + 2 * dt * p01 + dt2 * p11 + q * dt3 / 3;
      this.P[k + 1] = p01 + dt * p11 + q * dt2 / 2;
      this.P[k + 2] = p11 + q * dt;
    }
    this.t = t;
  }

  // Mahalanobis distance² of a measurement against the predicted state.
  gate(z) {
    let d2 = 0;
    for (let i = 0; i < this.n; i++) {
      const s = this.P[i * 3] + this.r;
      const e = z[i] - this.pos[i];
      d2 += (e * e) / s;
    }
    return d2;
  }

  update(z, t, rScale = 1) {
    if (!this.initialized) { this.init(z, t); return; }
    this.predict(t);
    const r = this.r * rScale;
    for (let i = 0; i < this.n; i++) {
      const k = i * 3;
      const p00 = this.P[k], p01 = this.P[k + 1], p11 = this.P[k + 2];
      const s = p00 + r;
      const k0 = p00 / s, k1 = p01 / s;
      const e = z[i] - this.pos[i];
      this.pos[i] += k0 * e;
      this.vel[i] += k1 * e;
      this.P[k] = (1 - k0) * p00;
      this.P[k + 1] = (1 - k0) * p01;
      this.P[k + 2] = p11 - k1 * p01;
    }
  }

  // Position extrapolated to time t without mutating the filter.
  extrapolate(t, out) {
    const dt = Math.max(0, t - this.t);
    for (let i = 0; i < this.n; i++) out[i] = this.pos[i] + this.vel[i] * dt;
    return out;
  }
}

// Bounded linear motion prediction. Extrapolates `pos` by `vel` over `dt`
// seconds, with the horizon clamped to [0, maxDt] and the displacement of
// every point clamped to `maxDisp` (same units as pos) so a bad velocity
// estimate can never fling the model across the screen.
export function predictBounded(pos, vel, dt, maxDt, maxDisp, out, stride = 2) {
  const h = Math.min(Math.max(dt, 0), maxDt);
  for (let i = 0; i < pos.length; i += stride) {
    let m2 = 0;
    for (let k = 0; k < stride; k++) {
      const d = vel[i + k] * h;
      m2 += d * d;
    }
    const m = Math.sqrt(m2);
    const s = m > maxDisp ? maxDisp / m : 1;
    for (let k = 0; k < stride; k++) out[i + k] = pos[i + k] + vel[i + k] * h * s;
  }
  return out;
}
