// 2D debug overlay (Canvas 2D): raw vs filtered vs predicted keypoints,
// skeleton, ROIs (rotated), detector boxes and per-hand labels.

import { EDGES, NUM_KP } from '../core/handModel.js';

const GESTURE_PT = {
  none: '—', open_palm: 'mão aberta', fist: 'punho', point: 'apontar', victory: 'vitória',
  rock: 'rock', call: 'telefone', thumbs_up: 'joinha', thumbs_down: 'polegar p/ baixo', pinch: 'pinça', ok: 'ok',
};
export const gestureLabel = (g) => GESTURE_PT[g] || g;

export class Overlay2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  clear() { this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); }

  draw(map, frame, hands, { debug, labels }) {
    const c = this.ctx;
    this.clear();
    const X = (x) => map.ox + x * map.s, Y = (y) => map.oy + y * map.s;
    if (debug && frame) {
      c.lineWidth = 1;
      c.strokeStyle = 'rgba(255,80,80,0.9)';
      for (const d of frame.dets || []) {
        c.strokeRect(X(d[0]), Y(d[1]), (d[2] - d[0]) * map.s, (d[3] - d[1]) * map.s);
        c.fillStyle = 'rgba(255,80,80,0.9)';
        c.font = '11px ui-monospace, monospace';
        c.fillText(`det ${d[4].toFixed(2)}`, X(d[0]) + 2, Y(d[1]) - 3);
      }
      for (const r of frame.rois || []) {
        c.save();
        c.translate(X(r.cx), Y(r.cy));
        c.rotate(r.theta);
        c.strokeStyle = r.fromDet ? 'rgba(255,170,40,0.9)' : 'rgba(80,200,255,0.9)';
        c.setLineDash([5, 4]);
        const s = r.size * map.s;
        c.strokeRect(-s / 2, -s / 2, s, s);
        c.restore();
      }
      c.setLineDash([]);
    }
    for (const h of hands) {
      const k = h.kp2dPred || h.kp2d;
      if (debug) {
        // raw model output (unfiltered)
        c.fillStyle = 'rgba(255,60,60,0.85)';
        for (let i = 0; i < NUM_KP; i++) dot(c, X(h.kpRaw[2 * i]), Y(h.kpRaw[2 * i + 1]), 2);
        c.strokeStyle = h.handedness === 'left' ? 'rgba(90,200,255,0.95)' : 'rgba(60,240,150,0.95)';
        c.lineWidth = 2;
        c.beginPath();
        for (const [a, b] of EDGES) { c.moveTo(X(k[2 * a]), Y(k[2 * a + 1])); c.lineTo(X(k[2 * b]), Y(k[2 * b + 1])); }
        c.stroke();
        for (let i = 0; i < NUM_KP; i++) {
          const s = h.kpScores[i];
          c.fillStyle = `hsl(${Math.round(120 * Math.min(1, s / 0.7))},90%,55%)`;
          dot(c, X(k[2 * i]), Y(k[2 * i + 1]), 3.2);
        }
      }
      if (labels) {
        const x = X(k[0]), y = Y(k[1]) + 22;
        const txt = `${h.handedness === 'left' ? 'Esquerda' : 'Direita'} · ${gestureLabel(h.gesture)} · ${(h.score * 100).toFixed(0)}%` +
          (h.state === 'coasting' ? ' · ocluída' : '') + (debug ? ` · #${h.id} ${h.facing === 'palm' ? 'palma' : 'dorso'} · z≈${h.depth.toFixed(2)}m` : '');
        c.font = '600 13px system-ui, sans-serif';
        const w = c.measureText(txt).width;
        c.fillStyle = 'rgba(0,0,0,0.55)';
        roundRect(c, x - w / 2 - 8, y - 14, w + 16, 22, 11);
        c.fill();
        c.fillStyle = '#fff';
        c.fillText(txt, x - w / 2, y + 2);
      }
    }
  }
}

function dot(c, x, y, r) { c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); }
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}
