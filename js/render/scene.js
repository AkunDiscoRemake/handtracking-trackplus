// three.js scene: camera whose projection matches the physical camera and the
// on-screen crop of the video, two rigged GLB hands, the aura and 3D debug.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HandRig } from './handRig.js';
import { Aura, LAYER_HAND, LAYER_SHELL, LAYER_DEBUG } from './aura.js';
import { handRimChunk } from './shaders.js';
import { EDGES, NUM_KP } from '../core/handModel.js';

export class HandScene {
  constructor(canvas, { maxPixelRatio = 2 } = {}) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, alpha: true, premultipliedAlpha: true, antialias: false,
      powerPreference: 'high-performance', preserveDrawingBuffer: false,
    });
    this.renderer.autoClear = false;
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.maxPixelRatio = maxPixelRatio;
    this.pixelScale = 1;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.02, 10);
    this.camera.matrixAutoUpdate = false;
    const hemi = new THREE.HemisphereLight(0xfff4ea, 0x2a2622, 1.6);
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.3, 0.8, 1.0);
    for (const l of [hemi, key]) { l.layers.enableAll(); this.scene.add(l); }
    this.aura = new Aura(this.renderer);
    this.rigs = {};
    this.mode = 'hand_aura'; // 'aura' | 'hand_aura' | 'debug'
    this.debugGroup = new THREE.Group();
    this.scene.add(this.debugGroup);
    this.view = { W: 1280, H: 720, f: 1000, fit: 'cover' };
    this.resize();
  }

  async load(baseUrl) {
    const loader = new GLTFLoader();
    const skin = new THREE.MeshStandardMaterial({ color: 0xc89b84, roughness: 0.55, metalness: 0.0 });
    skin.onBeforeCompile = (sh) => {
      sh.fragmentShader = sh.fragmentShader.replace('#include <dithering_fragment>', `#include <dithering_fragment>\n${handRimChunk}`);
    };
    for (const side of ['left', 'right']) {
      const gltf = await loader.loadAsync(new URL(`${side}_hand.glb`, baseUrl).href);
      const rig = new HandRig(gltf, side);
      rig.mesh.material = skin.clone();
      rig.mesh.material.onBeforeCompile = skin.onBeforeCompile;
      rig.mesh.layers.set(LAYER_HAND);
      this.aura.attach(rig);
      this.scene.add(rig.root);
      this.rigs[side] = rig;
      rig.debug = makeDebugSkeleton();
      this.debugGroup.add(rig.debug.group);
      rig.animations = gltf.animations;
    }
  }

  resize() {
    const w = this.canvas.clientWidth || 1, h = this.canvas.clientHeight || 1;
    const pr = Math.min(window.devicePixelRatio || 1, this.maxPixelRatio) * this.pixelScale;
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w, h, false);
    const buf = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.aura.resize(buf.x, buf.y);
    this.updateProjection();
  }

  setPixelScale(s) {
    if (Math.abs(s - this.pixelScale) < 1e-3) return;
    this.pixelScale = s;
    this.resize();
  }

  // W, H: processed frame size; f: focal length in px of that frame;
  // fit: 'cover' or 'contain' (must match the CSS object-fit of the media).
  setView(W, H, f, fit) {
    const v = this.view;
    if (v.W === W && v.H === H && Math.abs(v.f - f) < 1e-3 && v.fit === fit) return;
    Object.assign(v, { W, H, f, fit });
    this.updateProjection();
  }

  // Screen mapping of frame pixels (CSS px) — shared with the 2D overlay.
  frameToScreen() {
    const vw = this.canvas.clientWidth || 1, vh = this.canvas.clientHeight || 1;
    const { W, H, fit } = this.view;
    const s = fit === 'contain' ? Math.min(vw / W, vh / H) : Math.max(vw / W, vh / H);
    return { s, ox: (vw - W * s) / 2, oy: (vh - H * s) / 2, vw, vh };
  }

  updateProjection() {
    const { s, vw, vh } = this.frameToScreen();
    const { f } = this.view;
    const n = this.camera.near, fa = this.camera.far;
    const p = this.camera.projectionMatrix;
    p.set(
      (2 * s * f) / vw, 0, 0, 0,
      0, (2 * s * f) / vh, 0, 0,
      0, 0, -(fa + n) / (fa - n), (-2 * fa * n) / (fa - n),
      0, 0, -1, 0,
    );
    this.camera.projectionMatrixInverse.copy(p).invert();
    this.camera.matrixWorld.identity();
    this.camera.matrixWorldInverse.identity();
  }

  // hands: [{ handedness, kp3d, opacity }]
  update(hands, timeS) {
    const used = new Set();
    for (const h of hands) {
      let rig = this.rigs[h.handedness];
      if (used.has(h.handedness)) rig = this.rigs[h.handedness === 'left' ? 'right' : 'left'];
      if (!rig || used.has(rig.handedness)) continue;
      used.add(rig.handedness);
      rig.update(h.kp3d);
      rig.setVisible(h.opacity > 0.01);
      rig.mesh.material.transparent = h.opacity < 0.99;
      rig.mesh.material.opacity = h.opacity;
      this.aura.setOpacity(rig, h.opacity);
      updateDebugSkeleton(rig.debug, h.kp3d, this.mode === 'debug');
    }
    for (const side of ['left', 'right']) {
      if (!used.has(side) && this.rigs[side]) {
        this.rigs[side].setVisible(false);
        this.rigs[side].debug.group.visible = false;
      }
    }
    this.time = timeS;
    this.anyVisible = used.size > 0;
  }

  render() {
    const r = this.renderer;
    const cam = this.camera;
    r.setRenderTarget(null);
    r.setClearColor(0x000000, 0);
    r.clear(true, true, true);
    if (!this.anyVisible) return;
    const showHand = this.mode !== 'aura';
    this.aura.prepare(this.scene, cam, this.time || 0);
    this.aura.composite();
    // Hand pass: coloured in Hand+Aura/Debug, depth-only in Aura Only so the
    // shells are correctly occluded but the centre stays see-through.
    for (const rig of Object.values(this.rigs)) rig.mesh.material.colorWrite = showHand;
    cam.layers.set(LAYER_HAND);
    r.render(this.scene, cam);
    cam.layers.set(LAYER_SHELL);
    r.render(this.scene, cam);
    if (this.mode === 'debug') {
      r.clearDepth();
      cam.layers.set(LAYER_DEBUG);
      r.render(this.scene, cam);
    }
  }
}

function makeDebugSkeleton() {
  const group = new THREE.Group();
  group.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  const pos = new Float32Array(EDGES.length * 2 * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x35f0a0, depthTest: false }));
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NUM_KP * 3), 3));
  const pts = new THREE.Points(pgeo, new THREE.PointsMaterial({ color: 0xffd23a, size: 6, sizeAttenuation: false, depthTest: false }));
  for (const o of [lines, pts]) { o.layers.set(LAYER_DEBUG); o.frustumCulled = false; group.add(o); }
  group.visible = false;
  return { group, lines, pts };
}

function updateDebugSkeleton(d, kp3d, show) {
  d.group.visible = show;
  if (!show) return;
  const a = d.lines.geometry.attributes.position;
  EDGES.forEach(([i, j], e) => {
    a.array.set(kp3d.subarray(3 * i, 3 * i + 3), 6 * e);
    a.array.set(kp3d.subarray(3 * j, 3 * j + 3), 6 * e + 3);
  });
  a.needsUpdate = true;
  const p = d.pts.geometry.attributes.position;
  p.array.set(kp3d);
  p.needsUpdate = true;
}
