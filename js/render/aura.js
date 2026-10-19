// Black aura: (1) a screen-space silhouette field built from the skinned hand
// meshes (mask → downsample → separable Gaussian blurs → composite) and
// (2) volumetric Fresnel shells around the skinned meshes. Both are drawn with
// premultiplied alpha blending over the camera image.

import * as THREE from 'three';
import { maskShader, shellShader, blurShader, compositeShader } from './shaders.js';

export const LAYER_HAND = 1;
export const LAYER_MASK = 2;
export const LAYER_SHELL = 3;
export const LAYER_DEBUG = 4;

const SHELLS = [0.003, 0.0075];

export class Aura {
  constructor(renderer) {
    this.renderer = renderer;
    this.params = { color: new THREE.Color(0x000000), outer: 0.95, inner: 0.9, flow: 0.035, shells: 1.0 };
    // Half-float targets avoid the contour banding that 8-bit storage causes
    // after several blur passes; fall back to 8-bit where not renderable.
    const ext = renderer.extensions;
    const half = renderer.capabilities.isWebGL2 && (ext.has('EXT_color_buffer_half_float') || ext.has('EXT_color_buffer_float'));
    this.precision = half ? 'half-float' : '8-bit';
    const rtOpts = { type: half ? THREE.HalfFloatType : THREE.UnsignedByteType, depthBuffer: false, magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter };
    this.maskRT = new THREE.WebGLRenderTarget(4, 4, { ...rtOpts, type: THREE.UnsignedByteType, depthBuffer: true });
    // Blur pyramid: near at 1/4, far1 at 1/8, far2 at 1/16 of the buffer.
    this.levels = {
      near: [new THREE.WebGLRenderTarget(4, 4, rtOpts), new THREE.WebGLRenderTarget(4, 4, rtOpts)],
      far1: [new THREE.WebGLRenderTarget(4, 4, rtOpts), new THREE.WebGLRenderTarget(4, 4, rtOpts)],
      far2: [new THREE.WebGLRenderTarget(4, 4, rtOpts), new THREE.WebGLRenderTarget(4, 4, rtOpts)],
    };

    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quadGeo = new THREE.PlaneGeometry(2, 2);
    this.blurMat = new THREE.ShaderMaterial({
      ...blurShader,
      uniforms: { tInput: { value: null }, uDir: { value: new THREE.Vector2() } },
      depthTest: false, depthWrite: false,
    });
    this.compMat = new THREE.ShaderMaterial({
      ...compositeShader,
      uniforms: {
        tNear: { value: this.levels.near[1].texture },
        tFar1: { value: this.levels.far1[1].texture },
        tFar2: { value: this.levels.far2[1].texture },
        uFarSize1: { value: new THREE.Vector2(1, 1) },
        uFarSize2: { value: new THREE.Vector2(1, 1) },
        uColor: { value: this.params.color },
        uOuter: { value: this.params.outer },
        uInner: { value: this.params.inner },
        uTime: { value: 0 },
        uFlow: { value: this.params.flow },
        uAspect: { value: new THREE.Vector2(1, 1) },
      },
      transparent: true, premultipliedAlpha: true, blending: THREE.NormalBlending,
      depthTest: false, depthWrite: false,
    });
    this.quad = new THREE.Mesh(this.quadGeo, this.blurMat);
    this.quadScene = new THREE.Scene();
    this.quadScene.add(this.quad);
    this.shellMats = [];
    this.maskMats = [];
  }

  // Attaches mask + shell meshes to a hand rig.
  attach(rig) {
    const maskMat = new THREE.ShaderMaterial({
      ...maskShader,
      uniforms: { uInflate: { value: 0.0025 }, uOpacity: { value: 1 } },
    });
    rig.cloneSkinned(maskMat, LAYER_MASK);
    this.maskMats.push(maskMat);
    rig.maskMat = maskMat;
    rig.shellMats = SHELLS.map((off, i) => {
      const m = new THREE.ShaderMaterial({
        ...shellShader,
        uniforms: {
          uInflate: { value: off }, uColor: { value: this.params.color }, uOpacity: { value: 1 },
          uPower: { value: 2.8 + i * 1.0 }, uTime: { value: 0 }, uLayer: { value: i },
        },
        transparent: true, premultipliedAlpha: true, blending: THREE.NormalBlending,
        depthWrite: false, side: THREE.FrontSide,
      });
      const mesh = rig.cloneSkinned(m, LAYER_SHELL);
      mesh.renderOrder = 10 + i;
      this.shellMats.push(m);
      return m;
    });
  }

  setOpacity(rig, a) {
    rig.maskMat.uniforms.uOpacity.value = a;
    for (const m of rig.shellMats) m.uniforms.uOpacity.value = a * this.params.shells * 0.45;
  }

  resize(w, h) {
    // Mask at 1/2, blur pyramid at 1/4, 1/8, 1/16 of the drawing buffer: the
    // field is soft by design, so full resolution would only cost bandwidth.
    const sz = (d) => [Math.max(2, Math.round(w / d)), Math.max(2, Math.round(h / d))];
    this.maskRT.setSize(...sz(2));
    for (const [name, d] of [['near', 4], ['far1', 8], ['far2', 16]]) {
      for (const rt of this.levels[name]) rt.setSize(...sz(d));
    }
    this.compMat.uniforms.uFarSize1.value.set(...sz(8));
    this.compMat.uniforms.uFarSize2.value.set(...sz(16));
    this.compMat.uniforms.uAspect.value.set(w / h, 1);
  }

  pass(material, target) {
    this.quad.material = material;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.quadScene, this.quadCam);
  }

  // Two separable Gaussian iterations into level [tmp, dst]; the first pass
  // also downsamples `src` to the level size. Tap spacing stays ≤ ~1 texel
  // of the destination, which is what keeps the result free of ringing.
  blurLevel(src, level, iterations = 2) {
    const [tmp, dst] = level;
    const u = this.blurMat.uniforms;
    let input = src;
    for (let i = 0; i < iterations; i++) {
      u.tInput.value = input.texture;
      u.uDir.value.set(1 / tmp.width, 0);
      this.pass(this.blurMat, tmp);
      u.tInput.value = tmp.texture;
      u.uDir.value.set(0, 1 / dst.height);
      this.pass(this.blurMat, dst);
      input = dst;
    }
  }

  // Renders the mask and the blur pyramid. Call before drawing to the screen.
  prepare(scene, camera, timeS) {
    const r = this.renderer;
    const prevMask = camera.layers.mask;
    r.setRenderTarget(this.maskRT);
    r.setClearColor(0x000000, 0);
    r.clear(true, true, false);
    camera.layers.set(LAYER_MASK);
    r.render(scene, camera);
    camera.layers.mask = prevMask;
    this.blurLevel(this.maskRT, this.levels.near, 2);
    this.blurLevel(this.levels.near[1], this.levels.far1, 2);
    this.blurLevel(this.levels.far1[1], this.levels.far2, 2);
    const u = this.compMat.uniforms;
    u.uTime.value = timeS;
    u.uOuter.value = this.params.outer;
    u.uInner.value = this.params.inner;
    u.uFlow.value = this.params.flow;
    for (const m of this.shellMats) m.uniforms.uTime.value = timeS;
  }

  composite() {
    this.renderer.setRenderTarget(null);
    this.quad.material = this.compMat;
    this.renderer.render(this.quadScene, this.quadCam);
  }

  setColor(hex) { this.params.color.set(hex); }
}
