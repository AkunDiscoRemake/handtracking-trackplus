// Drives a rigged GLB hand from lifted 3D keypoints (camera frame, metres).
//
// Solve: the wrist (root) gets the rotation that maps the template palm frame
// to the observed palm frame; every other bone gets the minimal swing that
// turns its parent-propagated rest direction onto the observed direction of
// its child keypoint. Rotations are computed in the camera frame and then
// expressed locally (G_parent⁻¹ · G_bone) because the GLB bones have identity
// rest rotations.

import * as THREE from 'three';
import { NUM_KP, PARENT, BONE_NAMES, BONE_CHILD, template } from '../core/handModel.js';

const _v0 = new THREE.Vector3(), _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _m = new THREE.Matrix4();

// Orthonormal palm frame from wrist, index MCP, middle MCP, pinky MCP.
// Returns a rotation matrix whose columns are the model axes X, Y, Z.
export function palmFrame(p, handedness, out = new THREE.Matrix4()) {
  const P = (i) => new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
  const y = P(9).sub(P(0)).normalize();
  // Model +X points to the thumb for the right hand, to the pinky for the left
  // (mirrored geometry); index→pinky orientation flips accordingly.
  const x = handedness === 'left' ? P(17).sub(P(5)) : P(5).sub(P(17));
  const z = new THREE.Vector3().crossVectors(x, y).normalize();
  x.crossVectors(y, z).normalize();
  return out.makeBasis(x, y, z);
}

export function templateFlat(handedness) {
  const t = template(handedness);
  const a = new Float32Array(NUM_KP * 3);
  t.forEach((v, i) => a.set(v, 3 * i));
  return a;
}

export class HandRig {
  constructor(gltf, handedness) {
    this.handedness = handedness;
    this.root = new THREE.Group();         // camera frame → three world
    this.root.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    this.model = gltf.scene;
    this.root.add(this.model);
    this.bones = [];
    const side = handedness === 'left' ? 'L' : 'R';
    for (let j = 0; j < NUM_KP; j++) {
      const b = this.model.getObjectByName(`${BONE_NAMES[j]}_${side}`);
      if (!b) throw new Error(`bone ${BONE_NAMES[j]}_${side} missing in GLB`);
      this.bones.push(b);
    }
    this.mesh = null;
    this.model.traverse((o) => { if (o.isSkinnedMesh) this.mesh = o; });
    if (!this.mesh) throw new Error('GLB has no skinned mesh');
    this.mesh.frustumCulled = false;
    this.tpl = templateFlat(handedness);
    this.restFrame = palmFrame(this.tpl, handedness);
    this.restFrameInv = this.restFrame.clone().invert();
    this.restDir = [];
    for (let j = 0; j < NUM_KP; j++) {
      const c = BONE_CHILD[j];
      this.restDir.push(c < 0 ? null : new THREE.Vector3(
        this.tpl[3 * c] - this.tpl[3 * j], this.tpl[3 * c + 1] - this.tpl[3 * j + 1], this.tpl[3 * c + 2] - this.tpl[3 * j + 2]).normalize());
    }
    this.global = Array.from({ length: NUM_KP }, () => new THREE.Quaternion());
    this.visible = false;
    this.root.visible = false;
  }

  // Adds extra skinned meshes that share this rig's skeleton (mask, shells).
  cloneSkinned(material, layer) {
    const m = new THREE.SkinnedMesh(this.mesh.geometry, material);
    m.bind(this.mesh.skeleton, this.mesh.bindMatrix);
    m.frustumCulled = false;
    m.layers.set(layer);
    this.mesh.parent.add(m);
    return m;
  }

  // kp3d: Float32Array(63), camera frame (x right, y down, z forward).
  update(kp3d) {
    // Root: palm frame alignment.
    palmFrame(kp3d, this.handedness, _m).multiply(this.restFrameInv);
    this.global[0].setFromRotationMatrix(_m);
    const wrist = this.bones[0];
    wrist.position.set(kp3d[0], kp3d[1], kp3d[2]);
    wrist.quaternion.copy(this.global[0]);
    // Children in topological order (PARENT[j] < j for all j).
    for (let j = 1; j < NUM_KP; j++) {
      const p = PARENT[j];
      const gp = this.global[p];
      const rest = this.restDir[j];
      if (!rest) { this.global[j].copy(gp); this.bones[j].quaternion.identity(); continue; }
      const c = BONE_CHILD[j];
      _v0.copy(rest).applyQuaternion(gp);                                  // rest dir carried by parent
      _v1.set(kp3d[3 * c] - kp3d[3 * j], kp3d[3 * c + 1] - kp3d[3 * j + 1], kp3d[3 * c + 2] - kp3d[3 * j + 2]);
      if (_v1.lengthSq() < 1e-10) { this.global[j].copy(gp); this.bones[j].quaternion.identity(); continue; }
      _v1.normalize();
      _q.setFromUnitVectors(_v0, _v1);
      this.global[j].copy(_q).multiply(gp);
      // local = gp⁻¹ · g
      this.bones[j].quaternion.copy(gp).invert().multiply(this.global[j]);
    }
  }

  setVisible(v) {
    this.visible = v;
    this.root.visible = v;
  }
}

// Wrist world position in three.js coordinates (for debug / labels).
export function cameraToThree(x, y, z, out = _v2) { return out.set(x, -y, -z); }
