// Integration tests on the real assets: GLB validity + rig layout, and the
// real ONNX models on real photos (onnxruntime-web WASM in Node).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import validator from 'gltf-validator';
import { ROOT, loadModel, detectAndPose } from '../tools/lib/pipeline-node.mjs';
import { decodePng } from '../tools/lib/png.mjs';
import { BONE_NAMES, NUM_KP, TEMPLATE_RIGHT } from '../js/core/handModel.js';
import { classifyGesture } from '../js/core/gestures.js';
import { liftHand, palmChirality } from '../js/core/lift.js';

function parseGlb(buf) {
  assert.equal(buf.readUInt32LE(0), 0x46546c67, 'glTF magic');
  assert.equal(buf.readUInt32LE(4), 2);
  const jsonLen = buf.readUInt32LE(12);
  return JSON.parse(buf.toString('utf8', 20, 20 + jsonLen));
}

for (const side of ['left', 'right']) {
  test(`${side}_hand.glb: valid glTF 2.0 with a 21-joint anatomical skin`, async () => {
    const buf = await readFile(join(ROOT, 'assets', 'hands', `${side}_hand.glb`));
    const r = await validator.validateBytes(new Uint8Array(buf));
    assert.equal(r.issues.numErrors, 0);
    assert.equal(r.issues.numWarnings, 0);
    const g = parseGlb(buf);
    assert.equal(g.skins.length, 1);
    assert.equal(g.skins[0].joints.length, NUM_KP);
    const s = side === 'left' ? 'L' : 'R';
    BONE_NAMES.forEach((n, j) => {
      const node = g.nodes[g.skins[0].joints[j]];
      assert.equal(node.name, `${n}_${s}`);
      assert.equal(node.extras.landmark, j);
    });
    // Rest pose matches the template (mirrored for the left hand).
    const wristChildren = g.nodes[0].children.map((c) => g.nodes[c].name);
    assert.deepEqual(wristChildren, ['thumb_cmc', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinky_mcp'].map((n) => `${n}_${s}`));
    const idx = g.nodes[5].translation;
    assert.ok(Math.abs(idx[0] - (side === 'left' ? -1 : 1) * TEMPLATE_RIGHT[5][0]) < 1e-5);
    assert.ok(g.animations[0].channels.length >= 15, 'curl animation drives the finger joints');
    assert.ok(r.info.totalTriangleCount > 10000 && r.info.totalTriangleCount < 60000);
  });
}

const haveModels = existsSync(join(ROOT, 'models', 'manifest.json'));

test('real models detect and pose hands on real photos', { skip: !haveModels && 'run npm run setup first', timeout: 120000 }, async () => {
  const det = await loadModel('detector');
  const pose = await loadModel('pose');
  const expect = { 'vsign_pd.png': 'victory', 'move_pd.png': null };
  for (const [f, gesture] of Object.entries(expect)) {
    const img = decodePng(await readFile(join(ROOT, 'tests', 'fixtures', f)));
    const r = await detectAndPose(det, pose, img);
    assert.ok(r.boxes[0].score > 0.35, `${f}: hand detected`);
    assert.equal(r.hands.length >= 1, true);
    const h = r.hands[0];
    const mean = h.scores.reduce((a, b) => a + b, 0) / 21;
    assert.ok(mean > 0.4, `${f}: mean keypoint confidence ${mean}`);
    // keypoints inside the detection box (with margin)
    const b = r.boxes[0], m = 0.25 * (b.x2 - b.x1);
    for (let k = 0; k < 21; k++) {
      assert.ok(h.kp[2 * k] > b.x1 - m && h.kp[2 * k] < b.x2 + m && h.kp[2 * k + 1] > b.y1 - m && h.kp[2 * k + 1] < b.y2 + m);
    }
    if (gesture) {
      const f0 = Math.max(img.width, img.height) / (2 * Math.tan((67 / 2) * Math.PI / 180));
      const k3 = new Float32Array(63);
      liftHand(h.kp, palmChirality(h.kp) < 0 ? 'right' : 'left', { f: f0, cx: img.width / 2, cy: img.height / 2 }, null, k3);
      assert.equal(classifyGesture(k3, h.kp).name, gesture);
    }
  }
  // Background crop gives a low pose score (rejected by the tracker).
  const img = decodePng(await readFile(join(ROOT, 'tests', 'fixtures', 'vsign_pd.png')));
  const { runPose } = await import('../tools/lib/pipeline-node.mjs');
  const bg = await runPose(pose, img, [{ cx: 400, cy: 120, size: 150, theta: 0 }]);
  const bgMean = bg.res[0].scores.reduce((a, b) => a + b, 0) / 21;
  assert.ok(bgMean < 0.3, `background score ${bgMean}`);
});
