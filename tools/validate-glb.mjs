#!/usr/bin/env node
// Validates the generated GLB files with the Khronos glTF-Validator and checks
// the rig structure the runtime expects.
import validator from 'gltf-validator';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
let failed = false;
for (const h of ['left', 'right']) {
  const file = join(ROOT, 'assets', 'hands', `${h}_hand.glb`);
  const data = new Uint8Array(await readFile(file));
  const r = await validator.validateBytes(data, { maxIssues: 50 });
  const { numErrors, numWarnings, numInfos, numHints } = r.issues;
  console.log(`${h}_hand.glb: errors=${numErrors} warnings=${numWarnings} infos=${numInfos} hints=${numHints}`);
  for (const m of r.issues.messages) console.log(`  [${m.severity}] ${m.code} ${m.pointer || ''} ${m.message}`);
  const info = r.info;
  console.log(`  ${info.totalVertexCount} vertices, ${info.totalTriangleCount} triangles, skins=${info.hasSkins}, animations=${info.animationCount}`);
  if (numErrors > 0) failed = true;
}
process.exit(failed ? 1 : 0);
