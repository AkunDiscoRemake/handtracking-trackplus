// Execution-provider probing, validation and selection for ONNX Runtime Web.
//
// Preference order (NPU first, GPU complementary, CPU fallback):
//   webnn-npu → webnn-gpu → webgpu → wasm
// A backend is only accepted if (1) the session builds, (2) its output on a
// fixed calibration input matches the reference recorded with the WASM CPU
// backend (tools/make-calibration.mjs) — some drivers silently compute garbage —
// and (3) its measured median latency is within `npuSlack` of the fastest
// valid backend (so an NPU that is much slower than the GPU is not chosen
// just because it is an NPU).

export const CANDIDATES = [
  { id: 'webnn-npu', label: 'WebNN · NPU', ep: () => ({ name: 'webnn', deviceType: 'npu', powerPreference: 'low-power' }) },
  { id: 'webnn-gpu', label: 'WebNN · GPU', ep: () => ({ name: 'webnn', deviceType: 'gpu', powerPreference: 'high-performance' }) },
  { id: 'webgpu', label: 'WebGPU', ep: () => ({ name: 'webgpu' }) },
  { id: 'wasm', label: 'WASM (CPU)', ep: () => 'wasm' },
];

export async function probeEnvironment() {
  const env = {
    webnn: typeof self.navigator !== 'undefined' && !!navigator.ml && typeof navigator.ml.createContext === 'function',
    webgpu: false,
    gpu: null,
    crossOriginIsolated: !!self.crossOriginIsolated,
    threads: self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1,
    cores: navigator.hardwareConcurrency || 0,
    simd: true,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        env.webgpu = true;
        const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
        env.gpu = { vendor: info.vendor || '', architecture: info.architecture || '', device: info.device || '', description: info.description || '' };
        env.gpuFp16 = adapter.features?.has?.('shader-f16') || false;
      }
    } catch { /* blocked */ }
  }
  if (env.webnn) {
    // WebNN device names are not exposed; record which contexts can be created.
    env.webnnContexts = {};
    for (const deviceType of ['npu', 'gpu', 'cpu']) {
      try { await navigator.ml.createContext({ deviceType }); env.webnnContexts[deviceType] = true; } catch { env.webnnContexts[deviceType] = false; }
    }
  }
  return env;
}

// Best-effort SoC family from what the browser exposes (WebGPU adapter info
// and UA client hints). Used for display/diagnostics only.
export function socHint(env, uaModel = '') {
  const s = `${env.gpu?.vendor} ${env.gpu?.architecture} ${env.gpu?.description} ${uaModel}`.toLowerCase();
  if (/qualcomm|adreno/.test(s)) return 'Qualcomm Snapdragon (GPU Adreno)';
  if (/mali|arm/.test(s)) return 'ARM Mali (comum em MediaTek Dimensity / Exynos / Tensor)';
  if (/img|powervr|imagination/.test(s)) return 'Imagination PowerVR (comum em MediaTek Helio)';
  if (/samsung|xclipse/.test(s)) return 'Samsung Exynos (GPU Xclipse)';
  if (/apple/.test(s)) return 'Apple';
  if (/nvidia|amd|intel/.test(s)) return 'Desktop GPU';
  return 'desconhecido';
}

export function isAvailable(c, env) {
  if (c.id.startsWith('webnn')) return env.webnn && (env.webnnContexts?.[c.id.slice(6)] ?? true);
  if (c.id === 'webgpu') return env.webgpu;
  return true;
}

export function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
}

// Chooses the backend. results: [{ id, valid, medianMs }] in preference order.
export function chooseBackend(results, npuSlack = 1.3) {
  const valid = results.filter((r) => r.valid && Number.isFinite(r.medianMs));
  if (!valid.length) return null;
  const fastest = Math.min(...valid.map((r) => r.medianMs));
  return valid.find((r) => r.medianMs <= fastest * npuSlack) || valid[0];
}
