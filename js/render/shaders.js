// GLSL (WebGL2 / GLSL ES 3.0 via three.js) shaders for the aura.

// Skinned vertex stage shared by the silhouette mask and the 3D shells. The
// skinned surface is pushed outwards along its (skinned) normal by uInflate
// metres, so the aura hugs the deformed hand, not the rest pose.
const skinnedInflateVertex = /* glsl */ `
  #include <common>
  #include <skinning_pars_vertex>
  uniform float uInflate;
  varying vec3 vNormalV;
  varying vec3 vViewPos;
  void main() {
    #include <beginnormal_vertex>
    #include <skinbase_vertex>
    #include <skinnormal_vertex>
    #include <defaultnormal_vertex>
    #include <begin_vertex>
    #include <skinning_vertex>
    transformed += normalize(objectNormal) * uInflate;
    #include <project_vertex>
    vNormalV = normalize(transformedNormal);
    vViewPos = mvPosition.xyz;
  }
`;

export const maskShader = {
  vertexShader: skinnedInflateVertex,
  fragmentShader: /* glsl */ `
    uniform float uOpacity;
    varying vec3 vNormalV;
    varying vec3 vViewPos;
    void main() {
      // R: coverage weighted by tracking confidence.
      // G: facing ratio (1 = surface faces the camera), used to keep the
      //    centre of the hand transparent.
      float facing = abs(dot(normalize(vNormalV), normalize(-vViewPos)));
      gl_FragColor = vec4(uOpacity, facing * uOpacity, 0.0, 1.0);
    }
  `,
};

// Volumetric shells: several inflated copies of the skinned hand, each with
// a Fresnel opacity — opaque black where the surface turns away from the eye
// (the silhouette), fully transparent where it faces the eye (the centre).
export const shellShader = {
  vertexShader: skinnedInflateVertex,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor;
    uniform float uOpacity;
    uniform float uPower;
    uniform float uTime;
    uniform float uLayer;
    varying vec3 vNormalV;
    varying vec3 vViewPos;
    float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
    float noise(vec3 p) {
      vec3 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
    }
    void main() {
      vec3 n = normalize(vNormalV);
      vec3 v = normalize(-vViewPos);
      float rim = 1.0 - abs(dot(n, v));
      float a = pow(rim, uPower);
      // Slow turbulence so the aura "breathes" instead of looking like a hull.
      float t = noise(vViewPos * 45.0 + vec3(0.0, -uTime * 0.6, uTime * 0.25));
      a *= mix(0.65, 1.15, t);
      a *= uOpacity * (1.0 - uLayer * 0.22);
      a = clamp(a, 0.0, 1.0);
      gl_FragColor = vec4(uColor * a, a); // premultiplied alpha
    }
  `,
};

export const fullscreenVertex = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// 9-tap separable Gaussian (linear-sampling offsets ≈ 17-tap kernel).
export const blurShader = {
  vertexShader: fullscreenVertex,
  fragmentShader: /* glsl */ `
    uniform sampler2D tInput;
    uniform vec2 uDir;      // texel step * radius scale
    varying vec2 vUv;
    void main() {
      vec4 c = texture2D(tInput, vUv) * 0.1964825501511404;
      c += texture2D(tInput, vUv + uDir * 1.411764705882353) * 0.2969069646728344;
      c += texture2D(tInput, vUv - uDir * 1.411764705882353) * 0.2969069646728344;
      c += texture2D(tInput, vUv + uDir * 3.2941176470588234) * 0.09447039785044732;
      c += texture2D(tInput, vUv - uDir * 3.2941176470588234) * 0.09447039785044732;
      c += texture2D(tInput, vUv + uDir * 5.176470588235294) * 0.010381362401148057;
      c += texture2D(tInput, vUv - uDir * 5.176470588235294) * 0.010381362401148057;
      gl_FragColor = c;
    }
  `,
};

// Screen-space aura composite. Inputs come from a blur pyramid of the
// silhouette mask (R = coverage, G = coverage × facing): a narrow field
// (≈ signed distance to the edge near the silhouette) and two wide fields for
// the halo. Output: premultiplied near-black whose opacity peaks at the
// silhouette and fades smoothly to fully transparent both outwards and
// towards the centre of the hand. No hard mask edge is used, so there are no
// aliasing stairs at any render scale.
export const compositeShader = {
  vertexShader: fullscreenVertex,
  fragmentShader: /* glsl */ `
    uniform sampler2D tNear;
    uniform sampler2D tFar1;
    uniform sampler2D tFar2;
    uniform vec2 uFarSize1;
    uniform vec2 uFarSize2;
    uniform vec3 uColor;
    uniform float uOuter;
    uniform float uInner;
    uniform float uTime;
    uniform float uFlow;
    uniform vec2 uAspect;
    varying vec2 vUv;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float noise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
    }
    float fbm(vec2 p) {
      float v = 0.0, a = 0.5;
      for (int i = 0; i < 4; i++) { v += a * noise(p); p = p * 2.03 + 17.1; a *= 0.5; }
      return v;
    }
    // Cubic B-spline filtered lookup with 4 bilinear taps: smooth upsampling of
    // the low-resolution halo fields (no blocky bilinear diamonds).
    vec4 bicubic(sampler2D t, vec2 uv, vec2 size) {
      vec2 px = uv * size - 0.5;
      vec2 f = fract(px);
      px -= f;
      vec2 f2 = f * f, f3 = f2 * f;
      vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
      vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
      vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
      vec2 w3 = f3 / 6.0;
      vec2 s0 = w0 + w1, s1 = w2 + w3;
      vec2 o0 = (px - 0.5 + w1 / s0) / size, o1 = (px + 1.5 + w3 / s1) / size;
      return mix(mix(texture2D(t, vec2(o1.x, o1.y)), texture2D(t, vec2(o0.x, o1.y)), s0.x / (s0.x + s1.x)),
                 mix(texture2D(t, vec2(o1.x, o0.y)), texture2D(t, vec2(o0.x, o0.y)), s0.x / (s0.x + s1.x)),
                 s0.y / (s0.y + s1.y));
    }
    void main() {
      vec2 p = vUv * uAspect;
      // Upward-drifting turbulence displaces the halo lookup → smoky wisps.
      vec2 flow = vec2(fbm(p * 6.0 + vec2(0.0, -uTime * 0.35)), fbm(p * 6.0 + vec2(5.2, -uTime * 0.42))) - 0.5;
      vec2 uvFar = vUv + flow * uFlow / uAspect;
      vec4 nearS = texture2D(tNear, vUv);
      float n = nearS.r;
      float far = 0.55 * bicubic(tFar1, uvFar, uFarSize1).r + 0.45 * bicubic(tFar2, uvFar, uFarSize2).r;
      // Outside: halo rising towards the silhouette (n ≈ 0.5 on the edge).
      float outside = 1.0 - smoothstep(0.42, 0.58, n);
      float outer = smoothstep(0.015, 0.5, far) * outside;
      outer *= mix(0.72, 1.15, fbm(p * 9.0 + vec2(uTime * 0.2, -uTime * 0.5)));
      // Inside: opaque at the edge, transparent at the centre. Depth into the
      // silhouette comes from n (→ 1 deep inside the palm); the facing ratio
      // (G/R) keeps camera-facing surfaces of thin fingers see-through too.
      float facing = nearS.g / max(n, 1e-3);
      float centre = max(smoothstep(0.55, 0.96, n), smoothstep(0.62, 0.95, facing) * smoothstep(0.62, 0.82, n));
      float inner = smoothstep(0.22, 0.5, n) * pow(1.0 - centre, 1.5);
      float a = clamp(max(outer * uOuter, inner * uInner), 0.0, 1.0);
      gl_FragColor = vec4(uColor * a, a); // premultiplied alpha
    }
  `,
};

// Visible hand material tweak: rim darkening so the model blends into the aura.
export const handRimChunk = /* glsl */ `
  float rimF = 1.0 - abs(dot(normalize(vNormal), normalize(vViewPosition)));
  gl_FragColor.rgb *= mix(1.0, 0.25, pow(rimF, 2.5));
`;
