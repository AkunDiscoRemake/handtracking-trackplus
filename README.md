# Aura Hands

Rastreamento das duas mãos em tempo real no navegador, com modelos neurais reais de detecção e landmarks (sem MediaPipe), mãos 3D rigadas (GLB) animadas pelos landmarks e uma aura negra tridimensional renderizada em WebGL2. Feito só com HTML, CSS e JavaScript.

- **Detecção:** RTMDet-nano (hand), OpenMMLab, ONNX, 320×320.
- **Landmarks:** RTMPose-m SimCC (hand, 21 pontos), OpenMMLab, ONNX, 256×256, batch com as duas mãos.
- **Runtime:** ONNX Runtime Web 1.30.0 com seleção validada de backend: WebNN NPU → WebNN GPU → WebGPU → WASM (CPU).
- **Render:** three.js 0.186.0 (WebGL2), `left_hand.glb` e `right_hand.glb` gerados por este projeto, shaders próprios para a aura.
- **Modos:** Aura Only, Hand + Aura e Debug, com HUD de FPS, latência, jitter, confiança e backend.

## Como rodar

Requisitos: Node.js 18+ (só para preparar os arquivos e servir localmente) e um navegador Chromium recente (Chrome/Edge 113+ para WebGPU).

```bash
npm install            # three, onnxruntime-web, gltf-validator, playwright (só testes)
npm run setup          # copia o runtime para vendor/ e baixa os modelos (≈ 59 MB, SHA-256 verificado)
npm run serve          # http://localhost:8080
```

`npm run setup` baixa os pacotes oficiais do OpenMMLab (`download.openmmlab.com`), extrai `end2end.onnx` em JavaScript puro, confere o SHA-256 e divide o modelo de pose em partes de 14 MB (`models/*.bin`) para hospedagem estática. Os GLB já vêm prontos em `assets/hands/`; para regenerar: `npm run gen:glb`.

### Site público com câmera (GitHub Pages)

O workflow `.github/workflows/pages.yml` baixa os modelos, roda os testes, monta `build/site/` (`node tools/build-site.mjs`) e publica no GitHub Pages a cada push. O GitHub Pages serve HTTPS, então a câmera funciona no celular. Como ele não envia os cabeçalhos COOP/COEP, o service worker `coi-sw.js` os adiciona no navegador; a página recarrega uma vez no primeiro acesso para ativar o WASM multithread.

Configuração única no GitHub:

1. O Pages exige repositório **público** no plano gratuito (repositórios privados precisam de GitHub Pro/Team): Settings → General → Danger Zone → Change visibility.
2. Settings → Pages → Build and deployment → Source: **GitHub Actions**.
3. Actions → "Deploy site" → Run workflow (ou faça um push). O endereço fica `https://<usuário>.github.io/<repositório>/`.

### No celular (câmera traseira)

A câmera exige contexto seguro (HTTPS ou `localhost`). Duas opções:

1. **USB + Chrome:** `adb reverse tcp:8080 tcp:8080` e abra `http://localhost:8080` no Chrome do Android.
2. **HTTPS:** hospede a pasta (com `vendor/` e `models/`) em qualquer servidor estático com HTTPS. Para WASM multithread, o servidor precisa enviar `Cross-Origin-Opener-Policy: same-origin` e `Cross-Origin-Embedder-Policy: require-corp` (o `tools/serve.mjs` já faz isso).

Na primeira abertura os modelos são baixados e guardados no Cache Storage; as aberturas seguintes funcionam offline.

### Parâmetros de URL

| Parâmetro | Efeito |
|---|---|
| `?src=camera-env` / `camera-user` / `sample` / `synthetic` | inicia direto numa fonte |
| `?mode=aura` / `hand_aura` / `debug` | modo inicial |
| `?backend=webnn-npu` / `webnn-gpu` / `webgpu` / `wasm` | força um backend (com fallback para WASM) |
| `?ort=wasm` | usa o build de ONNX Runtime só-CPU (binário de 14 MB) |

## Testes e verificação

```bash
npm test                 # 22 testes: filtros, decodificação, lifting 3D, gestos, tracker, GLB, modelos reais em fotos reais
npm run validate:models  # roda os modelos nas fotos de teste e grava imagens anotadas em build/validation/
node tools/validate-glb.mjs   # Khronos glTF-Validator nos dois GLB (0 erros, 0 avisos)
npm run bench:node       # latência real dos modelos (WASM) nesta máquina
npm run e2e              # Chromium headless: foto, câmera falsa com vídeo de mão real, oclusão, 3 modos
```

O teste end-to-end alimenta a câmera falsa do Chromium com um clipe Y4M gerado a partir de uma foto real (domínio público), com a mão se movendo e uma oclusão de 10 quadros, e verifica rastreamento, estado "ocluída" e recuperação com o mesmo ID.

## Estrutura

```
index.html, css/app.css      interface
js/main.js                   fontes de vídeo, frame pacing, predição, render loop, HUD
js/core/                     lógica pura (roda no navegador e no Node)
  handModel.js               topologia de 21 pontos e template anatômico (metros)
  decode.js, warp.js         letterbox, recortes afins com rotação, SimCC, NMS, normalização
  filters.js                 One Euro, Kalman de velocidade constante, predição limitada
  lift.js                    2D → 3D monocular com prior anatômico
  tracker.js                 duas mãos, ROI tracking, associação, oclusão, lateralidade
  gestures.js                10 gestos com histerese
js/worker/                   ONNX Runtime, cache de modelos, seleção/validação de backend
js/render/                   cena three.js, rig, aura, shaders, overlay de debug
assets/hands/*.glb           mãos rigadas (21 ossos, animação "curl")
models/                      manifest.json, calibration.json (+ modelos após setup)
tools/                       geração de GLB, preparação de modelos, servidor, benchmark, e2e
tests/                       testes node:test + fotos de teste
docs/                        arquitetura, hardware, medições, limitações
```

## Documentação

- [docs/ARQUITETURA.md](docs/ARQUITETURA.md): pipeline, threads, buffers, filtros, lifting, rig, aura.
- [docs/HARDWARE.md](docs/HARDWARE.md): NPU/GPU/CPU, Snapdragon e MediaTek, o que o navegador expõe de fato.
- [docs/MEDICOES.md](docs/MEDICOES.md): números medidos neste ambiente e como medir no seu aparelho.
- [docs/LIMITACOES.md](docs/LIMITACOES.md): o que este projeto não faz.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): licenças de modelos, bibliotecas e imagens.

## Licença

Código e GLB: Apache-2.0 (veja `LICENSE`). Dependências e modelos têm licenças próprias, listadas em `THIRD_PARTY_NOTICES.md`.
