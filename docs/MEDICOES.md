# Medições

Todos os números abaixo foram medidos neste ambiente de desenvolvimento em 2026-10-19. **Nenhum foi medido em celular.** Não há números de NPU ou GPU de celular aqui porque nenhum aparelho estava disponível; meça no seu com as ferramentas do final.

## Ambiente

- CPU: Intel Xeon @ 2,10 GHz, 4 vCPUs, sem GPU física.
- Node.js v22.22.0, onnxruntime-web 1.30.0 (WASM SIMD).
- Chromium headless 141 (Playwright 1.56.1), WebGL e WebGPU por SwiftShader (emulação por software).

## Modelos em CPU (`npm run bench:node`, 20 execuções após 3 de aquecimento)

| Modelo | 1 thread (mediana / p95) | 4 threads (mediana / p95) |
|---|---|---|
| RTMDet-nano 320×320 | 59,0 / 62,6 ms | 24,1 / 40,8 ms |
| RTMPose-m 256×256, batch 1 | 270,9 / 302,7 ms | 90,2 / 129,8 ms |
| RTMPose-m 256×256, batch 2 | 538,9 / 583,9 ms | 261,4 / 312,4 ms |

## Navegador, pipeline completo (`npm run e2e`)

Câmera falsa do Chromium alimentada por vídeo 640×480 a 30 FPS de uma mão real (foto de domínio público em movimento, com oclusão de 10 quadros).

| Configuração | Inferência média (p95) | Captura→resultado médio | FPS de inferência |
|---|---|---|---|
| Sem isolamento (WASM 1 thread) | 307 ms (576) | 379 ms | 2,4 |
| COOP/COEP (WASM 4 threads) | 196 ms (404) | 339 ms | 2,9 |

Os FPS de render nesses testes (5–20) refletem o WebGL emulado por software competindo pela mesma CPU e não indicam o desempenho num aparelho com GPU.

Seleção de backend com WebGPU emulado (`--webgpu`): as sessões WebGPU foram criadas e **passaram na validação numérica** contra a referência da CPU nos dois modelos; o seletor as rejeitou pelo tempo medido (4.746 ms detector e 8.364 ms pose, contra 66 ms e 249 ms em WASM), como esperado de uma GPU emulada.

Rastreamento no mesmo teste: lateralidade e gesto corretos na foto do "V" (direita, vitória, confiança 0,54–0,55); no vídeo, a mão ocluída entrou em estado "ocluída" e foi recuperada com o mesmo ID (2–3 recuperações por execução).

## Como medir no seu aparelho

1. Abra o app no modo **Debug**: o HUD mostra o relatório por backend (válido/inválido e mediana), FPS de render/câmera/inferência, inferência média e p95 (detector, pose, pré-processamento), captura→resultado (com `captureTime` real da câmera), atraso residual após predição, jitter (desvio padrão dos intervalos de render, câmera e inferência), ruído de landmarks em px e quadros descartados.
2. **Ajustes → Benchmark (20×)** mede a mediana de cada modelo no backend escolhido.
3. **Ajustes → Gravar métricas / Baixar CSV** grava uma linha por quadro para análise.
