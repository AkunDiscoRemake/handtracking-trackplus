# Hardware: NPU, GPU e CPU

## O que um app web consegue usar

Uma página não acessa NNAPI, QNN (Qualcomm) ou NeuroPilot (MediaTek) diretamente. O único caminho padronizado do navegador até uma NPU é a **WebNN API**. Este projeto usa o ONNX Runtime Web, que encaminha os modelos para:

| Backend | O que é | Disponibilidade (out/2026) |
|---|---|---|
| `webnn` `deviceType: 'npu'` | WebNN pedindo o acelerador neural | Chrome/Edge com WebNN habilitado; implementações dependem do SO (DirectML/NPU no Windows, Core ML no macOS, LiteRT/TFLite no ChromeOS e Android). Em muitos aparelhos ainda exige flag (`chrome://flags/#web-machine-learning-neural-network`). |
| `webnn` `deviceType: 'gpu'` | WebNN na GPU | idem |
| `webgpu` | Shaders de computação WebGPU | Chrome Android 121+ na maioria das GPUs Adreno e Mali recentes |
| `wasm` | CPU, SIMD, multithread se a página for cross-origin isolated | universal |

O navegador **não informa** em qual unidade o WebNN realmente executou. O que o app mostra é: quais contextos WebNN puderam ser criados, se a sessão passou na validação numérica e quanto tempo levou. Isso é o que dá para afirmar sem adivinhar.

## Qualcomm Snapdragon

- **GPU Adreno:** WebGPU funciona no Chrome Android em Adreno 6xx/7xx recentes. É o caminho acelerado mais provável hoje.
- **NPU Hexagon/HTP:** só alcançável por WebNN, se a implementação do navegador no Android delegar para um runtime que use o HTP. Não há garantia; o app testa e mede.
- O HUD identifica o fornecedor da GPU via `GPUAdapter.info` (por exemplo `qualcomm` / `adreno-7xx`).

## MediaTek Dimensity / Helio

- **GPU Mali (Dimensity) ou PowerVR (alguns Helio):** WebGPU disponível na maioria dos Mali-G7x/G6xx recentes com Chrome atualizado; em PowerVR o suporte varia por driver.
- **APU:** mesma situação da Qualcomm, só por WebNN e sem garantia.
- O HUD mostra `arm` / `mali` ou `img-tec` quando o navegador expõe.

## Requisitos

- **Mínimo:** Chrome/Edge 113+ (Android 10+ ou desktop), WebGL2, 3 GB de RAM livre para o navegador, ≈ 60 MB de armazenamento para o cache de modelos.
- **Recomendado para tempo real:** WebGPU funcional (Chrome 121+ no Android) ou NPU via WebNN, e página servida com COOP/COEP para WASM multithread no fallback.
- **Câmera:** contexto seguro (HTTPS ou localhost). A câmera traseira é pedida com `facingMode: environment`; o FOV real não é exposto pelo navegador, por isso há um ajuste de FOV.

## Custo dos modelos

O RTMPose-m tem ≈ 13,6 M de parâmetros (55 MB em FP32) e é a parte cara do pipeline. Em CPU via WASM ele é lento (veja `MEDICOES.md`); com uma mão por quadro e ROI tracking, o custo é uma chamada de pose por quadro e o detector quase nunca roda. Em CPU pura, espere poucas inferências por segundo; a renderização continua na taxa da tela com predição entre os resultados.
