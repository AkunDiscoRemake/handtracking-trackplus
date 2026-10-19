# Arquitetura

## Visão geral

```
câmera (getUserMedia, traseira)            thread principal                         Web Worker
──────────────────────────────            ────────────────                         ──────────
<video>  ── requestVideoFrameCallback ──► timestamp de captura (captureTime)
                                          se worker livre e FPS-cap permite:
                                          createImageBitmap(redimensionado 640px) ──► OffscreenCanvas 2D
                                          (senão: quadro descartado e contado)        ├ letterbox 320 → tensor BGR (reusado)
                                                                                      ├ RTMDet (só quando necessário)
                                                                                      ├ recortes afins 256 (com rotação) → batch RGB (reusado)
                                                                                      ├ RTMPose (1 chamada para as 2 mãos)
                                                                                      ├ SimCC → keypoints na imagem
                                                                                      └ tracker: One Euro, Kalman, lifting 3D, gestos
                                          ◄── resultado + tempos ───────────────────┘
requestAnimationFrame:
  predição limitada até o timestamp do quadro exibido
  rig dos GLB → máscara → pirâmide de blur → composição da aura → mão → camadas 3D → overlay
```

## Ritmo de quadros e latência

- **Um quadro em voo.** A thread principal só envia um quadro ao worker quando o anterior terminou. Não há fila: um quadro velho nunca espera atrás de outro. Quadros que chegam com o worker ocupado são descartados e contados (`descartados` no HUD de debug).
- **Teto de FPS de inferência** (Ajustes) para economizar bateria; a renderização continua na taxa da tela.
- **Timestamps monotônicos.** Todo tempo vem de `performance.now()` (monotônico). Para câmeras, `requestVideoFrameCallback` fornece `captureTime` no mesmo relógio, então "captura→resultado" é uma medida real, não estimada. Para arquivos de vídeo não existe `captureTime`; o HUD mostra "n/d" em vez de inventar.
- **Predição até o quadro exibido.** O `<video>` mostra quadros com seu próprio atraso. A cada `requestAnimationFrame`, os landmarks são extrapolados do instante de captura do resultado até o instante de captura do quadro que está na tela. A predição é limitada em horizonte (padrão 80 ms) e em deslocamento (35 % do tamanho da palma em 2D, 5 cm em 3D). O que sobra acima do limite aparece como "atraso residual" no HUD.
- **ROI tracking.** Com as mãos rastreadas, o detector não roda: as ROIs do próximo quadro vêm de um Kalman sobre centro e tamanho da mão, com o recorte girado para deixar a mão "em pé". O detector roda a cada 3 quadros quando falta uma mão, a cada 45 quadros como checagem, em todo quadro enquanto alguma mão estiver ocluída e a ~8 Hz no modo ocioso (sem mãos há 1,5 s).
- **Buffers reutilizáveis.** Tensores de entrada (`Float32Array`) e canvases são alocados uma vez; o batch de pose usa `subarray` (sem cópia).
- **Resolução de render adaptativa.** Se a tela cai abaixo de 45 FPS por 2,5 s, o `pixelRatio` do WebGL desce em passos de 10 % (até 50 %) e volta a subir com folga. Máscara e blur da aura rodam em 1/2, 1/4, 1/8 e 1/16 da resolução.
- **Aba oculta:** nenhum quadro é enviado; o `requestAnimationFrame` para sozinho.

## Tracker

`js/core/tracker.js`

- **Associação:** cada ROI carrega o `trackId` da trilha que a gerou; detecções novas só viram ROI se não se sobrepõem a uma trilha existente. Duas ROIs que convergem para a mesma mão são fundidas (IoU > 0,5).
- **Confiança:** média dos 21 escores SimCC (mín(max_x, max_y), como no mmpose). Medido nas fotos de teste: mãos ≈ 0,50–0,55, recortes de fundo ≈ 0,16–0,21. Limiares: 0,38 para criar trilha, 0,32 para manter.
- **Oclusão:** quando a pose de uma trilha falha, ela entra em `coasting`: a posição é extrapolada com velocidade amortecida, a ROI cresce 25 %, o detector passa a rodar todo quadro e a confiança decai. Se a mão reaparecer em até 450 ms, a mesma trilha (mesmo ID, mesma lateralidade, mesmo estado de filtros) é retomada; saltos grandes reiniciam os filtros.
- **Lateralidade:** o RTMPose não informa esquerda/direita. A quiralidade do polígono da palma na imagem diz apenas "direita com palma para a câmera" ou "esquerda com dorso para a câmera". No nascimento da trilha, a ambiguidade é resolvida pela suposição configurável (palmas ou dorso para a câmera); com duas mãos, a segunda recebe a lateralidade oposta. Depois disso a lateralidade fica fixa na trilha e uma inversão de quiralidade passa a significar que a mão girou.

## Filtros

- **One Euro** (Casiez et al., 2012) sobre os 42 valores 2D, com velocidade normalizada pelo tamanho da palma em pixels, para que o mesmo `beta` funcione perto ou longe da câmera.
- **Kalman de velocidade constante** (por coordenada) sobre os keypoints, com ruído de medição inversamente proporcional à confiança: fornece a velocidade usada na predição e na travessia de oclusões. Outro Kalman acompanha centro e tamanho da ROI.
- O "ruído de landmarks" do HUD é o RMS (px) entre a saída bruta do modelo e a filtrada.

## 2D → 3D

`js/core/lift.js`. Uma câmera monocular não mede profundidade métrica. O lifting produz uma pose 3D plausível que **reprojeta exatamente** sobre os keypoints detectados:

1. Ajuste planar dos 5 pontos da palma ao template anatômico (fraca perspectiva, solução fechada): escala e rotação com a ambiguidade de inclinação de duas soluções. No primeiro quadro da trilha a escolha é pelo menor erro de reprojeção em perspectiva completa; depois, por continuidade temporal.
2. Profundidade da palma = `f / escala`, supondo mão do tamanho do template (adulto médio). O erro é proporcional à diferença de tamanho real.
3. Cada falange mantém o comprimento do template: o ponto filho fica na interseção do seu raio de câmera com a esfera de raio L em torno do pai (duas raízes = duas hipóteses). A raiz escolhida continua o osso anterior, dobra para o lado da palma e se mantém próxima do quadro anterior.

O FOV horizontal usado para `f` é configurável (padrão 67°, típico de câmera principal de celular). O alinhamento visual na tela não depende dele, só a profundidade estimada.

## Rig

`js/render/handRig.js`. O pulso recebe a rotação que leva o referencial da palma do template ao observado. Cada osso recebe o giro mínimo que leva a direção de repouso (transportada pelo pai) à direção observada do keypoint filho; a rotação local é `G_pai⁻¹ · G_osso`. Os GLB têm rotações de repouso identidade, então o resultado é direto.

### GLB

`tools/gen-hand-glb.mjs` gera os dois arquivos do zero:

- **Geometria:** campo de distância assinada (cones arredondados para as falanges com raios anatômicos, palma extrudada a partir de um contorno, eminências tenar e hipotenar, coto de antebraço), união suave, poligonização por Naive Surface Nets em grade de 2 mm, suavização de Taubin e reprojeção na isosuperfície; normais pelo gradiente do campo. ≈ 16,8 mil vértices e 33,6 mil triângulos por mão.
- **Esqueleto:** 21 juntas com a mesma topologia dos landmarks (`wrist`, `thumb_cmc`…`pinky_tip`), sufixo `_L`/`_R`, `extras.landmark` com o índice do keypoint.
- **Skin:** até 4 influências por vértice, pesos por soft-min da distância à superfície de cada segmento ósseo.
- **Animação:** clipe `curl` (abre → punho → abre) para conferir a articulação; veja `tools/glb-viewer.html`.
- Validado pelo Khronos glTF-Validator com 0 erros e 0 avisos.

## Aura

`js/render/aura.js`, `js/render/shaders.js`. Tudo com alpha pré-multiplicado sobre a imagem da câmera.

1. **Máscara:** as malhas com skin, infladas 2,5 mm ao longo da normal deformada, são desenhadas em 1/2 resolução: R = cobertura × confiança, G = cobertura × "facing" (|N·V|).
2. **Pirâmide de blur:** Gaussiano separável em 1/4, 1/8 e 1/16 da resolução, alvos half-float quando o aparelho permite (sem bandas), com espaçamento de amostras ≤ 1 texel por nível.
3. **Composição:** o campo estreito funciona como distância à borda. A opacidade é máxima na silhueta e cai suavemente até zero no centro da mão (profundidade dentro da silhueta e facing) e para fora (campo largo, lido com filtro bicúbico B-spline). Turbulência fBm desloca o halo para cima. Cor: preto absoluto `#000` ou quase preto `#08080a`.
4. **Camadas 3D:** duas cascas infladas (3 e 7,5 mm) com opacidade de Fresnel: opacas onde a superfície vira de lado (silhueta), transparentes onde encara a câmera (centro). São testadas contra a profundidade da mão, então partes atrás dos dedos ficam escondidas.

No modo Aura Only a mão é desenhada apenas no depth buffer, então o centro fica transparente e as cascas continuam corretamente ocultas.

## Seleção de backend

`js/worker/backends.js`. Para cada modelo, em ordem de preferência (WebNN NPU, WebNN GPU, WebGPU, WASM):

1. cria a sessão (falha → indisponível);
2. roda numa entrada fixa (foto real, tensor idêntico ao gravado no Node) e compara com a referência da CPU em `models/calibration.json` (tolerâncias: 6 px/0,06 na caixa do detector, 4 px/0,06 nos keypoints); divergência → rejeitado, porque alguns drivers calculam lixo sem erro;
3. mede a mediana de 5 execuções;
4. escolhe o primeiro válido cuja mediana esteja a até 1,3× do mais rápido (prioridade à NPU sem aceitar uma NPU muito mais lenta).

O relatório completo aparece no HUD de Debug. No WebNN a pose usa batch fixo 1 (formas estáticas); as duas mãos rodam em duas chamadas.
