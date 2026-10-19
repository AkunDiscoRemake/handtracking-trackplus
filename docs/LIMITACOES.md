# Limitações

- **Profundidade não é métrica.** A distância da mão vem da suposição de tamanho médio de mão adulta; para mãos menores ou maiores o erro é proporcional (tipicamente ±10–20 %). A pose 3D dos dedos é uma reconstrução plausível com prior anatômico, não uma medida. O que é exato é a reprojeção: o modelo 3D cai sobre os keypoints 2D.
- **Ambiguidade de profundidade dos dedos.** Um dedo apontado para a câmera ou para longe dela produz o mesmo 2D. O lifting escolhe a hipótese mais anatômica e temporalmente coerente, mas pode errar em poses incomuns.
- **Lateralidade é inferida.** O modelo de landmarks não classifica esquerda/direita. A primeira detecção usa a suposição de orientação (Ajustes). Se a suposição não corresponder à cena, as mãos nascem trocadas até sumirem e reaparecerem.
- **NPU não é garantida.** O acesso à NPU no navegador depende do WebNN, que em muitos aparelhos ainda está atrás de flag ou delega para GPU/CPU. Não há como confirmar pela página em qual unidade o modelo rodou.
- **CPU é lenta para o RTMPose-m.** Em WASM puro espere poucas inferências por segundo. A renderização e a predição continuam fluidas, mas a aura atrasa em movimentos rápidos.
- **Duas mãos no máximo**, sem tratamento especial para mãos entrelaçadas: dedos cruzados entre mãos podem trocar keypoints.
- **Gestos são regras** sobre extensão dos dedos (2D + 3D) com histerese, não um classificador treinado. Funcionam para gestos estáticos claros: mão aberta, punho, apontar, vitória, rock, telefone, joinha, polegar para baixo, pinça, ok.
- **A prévia hospedada** roda só em CPU (o binário WASM com WebGPU/WebNN tem 28 MB e o host da prévia não permite arquivos desse tamanho nem buscá-lo de CDN) e, em geral, sem câmera, porque o quadro sandboxed recusa `getUserMedia`. Use a foto de exemplo, um vídeo/imagem seu ou a demo sintética ali; a versão completa roda localmente ou em seu servidor.
- **Termos dos datasets.** Os pesos são Apache-2.0, mas os dados de treino têm termos próprios; veja `THIRD_PARTY_NOTICES.md`.
