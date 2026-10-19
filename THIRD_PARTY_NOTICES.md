# Avisos de terceiros

Licenças verificadas nos próprios pacotes/arquivos em 2026-10-19.

## Modelos neurais

| Modelo | Origem | Licença | SHA-256 do `end2end.onnx` |
|---|---|---|---|
| RTMDet-nano hand (`rtmdet_nano_8xb32-300e_hand-267f9c8f`) | OpenMMLab mmpose, `download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/` | Apache-2.0 | `568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2` |
| RTMPose-m SimCC hand5 (`rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320`) | idem | Apache-2.0 | `39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29` |

Os pesos são publicados pela OpenMMLab sob Apache-2.0 (licença do repositório mmpose). Segundo a documentação do RTMPose, o modelo "hand5" foi treinado com cinco conjuntos de mãos (COCO-WholeBody-Hand, OneHand10K, FreiHand2d, RHD2d e Halpe-Hand); o RTMDet-nano hand foi treinado com dados de mãos do mesmo projeto (consulte a página do modelo no mmpose). Esses conjuntos de dados têm termos próprios (alguns restritos a pesquisa). Antes de uso comercial, revise os termos dos datasets com sua assessoria jurídica; este projeto não faz afirmação sobre isso.

Nenhum modelo ou código do MediaPipe é usado.

## Bibliotecas (vendor/)

| Pacote | Versão | Licença |
|---|---|---|
| three (incl. `GLTFLoader`, `BufferGeometryUtils`, `SkeletonUtils`) | 0.186.0 | MIT, © three.js authors |
| onnxruntime-web (JS + binários WASM) | 1.30.0 | MIT, © Microsoft Corporation |

## Ferramentas de desenvolvimento (não distribuídas)

| Pacote | Versão | Licença |
|---|---|---|
| gltf-validator | 2.0.0-dev.3.10 | Apache-2.0, © The Khronos Group |
| playwright | 1.56.1 | Apache-2.0, © Microsoft |

## Imagens de teste (`tests/fixtures/`, `assets/samples/`)

| Arquivo | Origem | Licença |
|---|---|---|
| `vsign_pd.png` | Wikimedia Commons, "Mrs. Constance B. Motley, first woman Senator, 21st Senatorial District, N.Y., raising hand in V sign" (Library of Congress, New York World-Telegram & Sun collection) | Domínio público |
| `move_pd.png` | Wikimedia Commons, "PlayStation-Move-in-Hand.jpg" | Domínio público |

Ambas foram reduzidas para 640 px e convertidas para PNG.

## Modelos 3D

`assets/hands/left_hand.glb` e `right_hand.glb` são gerados do zero por `tools/gen-hand-glb.mjs` (campo de distância + Surface Nets) e distribuídos sob a licença deste projeto (Apache-2.0).
