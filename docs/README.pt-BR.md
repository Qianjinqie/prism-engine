[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | **Português (BR)** | [Русский](README.ru.md)

# Prism Engine

> Motor de mídia em Rust, determinístico e sem dependências — distribuído como **um único arquivo TXT**: baixe um texto de 17 MB, execute o script de desempacotamento e obtenha a árvore de código-fonte completa (28 crates), sem npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

(Os emblemas são estáticos e factuais; o critério dos números está abaixo; `crates-28` = número de crates **dentro do pacote de distribuição**; o repositório tem mais 1 crate de demonstração que não entra no pacote, portanto o número no repositório é 29. Após integrar ao CI, eles podem ser substituídos por emblemas dinâmicos.)

---

## O que é isto

Prism Engine é um motor de mídia escrito em Rust, sem dependências de código de terceiros: ler imagens, compor, desenhar quadros, montar linhas de tempo e produzir o filme final — tudo dentro de um único motor determinístico. Todas as 110 dependências de execução
(mais 20 dependências de teste) são dependências `path` deste próprio repositório — sem dependência de rede, sem gerenciador de pacotes,
sem risco de cadeia de suprimentos.

Adota **distribuição em arquivo único de código-fonte**: o motor inteiro (622 arquivos, 28 crates) é consolidado em um texto UTF-8
`prism-engine-v2.0-source.txt`. Basta ter esse arquivo para reconstruir localmente uma árvore de diretórios idêntica à do repositório.

---

## O que o motor faz

Resumo em uma frase: **uma imagem é o quadro em um instante; um vídeo é uma sequência de quadros ao longo do tempo** —
a mesma descrição de cena: congele um instante e você tem um quadro único; avance continuamente e grave, e você tem vídeo; imagem estática e imagem em movimento compartilham um único conjunto de capacidades.
Abaixo, dividido em cinco blocos por "o que você quer fazer": **ler → compor → desenhar → produzir o filme → integrar à sua própria toolchain**.

### O que lê (entrada)

- **Fachada de decodificação unificada**: uma única entrada `decode_image(bytes, ext)` despacha imagens estáticas — PNG (não entrelaçado, 8/16 bits),
  JPEG baseline (IDCT de ponto fixo, bit a bit determinístico entre plataformas), **magic number com prioridade, extensão como fallback** (decodifica mesmo com a extensão errada);
  formatos fora da matriz ou que exigem cooperação externa sempre dão erro em três partes (onde / o que aconteceu / como corrigir) e trazem sugestão de cooperação com ffmpeg,
  **nunca falhando em silêncio**.
- **Imagens animadas / áudio / contêineres**: decodificação quadro a quadro nativa de GIF e APNG (`prism-media`); WAV com decodificação embutida, mp3/aac/FLAC
  pelo caminho de detecção do ffmpeg (`prism-audio`); releitura de MP4-MJPEG (`decode_mp4_video`).
- **Pacote completo de EXR**: leitura e escrita, compressão, DWA, com **comparação cruzada de fluxo de bits** contra o OpenEXR oficial (não é escrever e ler o próprio formato).
- **RAW de câmera**: decodificação RAW de linha de base. **Limite**: `Compression=7` (lossless JPEG) de DNGs reais não está implementado,
  e DNGs gerados por câmeras reais não podem ser lidos — isso está registrado em “Limites honestos”, sem esconder nada.
- **Ativos 3D**: importação de glTF / OBJ, análise de USD (`.usda`) e pacote de parâmetros MaterialX; malhas (mesh) podem entrar na cena de renderização.
- **Fontes externas**: `load_font_file` carrega diretamente TTF/OTF de terceiros, a cadeia de fallback resolve nível a nível,
  e glifos ausentes geram diagnóstico claro em vez de "tofu".

### O que compõe (sintaxe de autoria)

- **Sintaxe completa em blocos da DSL v2**: um `.prism` descreve "o que você quer, como posicionar, como animar, como deve parecer
  e em que exportar"; o produto da compilação é um grafo de nós + canais de propriedades + linha de tempo.
- **Cinco combinadores**: `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` — combinadores de alto nível expandidos em grafo de nós; "um filme de 90 segundos" deixa de ser código de cena escrito à mão e passa a ser montagem.
- **Transições e morph**: Fade / Wipe / Dissolve / Push totalmente parametrizados (duração, easing, direção),
  Morph com deformação de polilinha, e a sequência de cenas pode ser amostrada em qualquer instante.
- **Composição de fórmulas**: `math_ast` + `math_layout` compõem sobrescrito / subscrito / frações / radicais / operadores grandes /
  matrizes; o bloco `formula` da DSL compõe diretamente, na mesma linha de base do texto corrido.

### O que desenha (renderização e simulação)

- **Renderização offline fotorrealista**: path tracing + BSDF fisicamente corretos (difuso / especular / vidro / metal);
  vidro e metal usam inverso estocástico (Monte Carlo com série de Neumann), com suporte a meios volumétricos e subsurface scattering.
- **Quatro famílias de primitivas curvas**: cilindro / cone / toro / superfície de revolução, todas com interseção analítica (equações quárticas pela
  solução fechada de Ferrari, raízes exatas); malha triangular arbitrária + BVH com SAH, com verificação de paridade entre duas trajetórias.
- **GPU compute**: camada semântica de compute independente de dispositivo + dispatch Vulkan + SPIR-V mínimo escrito à mão;
  roda integralmente em plataformas sem GPU sob o ICD de software Mesa lavapipe (verificável em headless).
- **Física e partículas**: solver unificado XPBD (corpos rígidos / soft body / tecido / juntas / fluidos / fratura),
  fumaça FLIP, sistema de partículas (AudioReactive ligado ao envelope de áudio).
- **Arte generativa e pós-processamento**: campos de fluxo / L-system / autômatos celulares / reação-difusão / fractais / Voronoi;
  blur / bloom / halation / grão de filme / quatro estilos NPR (toon / watercolor / halftone / oil) / redução de ruído tripla.
- **Color grading e vetores**: color grading secundário (seleção HSL + curvas RGB), presets de estilo cinematic / film / anime e
  três níveis de qualidade — rascunho / padrão / cinema (qualidade é parâmetro, não misticismo); rasterização de caminhos vetoriais, draw-on e morph de caminhos.
- **Processamento de áudio**: normalização de loudness (loudnorm), detecção de beat, análise de espectro, extração de envelope —
  pode se ligar à imagem (o AudioReactive das partículas segue o envelope de áudio).
- **O backend de referência em CPU é cidadão de primeira classe**: máquinas sem GPU geram imagens corretas do mesmo jeito, e ele também serve de referência de acerto para a GPU.

### O que produz (entrega)

- **`render_movie` direto**: uma única chamada sobre o produto compilado da DSL v2 percorre "avaliação quadro a quadro → renderização em tile →
  entrega", e um `.prism` gera o filme com um comando (`prism-app render demo.prism`); nós não cobertos dão erro honesto, sem aproximação silenciosa.
- **Ciclo de preview**: `prism-app preview` inicia o serviço de preview; no editor, preview em dois níveis — Draft (padrões geométricos para preservar a taxa de quadros) /
  Real (renderização real com spp baixo, atualização assíncrona).
- **Codificador próprio + controle de taxa**: codificador temporal no domínio JPEG com quadros I/P em contêiner privado `PRMV` (duas gerações de
  schema, v2/v3), controle de taxa por water-filling + estimação de movimento subpixel qpel (medido: PSNR +2.577 dB,
  tamanho −21.6%).
- **Cadeia de cooperação com ffmpeg**: com ffmpeg → converte para H.264/HEVC via pipe rawvideo em stdin;
  sem ffmpeg → degrada automaticamente para MJPEG, **o motor sempre consegue produzir o filme**; não empacota nenhum binário.
- **Entrega de sequência EXR**: a sequência de quadros vai direto para EXR, entrando no pipeline de pós-produção.
- **Farm distribuída**: distribuição por quadro + N workers em processo + mesclagem determinística — MD5 paralelo == MD5 serial,
  com suporte a retomada após interrupção.

### Como integrar (ecossistema)

- **Superfície de ferramentas MCP**: três ferramentas JSON-RPC sobre stdio — `scene_check` (compilação + render de teste + estimativa de quadros),
  `render_submit` (job id determinístico + resumo MD5 por quadro), `diagnose` (localização de erro em três partes).
- **SDK de nós**: negociação de versão semântica (incluindo o caso especial 0.x), com pacote de nós de exemplo incluído
  (`examples/sdk-nodes/cinematic-basic/`, compilável / carregável / negociável).
- **Esqueleto de detecção de hardware encoding**: cadeia de detecção do `prism-ffi-hwenc` em três plataformas (VideoToolbox / NVENC / VAAPI)
  + matriz de feature gates; plataforma incompatível reporta claramente que não é suportada.

> Para os limites detalhados de capacidade (o que **não é possível fazer**), veja a seção “Limites honestos” do `开发者指南.md` (guia do desenvolvedor) após desempacotar —
> ali está registrado o estado real atual (primitivas curvas só em CPU, o ganho da GPU precisa ser medido no hardware alvo, hardware encoding conectado conforme a
> plataforma alvo…); este documento apresenta apenas o panorama, sem exageros.

---

## Por que distribuição em arquivo único

- **Auditável**: um único arquivo de texto, legível a olho nu, possível de `diff` e de verificação por hash; publicado, congelado.
- **Reprodutível**: o empacotamento é determinístico — o mesmo input empacotado duas vezes gera SHA-256 idêntico.
- **Portátil**: sem precisar clonar um repositório enorme; um único arquivo pode ser enviado por e-mail ou chat, e a compilação pode ser reconstruída até offline.
- **Nota de trade-off**: a distribuição em arquivo único abre mão da conveniência de "navegar pelo código diretamente no GitHub" em troca do determinismo de "levar tudo em um arquivo, verificável byte a byte". Após desempacotar, o resultado é um Cargo workspace comum, indistinguível de um projeto Rust convencional.

---

## Início rápido

Pré-requisitos: instalar [Rust](https://www.rust-lang.org/) (com `cargo`) e [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Baixe prism-engine-v2.0-source.txt do repositório e salve como arquivo local
# 2. Abra em um editor de texto e salve o script Node.js entre "UNPACK SCRIPT BEGIN" e
#    "UNPACK SCRIPT END" como unpack.cjs no mesmo diretório
# 3. Execute o desempacotamento (alguns segundos, restaura 622 arquivos)
node unpack.cjs
# 4. No diretório restaurado (a raiz do workspace), rode a suíte completa de testes
cargo test --workspace --release
# Esperado: 3714 passed / 0 failed
```

Após desempacotar, obtém-se um Cargo workspace padrão (28 crates sob `src/`); os comandos `cargo` seguintes são os mesmos de um projeto Rust convencional.

> Caminho de verificação wasm no celular: `rustup target add wasm32-unknown-unknown` e depois `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Estrutura do repositório (após desempacotar)

| Caminho | Descrição |
|------|------|
| `src/` | código-fonte dos **28 crates** (no repositório são 29; o nome original era `crates/`, reorganizado como `src/` na distribuição; o crate de demonstração `prism-anim-demo` não entra no pacote) |
| `Compliance/` | declarações de conformidade: verificação de dependência zero / avisos de terceiros / registro de ativos (nome original no repositório: `docs/合规/`, diretório de conformidade) |
| `开发者指南.md` | documentação de desenvolvedor autocontida (guia do desenvolvedor): primeiros passos / arquitetura / núcleo matemático / gates / extensão (nome original no repositório: `docs/开发者指南.md`) |
| `tools/` | scripts de gate (defesa contra escrita fantasma, varredura de higiene, autoverificação de conformidade) |
| `.github/workflows/` | configuração de CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | itens necessários à compilação (workspace members já apontam para `src/*`) |
| `examples/` | amostras para os testes do `prism-dsl` e pacotes de nós de exemplo (lidos com `include_str!`; sem eles a compilação falha) |

> A documentação de autoridade de design (`docs/设计/`, documentos de design) **não acompanha a distribuição** e permanece apenas no repositório completo do publicador.

---

## Visão geral dos módulos

Os **28 crates** dentro do pacote de distribuição são agrupados por responsabilidade (o repositório tem ainda o crate de demonstração `prism-anim-demo`, que existe apenas
como exemplo reproduzível no repositório e não entra no pacote):

| Categoria | crate |
|------|-------|
| Núcleo | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S de mídia | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Renderização | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animação/geração | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texto/UI | prism-text · prism-rt · prism-app |
| Avançado | prism-ai · prism-dsl · prism-infra · prism-author |
| Bypass (feature gate) | prism-ffi-hwenc |

---

## Compilação e testes

```bash
# Smoke test de todo o repositório (release, recomendado)
cargo test --workspace --release
# Alias equivalente
cargo xtest
# Verificação rápida de um único crate
cargo test -p prism-core --release
```

- Suíte completa: **3714 passed / 0 failed / 0 ignored** (esta versão foi de fato executada na **árvore desempacotada**: 166 blocos de teste;
  no repositório são 167 blocos — a diferença de 1 bloco é o bloco de teste vazio do crate de demonstração, com o mesmo número de casos).
- CI: **6 jobs** — defesa contra escrita fantasma (ghost-guard) / compilação + suíte completa (native) / build wasm32 (wasm) /
  checagem estática de WGSL (wasm-verify) / comparação cruzada de fluxo de bits com o OpenEXR oficial (interop) / verificação cruzada no host node (lint).
- 3 scripts locais de gate (totalmente sem dependências, usando apenas `node`, abaixo) + 1 script de empacotamento;
  destes, a autoverificação de conformidade passa **9/9** e a autoprova inversa acusa vermelho em **8/8**.

Para rodar os gates localmente (requer Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Defesa contra escrita fantasma: G1 definição duplicada/G2 superconjunto estrito/G3 documentação no topo/G4 compilação
node tools/hygiene_scan.mjs               # Higiene open source: lista item a item termos internos remanescentes (arquivo:linha); qualquer ocorrência → exit 1
node tools/compliance_check.mjs           # Autoverificação de conformidade: deve passar 9/9
node tools/compliance_check.mjs --self-test   # Autoprova inversa: deve acusar vermelho, provando que o critério tem poder de discriminação
```

> **A varredura de higiene open source ainda tem ocorrências existentes hoje**: há termos internos antigos remanescentes no repositório, dos quais
> **a camada de documentação distribuída no pacote já está zerada** (README / guia do desenvolvedor / Compliance); o restante está em **comentários de código dentro do pacote de distribuição**
> e em **arquivos do repositório que não acompanham o pacote**, já classificados conforme o caso e listados como itens de limpeza independentes. O script existe justamente para tornar
> o resíduo **visível e limpável item a item**, em vez de fingir que está limpo — não interprete "contagem de ocorrências diferente de 0" como falha do gate.

---

## Conformidade

O motor promete **zero dependências de código de terceiros**, e isso é verificado por autoverificação:

- **C1 Zero dependências de código de terceiros**: todas as 130 dependências (110 de execução + 20 de teste) são dependências `path` deste repositório; sem `build.rs`, sem diretivas de linkedição nativa, e o `Cargo.lock` fixa 29 pacotes deste repositório.
- **C2 Registro de ativos completo**: os campos da tabela principal dos 5 ativos estão completos e o estado de internalização está em vocabulário controlado; o texto integral da licença da fonte OFL 1.1 acompanha a fonte em `src/prism-text/font_data/LICENSE`.
- **C3 Aviso de atribuição de terceiros**: `Compliance/THIRD-PARTY-NOTICES.md` está presente e distingue explicitamente "ativo de dados ≠ dependência de código".

> Nota: este repositório distribui **código-fonte**. O ativo de fonte (subconjunto Prism Sans SC) é licenciado sob SIL OFL 1.1, e o **texto integral da licença acompanha a fonte em** `src/prism-text/font_data/LICENSE`; a origem dos ativos e o registro de licenças estão em `Compliance/assets/资产登记.md` (registro de ativos).

---

## Empacotamento determinístico (lado do publicador)

Para regerar o TXT de distribuição a partir do repositório completo (a versão pode ser passada; por padrão sai a versão atual v2.0):

```bash
node tools/pack_source_txt.mjs          # Padrão → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # Forma equivalente (versão passada explicitamente)
node tools/pack_source_txt.mjs 1.0      # Volta ao nome de artefato da era v1.0 (rodar sobre o código v1.0 reproduz o pacote antigo byte a byte)
node tools/pack_source_txt.mjs --verify # Validação de ida e volta logo após empacotar (compara a árvore de código arquivo por arquivo, byte a byte)
# Rode duas vezes seguidas; SHA-256 idêntico prova que o empacotamento determinístico vale
```

O escopo do empacotamento é estritamente: `crates/` (→ `src/`), `docs/合规/` (→ `Compliance/`), `docs/开发者指南.md`, `tools/`, `.github/`, `.cargo/`, `examples/`, `Cargo.toml`, `Cargo.lock`. Ficam deliberadamente **excluídos** `docs/设计/`, `target/`, `node_modules/`, o próprio `dist/` e todo artefato intermediário.

---

## Contribuição

Issues e Pull Requests são bem-vindos. Antes de enviar, mantenha:

- `cargo test --workspace --release` totalmente verde;
- `node tools/hygiene_scan.mjs` **sem novas** ocorrências de termos internos (o script lista todas as ocorrências, incluindo as existentes);
- `node tools/compliance_check.mjs` passando 9/9.

Guia mais detalhado de arquitetura e extensão no `开发者指南.md` (guia do desenvolvedor) após desempacotar.

---

## Licença

Este projeto é publicado sob a **Apache License 2.0**. Veja o arquivo [`LICENSE`](../LICENSE) para detalhes.

> Aviso de conformidade: a Apache-2.0 exige a preservação dos avisos de copyright e de licença; ao redistribuir com base neste motor, preserve os avisos de atribuição e o registro de ativos em `Compliance/`.
