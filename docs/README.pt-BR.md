[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | **Português (BR)** | [Русский](README.ru.md)


# Prism Engine

> Mecanismo de mídia em Rust, determinístico e sem dependências — distribuído como **arquivo TXT único**: baixe um texto de 18 MB, execute o script de desempacotamento e obtenha a árvore de código-fonte completa (28 crates), sem npm / pip / ffmpeg / OpenEXR.

![Licença](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Linguagem](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependências](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribuição](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Testes](https://img.shields.io/badge/tests-3929%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## O que é isto

Prism Engine é um mecanismo de mídia escrito em Rust, sem nenhuma dependência de código de terceiros: ler imagens, compor, desenhar quadros, cortar a linha do tempo e produzir o filme final — tudo dentro de um único motor determinístico. Todas as 110 dependências de tempo de execução
(mais 20 dependências de teste) são dependências `path` deste repositório — sem depender da rede, sem gerenciador de pacotes,
sem risco de cadeia de suprimentos.

Adota **distribuição de código-fonte em arquivo único**: o motor inteiro (642 arquivos, 28 crates) é consolidado em um único
texto UTF-8 `prism-engine-v2.1-source.txt`. Com esse arquivo em mãos, é possível restaurar localmente uma árvore de diretórios
idêntica à do repositório.

---

## O que o motor faz

Visão geral em uma frase: **uma imagem é o quadro como ele está em um instante; um vídeo são os quadros em sequência ao longo de um período** —
a mesma descrição de cena, congelada em um momento, é um quadro único; avançada continuamente e gravada, é um vídeo; imagem estática e imagem em movimento compartilham o mesmo conjunto de capacidades.
A seguir, dividido em cinco blocos por "o que você quer fazer": **ler → compor → desenhar → entregar o filme final → conectar à sua própria cadeia de ferramentas**.

### O que se lê (entrada)

- **Fachada de decodificação unificada**: uma entrada `decode_image(bytes, ext)` despacha imagens estáticas — PNG (não entrelaçado 8/16 bits),
  JPEG baseline (IDCT de ponto fixo, determinístico bit a bit entre plataformas), **número mágico primeiro, extensão como fallback** (mesmo com a extensão errada é possível decodificar);
  formatos fora da matriz ou que exigem cooperação externa sempre retornam erro em três partes (onde / o que / como corrigir) com sugestão de cooperação com ffmpeg,
  **nunca falhando em silêncio**.
- **GIF animado / áudio / contêiner**: GIF e APNG com decodificação nativa quadro a quadro (`prism-media`); WAV com decodificação embutida, mp3/aac/FLAC
  pelo caminho de detecção do ffmpeg (`prism-audio`); releitura de MP4-MJPEG (`decode_mp4_video`).
- **Conjunto completo de EXR**: leitura e escrita, compressão, DWA, e **comparação de bitstream entre implementações** com o OpenEXR oficial (não é escrever e ler o próprio formato).
- **RAW de câmera**: decodificação RAW baseline. **Limite**: `Compression=7` (lossless JPEG) de DNG reais não está implementado,
  e DNGs produzidos por câmeras reais não podem ser lidos — isso está registrado em "Limites honestos", sem esconder nada.
- **Ativos tridimensionais**: importação de glTF / OBJ, análise de USD (`.usda`) e pacote de parâmetros MaterialX; malhas podem entrar na cena de renderização.
- **Fontes externas**: `load_font_file` carrega diretamente TTF/OTF de terceiros, a cadeia de fallback resolve nível a nível,
  e a falta de glifos gera um diagnóstico explícito em vez de quadradinhos.

### O que se compõe (sintaxe de criação)

- **Sintaxe completa de blocos da DSL v2**: um `.prism` descreve "o que se quer, como se posiciona, como se move, com que aparência,
  em que se exporta"; o produto da compilação é um grafo de nós + canais de propriedades + linha do tempo.
- **Cinco combinadores**: `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` — combinadores de alto nível que se expandem em grafos de nós, reduzindo o "filme de 90 segundos" de código de cena escrito à mão para montagem.
- **Transições e morph**: Fade / Wipe / Dissolve / Push totalmente parametrizados (duração, easing, direção),
  Morph com deformação de polilinha, e sequências de cena amostráveis em qualquer instante.
- **Composição de fórmulas**: `math_ast` + `math_layout` compõem sobrescrito/subscrito / frações / radicais / operadores grandes /
  matrizes; o bloco `formula` da DSL compõe diretamente, na mesma linha de base do texto corrido.

### O que se desenha (renderização e simulação)

- **Renderização offline com qualidade fotográfica**: path tracing + BSDF fisicamente corretos (difuso / especular / vidro / metal),
  vidro e metal pelo inverso estocástico (série de Neumann por Monte Carlo), com suporte a meios volumétricos e subsurface scattering.
- **Quatro famílias de primitivas curvas**: cilindro / cone / toro / superfície de revolução, todas com interseção analítica (equação quártica via
  solução fechada de Ferrari, raízes exatas); qualquer malha triangular + SAH-BVH, com comparação de paridade por dois caminhos.
- **GPU compute**: camada semântica de compute independente de dispositivo + dispatch Vulkan + SPIR-V mínimo escrito à mão;
  funciona por completo em plataformas sem GPU sob o ICD de software Mesa lavapipe (verificável em headless).
- **Física e partículas**: solucionador unificado XPBD (corpos rígidos / corpos moles / tecidos / juntas / fluidos / fratura),
  fumaça FLIP, sistema de partículas (AudioReactive ligado ao envelope de áudio).
- **Arte generativa e pós-processamento**: campos de fluxo / sistemas L / autômatos celulares / reação-difusão / fractais / Voronoi;
  blur / bloom / halation / grão de filme / quatro estilos NPR (toon / watercolor / halftone / oil) / denoising três em um.
- **Gradação de cor e vetores**: gradação secundária (seleção HSL + curvas RGB), presets de estilo cinematic / film / anime e
  três níveis de qualidade — rascunho / padrão / cinema (qualidade é parâmetro, não misticismo); rasterização de caminhos vetoriais, draw-on e morph de caminhos.
- **Processamento de áudio**: normalização de loudness (loudnorm), detecção de batida, análise de espectro, extração de envelope —
  pode ser ligado à imagem (partículas AudioReactive seguem o envelope de áudio).
- **O backend de referência em CPU é cidadão de primeira classe**: máquinas sem GPU também geram imagens corretas, e ele é a referência de verificação para a GPU.

### O que se entrega (resultado final)

- **`render_movie` direto**: o produto da compilação da DSL v2 percorre em uma única chamada "avaliação quadro a quadro → renderização em tile →
  entrega", e um único comando `.prism` gera o filme final (`prism-app render demo.prism`); nós não cobertos reportam erro fielmente, sem aproximação silenciosa.
- **Ciclo fechado de pré-visualização**: `prism-app preview` inicia o serviço de pré-visualização; no editor, modo Draft (padrões geométricos mantêm a taxa de quadros) /
  modo Real (renderização real com baixo spp, atualização assíncrona), pré-visualização em dois níveis.
- **Codificador próprio + controle de taxa**: codificador temporal de domínio JPEG com quadros I/P em contêiner privado `PRMV` (duas gerações de
  schema, v2/v3), controle de taxa water-filling + estimativa de movimento subpixel qpel (PSNR medido +2,577 dB,
  tamanho −21,6%).
- **Cadeia de cooperação com ffmpeg**: com ffmpeg → conversão para H.264/HEVC via pipe stdin rawvideo;
  sem ffmpeg → degradação automática para MJPEG, **o motor sempre consegue gerar o filme**; nenhum binário é empacotado.
- **Entrega de sequência EXR**: a sequência de quadros é escrita diretamente em EXR e entra no pipeline de pós-produção.
- **Farm distribuída**: distribuição por quadro + workers em N processos + mesclagem determinística — MD5 paralelo == MD5 serial,
  com suporte a retomada de cálculo.

### Como integrar (ecossistema)

- **Superfície de ferramentas MCP**: três ferramentas stdio JSON-RPC — `scene_check` (compilação + renderização de teste + estimativa de quadros),
  `render_submit` (job id determinístico + resumo MD5 por quadro), `diagnose` (localização de erro em três partes).
- **SDK de nós**: negociação de versão semântica (incluindo o caso especial 0.x), com pacote de nós de exemplo junto ao pacote
  (`examples/sdk-nodes/cinematic-basic/`, compilável / carregável / negociável).
- **Esqueleto de detecção de codificação por hardware**: `prism-ffi-hwenc` em três plataformas (VideoToolbox / NVENC / VAAPI),
  cadeia de detecção + matriz de feature gate; plataforma incompatível reporta explicitamente não suportado.

> Os limites detalhados de capacidade (o que **não** é possível fazer) estão na seção "Limites honestos" do `开发者指南.md` incluído no pacote —
> lá está registrado o estado real atual (primitivas curvas só em CPU, a razão de aceleração por GPU precisa ser medida no hardware alvo, codificação por hardware
> integrada conforme a plataforma alvo…); este documento oferece apenas a visão geral, sem exageros.

---

---

## Capacidades internas (além da lista de recursos)

Algumas capacidades não cabem na lista "ler / compor / desenhar / entregar / integrar" — são os ativos profundos do motor. Este capítulo mantém a lista principal enxuta.

### Áudio: um núcleo de DAW embutido

`prism-audio` não é um módulo periférico "que sabe ler WAV", mas uma cadeia completa da composição à mixagem; cada função é uma função pura de seus parâmetros (mesma entrada ⇒ bits idênticos entre plataformas):

- **Síntese**: família de osciladores PolyBLEP sem aliasing + envelopes ADSR em forma fechada + 4 presets de clima;
- **Teoria musical**: biblioteca de escalas (maior/menor, pentatônica, modos), 12-TET altura↔frequência, máquina de estados de gramática funcional para progressões + condução de vozes;
- **Sequenciamento**: arpejador (padrão/velocidade/oitava), caixa de ritmos por passos, partitura em grade de compassos/pulsos;
- **Material**: parser SMF escrito à mão (importação MIDI), sampler SFZ, síntese granular;
- **Design de som**: modelagem física, panning espacial, cadeia de efeitos;
- **Mixagem**: mixagem linear multipista + fades + limitação de picos;
- **Análise e vínculo**: detecção de batida/onset, espectro FFT, extração de envelopes, bake AudioReactive (o quadro N lê uma janela fixa; função pura por quadro, equivalente a seek), normalização de loudness.

> Limite: AAC/mp3 não são desenvolvidos aqui (histórico de patentes + esforço); áudio geral passa por ffmpeg externo.

### Biblioteca de kernels definíveis (registro de soluções especiais)

Cada tipo de nó é uma **solução especial** de uma tripla "kernel + reticulado + dualidade". Terceiros (ou você) podem registrar novas soluções com `registry!`, sem tocar no núcleo:

- Coleta estática (a macro expande no crate que define; sem `inventory`/proc-macro, zero dependências);
- **Validação de três perguntas** no registro (forma do kernel / reticulado / dualidade): respostas incompletas são rejeitadas com erro em três partes;
- Soluções oficiais e personalizadas compartilham o mesmo registro; sem diferença em tempo de execução;
- Matematicamente é a equação de operador `f = g + Kf` com seis estratégias de inversão — exata / estocástica / local / regularizada / iterativa / espectral. Adicionar uma capacidade é declarar uma tripla, não remendar o laço de renderização.

### Infraestrutura de determinismo e reprodutibilidade

- **Fechamento de assinatura**: grafos de nós / chaves de cache / chaves de snapshot / assinaturas de quadro reduzem-se a um "hash de configuração quadridimensional"; mudar um parâmetro muda a chave, nada é reutilizado por engano; nós iterativos usam `(ε, n)`, nós de fluxo usam o número do quadro;
- **Pontos fixos de bytes**: USD exportar→importar→exportar é idêntico byte a byte; WAV 16/24/32 bits faz ida e volta exata; EXR é comparado bit a bit com a implementação oficial;
- **Empacotamento determinístico em arquivo único**: o mesmo código gera o mesmo SHA-256 duas vezes; o round-trip de desempacotamento compara cada arquivo byte a byte.

### Backends de GPU (declarados com honestidade)

- **Vulkan**: os pipelines de compute e de cena com ray tracing rodam de verdade (verificável sem GPU no ICD de software lavapipe); a saída de `render_scene` é **idêntica byte a byte à referência de CPU** (0 pixels diferentes); `is_available()` exige autoteste de compute E render;
- **Metal**: port isomórfico + compilação condicional prontos, **não verificado em hardware, não aceso** (`is_available()` sempre false);
- **WebGPU**: apenas conclusão de viabilidade (4–6 pessoas-dia), sem caminho de renderização.

### Portões anti-verde-falso

Quatro scripts locais (zero dependências, apenas `node`): defesa contra escrita fantasma (com autoprova reversa), varredura de higiene open source, autoverificação de conformidade (9/9, autoprova reversa 8/8 vermelho) e verificação de round-trip do empacotamento. Um portão só dá "passa" ou "vermelho"; "parece bom" não vale. **Verde falso é mais perigoso que falha.**

---

## Por que distribuição em arquivo único

- **Auditável**: um arquivo de texto, legível a olho nu, sujeito a `diff` e à verificação por hash; publicado, congelado.
- **Reprodutível**: o empacotamento é determinístico — o mesmo input empacotado duas vezes gera SHA-256 idêntico.
- **Portátil**: sem clonar um repositório enorme, um arquivo pode ser enviado por e-mail ou chat, e é possível restaurar e compilar offline.
- **Nota sobre o trade-off**: a distribuição em arquivo único abre mão da conveniência de "navegar o código diretamente no GitHub" em troca da determinabilidade de "levar tudo em um arquivo, verificável byte a byte". Após desempacotar obtém-se um Cargo workspace comum, indistinguível de um projeto Rust normal.

---

## Início rápido

Pré-requisitos: instalar [Rust](https://www.rust-lang.org/) (com `cargo`) e [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Baixe o prism-engine-v2.1-source.txt do repositório e salve como arquivo local
# 2. Abra-o em um editor de texto e salve o script Node.js entre "UNPACK SCRIPT BEGIN" e "UNPACK SCRIPT END"
#    como unpack.cjs no mesmo diretório
# 3. Execute o desempacotamento (cerca de alguns segundos, restaura 642 arquivos)
node unpack.cjs
# 4. No diretório restaurado (a raiz do workspace), execute a suíte completa de testes
cargo test --workspace --release
# Esperado: 3929 passed / 0 failed
```

Após desempacotar obtém-se um Cargo workspace padrão (28 crates em `src/`); os comandos `cargo` seguintes são os mesmos de um projeto Rust comum.

> Cadeia de verificação wasm no celular: `rustup target add wasm32-unknown-unknown`, depois `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Estrutura do repositório (após desempacotar)

| Caminho | Descrição |
|------|------|
| `src/` | código-fonte dos **28 crates** (29 no repositório, originalmente `crates/`, reorganizado como `src/` na distribuição) |
| `Compliance/` | declarações de conformidade: verificação de zero dependências / declarações de terceiros / registro de ativos (originalmente `docs/合规/` no repositório) |
| `开发者指南.md` | documentação de desenvolvedor autocontida: primeiros passos / arquitetura / núcleo matemático / gates / extensão (originalmente `docs/开发者指南.md` no repositório) |
| `tools/` | scripts de gate (defesa contra escrita fantasma, varredura de higiene, autoverificação de conformidade) |
| `.github/workflows/` | configuração de CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | itens necessários à build (os workspace members já apontam para `src/*`) |
| `examples/` | amostras para testes do `prism-dsl` e pacotes de nós de exemplo (lidos via `include_str!`, a compilação falha sem eles) |

---

## Visão geral dos módulos

Os **28 crates** no pacote de distribuição, agrupados por responsabilidade:

| Categoria | crate |
|------|-------|
| Núcleo | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| I/O de mídia | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Renderização | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animação/geração | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texto/UI | prism-text · prism-rt · prism-app |
| Avançado | prism-ai · prism-dsl · prism-infra · prism-author |
| Bypass (feature gate) | prism-ffi-hwenc |

---

## Build e testes

```bash
# Smoke test do repositório inteiro (release, recomendado)
cargo test --workspace --release
# Alias equivalente
cargo xtest
# Verificação rápida de um único crate
cargo test -p prism-core --release
```

- Suíte completa: **3929 passed / 0 failed / 0 ignored** (executado em release no repositório: 179 suítes, 171,5 s com todos os núcleos; o pacote é idêntico byte a byte ao repositório — apenas o crate de demonstração é excluído)
- CI: **6 jobs** — defesa contra escrita fantasma (ghost-guard) / compilação + suíte completa (native) / build wasm32 (wasm) /
  verificação estática de WGSL (wasm-verify) / comparação de bitstream entre implementações com o OpenEXR oficial (interop) / verificação cruzada no host node (lint).
- 3 scripts locais de gate (totalmente zero dependências, apenas `node`, abaixo) + 1 script de empacotamento;
  entre eles, a autoverificação de conformidade passa **9/9** e a prova reversa acusa falha em **8/8**.

Executar os gates localmente (requer Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Defesa contra escrita fantasma: G1 definição duplicada/G2 superconjunto estrito/G3 doc sem indentação/G4 compilação
node tools/hygiene_scan.mjs               # Higiene open source: lista item a item termos internos remanescentes (arquivo:linha), exit 1 em caso de ocorrência
node tools/compliance_check.mjs           # Autoverificação de conformidade: deve passar 9/9
node tools/compliance_check.mjs --self-test   # Prova reversa: deve acusar falha, provando que o critério tem poder discriminatório
```

> **A varredura de higiene open source ainda tem ocorrências remanescentes**: restam no repositório termos internos antigos, dos quais
> **a camada de documentação distribuída no pacote já está zerada** (README / guia do desenvolvedor / Compliance), e os demais estão em **comentários de código-fonte dentro do pacote de distribuição**
> e em **arquivos do repositório que não são distribuídos no pacote**, já classificados conforme o fato e listados como itens de limpeza independentes. A existência do script serve exatamente para tornar os
> resíduos **visíveis e removíveis item a item**, e não para fingir que está limpo — não interprete "contagem de ocorrências diferente de 0" como falha do gate.

---

## Conformidade

O motor promete **zero dependências de código de terceiros**, verificado por autoverificação:

- **C1 Zero dependências de código de terceiros**: todas as 130 dependências (110 de tempo de execução + 20 de teste) são dependências `path` deste repositório; sem `build.rs`, sem diretivas de link nativo, `Cargo.lock` fixa 29 pacotes deste repositório.
- **C2 Registro de ativos completo**: os campos da tabela principal dos 5 ativos estão completos e o estado de internalização está em vocabulário controlado; o texto integral da licença OFL 1.1 da fonte acompanha a fonte em `src/prism-text/font_data/LICENSE`.
- **C3 Declaração de atribuição de terceiros**: `Compliance/THIRD-PARTY-NOTICES.md` está presente e distingue explicitamente "ativo de dados ≠ dependência de código".

> Nota: este repositório distribui **código-fonte**. O ativo de fonte (subconjunto Prism Sans SC) é licenciado sob SIL OFL 1.1, e **o texto integral da licença acompanha a fonte** em `src/prism-text/font_data/LICENSE`; a origem e o registro de licença do ativo estão em `Compliance/assets/资产登记.md`.


---

## Licença

Este projeto é publicado sob a **Apache License 2.0**. Veja o arquivo [`LICENSE`](../LICENSE).

> Aviso de conformidade: a Apache-2.0 exige a preservação dos avisos de copyright e de licença; ao redistribuir com base neste motor, preserve as declarações de atribuição e o registro de ativos em `Compliance/`.

**Observação: caso outras versões tenham erros de tradução, prevaleça esta versão em chinês simplificado.**
