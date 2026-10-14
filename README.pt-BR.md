[English](README.en.md) | [简体中文](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | **Português (BR)** | [Русский](README.ru.md)

# Prism Engine

> Motor de mídia em Rust, sem dependências e determinístico — distribuído como **TXT de arquivo único**: baixe um texto de 14 MB, execute o script de desempacotamento e obtenha a árvore de código-fonte completa com 26 crates, sem necessidade de npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## O que é

Prism Engine é um motor de mídia escrito em Rust com zero dependências de código de terceiros, cobrindo módulos de imagem, áudio, geometria, física, renderização, DSL e outros. Todas as 103 dependências são dependências `path` deste próprio repositório — sem depender de rede, sem precisar de gerenciador de pacotes, sem risco de cadeia de suprimentos.

Adota a **distribuição de código-fonte em arquivo único**: todo o motor (481 arquivos, 26 crates) é fundido em um único texto UTF-8, `prism-engine-v1.0-source.txt`. Com esse arquivo em mãos, é possível restaurar localmente uma árvore de diretórios idêntica à do repositório.

## Por que distribuição em arquivo único

- **Auditável**: um único arquivo de texto, legível a olho nu, comparável com `diff` e verificável por hash; fica congelado no momento da publicação.
- **Reproduzível**: o empacotamento é determinístico — a mesma entrada empacotada duas vezes gera exatamente o mesmo SHA-256.
- **Portátil**: sem necessidade de clonar um repositório enorme; um único arquivo pode ser enviado por e-mail ou chat, e o build pode ser restaurado até offline.
- **Trade-off**: a distribuição em arquivo único abre mão da conveniência de "navegar pelo código diretamente no GitHub" em troca da garantia determinística de "levar tudo em um único arquivo, verificável byte a byte". Após o desempacotamento, obtém-se um Cargo workspace comum, igual ao de qualquer projeto Rust convencional.
- **O mais importante**: eu só preciso enviar um único arquivo, e pronto (doge

---

## Início rápido

Pré-requisitos: instale o [Rust](https://www.rust-lang.org/) (com `cargo`) e o [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Baixe o prism-engine-v1.0-source.txt do repositório e salve-o como um arquivo local
# 2. Abra-o em um editor de texto e salve o script Node.js entre "UNPACK SCRIPT BEGIN" e "UNPACK SCRIPT END"
#    como unpack.cjs no mesmo diretório
# 3. Execute o desempacotamento (leva alguns segundos, restaura 481 arquivos)
node unpack.cjs
# 4. Entre na árvore de código-fonte restaurada e rode a suíte completa de testes
cd src/..
cargo test --workspace --release
# Esperado: 3271 passed / 0 failed
```

Após o desempacotamento, você obtém um Cargo workspace padrão (26 crates sob `src/`); os comandos `cargo` subsequentes são os mesmos de qualquer projeto Rust convencional.

> Cadeia de verificação wasm no celular: `rustup target add wasm32-unknown-unknown` e, em seguida, `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Estrutura do repositório (após desempacotar)

| Caminho | Descrição |
|------|------|
| `src/` | Código-fonte dos 26 crates (originalmente `crates/` no repositório, reorganizado como `src/` na distribuição) |
| `Compliance/` | Declarações de conformidade: verificação de zero dependências / declaração de terceiros / registro de ativos (originalmente `docs/合规/` no repositório) |
| `开发者指南.md` | Documentação autocontida para desenvolvedores: primeiros passos / arquitetura / núcleo matemático / gates / extensão (originalmente `docs/开发者指南.md` no repositório) |
| `tools/` | Scripts de gate (defesa contra gravação fantasma, varredura de higiene, autoteste de conformidade) |
| `.github/workflows/` | Configuração de CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Arquivos necessários para o build (os workspace members já apontam para `src/*`) |
| `examples/` | Exemplos usados nos testes do `prism-dsl` (lidos via `include_str!`; sem eles não compila) |

> Os documentos de design autoritativos (`docs/设计/`) **não acompanham a distribuição** e permanecem apenas no repositório completo de quem publica.

---

## Visão geral dos módulos

Os 26 crates agrupados por responsabilidade:

| Categoria | crate |
|------|-------|
| Núcleo | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S de mídia | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Renderização | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animação/geração | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texto/UI | prism-text · prism-rt · prism-app |
| Avançado | prism-ai · prism-dsl · prism-infra |

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

- Suíte completa: **3271 passed / 0 failed** (validada em execução real sobre uma árvore recém-desempacotada).
- Gates: 14 no total, incluindo a defesa contra gravação fantasma (G1~G4), a varredura de higiene open source (0 vestígios internos), o autoteste de conformidade (9/9), a contraprova de conformidade (8/8, violações sempre em vermelho) e a cadeia de build wasm32.

Para rodar os gates localmente (requer Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Defesa contra gravação fantasma: G1 definições duplicadas/G2 superconjunto estrito/G3 documentação no topo/G4 compilação
node tools/hygiene_scan.mjs               # Higiene open source: deve reportar “limpo (0 vestígios internos)”
node tools/compliance_check.mjs           # Autoteste de conformidade: deve passar 9/9
node tools/compliance_check.mjs --self-test   # Contraprova: deve ficar vermelho, provando que os critérios têm poder de discriminação
```

---

## Conformidade

O motor se compromete com **zero dependências de código de terceiros**, verificado por autoteste:

- **C1 Zero dependências de código de terceiros**: todas as 103 dependências são dependências `path` deste próprio repositório; sem `build.rs`, sem instruções de linkagem nativa, e o `Cargo.lock` fixa os 26 pacotes deste repositório.
- **C2 Registro de ativos completo**: as 5 entradas da tabela principal de ativos têm todos os campos preenchidos e o estado de desenvolvimento próprio dentro de um vocabulário controlado; o texto integral da licença de fontes OFL 1.1 acompanha as fontes em `src/prism-text/font_data/LICENSE`.
- **C3 Declaração de atribuição de terceiros**: `Compliance/THIRD-PARTY-NOTICES.md` está presente e distingue claramente "ativos de dados ≠ dependências de código".

> Observação: o que este repositório distribui é o **código-fonte**. Os ativos de fonte (subconjunto do Prism Sans SC) são licenciados sob a SIL OFL 1.1, e **o texto integral da licença acompanha as fontes em** `src/prism-text/font_data/LICENSE`; para a origem dos ativos e o registro de licenças, veja `Compliance/assets/资产登记.md`.

---

## Licença

Este projeto é publicado sob a **Apache License 2.0**. Consulte o arquivo [`LICENSE`](LICENSE).

> Aviso de conformidade: a Apache-2.0 exige a preservação dos avisos de direitos autorais e de licença; ao redistribuir com base neste motor, mantenha as declarações de atribuição e o registro de ativos sob `Compliance/`.
