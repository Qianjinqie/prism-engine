[English](README.en.md) | [简体中文](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> Motor multimedia en Rust, sin dependencias y determinista — distribuido como **TXT de archivo único**: descarga un texto de 14 MB, ejecuta el script de desempaquetado y obtienes el árbol completo de código fuente de 26 crates, sin npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## Qué es esto

Prism Engine es un motor multimedia escrito en Rust con cero dependencias de código de terceros, que cubre módulos de imagen, audio, geometría, física, renderizado, DSL y más. Las 103 dependencias son todas dependencias `path` de este repositorio — sin depender de la red, sin gestores de paquetes y sin riesgos de cadena de suministro.

Adopta la **distribución de código fuente en archivo único**: todo el motor (481 archivos, 26 crates) se fusiona en un único texto UTF-8 `prism-engine-v1.0-source.txt`. Quien posea ese archivo puede reconstruir localmente un árbol de directorios idéntico al del repositorio.

## Por qué la distribución en archivo único

- **Auditable**: un solo archivo de texto, legible a simple vista, apto para `diff` y verificable por hash; queda congelado en el momento de la publicación.
- **Reproducible**: el empaquetado es determinista — dos empaquetados de la misma entrada producen un SHA-256 exactamente idéntico.
- **Portable**: no hace falta clonar un repositorio enorme; un único archivo se puede enviar por correo o chat, y la compilación puede reconstruirse incluso sin conexión.
- **Nota sobre compromisos**: la distribución en archivo único sacrifica la comodidad de "navegar el código directamente en GitHub" a cambio del determinismo de "un solo archivo que se lleva todo, y verificable byte a byte". Tras el desempaquetado se obtiene un workspace de Cargo normal, igual que cualquier proyecto Rust convencional.
- **Lo más importante**: solo necesito subir un único archivo y ya está (doge

---

## Inicio rápido

Requisitos previos: instalar [Rust](https://www.rust-lang.org/) (con `cargo` incluido) y [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Descarga prism-engine-v1.0-source.txt del repositorio y guárdalo como archivo local
# 2. Ábrelo con un editor de texto y guarda el script de Node.js situado entre
#    "UNPACK SCRIPT BEGIN" y "UNPACK SCRIPT END" como unpack.cjs en el mismo directorio
# 3. Ejecuta el desempaquetado (unos segundos; restaura 481 archivos)
node unpack.cjs
# 4. Entra en el árbol de código fuente restaurado y ejecuta la suite completa de pruebas
cd src/..
cargo test --workspace --release
# Resultado esperado: 3271 passed / 0 failed
```

Tras el desempaquetado se obtiene un workspace de Cargo estándar (26 crates bajo `src/`); los comandos `cargo` posteriores son los mismos que en cualquier proyecto Rust convencional.

> Cadena de verificación wasm en móvil: `rustup target add wasm32-unknown-unknown` y después `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Estructura del repositorio (tras el desempaquetado)

| Ruta | Descripción |
|------|------|
| `src/` | Código fuente de los 26 crates (nombre original en el repositorio: `crates/`; reorganizado como `src/` en la distribución) |
| `Compliance/` | Declaraciones de cumplimiento: verificación de cero dependencias / declaración de terceros / registro de activos (nombre original en el repositorio: `docs/合规/`) |
| `开发者指南.md` | Documentación para desarrolladores autocontenida: primeros pasos / arquitectura / núcleo matemático / gates / extensiones (nombre original en el repositorio: `docs/开发者指南.md`) |
| `tools/` | Scripts de gates (defensa contra escrituras fantasma, escaneo de higiene, autoverificación de cumplimiento) |
| `.github/workflows/` | Configuración de CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Piezas imprescindibles para la compilación (los workspace members ya apuntan a `src/*`) |
| `examples/` | Ejemplos usados en las pruebas de `prism-dsl` (se leen con `include_str!`; si faltan, la compilación falla) |

> La documentación de diseño autoritativa (`docs/设计/`) **no se incluye en la distribución**; permanece únicamente en el repositorio completo del publicador.

---

## Vista general de los módulos

Los 26 crates se agrupan por responsabilidad:

| Categoría | crate |
|------|-------|
| Núcleo | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S multimedia | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Renderizado | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animación/Generación | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texto/UI | prism-text · prism-rt · prism-app |
| Avanzado | prism-ai · prism-dsl · prism-infra |

---

## Compilación y pruebas

```bash
# Prueba de humo de todo el workspace (release, recomendado)
cargo test --workspace --release
# Alias equivalente
cargo xtest
# Verificación rápida de un solo crate
cargo test -p prism-core --release
```

- Suite completa: **3271 passed / 0 failed** (verificado con una ejecución real sobre un árbol recién desempaquetado).
- Gates: 14 en total, incluidos la defensa contra escrituras fantasma (G1~G4), el escaneo de higiene de código abierto (0 rastros internos), la autoverificación de cumplimiento (9/9), la autoprueba inversa de cumplimiento (8/8, rojo obligatorio ante cualquier infracción) y la cadena de compilación wasm32.

Ejecución local de los gates (requiere Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Defensa contra escrituras fantasma: G1 definiciones duplicadas / G2 superconjunto estricto / G3 documentación al inicio / G4 compilación
node tools/hygiene_scan.mjs               # Higiene de código abierto: debe informar «limpio (0 rastros internos)»
node tools/compliance_check.mjs           # Autoverificación de cumplimiento: debe pasar 9/9
node tools/compliance_check.mjs --self-test   # Autoprueba inversa: debe dar rojo, demostrando que los criterios tienen capacidad de discriminación
```

---

## Cumplimiento

El motor se compromete a **cero dependencias de código de terceros**, verificado mediante autoevaluación:

- **C1 Cero dependencias de código de terceros**: las 103 dependencias son todas dependencias `path` de este repositorio; sin `build.rs`, sin instrucciones de enlazado nativo, y `Cargo.lock` bloquea los 26 paquetes de este repositorio.
- **C2 Registro de activos completo**: los 5 activos tienen completos todos los campos de la tabla principal y su estado de desarrollo propio queda registrado en un vocabulario controlado; el texto íntegro de la licencia de fuentes OFL 1.1 acompaña a las fuentes en `src/prism-text/font_data/LICENSE`.
- **C3 Declaración de atribución de terceros**: `Compliance/THIRD-PARTY-NOTICES.md` está presente y distingue claramente "activos de datos ≠ dependencias de código".

> Nota: este repositorio distribuye **código fuente**. Los activos de fuentes (subconjunto de Prism Sans SC) se licencian bajo SIL OFL 1.1, y **el texto íntegro de la licencia acompaña a las fuentes en** `src/prism-text/font_data/LICENSE`; el origen de los activos y su registro de licencias están en `Compliance/assets/资产登记.md`.

---

## Licencia

Este proyecto se publica bajo la **Apache License 2.0**. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.

> Aviso de cumplimiento: Apache-2.0 exige conservar los avisos de copyright y licencia; al redistribuir basado en este motor, conserva las declaraciones de atribución y el registro de activos que están bajo `Compliance/`.
