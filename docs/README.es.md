[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> Motor de medios en Rust, determinista y sin dependencias — distribuido como **un único archivo TXT**: descarga un texto de 17 MB, ejecuta el script de desempaquetado y obtienes el árbol de código fuente completo (28 crates), sin necesidad de npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

(Las insignias son insignias de hechos estáticos; el criterio de las cifras se detalla más abajo; `crates-28` = número de crates **dentro del paquete de distribución**,
el repositorio contiene además 1 crate de demostración que no entra en el paquete, por lo que el total del repositorio es 29. Tras integrarse en CI pueden sustituirse por insignias dinámicas.)

---

## Qué es esto

Prism Engine es un motor de medios escrito en Rust sin dependencias de código de terceros: leer imágenes, componer tipografía,
dibujar cuadros, montar líneas de tiempo y producir la pieza final, todo dentro de un único motor determinista. Las 110 dependencias en tiempo de ejecución
(más otras 20 dependencias de pruebas) son todas dependencias `path` de este repositorio: no dependen de la red, no necesitan gestor de paquetes
y no presentan riesgo de cadena de suministro.

Adopta una **distribución de código fuente en un solo archivo**: todo el motor (622 archivos, 28 crates) se fusiona en un texto UTF-8,
`prism-engine-v2.0-source.txt`. Con solo tener ese archivo puedes reconstruir localmente un árbol de directorios idéntico al del repositorio.

---

## Qué puede hacer el motor

Resumen en una frase: **una imagen es el aspecto de una escena en un instante; un vídeo es la sucesión de ese aspecto a lo largo de un intervalo** —
con una misma descripción de escena, fijar un instante da un fotograma único, y avanzar de forma continua grabando da un vídeo; la imagen estática y la imagen en movimiento comparten un mismo conjunto de capacidades.
A continuación, cinco bloques según «lo que quieres hacer»: **leer → componer → dibujar → entregar → integrar en tu propia cadena de herramientas**.

### Qué se lee (entrada)

- **Fachada unificada de decodificación**: una única entrada `decode_image(bytes, ext)` distribuye las imágenes estáticas — PNG (no entrelazado 8/16 bit),
  JPEG baseline (IDCT de coma fija, determinista bit a bit entre plataformas), **prioridad al número mágico, extensión como respaldo** (se decodifica aunque la extensión esté mal);
  los formatos fuera de la matriz o que requieren colaboración externa informan siempre con un error en tres partes (dónde / qué ocurre / cómo corregirlo) y ofrecen una recomendación de colaboración con ffmpeg;
  **nunca fallan en silencio**.
- **Animación / audio / contenedores**: GIF y APNG con decodificación nativa fotograma a fotograma (`prism-media`); WAV con decodificación integrada, mp3/aac/FLAC
  por la ruta de detección de ffmpeg (`prism-audio`); relectura de MP4-MJPEG (`decode_mp4_video`).
- **La familia completa EXR**: lectura y escritura, compresión, DWA, y **contraste de flujo de bits entre implementaciones** con el OpenEXR oficial (no se escribe y se lee uno mismo).
- **RAW de cámara**: decodificación RAW de referencia. **Límite**: `Compression=7` (JPEG lossless) de los DNG reales no está implementado,
  así que un DNG producido por una cámara real no se puede leer — está escrito en «límites honestos», no se oculta.
- **Activos 3D**: importación de glTF / OBJ, análisis de USD (`.usda`) y paquete de parámetros MaterialX; las mallas pueden entrar en la escena de render.
- **Tipografías externas**: `load_font_file` carga directamente TTF/OTF de terceros; la cadena de respaldo resuelve por niveles y,
  si falta un glifo, da un diagnóstico claro en lugar de un cuadro vacío.

### Qué se compone (sintaxis de autoría)

- **Gramática completa de bloques DSL v2**: un `.prism` describe «qué se quiere, cómo se coloca, cómo se mueve, qué aspecto tiene
  y en qué se exporta»; el producto compilado es un grafo de nodos + canales de propiedades + línea de tiempo.
- **Cinco combinadores**: `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between`, combinadores de alto nivel que se expanden a grafos de nodos; «una pieza de 90 segundos» pasa de escribir código de escena a mano a solo ensamblar.
- **Transiciones y morph**: Fade / Wipe / Dissolve / Push totalmente parametrizados (duración, easing, dirección),
  Morph de deformación poligonal, y cualquier instante de una secuencia de escenas se puede muestrear.
- **Composición tipográfica de fórmulas**: `math_ast` + `math_layout` componen subíndices/superíndices / fracciones / raíces / operadores grandes /
  matrices; el bloque `formula` del DSL compone directamente, con la misma línea base que el texto.

### Qué se dibuja (render y simulación)

- **Render offline de nivel fotográfico**: path tracing + BSDF físicamente correctos (difuso / especular / vidrio / metal);
  el vidrio y el metal usan inversión estocástica (serie de Neumann por Monte Carlo), con soporte de medios volumétricos y subsurface scattering.
- **Cuatro familias de primitivas curvas**: cilindro / cono / toro / superficie de revolución, todas con intersección analítica (la ecuación cuártica se resuelve con la
  solución cerrada de Ferrari, raíces exactas); cualquier malla triangular + SAH-BVH, con contraste de paridad por doble ruta.
- **GPU compute**: capa semántica de compute independiente del dispositivo + dispatch Vulkan + SPIR-V mínimo escrito a mano;
  bajo el ICD software Mesa lavapipe funciona completo en plataformas sin GPU (verificable en headless).
- **Física y partículas**: solver unificado XPBD (cuerpos rígidos / blandos / telas / articulaciones / fluidos / fractura),
  humo FLIP, sistema de partículas (AudioReactive ligado a la envolvente de audio).
- **Arte generativo y postproceso**: campos de flujo / sistemas L / autómatas celulares / reacción-difusión / fractales / Voronoi;
  blur / bloom / halation / grano de película / cuatro estilos NPR (toon / watercolor / halftone / oil) / reducción de ruido tres en uno.
- **Color grading y vectores**: grading secundario (selección HSL + curvas RGB), presets de estilo cinematic / film / anime y
  tres niveles de calidad — borrador / estándar / cine (la calidad es un parámetro, no magia); rasterización de trazos vectoriales, draw-on y morph de trazos.
- **Procesamiento de audio**: normalización de sonoridad (loudnorm), detección de tempo, análisis de espectro, extracción de envolvente —
  todo ello vinculable a la imagen (las partículas AudioReactive siguen la envolvente de audio).
- **El backend de referencia por CPU es ciudadano de primera**: una máquina sin GPU produce igualmente imágenes correctas, y a la vez sirve de referencia para validar la GPU.

### Qué se entrega

- **`render_movie` directo**: el producto compilado de DSL v2 completa en una sola llamada «evaluación fotograma a fotograma → render por tile →
  entrega»; una sola orden sobre el `.prism` genera la pieza (`prism-app render demo.prism`); los nodos no cubiertos informan del error tal cual, sin aproximaciones silenciosas.
- **Bucle cerrado de previsualización**: `prism-app preview` levanta el servicio de previsualización; el editor ofrece previsualización en dos niveles — borrador (formas geométricas para mantener la tasa de fotogramas) /
  real (render real con spp bajo, refresco asíncrono).
- **Codificador propio + control de tasa**: codificador temporal en dominio JPEG de fotogramas I/P sobre el contenedor privado `PRMV` (dos generaciones de esquema,
  v2/v3), control de tasa water-filling + estimación de movimiento subpíxel qpel (PSNR medido +2.577 dB,
  volumen −21.6%).
- **Cadena de colaboración con ffmpeg**: con ffmpeg → conversión a H.264/HEVC mediante tubería rawvideo por stdin;
  sin ffmpeg → degradación automática a MJPEG, **el motor siempre puede entregar la pieza**; no se empaqueta ningún binario.
- **Entrega de secuencias EXR**: las secuencias de fotogramas se escriben directamente en EXR y entran en el pipeline de postproducción.
- **Granja distribuida**: distribución por fotogramas + workers de N procesos + fusión determinista — el MD5 en paralelo == el MD5 en serie,
  con soporte de reanudación.

### Cómo se integra (ecosistema)

- **Superficie de herramientas MCP**: tres herramientas stdio JSON-RPC — `scene_check` (compilación + render de prueba + estimación de fotogramas),
  `render_submit` (job id determinista + resumen MD5 por fotograma), `diagnose` (localización del error en tres partes).
- **SDK de nodos**: negociación de versiones semánticas (incluida la excepción 0.x), con un paquete de nodos de ejemplo adjunto
  (`examples/sdk-nodes/cinematic-basic/`, compilable / cargable / negociable).
- **Esqueleto de detección de codificación por hardware**: `prism-ffi-hwenc` con cadena de detección en tres plataformas (VideoToolbox / NVENC / VAAPI)
  y matriz de feature gates; si la plataforma no coincide, informa claramente de que no está soportada.

> Para los límites detallados de capacidades (qué **no se puede** hacer), véase la sección «límites honestos» de `开发者指南.md` tras descomprimir el paquete —
> allí consta el estado real actual (las primitivas curvas solo van por CPU, la mejora de la GPU debe medirse en el hardware objetivo, la codificación por hardware se integra según
> la plataforma objetivo…); este documento solo ofrece el panorama general, sin exageraciones.

---

## Por qué una distribución de un solo archivo

- **Auditable**: un archivo de texto, legible a simple vista, `diff`-able, verificable por hash; lo publicado queda congelado.
- **Reproducible**: el empaquetado es determinista — el mismo insumo empaquetado dos veces da un SHA-256 idéntico.
- **Portátil**: no hace falta clonar un repositorio enorme; un archivo se puede enviar por correo o por chat, y también se puede restaurar y compilar sin conexión.
- **Sobre el compromiso**: la distribución de un solo archivo cede la comodidad de «navegar el código directamente en GitHub» a cambio de la determinación de «llevarse todo en un archivo y poder verificarlo byte a byte». Lo que se obtiene tras descomprimir es un Cargo workspace normal, indistinguible de un proyecto Rust convencional.

---

## Inicio rápido

Requisitos previos: instalar [Rust](https://www.rust-lang.org/) (con `cargo`) y [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Descarga prism-engine-v2.0-source.txt del repositorio y guárdalo como archivo local
# 2. Ábrelo con un editor de texto y guarda el script Node.js comprendido entre "UNPACK SCRIPT BEGIN" y "UNPACK SCRIPT END"
#    como unpack.cjs en el mismo directorio
# 3. Ejecuta el desempaquetado (unos segundos, restaura 622 archivos)
node unpack.cjs
# 4. En el directorio restaurado (es decir, la raíz del workspace) ejecuta la batería completa de pruebas
cargo test --workspace --release
# Esperado: 3714 passed / 0 failed
```

Tras descomprimir se obtiene un Cargo workspace estándar (28 crates bajo `src/`); los comandos `cargo` posteriores son los mismos que en un proyecto Rust convencional.

> Cadena de verificación wasm en móvil: `rustup target add wasm32-unknown-unknown`, y luego `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Estructura del repositorio (tras descomprimir)

| Ruta | Descripción |
|------|-------------|
| `src/` | Código fuente de los **28 crates** (29 en el repositorio; el nombre original era `crates/`, reorganizado como `src/` al distribuir; el crate de demostración `prism-anim-demo` no entra en el paquete) |
| `Compliance/` | Declaraciones de cumplimiento: verificación de cero dependencias / avisos de terceros / registro de activos (nombre original en el repositorio: `docs/合规/`) |
| `开发者指南.md` | Documentación de desarrollador autocontenida: primeros pasos / arquitectura / núcleo matemático / controles / extensión (nombre original en el repositorio: `docs/开发者指南.md`) |
| `tools/` | Scripts de control (defensa contra escrituras fantasma, escaneo de higiene, autoverificación de cumplimiento) |
| `.github/workflows/` | Configuración de CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Elementos necesarios para compilar (los workspace members ya apuntan a `src/*`) |
| `examples/` | Ejemplos para las pruebas de `prism-dsl` y paquete de nodos de ejemplo (se leen con `include_str!`; si faltan, no compila) |

> Los documentos de diseño autoritativos (`docs/设计/`) **no se incluyen en la distribución**; permanecen solo en el repositorio completo del publicador.

---

## Resumen de módulos

Los **28 crates** del paquete de distribución, agrupados por responsabilidad (el repositorio tiene además el crate de demostración `prism-anim-demo`, que solo existe en el repositorio
como ejemplo reproducible y no entra en el paquete):

| Categoría | crate |
|-----------|-------|
| Núcleo | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S de medios | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Render | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animación/generación | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texto/UI | prism-text · prism-rt · prism-app |
| Avanzado | prism-ai · prism-dsl · prism-infra · prism-author |
| Auxiliar (feature gate) | prism-ffi-hwenc |

---

## Compilación y pruebas

```bash
# Prueba de humo de todo el repositorio (release, recomendado)
cargo test --workspace --release
# Alias equivalente
cargo xtest
# Verificación rápida de un solo crate
cargo test -p prism-core --release
```

- Pruebas completas: **3714 passed / 0 failed / 0 ignored** (esta versión se ha ejecutado realmente sobre el **árbol descomprimido**: 166 bloques de pruebas;
  en el repositorio hay 167 bloques, la diferencia de 1 bloque es el bloque de pruebas vacío del crate de demostración, con el mismo número de casos).
- CI: **6 jobs** — defensa contra escrituras fantasma (ghost-guard) / compilación + pruebas completas (native) / compilación wasm32 (wasm) /
  comprobación estática de WGSL (wasm-verify) / contraste de flujo de bits entre implementaciones con el OpenEXR oficial (interop) / validación cruzada del host node (lint).
- 3 scripts locales de control (totalmente sin dependencias, solo usan `node`, más abajo) + 1 script de empaquetado;
  de ellos, la autoverificación de cumplimiento pasa **9/9** y la autocomprobación inversa da **8/8 en rojo**.

Ejecución local de los controles (requiere Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Defensa contra escrituras fantasma: G1 definición duplicada / G2 superconjunto estricto / G3 documento al ras / G4 compilación
node tools/hygiene_scan.mjs               # Higiene de código abierto: enumera línea por línea el lenguaje interno residual (archivo:línea); si hay coincidencia, exit 1
node tools/compliance_check.mjs           # Autoverificación de cumplimiento: debe pasar 9/9
node tools/compliance_check.mjs --self-test   # Autocomprobación inversa: debe fallar, lo que demuestra que el criterio discrimina
```

> **El escaneo de higiene de código abierto aún tiene coincidencias heredadas**: en el repositorio quedan expresiones internas de etapas tempranas, de las cuales
> **la capa de documentación distribuida en el paquete ya está a cero** (README / guía del desarrollador / Compliance), y el resto está en **comentarios del código fuente dentro del paquete**
> y en **archivos del repositorio que no se distribuyen**, ya clasificados según los hechos y registrados como partidas de limpieza independientes. La existencia del script sirve precisamente para que
> lo residual sea **visible y limpiable una por una**, y no para fingir que está limpio — no interprete «número de coincidencias distinto de 0» como un fallo del control.

---

## Cumplimiento

El motor garantiza **cero dependencias de código de terceros**, verificado por autocomprobación:

- **C1 Cero dependencias de código de terceros**: las 130 dependencias totales (110 en tiempo de ejecución + 20 de pruebas) son dependencias `path` de este repositorio; no hay `build.rs`, no hay directivas de enlace nativo y `Cargo.lock` fija 29 paquetes de este repositorio.
- **C2 Registro de activos completo**: la tabla maestra de los 5 activos tiene todos los campos y el estado de internalización está dentro del vocabulario controlado; el texto completo de la licencia de fuentes OFL 1.1 se entrega junto a la fuente en `src/prism-text/font_data/LICENSE`.
- **C3 Aviso de atribución de terceros**: `Compliance/THIRD-PARTY-NOTICES.md` está presente y distingue explícitamente «activo de datos ≠ dependencia de código».

> Nota: este repositorio distribuye **código fuente**. El activo de fuente (subconjunto Prism Sans SC) se licencia bajo SIL OFL 1.1 y **el texto completo de la licencia se entrega junto a la fuente** en `src/prism-text/font_data/LICENSE`; el origen del activo y su registro de licencia están en `Compliance/assets/资产登记.md`.

---

## Empaquetado determinista (publicador)

Si necesitas regenerar el TXT de distribución desde el repositorio completo (el número de versión es un parámetro; por defecto publica la versión actual v2.0):

```bash
node tools/pack_source_txt.mjs          # Por defecto → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # Forma equivalente (versión explícita)
node tools/pack_source_txt.mjs 1.0      # Vuelve al nombre de producto de la era v1.0 (ejecutado sobre el código v1.0 reproduce el paquete antiguo byte a byte)
node tools/pack_source_txt.mjs --verify # Tras empaquetar, descomprime de inmediato para la verificación de ida y vuelta (comparación byte a byte archivo por archivo contra el árbol de origen)
# Ejecútalo dos veces: si el SHA-256 de salida coincide, queda demostrado el empaquetado determinista
```

El alcance del empaquetado está estrictamente limitado a: `crates/` (→ `src/`), `docs/合规/` (→ `Compliance/`), `docs/开发者指南.md`, `tools/`, `.github/`, `.cargo/`, `examples/`, `Cargo.toml`, `Cargo.lock`. Se **excluyen** deliberadamente `docs/设计/`, `target/`, `node_modules/`, el propio `dist/` y cualquier producto intermedio.

---

## Contribución

Se aceptan Issues y Pull Requests. Antes de enviar, mantén:

- `cargo test --workspace --release` completamente en verde;
- `node tools/hygiene_scan.mjs` **sin añadir** expresiones internas (el script lista todas las coincidencias, incluidas las heredadas);
- `node tools/compliance_check.mjs` pasando 9/9.

Para una guía más detallada de arquitectura y extensión, véase `开发者指南.md` tras descomprimir el paquete.

---

## Licencia

Este proyecto se publica bajo la **Apache License 2.0**. Véase el archivo [`LICENSE`](../LICENSE).

> Aviso de cumplimiento: Apache-2.0 exige conservar los avisos de copyright y de licencia; al redistribuir a partir de este motor, conserva los avisos de atribución y el registro de activos bajo `Compliance/`.
