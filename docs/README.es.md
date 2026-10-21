[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)


# Prism Engine

> Motor multimedia en Rust, determinista y sin dependencias — **distribuido como un único archivo TXT**: descarga un texto de 17 MB, ejecuta el script de descompresión y obtienes el árbol de código fuente completo (28 crates), sin necesidad de npm / pip / ffmpeg / OpenEXR.

![Licencia](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Lenguaje](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencias](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribución](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Pruebas](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## Qué es esto

Prism Engine es un motor multimedia escrito en Rust con cero dependencias de código de terceros:
leer imágenes, componer, dibujar fotogramas, montar la línea de tiempo y producir el montaje final,
todo dentro de un único motor determinista. Sus 110 dependencias de ejecución
(más 20 dependencias de prueba) son todas dependencias `path` de este mismo repositorio: no
dependen de la red, no necesitan gestor de paquetes y no implican riesgo de cadena de suministro.

Adopta la **distribución del código fuente en un solo archivo**: todo el motor (622 archivos, 28 crates)
se fusiona en un único texto UTF-8, `prism-engine-v2.0-source.txt`. Basta con disponer de ese archivo
para restaurar en local un árbol de directorios idéntico al del repositorio.

---

## Qué puede hacer el motor

Resumen en una frase: **una imagen es el aspecto de una escena en un instante, y un vídeo es la
sucesión de ese aspecto a lo largo de un periodo de tiempo** — con una misma descripción de escena,
fijar un momento da un fotograma único, y avanzar de forma continua grabando da un vídeo; la imagen
estática y la imagen en movimiento comparten un mismo conjunto de capacidades.
A continuación, cinco bloques según "lo que quieres hacer": **leer → componer → dibujar → producir el montaje → integrarlo en tu propia cadena de herramientas**.

### Qué lee (entrada)

- **Fachada de decodificación unificada**: una única entrada `decode_image(bytes, ext)` despacha las imágenes estáticas — PNG (8/16 bits no entrelazado),
  JPEG de línea base (IDCT de coma fija, determinista bit a bit entre plataformas), **prioridad del número mágico y la extensión como respaldo** (también decodifica si la extensión está mal escrita);
  los formatos fuera de la matriz o que requieren colaboración externa devuelven siempre un error en tres partes (dónde / qué ocurre / cómo corregirlo) junto con una sugerencia de colaboración con ffmpeg;
  **nunca fallan en silencio**.
- **Animación / audio / contenedores**: decodificación nativa fotograma a fotograma de GIF y APNG (`prism-media`); decodificación WAV integrada, y mp3/aac/FLAC mediante la ruta de detección de ffmpeg (`prism-audio`); relectura de MP4-MJPEG (`decode_mp4_video`).
- **Todo el ecosistema EXR**: lectura y escritura, compresión, DWA, además de una **comparación de flujos de bits entre implementaciones** con el OpenEXR oficial (no se limita a escribir y leer lo propio).
- **RAW de cámara**: decodificación RAW de línea base. **Límite**: no está implementado `Compression=7` (lossless JPEG) de los DNG reales,
  por lo que un DNG capturado por una cámara real no se puede leer; está documentado en «Límites honestos», sin ocultarlo.
- **Activos 3D**: importación de glTF / OBJ, análisis de USD (`.usda`) y paquetes de parámetros MaterialX; las mallas pueden entrar en la escena de renderizado.
- **Fuentes externas**: `load_font_file` carga directamente TTF/OTF de terceros, la cadena de respaldo se resuelve nivel a nivel y,
  cuando falta un glifo, se ofrece un diagnóstico explícito en lugar de un recuadro vacío.

### Qué compone (sintaxis de creación)

- **Sintaxis de bloques completa de DSL v2**: un archivo `.prism` describe "qué se quiere, cómo se coloca, cómo se mueve, qué aspecto tiene
  y en qué se exporta"; el producto de la compilación es un grafo de nodos + canales de propiedades + línea de tiempo.
- **Cinco combinadores**: `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between`, combinadores de alto nivel que se expanden a grafos de nodos; "un montaje de 90 segundos" pasa de escribir código de escena a mano a solo ensamblar.
- **Transiciones y morph**: Fade / Wipe / Dissolve / Push totalmente parametrizados (duración, easing, dirección),
  deformación de polilíneas Morph, y la secuencia de escenas se puede muestrear en cualquier instante.
- **Composición de fórmulas**: `math_ast` + `math_layout` componen superíndices y subíndices / fracciones / radicales / operadores grandes /
  matrices; el bloque `formula` del DSL las compone directamente, con la misma línea base que el texto.

### Qué dibuja (renderizado y simulación)

- **Renderizado offline con calidad fotográfica**: trazado de rutas + BSDF físicamente correctos (difuso / especular / vidrio / metal);
  el vidrio y el metal usan inversión estocástica (Monte Carlo por series de Neumann) y se admiten medios volumétricos y dispersión subsuperficial.
- **Cuatro familias de primitivas curvas**: cilindro / cono / toro / superficie de revolución, todas con intersección analítica (la ecuación cuártica se resuelve
  con la solución cerrada de Ferrari, con raíces exactas); mallas triangulares arbitrarias + SAH-BVH, con verificación de paridad por doble ruta.
- **GPU compute**: capa semántica de compute independiente del dispositivo + despacho Vulkan + SPIR-V mínimo escrito a mano;
  con el ICD software Mesa lavapipe funciona por completo en plataformas sin GPU (verificable en headless).
- **Física y partículas**: solver unificado XPBD (cuerpos rígidos / blandos / tela / articulaciones / fluidos / fractura),
  humo FLIP, sistemas de partículas (AudioReactive vinculado a la envolvente de audio).
- **Arte generativo y posprocesado**: campos de flujo / sistemas L / autómatas celulares / reacción-difusión / fractales / Voronoi;
  blur / bloom / halation / grano de película / cuatro estilos NPR (toon / watercolor / halftone / oil) / reducción de ruido triple.
- **Etalonaje y vectores**: corrección secundaria (selección HSL + curvas RGB), presets de estilo cinematic / film / anime y
  tres niveles de calidad borrador / estándar / cine (la calidad es un parámetro, no magia); rasterización de trazados vectoriales, draw-on y morph de trazados.
- **Procesado de audio**: normalización de sonoridad (loudnorm), detección de tempo, análisis de espectro, extracción de envolvente —
  todo puede vincularse a la imagen (las partículas AudioReactive siguen la envolvente de audio).
- **El backend de referencia en CPU es un ciudadano de primera**: las máquinas sin GPU producen igualmente imágenes correctas y, además, sirve de referencia para validar la GPU.

### Qué produce (entrega)

- **`render_movie` directo**: el producto compilado de DSL v2 completa en una sola llamada "evaluación fotograma a fotograma → renderizado por tile →
  entrega"; un `.prism` produce el montaje con un solo comando (`prism-app render demo.prism`); los nodos no cubiertos devuelven un error explícito, sin aproximaciones silenciosas.
- **Ciclo de previsualización**: `prism-app preview` levanta el servicio de previsualización; el editor ofrece dos modos de previsualización,
  Draft (patrones geométricos para mantener los fps) / Real (renderizado real con pocos spp y refresco asíncrono).
- **Codificador propio + control de tasa**: un codificador temporal en el dominio JPEG con fotogramas I/P sobre el contenedor privado `PRMV` (dos generaciones de
  schema, v2/v3), control de tasa water-filling + estimación de movimiento subpíxel qpel (PSNR +2.577 dB y tamaño −21.6% medidos).
- **Cadena de colaboración con ffmpeg**: con ffmpeg → conversión a H.264/HEVC mediante una tubería rawvideo por stdin;
  sin ffmpeg → degradación automática a MJPEG, **el motor siempre puede producir el montaje**; no se incluye ningún binario.
- **Entrega de secuencias EXR**: la secuencia de fotogramas se escribe directamente en EXR y entra en la tubería de posproducción.
- **Granja distribuida**: reparto a nivel de fotograma + N workers en procesos + fusión determinista — el MD5 en paralelo es igual al MD5 en serie,
  con reanudación desde puntos de control.

### Cómo se integra (ecosistema)

- **Superficie de herramientas MCP**: tres herramientas JSON-RPC por stdio — `scene_check` (compilación + renderizado de prueba + estimación de fotogramas),
  `render_submit` (job id determinista + resumen MD5 por fotograma), `diagnose` (localización de errores en tres partes).
- **SDK de nodos**: negociación de versiones semánticas (incluida la excepción 0.x), con un paquete de nodos de ejemplo incluido
  (`examples/sdk-nodes/cinematic-basic/`, compilable / cargable / negociable).
- **Esqueleto de detección de codificación por hardware**: `prism-ffi-hwenc` con cadena de detección en tres plataformas (VideoToolbox / NVENC / VAAPI)
  + matriz de feature gates; si la plataforma no coincide, se informa explícitamente de que no está soportada.

> Para conocer en detalle los límites de capacidad (qué **no se puede** hacer), consulta la sección «Límites honestos» de `开发者指南.md` dentro del paquete descomprimido:
> allí se registra el estado real actual (las primitivas curvas solo van por CPU, la aceleración de la GPU debe medirse en el hardware objetivo, la codificación por hardware
> se integra según la plataforma objetivo…); este documento solo ofrece el panorama general, sin exagerar.

---

## Por qué la distribución en un solo archivo

- **Auditable**: un único archivo de texto, legible a simple vista, comparable con `diff` y verificable por hash; lo publicado queda congelado.
- **Reproducible**: el empaquetado es determinista — dos empaquetados de la misma entrada dan un SHA-256 idéntico.
- **Portable**: no hace falta clonar un repositorio enorme; un solo archivo se puede enviar por correo o por chat y también permite restaurar y compilar sin conexión.
- **Compromiso asumido**: la distribución en un solo archivo renuncia a la comodidad de "navegar el código directamente en GitHub" a cambio de la determinación de "llevarse todo en un archivo y verificarlo byte a byte". Tras descomprimir se obtiene un Cargo workspace normal, idéntico a cualquier proyecto Rust convencional.

---

## Inicio rápido

Requisitos: instalar [Rust](https://www.rust-lang.org/) (con `cargo`) y [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Descarga el archivo prism-engine-v2.0-source.txt del repositorio y guárdalo como archivo local
# 2. Ábrelo con un editor de texto y guarda el script Node.js que está entre "UNPACK SCRIPT BEGIN" y "UNPACK SCRIPT END"
#    como unpack.cjs en el mismo directorio
# 3. Ejecuta la descompresión (unos segundos; restaura 622 archivos)
node unpack.cjs
# 4. Ejecuta toda la batería de pruebas en el directorio restaurado (la raíz del workspace)
cargo test --workspace --release
# Esperado: 3714 passed / 0 failed
```

Tras descomprimir se obtiene un Cargo workspace estándar (28 crates bajo `src/`); los comandos `cargo` posteriores son iguales que en cualquier proyecto Rust convencional.

> Ruta de verificación de wasm en el móvil: `rustup target add wasm32-unknown-unknown` y después `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Estructura del repositorio (tras descomprimir)

| Ruta | Descripción |
|------|------|
| `src/` | Código fuente de los **28 crates** (29 en el repositorio, con el nombre original `crates/`, reorganizados como `src/` al distribuir) |
| `Compliance/` | Declaraciones de cumplimiento: verificación de cero dependencias / avisos de terceros / registro de activos (nombre original en el repositorio `docs/合规/`) |
| `开发者指南.md` | Documentación de desarrollo autocontenida: primeros pasos / arquitectura / núcleo matemático / puertas de calidad / extensión (nombre original en el repositorio `docs/开发者指南.md`) |
| `tools/` | Scripts de puertas de calidad (defensa contra escrituras fantasma, escaneo de higiene, autocomprobación de cumplimiento) |
| `.github/workflows/` | Configuración de CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Elementos necesarios para compilar (los workspace members ya apuntan a `src/*`) |
| `examples/` | Ejemplos para las pruebas de `prism-dsl` y paquetes de nodos de ejemplo (se leen con `include_str!`; si faltan, no compila) |

---

## Resumen de módulos

Los **28 crates** del paquete de distribución, agrupados por responsabilidad:

| Categoría | crate |
|------|-------|
| Núcleo | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S de medios | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Renderizado | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
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

- Batería completa: **3714 passed / 0 failed / 0 ignored** (esta versión se ejecutó realmente sobre el **árbol descomprimido**: 166 bloques de pruebas)
- CI: **6 jobs** — defensa contra escrituras fantasma (ghost-guard) / compilación + pruebas completas (native) / compilación wasm32 (wasm) /
  comprobación estática de WGSL (wasm-verify) / comparación de flujos de bits entre implementaciones con el OpenEXR oficial (interop) / validación cruzada del host node (lint).
- 3 scripts locales de puertas de calidad (todos con cero dependencias y solo `node`, ver abajo) + 1 script de empaquetado;
  en ellos, la autocomprobación de cumplimiento pasa **9/9** y la autodemostración inversa da rojo **8/8**.

Ejecución local de las puertas de calidad (requiere Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Defensa contra escrituras fantasma: G1 definición duplicada / G2 superconjunto estricto / G3 documentación sin sangría / G4 compilación
node tools/hygiene_scan.mjs               # Higiene de código abierto: enumera uno a uno los términos internos residuales (archivo:línea); si hay coincidencia, exit 1
node tools/compliance_check.mjs           # Autocomprobación de cumplimiento: debe pasar 9/9
node tools/compliance_check.mjs --self-test   # Autodemostración inversa: debe dar rojo, lo que demuestra que el criterio tiene capacidad de discriminación
```

> **El escaneo de higiene de código abierto aún encuentra coincidencias heredadas**: en el repositorio quedan términos internos de una etapa temprana,
> de los cuales **la capa de documentación distribuida con el paquete ya está a cero** (README / guía de desarrollo / Compliance); el resto está en los
> **comentarios del código fuente dentro del paquete de distribución** y en **archivos del repositorio que no se distribuyen**, ya clasificados según su estado real
> y registrados como elementos de limpieza independientes. La existencia del script sirve precisamente para hacer el residuo **visible y limpiable uno a uno**,
> y no para fingir que está limpio: no interpretes "que el número de coincidencias no sea 0" como un fallo de la puerta de calidad.

---

## Cumplimiento

El motor promete **cero dependencias de código de terceros**, y lo verifica mediante autocomprobación:

- **C1 Cero dependencias de código de terceros**: las 130 dependencias (110 de ejecución + 20 de prueba) son todas dependencias `path` de este repositorio; no hay `build.rs`, ni directivas de enlace nativo, y `Cargo.lock` fija 29 paquetes del propio repositorio.
- **C2 Registro de activos completo**: los campos de la tabla principal de los 5 activos están completos y el estado de desarrollo propio se ajusta al vocabulario controlado; el texto completo de la licencia de fuente OFL 1.1 acompaña a la fuente en `src/prism-text/font_data/LICENSE`.
- **C3 Declaración de atribución de terceros**: `Compliance/THIRD-PARTY-NOTICES.md` está presente y distingue explícitamente "activo de datos ≠ dependencia de código".

> Nota: este repositorio distribuye **código fuente**. El activo de fuente (subconjunto de Prism Sans SC) se licencia bajo SIL OFL 1.1, y **el texto completo de la licencia acompaña a la fuente en** `src/prism-text/font_data/LICENSE`; el origen de los activos y el registro de licencias se encuentran en `Compliance/assets/资产登记.md`.


---

## Licencia

Este proyecto se publica bajo la **Apache License 2.0**. Consulta el archivo [`LICENSE`](../LICENSE) para más detalles.

> Aviso de cumplimiento: Apache-2.0 exige conservar los avisos de copyright y de licencia; si redistribuyes basándote en este motor, conserva las declaraciones de atribución y el registro de activos bajo `Compliance/`.

**Nota: si otras versiones contienen errores de traducción, la versión en chino simplificado es la referencia.**
