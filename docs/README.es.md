[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)


# Prism Engine

> Motor multimedia en Rust, determinista y sin dependencias — **distribuido como un único archivo TXT**: descarga un texto de 18 MB, ejecuta el script de descompresión y obtienes el árbol de código fuente completo (28 crates), sin necesidad de npm / pip / ffmpeg / OpenEXR.

![Licencia](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Lenguaje](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencias](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribución](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Pruebas](https://img.shields.io/badge/tests-3929%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## Qué es esto

Prism Engine es un motor multimedia escrito en Rust con cero dependencias de código de terceros:
leer imágenes, componer, dibujar fotogramas, montar la línea de tiempo y producir el montaje final,
todo dentro de un único motor determinista. Sus 110 dependencias de ejecución
(más 20 dependencias de prueba) son todas dependencias `path` de este mismo repositorio: no
dependen de la red, no necesitan gestor de paquetes y no implican riesgo de cadena de suministro.

Adopta la **distribución del código fuente en un solo archivo**: todo el motor (642 archivos, 28 crates)
se fusiona en un único texto UTF-8, `prism-engine-v2.1-source.txt`. Basta con disponer de ese archivo
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

---

## Capacidades internas (más allá de la lista de funciones)

Algunas capacidades no encajan en la lista «leer / componer / dibujar / entregar / integrar»: son los activos profundos del motor. Este capítulo mantiene limpia la lista principal.

### Audio: un núcleo DAW integrado

`prism-audio` no es un módulo marginal que «sabe leer WAV», sino una cadena completa desde la composición hasta la mezcla; cada función es una función pura de sus parámetros (misma entrada ⇒ bits idénticos entre plataformas):

- **Síntesis**: familia de osciladores PolyBLEP sin aliasing + envolventes ADSR en forma cerrada + 4 presets de ambiente;
- **Teoría musical**: biblioteca de escalas (mayor/menor, pentatónica, modos), 12-TET altura↔frecuencia, máquina de estados de gramática funcional para progresiones + conducción de voces;
- **Secuenciación**: arpegiador (patrón/velocidad/octava), caja de ritmos por pasos, partitura en rejilla de compases/pulsos;
- **Material**: parser SMF escrito a mano (importación MIDI), sampler SFZ, síntesis granular;
- **Diseño sonoro**: modelado físico, paneo espacial, cadena de efectos;
- **Mezcla**: mezcla lineal multipista + fundidos + limitación de picos;
- **Análisis y vínculo**: detección de pulso/onset, espectro FFT, extracción de envolventes, horneado AudioReactive (el fotograma N lee una ventana fija; función pura por fotograma, equivalente a seek), normalización de sonoridad.

> Límite: AAC/mp3 no se desarrollan aquí (historial de patentes + esfuerzo); el audio general pasa por ffmpeg externo.

### Biblioteca de kernels definibles (registro de soluciones especiales)

Cada tipo de nodo es una **solución especial** de una terna «kernel + retícula + dualidad». Terceros (o tú) pueden registrar nuevas soluciones con `registry!` sin tocar el núcleo:

- Recolección estática (la macro se expande en el crate que define; sin `inventory`/proc-macro, cero dependencias);
- **Validación de tres preguntas** al registrar (forma del kernel / retícula / dualidad): respuestas incompletas se rechazan con un error en tres partes;
- Las soluciones oficiales y las personalizadas comparten el mismo registro; sin diferencia en tiempo de ejecución;
- Matemáticamente es la ecuación de operador `f = g + Kf` con seis estrategias de inversión — exacta / estocástica / local / regularizada / iterativa / espectral. Añadir una capacidad es declarar una terna, no parchear el bucle de render.

### Infraestructura de determinismo y reproducibilidad

- **Cierre de firmas**: grafos de nodos / claves de caché / claves de instantánea / firmas de fotograma se reducen a un «hash de configuración cuatridimensional»; cambiar un parámetro cambia la clave y nada se reutiliza por error; los nodos iterativos usan `(ε, n)`, los nodos de flujo usan el número de fotograma;
- **Puntos fijos de bytes**: USD exportar→importar→exportar es idéntico byte a byte; WAV 16/24/32 bits va y vuelve exacto; EXR se compara bit a bit contra la implementación oficial;
- **Empaquetado determinista en un solo archivo**: el mismo código produce el mismo SHA-256 dos veces; el round-trip de desempaquetado compara cada archivo byte a byte.

### Backends GPU (declarados con honestidad)

- **Vulkan**: las canalizaciones de cómputo y de escena con trazado de rayos se ejecutan de verdad (verificable sin GPU con el ICD software lavapipe); la salida de `render_scene` es **idéntica byte a byte a la referencia CPU** (0 píxeles distintos); `is_available()` exige autoprueba de cómputo y de render;
- **Metal**: port isomorfo + compilación condicionada listos, **no verificado en hardware, no activado** (`is_available()` siempre false);
- **WebGPU**: solo conclusión de viabilidad (4–6 días-persona), sin ruta de render.

### Puertas anti-falso-verde

Cuatro scripts locales (cero dependencias, solo `node`): defensa contra escrituras fantasma (con autoprueba inversa), escaneo de higiene de código abierto, autocomprobación de cumplimiento (9/9, autoprueba inversa 8/8 en rojo) y verificación de round-trip del empaquetado. Una puerta solo da «pasa» o «rojo»; «parece bien» no cuenta. **El verde falso es más peligroso que un fallo.**

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
# 1. Descarga el archivo prism-engine-v2.1-source.txt del repositorio y guárdalo como archivo local
# 2. Ábrelo con un editor de texto y guarda el script Node.js que está entre "UNPACK SCRIPT BEGIN" y "UNPACK SCRIPT END"
#    como unpack.cjs en el mismo directorio
# 3. Ejecuta la descompresión (unos segundos; restaura 642 archivos)
node unpack.cjs
# 4. Ejecuta toda la batería de pruebas en el directorio restaurado (la raíz del workspace)
cargo test --workspace --release
# Esperado: 3929 passed / 0 failed
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

- Batería completa: **3929 passed / 0 failed / 0 ignored** (ejecutado en release dentro del repositorio: 179 suites, 171,5 s con todos los núcleos; el paquete es idéntico byte a byte al repositorio — solo se excluye el crate de demostración)
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
