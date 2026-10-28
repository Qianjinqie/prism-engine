<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Qianjinqie/prism-engine@main/docs/assets/logo.png" width="132" alt="Prism Engine">
</p>

<h1 align="center">Prism Engine</h1>

<h3 align="center">Todo el pipeline multimedia, en un solo árbol de código</h3>

<p align="center">
  <sub>Decodificar · Componer · Path tracing · Simular · Sintetizar · Codificar &nbsp;|&nbsp; Cero dependencias · Determinista bit a bit</sub>
</p>

<p align="center">
  <a href="../README.md">中文</a>
  · <a href="README.en.md">English</a>
  · <a href="README.de.md">Deutsch</a>
  · <b><a href="README.es.md">Español</a></b>
  · <a href="README.fr.md">Français</a>
  · <a href="README.ja.md">日本語</a>
  · <a href="README.ko.md">한국어</a>
  · <a href="README.pt-BR.md">Português</a>
  · <a href="README.ru.md">Русский</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-3B82F6?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/tests-3929%20passed-22C55E?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/dependencies-zero-16A34A?style=for-the-badge" alt="Dependencies">
  <img src="https://img.shields.io/badge/crates-28-8B5CF6?style=for-the-badge" alt="Crates">
  <img src="https://img.shields.io/badge/size-18.9%20MB-64748B?style=for-the-badge" alt="Size">
  <a href="https://github.com/Qianjinqie/prism-engine/releases"><img src="https://img.shields.io/badge/release-v2.1-6366F1?style=for-the-badge" alt="Release"></a>
</p>

<p align="center">
  <a href="https://github.com/Qianjinqie/prism-engine/releases/latest"><img src="https://img.shields.io/badge/DOWNLOAD-v2.1-6C5CE7?style=for-the-badge" alt="Download"></a>
  <a href="#en-marcha-en-tres-minutos"><img src="https://img.shields.io/badge/QUICK_START-3%20minutes-0EA5E9?style=for-the-badge" alt="Quick start"></a>
  <a href="#arquitectura"><img src="https://img.shields.io/badge/ARCHITECTURE-28%20crates-334155?style=for-the-badge" alt="Architecture"></a>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Qianjinqie/prism-engine@main/docs/assets/still-life.png" width="880" alt="Bodegón renderizado por Prism Engine">
</p>

<p align="center">
  <sub>No es un montaje, es salida del motor: 1200×675 · 256 spp · trazado de rayos + BSDF físico + mapeo tonal ACES · 4 hilos, 158 s. Código de la escena: <a href="gallery/gallery.rs"><code>docs/gallery/gallery.rs</code></a>.</sub>
</p>

<p align="center">
  <sub><b>642</b> archivos · <b>28</b> crates · <b>261 mil</b> líneas de Rust · <b>3929</b> pruebas · <b>0</b> dependencias de terceros · <b>18,9 MB</b> en un solo archivo</sub>
</p>

## Por qué existe este motor

Una tubería de medios convencional se ensambla: ffmpeg decodifica, Blender renderiza, ImageMagick escribe imágenes fijas, un DAW se encarga del sonido y los scripts lo pegan todo con archivos temporales. Cada herramienta es excelente, pero en las costuras se pierden dos cosas: **determinismo** y **auditabilidad**. Cambia de máquina y cambia la ruta de coma flotante; cambia de versión y se mueven los valores por defecto; cuando algo falla, quedan cinco cajas negras y una pila de archivos intermedios.

| | Pipeline convencional | Prism Engine |
| --- | --- | --- |
| Cadena | cinco o seis herramientas + scripts de pegamento + archivos temporales | un árbol de código, una compilación |
| Otra máquina | las rutas de coma flotante cambian, los píxeles cambian | misma entrada, mismos bytes |
| Algo falla | cinco cajas negras y un montón de intermedios | cada píxel se rastrea hasta una línea de código |
| Dependencias | decenas de bibliotecas nativas, cada una con su matriz de versiones | 0 |

Prism Engine lleva toda la cadena a un solo árbol de código. La misma escena, los mismos parámetros, cualquier máquina, cualquier ejecución: píxeles idénticos y bytes idénticos. No es una promesa; 3929 pruebas lo sujetan.

No pretende reemplazar a ffmpeg ni a Blender. Reemplaza el pegamento entre ellos y el determinismo que ese pegamento tira en silencio.

El público es estrecho a propósito: tuberías que necesitan «misma entrada, cualquier máquina, misma salida» — renderizado por lotes, investigación gráfica reproducible, renderizado dentro de CI.

### Tres restricciones duras

| Restricción | Cómo se aplica | Qué cuesta |
| --- | --- | --- |
| **Cero código de terceros** | Las 110 dependencias de ejecución y 20 de prueba son todas `path` dentro del repositorio; sin `build.rs`, sin enlazado nativo, sin red al compilar | Cada formato, algoritmo, fuente y códec hay que escribirlo aquí |
| **Determinismo bit a bit** | IDCT de coma fija, secuencias de muestreo fijas, un hash de configuración tetradimensional; idéntico entre plataformas para la misma entrada | Renunciar a atajos rápidos de coma flotante y a depender del orden de iteración de tablas hash |
| **Distribución en un archivo** | 642 archivos de 28 crates fusionados en un texto UTF-8 de 18,9 MB, restaurado con `node unpack.cjs` | Se pierde la comodidad de hojear el código directamente en GitHub |

## En marcha en tres minutos

```bash
# 1. Descargar el paquete de código y verificar su SHA-256
curl -L -O https://github.com/Qianjinqie/prism-engine/releases/download/v2.1/prism-engine-v2.1-source.txt
curl -L -O https://github.com/Qianjinqie/prism-engine/releases/download/v2.1/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt

# 2. Extraer el script de desempaquetado incrustado (solo Node 18+, sin npm install)
awk '/^UNPACK SCRIPT BEGIN$/{f=1;next} /^UNPACK SCRIPT END$/{f=0} f' \
  prism-engine-v2.1-source.txt > unpack.cjs
node unpack.cjs                    # restaura 642 archivos

# 3. Compilar y ejecutar la suite completa
cargo test --workspace --release     # 3929 passed / 0 failed

# 4. Sin escribir nada, verlo renderizar
cargo run -p prism-app --release     # escena de demostración integrada
```

El resultado es un workspace de Cargo normal (28 crates bajo `src/`); el resto de comandos `cargo` funcionan como en cualquier proyecto Rust. La primera compilación tarda unos minutos.

## Un plano, un texto

`.prism` es el lenguaje de descripción de escenas del motor. Lo siguiente es un plano completo y renderizable de tres segundos: esfera, material, cámara orbital, luz direccional, duración, etalonaje y especificación de salida:

```prism
scene "spin_red_ball" {
    sphere ball {
        radius 1.0
        material {
            albedo #c0392b
            roughness 0.3
            metalness 0.05
        }
    }

    camera main {
        orbit {
            center 0,0,0
            radius 8.0
        }
        fov 45
    }

    light sun {
        direction (1.0, -1.0, -0.5)
        color #fff4e6
        intensity 3.0
    }
}

timeline {
    duration 3s
    fps 12
}

aesthetic {
    teal-orange
    contrast 1.12
    filmic
}

output {
    resolution 480x270
    format frames
}
```

```bash
cargo run -p prism-app --release -- examples/spin_red_ball.prism          # secuencia de fotogramas
cargo run -p prism-app --release -- render examples/w2d_showcase.prism \
    --out out/demo.mp4                                                     # película terminada en una llamada
```

`render` toma la ruta directa v2: compilar el grafo de nodos → evaluar fotograma a fotograma → codificar en un contenedor, sin fotogramas intermedios en disco. Los nodos no soportados producen un error explícito `R5xxx` en lugar de omitirse en silencio.

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Qianjinqie/prism-engine@main/docs/assets/dsl-title.png" width="760" alt="Fotograma de título renderizado desde DSL">
  <br>
  <sub>Fotograma 10 de `examples/w2d_showcase.prism`: una esfera 3D, una capa de texto, un etalonaje y una pista de fundido comparten una misma línea de tiempo.</sub>
</p>

## O simplemente la API de Rust

Si prefieres no escribir DSL, llama a la biblioteca directamente. Este es el esqueleto de la imagen principal (la escena completa tiene 190 líneas, ver [`docs/gallery/gallery.rs`](gallery/gallery.rs)):

```rust
use prism_core::node::EvalCtx;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, MaterialKind, RenderParams, Scene, Sphere};

let scene = Scene {
    spheres: vec![(
        Sphere { center: Vec3::new(0.0, 0.6, 0.0), radius: 0.6 },
        MaterialKind::Lambert { albedo: Vec3::new(0.8, 0.2, 0.15) },
    )],
    ..Default::default()
};
let cam = Camera::new(
    Vec3::new(0.0, 1.2, 4.8),   // ojo
    Vec3::new(0.0, 0.72, 0.0),  // objetivo
    Vec3::new(0.0, 1.0, 0.0),   // arriba
    30.0, 0.05, 4.85, 0.0, 1.0 / 48.0,
);
let ctx = EvalCtx::new(20261028, 0, 0);
let params = RenderParams::new(256, 10);  // 256 muestras por píxel, hasta 10 rebotes

// Un píxel: escena, cámara, contexto de muestreo, coordenada, resolución, parámetros
let color = render_pixel(&scene, &cam, &ctx, (640, 360), (1280, 720), &params);
```

`render_pixel` es una función pura: entran escena, cámara, contexto de muestreo y coordenada de píxel; sale RGB lineal. Con los mismos parámetros devuelve bits idénticos sin importar el hilo ni el orden de llamada; por eso el renderizado en paralelo no necesita sincronización alguna.

## Mapa de capacidades

| Capa | Qué hace |
| --- | --- |
| **Leer** | PNG (8/16 bit), JPEG baseline (IDCT de coma fija), GIF/APNG fotograma a fotograma, EXR (incluido DWA), RAW baseline, WAV; activos glTF / OBJ / USDA, parámetros MaterialX, fuentes TTF/OTF externas |
| **Componer** | Compilación DSL v1/v2, cinco combinadores de alto nivel (tarjeta de título / lower third / pantalla dividida / Ken Burns / morph entre escenas), cuatro transiciones + morph, composición de fórmulas (subíndices, fracciones, raíces, matrices, operadores grandes), muestreo de la línea de tiempo en cualquier instante |
| **Renderizar** | Trazado de rayos + BSDF físico (difuso / especular / vidrio / metal / subsuperficie), volúmenes heterogéneos Woodcock, intersección analítica de cilindros / conos / toros / superficies de revolución (cuártica por forma cerrada de Ferrari), mallas triangulares con SAH-BVH, backend de referencia en CPU, doble tubería Vulkan compute + ray tracing |
| **Simular** | Solver unificado XPBD (rígidos / blandos / tela / articulaciones / fluidos / fractura), humo FLIP, partículas (opcionalmente ligadas a la envolvente de audio), campos de flujo / sistemas L / autómatas celulares / reacción-difusión / fractales / Voronoi |
| **Etalonar** | Etalonaje secundario (selecciones HSL + curvas RGB), presets cinematic / film / anime, blur / bloom / halation / grano, cuatro estilos NPR (toon / acuarela / semitono / óleo), denoiser triple, rasterización de trazados vectoriales y draw-on |
| **Sonar** | Sintetizadores PolyBLEP, ADSR, máquinas de estado de escalas y progresiones de acordes, arpegiador, caja de ritmos por pasos, análisis SMF, sampler SFZ, síntesis granular, mezcla multipista, normalización de sonoridad, análisis de beat / onset / FFT |
| **Entregar** | Contenedor propio `PRMV` (v2/v3) con codificación en dominio JPEG de tramas I/P, control de tasa water-filling, estimación de movimiento qpel; tubería ffmpeg (H.264/HEVC si existe, degradación automática a MJPEG si no); secuencias EXR; granja de fotogramas multiproceso (reanudable, MD5 en paralelo == MD5 en serie) |
| **Integrar** | Tres herramientas MCP stdio (`scene_check` / `render_submit` / `diagnose`), SDK de nodos con negociación de versión semántica, servicio HTTP/1.1 de renderizado, renderizado por lotes con manifiesto cartesiano, esqueleto de sondeo de codificadores hardware en tres plataformas |

> La tabla es solo un índice. Los parámetros exactos y el «qué no puede hacer» de cada capacidad están en `开发者指南.md`, dentro del paquete.

## Capacidades que no caben en la lista

Hay cosas que no entran en la tabla anterior: no son «un formato más», sino infraestructura que el motor cultivó para sí mismo.

### Audio: un núcleo de DAW incorporado

`prism-audio` no es un módulo periférico que resulta que lee WAV. De la composición a la mezcla es una cadena completa, y cada función es una función pura de sus parámetros (bits idénticos entre plataformas para la misma entrada):

- **Síntesis**: familia de osciladores PolyBLEP antialias, envolventes ADSR en forma cerrada, cuatro presets de ánimo;
- **Teoría**: biblioteca de escalas (mayor/menor, pentatónica, modos), clase de altura 12-TET ↔ frecuencia, máquina de estados con gramática funcional para progresiones de acordes y conducción de voces;
- **Arreglo**: arpegiador (patrón / tasa / octava), caja de ritmos por pasos, rejillas de compás y pulso;
- **Material**: análisis SMF escrito a mano (importación MIDI), sampler SFZ, síntesis granular;
- **Diseño sonoro**: modelado físico, paneo espacial, cadenas de efectos;
- **Mezcla**: mezcla lineal multipista, fundidos, protección de picos;
- **Análisis y vínculo**: detección de beat / onset, espectros FFT, extracción de envolvente, normalización de sonoridad; el horneado AudioReactive lee una ventana fija en el fotograma N, lo que lo hace función pura del fotograma y equivalente a seek.

> Límite: la codificación AAC / mp3 no se implementa aquí (lastre histórico de patentes más esfuerzo); pasa por ffmpeg externo.

### Un registro de kernels definibles

Cada tipo de nodo del motor es una **solución particular** de una terna «kernel + retícula + dual». Terceros (o tú) pueden registrar soluciones nuevas estáticamente con `registry!`, **sin tocar el núcleo del motor**:

- Inclusión estática: la macro se expande dentro del crate que define; se prohíben `inventory` y los proc-macros para mantener cero dependencias;
- El registro hace una **verificación de tres preguntas** (forma del kernel / retícula / dual); respuestas incompletas se rechazan con un error en tres partes;
- Las soluciones oficiales y las propias comparten un mismo registro y son indistinguibles en tiempo de ejecución;
- Matemáticamente corresponde a las seis estrategias de inversión de la ecuación de operadores `f = g + Kf` (exacta / estocástica / local / regularizada / iterativa / espectral): añadir una capacidad es declarar una terna, no parchear el bucle de renderizado.

### GPU: tres rutas, tres estados

| Backend | Estado | Evidencia |
| --- | --- | --- |
| **Vulkan** | Funciona de verdad | Tuberías compute + ray tracing; verificable sin GPU sobre el ICD software lavapipe de Mesa; la salida de `render_scene` es **idéntica byte a byte (0 diferencia)** a la referencia de CPU; `is_available()` es una autocomprobación compute ∧ render |
| **Metal** | Compilado tras una puerta, sin activar | Port isomorfo terminado, no verificado en hardware, `is_available()` siempre false |
| **WebGPU** | Solo conclusión | Evaluación de viabilidad hecha (4–6 días-persona de port), ruta de renderizado no implementada |

### Puertas anti-falso-verde

Cuatro scripts de puerta sin dependencias (solo Node): defensa contra escrituras fantasma (con autoprueba inversa), escaneo de higiene de código abierto, autocomprobación de cumplimiento (9/9 aprobadas más 8/8 pruebas de que los criterios fallan cuando deben) y verificación de ida y vuelta del empaquetado. Una puerta tiene exactamente dos resultados — pasa o falla — y «parece que está bien» no es uno de ellos. **Un verde falso es más peligroso que un fallo.**

## El determinismo no es un eslogan

Cuatro capas, desde los parámetros hasta los bytes:

1. **Cierre de firmas**: grafos de nodos, claves de caché, claves de instantánea y firmas de fotograma comparten un hash de configuración tetradimensional. Cambia un parámetro ⇒ la clave debe cambiar ⇒ no hay reutilización indebida. Los nodos iterativos llevan `(ε, n)` en la clave; los nodos de flujo, el número de fotograma; con `None` todo degrada bit a bit al comportamiento anterior.
2. **Puntos fijos de bytes**: exportar→importar→exportar USD es idéntico byte a byte; WAV 16/24/32 bit va y vuelve sin pérdida; EXR se compara bitstream contra bitstream con el OpenEXR oficial.
3. **Determinismo del empaquetado**: empaquetar dos veces el mismo código da el mismo SHA-256; la ida y vuelta de desempaquetado compara cada archivo byte a byte.
4. **Las pruebas como criterio**: muchas de las 3929 pruebas fijan valores de oro (tolerancia ajustada de 1e-2 a 1e-6, residuo medido 4,53e-7); la fidelidad de forma se acota con distancias de Hausdorff en lugar de «se parece bastante».

## Arquitectura

```mermaid
flowchart LR
    IN["Entrada<br/>PNG · JPEG · EXR · RAW<br/>GIF · APNG · WAV<br/>glTF · OBJ · USDA"] --> DSL["Composición<br/>prism-dsl<br/>grafo de nodos · canales · línea de tiempo"]
    DSL --> RT["Evaluación<br/>raster 2D · trazado 3D<br/>física · partículas · arte generativo"]
    DSL --> AU["Audio<br/>prism-audio<br/>síntesis · arreglo · mezcla"]
    RT --> FX["Post<br/>etalonaje · bloom · denoise · NPR"]
    FX --> OUT["Entrega<br/>PRMV · ffmpeg · secuencias EXR"]
    AU --> OUT
```

28 crates agrupados por responsabilidad:

| Grupo | Crates |
| --- | --- |
| Núcleo | `prism-core` · `prism-math` · `prism-bytes` · `prism-geometry` · `prism-noise` |
| E/S de medios | `prism-io` · `prism-media` · `prism-mux` · `prism-audio` · `prism-texture` |
| Renderizado | `prism-render` · `prism-raster` · `prism-gpu` · `prism-postfx` · `prism-vector` · `prism-particles` |
| Animación / generativo | `prism-anim` · `prism-genart` · `prism-aesthetic` · `prism-physics` |
| Texto / UI | `prism-text` · `prism-rt` · `prism-app` |
| Avanzado | `prism-ai` · `prism-dsl` · `prism-infra` · `prism-author` |
| Sidecar (feature gate) | `prism-ffi-hwenc` |

Estructura tras desempaquetar:

| Ruta | Contenido |
| --- | --- |
| `src/` | Código de los 28 crates (en el repositorio `crates/`, reorganizado para la distribución) |
| `Compliance/` | Verificación de cero dependencias, avisos de terceros, registro de activos |
| `开发者指南.md` | Guía del desarrollador autónoma: inicio / arquitectura / núcleo matemático / puertas / extensión |
| `tools/` | Scripts de puertas y empaquetado |
| `examples/` | Ejemplos `.prism` y paquetes de nodos (leídos con `include_str!`; si falta uno, no compila) |
| `.github/workflows/` | Configuración de CI |

> Los documentos de diseño (`docs/设计/`) solo existen en el repositorio completo del editor y no se distribuyen; el crate de demostración `prism-anim-demo` tampoco va en el paquete.

## Compilar, probar y puertas

```bash
cargo test --workspace --release      # 3929 passed / 0 failed / 0 ignored
cargo xtest                           # alias equivalente
cargo test -p prism-core --release    # prueba rápida de un crate

node tools/ghost_guard.mjs --check    # defensa de escritura fantasma: G1 redefinición / G2 superconjunto estricto / G3 documentación superior / G4 compilación
node tools/hygiene_scan.mjs           # higiene: lista cada término interno restante (archivo:línea), exit 1 si hay coincidencia
node tools/compliance_check.mjs       # autocomprobación de cumplimiento: debe pasar 9/9
node tools/compliance_check.mjs --self-test  # prueba inversa: debe fallar, demostrando que el criterio discrimina
node tools/pack_source_txt.mjs --verify      # ida y vuelta de empaquetado, byte a byte
```

- Suite completa: **3929 passed / 0 failed / 0 ignored**, 179 secciones de prueba, 171,5 s con todos los núcleos (release).
- La CI tiene seis jobs: defensa de escritura fantasma, compilación + pruebas completas, compilación wasm32, comprobación estática WGSL, interoperabilidad de bitstream con el OpenEXR oficial, verificación cruzada en host Node.
- El escaneo de higiene aún tiene coincidencias: la capa de documentación distribuida está limpia, el resto está en comentarios de código y archivos solo del repositorio. El script existe para que lo restante siga **visible y corregible pieza a pieza**, no para fingir que todo está limpio.

## Límites honestos

| Límite | Estado | Salida |
| --- | --- | --- |
| DNG `Compression=7` (lossless JPEG) | No implementado; los DNG de cámaras reales no se pueden leer | Convertir antes a TIFF/PNG de 16 bit |
| mp3 / AAC / FLAC | Decodificación vía sonda de ffmpeg externo; codificación no implementada | Instalar ffmpeg, o quedarse en WAV |
| Primitivas curvas en GPU | Solo ruta de CPU | Triangular y pasar por el SAH-BVH |
| Metal | Puerta compilada, no verificada en hardware, sin activar | Conectar cuando haya hardware Apple |
| WebGPU | Viabilidad concluida, ruta de renderizado no implementada | 4–6 días-persona |
| Codificación hardware | Esqueleto de sondeo para tres plataformas, sin implementación | Rellenar por plataforma objetivo |
| Aceleración GPU | Corrección verificada solo en lavapipe, nunca medida en GPU real | Medir en el hardware objetivo |
| Escaneo de higiene | Coincidencias existentes en comentarios | Limpiar una a una; el script las rastrea |

No son puntos de «ya veremos». Hoy no están soportados, explícitamente. Mejor aquí que escondidos en un issue.

## Cumplimiento

El motor promete cero dependencias de código de terceros y lo verifica por autocomprobación:

- **C1 Cero código de terceros**: las 130 dependencias (110 de ejecución + 20 de prueba) son `path` dentro del repositorio; sin `build.rs`, sin directivas de enlace nativo; `Cargo.lock` fija 29 paquetes del repositorio.
- **C2 Registro de activos completo**: las cinco entradas de activos tienen completos los campos de la tabla principal; las fuentes se distribuyen bajo SIL OFL 1.1 con el texto completo de la licencia junto a la fuente en `src/prism-text/font_data/LICENSE`.
- **C3 Avisos de terceros**: `Compliance/THIRD-PARTY-NOTICES.md` separa explícitamente «activos de datos ≠ dependencias de código».

## Licencia

Apache License 2.0, ver [`LICENSE`](../LICENSE).

Al redistribuir, conserva los avisos y el registro de activos bajo `Compliance/`.

<sub>La imagen principal y la captura de DSL fueron renderizadas por el propio motor; los parámetros están en los pies de foto y en [`docs/gallery/gallery.rs`](gallery/gallery.rs).</sub>
