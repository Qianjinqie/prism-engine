[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch** | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> Eine deterministische Rust-Medien-Engine ohne Abhängigkeiten – verteilt als **einzelne TXT-Datei**: eine 17 MB große Textdatei herunterladen, das Entpackungsskript ausführen und man erhält den vollständigen Quellbaum (28 crates), ohne npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

(Die Badges sind statische Fakten-Badges, die Zahlenbasis steht weiter unten; `crates-28` = die Anzahl der crates **im Distributionspaket**, im Repository liegt zusätzlich 1 Demo-crate, das nicht ins Paket geht, daher beträgt die Repository-Zahl 29. Nach Anbindung an die CI lassen sie sich durch dynamische Badges ersetzen.)

---

## Was ist das

Prism Engine ist eine in Rust geschriebene Medien-Engine ohne Code-Abhängigkeiten von Drittanbietern: Bilder lesen, Satz, Bildgestaltung, Timeline-Schnitt und fertiger Film – alles in einer deterministischen Engine. Alle 110 Laufzeitabhängigkeiten (dazu 20 Testabhängigkeiten) sind `path`-Abhängigkeiten dieses Repositories – keine Netzabhängigkeit, kein Paketmanager, kein Supply-Chain-Risiko.

Verteilt wird per **Single-File-Quelltextverteilung**: die gesamte Engine (622 Dateien, 28 crates) ist in eine UTF-8-Textdatei `prism-engine-v2.0-source.txt` zusammengeführt. Wer diese Datei besitzt, kann lokal einen Verzeichnisbaum wiederherstellen, der exakt dem Repository entspricht.

---

## Was die Engine kann

Überblick in einem Satz: **Ein Bild ist ein Motiv zu einem Zeitpunkt, ein Video ist eine Folge von Motiven über einen Zeitraum** – dieselbe Szenenbeschreibung ergibt, an einem Moment festgehalten, ein Einzelbild; kontinuierlich weitergeschoben und aufgezeichnet ergibt sie ein Video. Statisches und bewegtes Bild teilen sich denselben Funktionsumfang. Im Folgenden fünf Blöcke nach dem, was Sie tun wollen: **einlesen → anordnen → zeichnen → zum fertigen Film bringen → in Ihre eigene Toolchain einbinden**.

### Was gelesen wird (Eingabe)

- **Einheitliche Dekodier-Fassade**: ein Einstieg `decode_image(bytes, ext)` verteilt statische Bilder – PNG (nicht interlaced, 8/16 bit), Baseline-JPEG (Integer-IDCT, plattformübergreifend bitgenau), **Magic Number zuerst, Dateiendung als Rückfallebene** (auch bei falscher Endung dekodierbar); Formate außerhalb der Matrix oder solche, die externe Mithilfe brauchen, liefern stets einen dreiteiligen Fehler (wo / was / wie ändern) samt ffmpeg-Kooperationshinweis, **niemals stilles Scheitern**.
- **Animierte Bilder / Audio / Container**: GIF, APNG native Dekodierung Frame für Frame (`prism-media`); WAV mit eingebauter Dekodierung, mp3/aac/FLAC über den ffmpeg-Probing-Pfad (`prism-audio`); MP4-MJPEG-Rücklesen (`decode_mp4_video`).
- **EXR komplett**: Lesen und Schreiben, Kompression, DWA, dazu **Bitstrom-Abgleich über Implementierungsgrenzen hinweg** mit dem offiziellen OpenEXR (kein Selbstschreiben-und-selbst-Lesen).
- **Kamera-RAW**: Baseline-RAW-Dekodierung. **Grenze**: `Compression=7` (lossless JPEG) echter DNGs ist nicht implementiert, von echten Kameras erzeugte DNGs lassen sich nicht lesen – das steht in den „Ehrlichen Grenzen" und wird nicht versteckt.
- **3D-Assets**: glTF / OBJ-Import, USD-(`.usda`)-Parsing und MaterialX-Parameterpakete, meshes lassen sich in die Render-Szene übernehmen.
- **Externe Schriften**: `load_font_file` lädt TTF/OTF von Drittanbietern direkt, die Fallback-Kette wird stufenweise aufgelöst, fehlende Glyphen liefern eine klare Diagnose statt Tofu-Kästchen.

### Was angeordnet wird (Autorensyntax)

- **Vollständige Blocksyntax von DSL v2**: eine `.prism`-Datei beschreibt, „was gewünscht ist, wie es angeordnet ist, wie es sich bewegt, wie es aussieht, wohin exportiert wird"; das Kompilat ist ein Node-Graph + Eigenschaftskanäle + Timeline.
- **Fünf Kombinatoren**: die High-Level-Kombinatoren `title_card` / `lower_third` / `split_screen` / `ken_burns` / `morph_between` expandieren zu einem Node-Graph; „90-Sekunden-Film" sinkt von handgeschriebenem Szenencode auf Zusammenstecken.
- **Übergänge und Morph**: Fade / Wipe / Dissolve / Push vollständig parametrisiert (Dauer, Easing, Richtung), Morph als Verformung von Polylinien, Szenensequenzen zu jedem Zeitpunkt abtastbar.
- **Formelsatz**: `math_ast` + `math_layout` setzen Hoch- und Tiefstellung / Brüche / Wurzelzeichen / große Operatoren / Matrizen; der DSL-Block `formula` setzt direkt, auf derselben Grundlinie wie der Fließtext.

### Was gezeichnet wird (Rendering und Simulation)

- **Offline-Rendering in Fotoqualität**: Path Tracing + physikalisch korrekte BSDFs (diffus / spiegelnd / Glas / Metall), Glas und Metall über stochastische Inversion (Neumann-Reihe, Monte Carlo), Unterstützung für volumetrische Medien und Subsurface Scattering.
- **Vier Familien gekrümmter Primitive**: Zylinder / Kegel / Torus / Rotationsfläche, alle analytisch geschnitten (quartische Gleichung über die geschlossene Ferrari-Lösung, exakte Wurzeln); beliebige Dreiecks-meshes + SAH-BVH, Parity-Abgleich über zwei Pfade.
- **GPU compute**: geräteunabhängige Compute-Semantikschicht + Vulkan dispatch + handgeschriebenes minimales SPIR-V; unter der Software-ICD Mesa lavapipe vollständig lauffähig auf Plattformen ohne GPU (headless verifizierbar).
- **Physik und Partikel**: einheitlicher XPBD-Solver (Starrkörper / Soft Bodies / Stoff / Gelenke / Fluide / Bruch), Rauch mit FLIP, Partikelsysteme (AudioReactive an die Audio-Hüllkurve gekoppelt).
- **Generative Kunst und Postprocessing**: Strömungsfelder / L-system / zelluläre Automaten / Reaktions-Diffusion / Fraktale / Voronoi; blur / bloom / halation / Filmkorn / vier NPR-Stile (toon / watercolor / halftone / oil) / dreifach kombiniertes Denoising.
- **Color Grading und Vektoren**: sekundäres Grading (HSL-Auswahl + RGB-Kurven), Stil-Presets cinematic / film / anime und drei Qualitätsstufen Entwurf/Standard/Film (Qualität ist ein Parameter, keine Esoterik); Rasterung von Vektorpfaden, Pfad-draw-on und morph.
- **Audioverarbeitung**: Loudness-Normalisierung (loudnorm), Beat-Erkennung, Spektralanalyse, Hüllkurvenextraktion – koppelbar mit dem Bild (Partikel-AudioReactive folgt der Audio-Hüllkurve).
- **Das CPU-Referenzbackend ist ein Bürger erster Klasse**: auch ohne GPU entstehen korrekte Bilder, und zugleich ist es der Korrektheitsmaßstab für die GPU.

### Was herauskommt (Auslieferung)

- **`render_movie` direkt**: das DSL-v2-Kompilat durchläuft in einem Aufruf „Auswertung Frame für Frame → tile-Rendering → Auslieferung", eine einzelne `.prism`-Datei erzeugt per Einzelbefehl den Film (`prism-app render demo.prism`); nicht abgedeckte Nodes melden ehrlich einen Fehler statt still zu approximieren.
- **Vorschau-Kreislauf**: `prism-app preview` startet den Vorschau-Dienst; der Editor bietet zwei Vorschaustufen – Draft (geometrische Muster zur Erhaltung der Framerate) / Real (echtes Rendering mit niedriger spp, asynchron aktualisiert).
- **Eigener Encoder + Ratenkontrolle**: ein zeitlicher I/P-Frame-Encoder im JPEG-Bereich schreibt in den privaten `PRMV`-Container (zwei Schema-Generationen, v2/v3), Water-Filling-Ratenkontrolle + qpel-Subpixel-Bewegungsschätzung (gemessen: PSNR +2.577 dB, Volumen −21.6 %).
- **ffmpeg-Kooperationskette**: mit ffmpeg → über eine stdin-rawvideo-Pipe nach H.264/HEVC; ohne ffmpeg → automatischer Rückfall auf MJPEG, **die Engine liefert immer einen Film**; es werden keine Binaries gebündelt.
- **EXR-Sequenzauslieferung**: die Bildfolge wird direkt als EXR geschrieben und geht in die Postproduktions-Pipeline.
- **Verteilte Farm**: Verteilung auf Frame-Ebene + N Prozess-worker + deterministische Zusammenführung – paralleles MD5 == serielles MD5, Fortsetzung nach Unterbrechung möglich.

### Wie integriert wird (Ökosystem)

- **MCP-Tool-Oberfläche**: drei stdio-JSON-RPC-Tools – `scene_check` (Kompilieren + Testrender + Frame-Zahlenschätzung), `render_submit` (deterministische job id + MD5-Digest pro Frame), `diagnose` (dreiteilige Fehlerlokalisierung).
- **Node-SDK**: semantische Versionsaushandlung (inklusive 0.x-Sonderfall), mitgeliefert wird ein Beispiel-Node-Paket (`examples/sdk-nodes/cinematic-basic/`, kompilierbar / ladbar / aushandelbar).
- **Skelett der Hardware-Encoding-Erkennung**: `prism-ffi-hwenc` mit Erkennungskette für drei Plattformen (VideoToolbox / NVENC / VAAPI) + Feature-Gate-Matrix; bei unpassender Plattform wird klar „nicht unterstützt" gemeldet.

> Die detaillierten Fähigkeitsgrenzen (was **nicht** geht) stehen im Abschnitt „Ehrliche Grenzen" der `开发者指南.md` im entpackten Paket –
> dort ist der aktuelle reale Stand festgehalten (gekrümmte Primitive laufen nur über die CPU, das GPU-Beschleunigungsverhältnis muss auf der Zielhardware gemessen werden, Hardware-Encoding wird je nach
> Zielplattform angebunden …); dieses Dokument gibt nur den Überblick und übertreibt nicht.

---

## Warum Single-File-Verteilung

- **Auditierbar**: eine Textdatei, mit bloßem Auge lesbar, `diff`-bar, per Hash prüfbar; die Veröffentlichung ist damit eingefroren.
- **Reproduzierbar**: Das Packen ist deterministisch – zweimaliges Packen derselben Eingabe ergibt identische SHA-256.
- **Transportabel**: kein Klonen eines großen Repositories nötig, eine Datei lässt sich per Mail oder Chat verschicken, auch offline ist Wiederherstellen und Bauen möglich.
- **Zur Abwägung**: Die Single-File-Verteilung gibt den Komfort „Code direkt auf GitHub durchstöbern" auf und gewinnt dafür die Determinismus-Garantie „eine Datei trägt alles, byteweise verifizierbar". Nach dem Entpacken liegt ein normales Cargo workspace vor, nicht anders als bei einem üblichen Rust-Projekt.

---

## Schnellstart

Voraussetzungen: [Rust](https://www.rust-lang.org/) (inklusive `cargo`) und [Node.js](https://nodejs.org/) 18+ installieren.

```bash
# 1. Die Datei prism-engine-v2.0-source.txt aus dem Repository herunterladen und lokal speichern
# 2. In einem Texteditor öffnen und das Node.js-Skript zwischen "UNPACK SCRIPT BEGIN" und
#    "UNPACK SCRIPT END" als unpack.cjs im selben Verzeichnis speichern
# 3. Entpacken ausführen (wenige Sekunden, stellt 622 Dateien wieder her)
node unpack.cjs
# 4. Im wiederhergestellten Verzeichnis (also der workspace-Wurzel) die vollständigen Tests laufen lassen
cargo test --workspace --release
# Erwartet: 3714 passed / 0 failed
```

Nach dem Entpacken erhält man ein standardmäßiges Cargo workspace (28 crates unter `src/`); alle weiteren `cargo`-Befehle entsprechen denen eines normalen Rust-Projekts.

> wasm-Verifikationskette für Mobilgeräte: `rustup target add wasm32-unknown-unknown`, danach `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Repository-Struktur (nach dem Entpacken)

| Pfad | Beschreibung |
|------|------|
| `src/` | Quellcode der **28 crates** (im Repository 29; ursprünglicher Name `crates/`, bei der Distribution zu `src/` umgruppiert; das Demo-crate `prism-anim-demo` ist nicht im Paket) |
| `Compliance/` | Compliance-Erklärungen: Prüfung der Null-Abhängigkeit / Drittanbieter-Erklärungen / Asset-Register (ursprünglicher Name im Repository: `docs/合规/`) |
| `开发者指南.md` | In sich geschlossene Entwicklerdokumentation: Einstieg / Architektur / mathematischer Kern / Gates / Erweiterung (ursprünglicher Name im Repository: `docs/开发者指南.md`) |
| `tools/` | Gate-Skripte (Ghost-Write-Abwehr, Hygiene-Scan, Compliance-Selbstprüfung) |
| `.github/workflows/` | CI-Konfiguration |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | zum Bauen erforderliche Teile (workspace members zeigen bereits auf `src/*`) |
| `examples/` | Beispiele für `prism-dsl`-Tests und Beispiel-Node-Pakete (per `include_str!` gelesen, fehlen sie, kompiliert es nicht) |

> Die maßgebliche Design-Dokumentation (`docs/设计/`) wird **nicht mitverteilt** und bleibt ausschließlich im vollständigen Repository des Herausgebers.

---

## Modulübersicht

Die **28 crates** im Distributionspaket, nach Zuständigkeit gruppiert (im Repository gibt es zusätzlich das Demo-crate `prism-anim-demo`, das nur als reproduzierbares Beispiel im Repository liegt und nicht ins Paket geht):

| Kategorie | crate |
|------|-------|
| Kernel | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Medien-I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendering | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Generierung | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Text/UI | prism-text · prism-rt · prism-app |
| Erweitert | prism-ai · prism-dsl · prism-infra · prism-author |
| Bypass (feature gate) | prism-ffi-hwenc |

---

## Bauen und Testen

```bash
# Smoke-Test über das gesamte Repository (release, empfohlen)
cargo test --workspace --release
# gleichwertiger Alias
cargo xtest
# schnelle Prüfung eines einzelnen crates
cargo test -p prism-core --release
```

- Vollständige Tests: **3714 passed / 0 failed / 0 ignored** (diese Version wurde tatsächlich auf dem **entpackten Baum** ausgeführt: 166 Test-Blöcke; im Repository sind es 167 Blöcke, die Differenz von 1 Block ist der leere Test-Block des Demo-crates, die Zahl der Testfälle ist gleich).
- CI: **6 jobs** – Ghost-Write-Abwehr (ghost-guard) / Kompilieren + vollständige Tests (native) / wasm32-Build (wasm) / statische WGSL-Prüfung (wasm-verify) / Bitstrom-Abgleich mit dem offiziellen OpenEXR über Implementierungsgrenzen hinweg (interop) / Kreuzabgleich mit dem node-Host (lint).
- 3 lokale Gate-Skripte (vollständig ohne Abhängigkeiten, nur `node`, siehe unten) + 1 Packskript; davon besteht die Compliance-Selbstprüfung **9/9**, der Gegentest wird **8/8 als rot** gewertet.

Gate-Skripte lokal ausführen (Node 18+ erforderlich):

```bash
node tools/ghost_guard.mjs --check        # Ghost-Write-Abwehr: G1 doppelte Definition / G2 echte Obermenge / G3 Dokumentation ohne Einrückung / G4 Kompilierung
node tools/hygiene_scan.mjs               # Open-Source-Hygiene: listet jedes verbliebene interne Wording einzeln auf (Datei:Zeile), bei Treffer exit 1
node tools/compliance_check.mjs           # Compliance-Selbstprüfung: sollte 9/9 bestehen
node tools/compliance_check.mjs --self-test   # Gegentest: sollte rot werden und damit belegen, dass die Kriterien unterscheidungsfähig sind
```

> **Der Open-Source-Hygiene-Scan hat derzeit noch Altbestände an Treffern**: im Repository verbleibt frühes internes Wording, davon ist
> **die mitgelieferte Dokumentationsebene auf null** (README / Entwicklerhandbuch / Compliance); der Rest steckt in **Quellkommentaren
> innerhalb des Distributionspakets** und in **Repository-Dateien, die nicht mitverteilt werden**, wurde nach tatsächlichem Befund klassifiziert
> und als eigener Aufräumpunkt geführt. Das Skript existiert gerade dazu, Rückstände **sichtbar und einzeln abarbeitbar** zu machen, statt Sauberkeit vorzutäuschen – bitte verstehen Sie „Trefferzahl ungleich 0" nicht als Versagen des Gates.

---

## Compliance

Die Engine verspricht **null Code-Abhängigkeiten von Drittanbietern**, geprüft durch Selbstprüfung:

- **C1 Null Code-Abhängigkeiten von Drittanbietern**: alle 130 Abhängigkeiten (110 Laufzeit + 20 Test) sind `path`-Abhängigkeiten dieses Repositories; kein `build.rs`, keine nativen Linker-Anweisungen, `Cargo.lock` sperrt 29 Pakete dieses Repositories.
- **C2 Vollständiges Asset-Register**: die Haupttabelle der 5 Assets hat alle Felder, der Status der Eigenentwicklung liegt in einem kontrollierten Vokabular; der vollständige Text der OFL-1.1-Schriftlizenz liegt bei der Schrift unter `src/prism-text/font_data/LICENSE`.
- **C3 Drittanbieter-Zuordnungsanzeigen**: `Compliance/THIRD-PARTY-NOTICES.md` ist vorhanden und unterscheidet ausdrücklich „Datenasset ≠ Code-Abhängigkeit".

> Hinweis: Dieses Repository verteilt **Quellcode**. Das Schriftasset (Prism Sans SC Subset) steht unter SIL OFL 1.1, **der vollständige Lizenztext liegt bei der Schrift unter** `src/prism-text/font_data/LICENSE`; Herkunft und Lizenzregistrierung der Assets siehe `Compliance/assets/资产登记.md`.

---

## Deterministisches Packen (für Herausgeber)

Um die Distributions-TXT aus dem vollständigen Repository neu zu erzeugen (Versionsnummer kann übergeben werden, standardmäßig wird die aktuelle Version v2.0 erzeugt):

```bash
node tools/pack_source_txt.mjs          # Standard → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # gleichwertige Schreibweise (Versionsnummer explizit übergeben)
node tools/pack_source_txt.mjs 1.0      # zurück zum Produktnamen der v1.0-Zeit (auf v1.0-Quellen ausgeführt wird das alte Paket byteweise reproduziert)
node tools/pack_source_txt.mjs --verify # nach dem Packen sofortiger Entpack-Rundlauf zur Prüfung (Datei für Datei, Byte für Byte gegen den Quellbaum)
# zweimal hintereinander ausführen; identische SHA-256 belegen, dass deterministisches Packen gilt
```

Der Packumfang ist strikt begrenzt auf: `crates/` (→ `src/`), `docs/合规/` (→ `Compliance/`), `docs/开发者指南.md`, `tools/`, `.github/`, `.cargo/`, `examples/`, `Cargo.toml`, `Cargo.lock`. Bewusst **ausgeschlossen** sind `docs/设计/`, `target/`, `node_modules/`, `dist/` selbst sowie sämtliche Zwischenprodukte.

---

## Mitwirken

Issues und Pull Requests sind willkommen. Bitte vor dem Einreichen sicherstellen:

- `cargo test --workspace --release` läuft vollständig grün;
- `node tools/hygiene_scan.mjs` führt **kein neues** internes Wording ein (das Skript listet alle Treffer inklusive Altbestand);
- `node tools/compliance_check.mjs` besteht 9/9.

Ausführlichere Hinweise zu Architektur und Erweiterung finden sich in der `开发者指南.md` nach dem Entpacken.

---

## Lizenz

Dieses Projekt wird unter der **Apache License 2.0** veröffentlicht. Details siehe Datei [`LICENSE`](../LICENSE).

> Compliance-Hinweis: Apache-2.0 verlangt die Beibehaltung von Copyright- und Lizenzhinweisen; bei einer Weitergabe auf Basis dieser Engine bitte die Zuordnungsanzeigen und das Asset-Register unter `Compliance/` beibehalten.
