[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch** | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> Ein abhängigkeitsfreier, deterministischer Media-Engine in Rust — verteilt als **einzelne TXT-Datei**: eine 18 MB große Textdatei herunterladen, das Entpack-Skript ausführen und schon liegt der vollständige Quellbaum vor (28 Crates) — ohne npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3929%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## Was ist das

Prism Engine ist ein in Rust geschriebener Media-Engine ohne Abhängigkeiten von Fremdcode: Bilder lesen, Satz setzen, Bilder zeichnen, Timelines schneiden, fertige Filme ausgeben — alles in einer einzigen deterministischen Engine. Alle 110 Laufzeitabhängigkeiten
(dazu 20 Testabhängigkeiten) sind `path`-Abhängigkeiten dieses Repositorys — kein Netzwerk, kein Paketmanager,
kein Supply-Chain-Risiko.

Verteilt wird als **Single-File-Quellcode**: die gesamte Engine (642 Dateien, 28 Crates) ist in eine einzige UTF-8-Textdatei
`prism-engine-v2.1-source.txt` zusammengeführt. Wer diese Datei besitzt, kann daraus lokal einen Verzeichnisbaum
rekonstruieren, der dem Repository exakt entspricht.

---

## Was die Engine kann

Überblick in einem Satz: **Ein Bild ist die Ansicht einer Szene zu einem bestimmten Zeitpunkt, ein Video ist eine Folge solcher Ansichten über einen Zeitraum** —
dieselbe Szenenbeschreibung ergibt, an einem Moment festgehalten, ein Einzelbild, und kontinuierlich fortgeschrieben und aufgezeichnet ein Video; statisches und bewegtes Bild teilen sich denselben Funktionsumfang.
Im Folgenden fünf Blöcke entlang dessen, was man tun will: **einlesen → anordnen → zeichnen → ausliefern → in die eigene Toolchain einbinden**.

### Was wird gelesen (Eingabe)

- **Einheitliche Decoder-Fassade**: ein Einstiegspunkt `decode_image(bytes, ext)` verteilt statische Bilder — PNG (nicht interlaced, 8/16 Bit),
  Baseline-JPEG (Fixed-Point-IDCT, plattformübergreifend bitgenau deterministisch), **Magic Number zuerst, Dateiendung als Rückfallebene** (auch bei falscher Endung dekodierbar);
  Formate außerhalb der Matrix oder mit Bedarf an externer Zusammenarbeit führen stets zu einem dreiteiligen Fehler (wo / was / wie beheben) samt Empfehlung zur ffmpeg-Zusammenarbeit,
  **niemals stilles Scheitern**.
- **Animierte Bilder / Audio / Container**: GIF, APNG native Dekodierung Frame für Frame (`prism-media`); WAV integriert dekodiert, mp3/aac/FLAC
  über den ffmpeg-Probe-Pfad (`prism-audio`); MP4-MJPEG Rücklesen (`decode_mp4_video`).
- **EXR komplett**: Lesen und Schreiben, Kompression, DWA, und **Bitstrom-Abgleich über Implementierungsgrenzen** mit dem offiziellen OpenEXR (nicht Selbstschreiben und Selbstlesen).
- **Kamera-RAW**: Baseline-RAW-Dekodierung. **Grenze**: `Compression=7` (lossless JPEG) echter DNGs ist nicht implementiert,
  von echten Kameras erzeugte DNGs lassen sich nicht lesen — das steht unter „Ehrliche Grenzen“ und wird nicht versteckt.
- **3D-Assets**: glTF / OBJ-Import, USD (`.usda`)-Parsing und MaterialX-Parameterpakete, Meshes gelangen in die Render-Szene.
- **Externe Schriften**: `load_font_file` lädt TTF/OTF von Drittanbietern direkt, die Fallback-Kette wird stufenweise aufgelöst,
  fehlende Glyphen ergeben eine klare Diagnose statt Tofu-Kästchen.

### Was wird angeordnet (Autorensyntax)

- **Vollständige Block-Syntax von DSL v2**: ein Satz `.prism` beschreibt „was gewünscht ist, wie es angeordnet wird, wie es sich bewegt, wie es aussieht,
  wohin es exportiert wird“; das Kompilat ist ein Node-Graph + Property-Kanäle + Timeline.
- **Fünf Kombinatoren**: `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` — High-Level-Kombinatoren expandieren zu Node-Graphen; „ein 90-Sekunden-Film“ sinkt vom handgeschriebenen Szenencode auf bloßes Zusammensetzen.
- **Übergänge und Morph**: Fade / Wipe / Dissolve / Push voll parametrisiert (Dauer, Easing, Richtung),
  Morph als Verformung von Polylinien, Szenensequenzen zu jedem Zeitpunkt abtastbar.
- **Formelsatz**: `math_ast` + `math_layout` setzen Hoch- und Tiefstellung, Brüche, Wurzeln, große Operatoren,
  Matrizen; der DSL-Block `formula` setzt direkt, auf derselben Grundlinie wie der Fließtext.

### Was wird gezeichnet (Rendering und Simulation)

- **Offline-Rendering in Fotoqualität**: Path Tracing + physikalisch korrekte BSDF (diffus / spiegelnd / Glas / Metall),
  Glas und Metall über stochastische Inversion (Monte-Carlo über die Neumann-Reihe), Unterstützung für volumetrische Medien und Subsurface Scattering.
- **Vier Familien von Kurven-Primitiven**: Zylinder / Kegel / Torus / Rotationsfläche, alle analytisch geschnitten (Gleichung vierten Grades über
  die geschlossene Lösung nach Ferrari, exakte Wurzeln); beliebige Dreiecks-Meshes + SAH-BVH, Parity-Abgleich über zwei Pfade.
- **GPU compute**: geräteunabhängige Compute-Semantikschicht + Vulkan-Dispatch + handgeschriebenes minimales SPIR-V;
  unter der Mesa-lavapipe-Software-ICD vollständig lauffähig auch auf Plattformen ohne GPU (headless verifizierbar).
- **Physik und Partikel**: einheitlicher XPBD-Solver (starre Körper / Soft Bodies / Cloth / Gelenke / Fluide / Bruch),
  Rauch über FLIP, Partikelsystem (AudioReactive an die Audio-Hüllkurve gekoppelt).
- **Generative Kunst und Post-Processing**: Strömungsfeld / L-Systeme / zelluläre Automaten / Reaktions-Diffusion / Fraktale / Voronoi;
  blur / bloom / halation / Filmkorn / vier NPR-Stile (toon / watercolor / halftone / oil) / 3-in-1-Entrauschung.
- **Color Grading und Vektoren**: sekundäres Grading (HSL-Auswahl + RGB-Kurven), Stil-Presets cinematic / film / anime und
  drei Qualitätsstufen Entwurf/Standard/Film (Qualität ist ein Parameter, keine Esoterik); Rasterung von Vektorpfaden, Pfad-Draw-on und Morph.
- **Audiobearbeitung**: Loudness-Normalisierung (loudnorm), Beat-Erkennung, Spektralanalyse, Hüllkurvenextraktion —
  koppelbar mit dem Bild (Partikel AudioReactive folgen der Audio-Hüllkurve).
- **Das CPU-Referenz-Backend ist vollwertig**: Maschinen ohne GPU erzeugen trotzdem korrekte Bilder und dienen zugleich als Referenz für die Korrektheit der GPU.

### Was wird ausgeliefert (Delivery)

- **`render_movie` direkt**: Das Kompilat von DSL v2 durchläuft in einem Aufruf „Frame-Auswertung → tile-Rendering →
  Delivery“; ein einzelnes `.prism` ergibt mit einem Befehl den Film (`prism-app render demo.prism`); nicht abgedeckte Nodes melden ehrlich einen Fehler, statt stillschweigend zu approximieren.
- **Geschlossener Vorschau-Kreislauf**: `prism-app preview` startet den Vorschau-Dienst; der Editor bietet zwei Vorschau-Stufen — Entwurf (geometrische Muster, Bildrate bleibt erhalten) /
  Real (echtes Rendering mit niedriger spp, asynchrones Refresh).
- **Eigener Encoder + Ratensteuerung**: I/P-Frame-Zeitkodierer im JPEG-Bereich über den privaten `PRMV`-Container (zwei Schema-Generationen
  v2/v3), Water-Filling-Ratensteuerung + qpel-Subpixel-Bewegungsschätzung (gemessen PSNR +2,577 dB,
  Volumen −21,6 %).
- **ffmpeg-Kooperationskette**: mit ffmpeg → über stdin-rawvideo-Pipe nach H.264/HEVC;
  ohne ffmpeg → automatischer Rückfall auf MJPEG, **die Engine liefert immer ein Ergebnis**; es werden keinerlei Binärdateien gebündelt.
- **EXR-Sequenz-Delivery**: Frame-Sequenzen werden direkt als EXR geschrieben und gehen so in die Postproduktions-Pipeline.
- **Verteilte Farm**: Verteilung auf Frame-Ebene + N-Prozess-Worker + deterministische Zusammenführung — parallele MD5 == serielle MD5,
  inklusive Fortsetzen nach Unterbrechung.

### Wie die Integration aussieht (Ökosystem)

- **MCP-Tool-Oberfläche**: stdio-JSON-RPC mit drei Tools — `scene_check` (Kompilieren + Testrendern + Frame-Schätzung),
  `render_submit` (deterministische job id + MD5-Digest pro Frame), `diagnose` (dreiteilige Fehlerlokalisierung).
- **Node-SDK**: Aushandlung semantischer Versionen (inklusive Sonderfall 0.x), Beispiel-Node-Paket beigelegt
  (`examples/sdk-nodes/cinematic-basic/`, kompilierbar / ladbar / verhandelbar).
- **Gerüst zur Erkennung von Hardware-Encodern**: `prism-ffi-hwenc` Erkennungskette für drei Plattformen (VideoToolbox / NVENC / VAAPI)
  + feature-gate-Matrix, bei nicht passender Plattform eine klare Meldung „nicht unterstützt“.

> Detaillierte Fähigkeitsgrenzen (was **nicht** geht) stehen nach dem Entpacken in `开发者指南.md` im Abschnitt „Ehrliche Grenzen“ —
> dort ist der aktuelle reale Stand notiert (Kurven-Primitive laufen nur auf der CPU, Beschleunigungsverhältnisse der GPU müssen auf der
> Zielhardware gemessen werden, Hardware-Encoding wird je nach Zielplattform angebunden …); dieser Text gibt nur den Überblick und übertreibt nicht.

---

---

## Engine-Interna (jenseits der Funktionsliste)

Manche Fähigkeiten passen nicht in die Liste „Lesen / Anordnen / Zeichnen / Ausliefern / Integrieren" — sie sind die tieferen Assets der Engine. Dieses Kapitel hält die Hauptliste übersichtlich.

### Audio: ein eingebauter DAW-Kern

`prism-audio` ist kein Randmodul, das „WAV lesen kann", sondern eine vollständige Kette vom Komponieren bis zum Mixing; jede Funktion ist eine reine Funktion ihrer Parameter (gleiche Eingabe ⇒ plattformübergreifend bit-identisch):

- **Synthese**: PolyBLEP-aliasingfreie Oszillatorfamilie + geschlossene ADSR-Hüllkurven + 4 Stimmungs-Presets;
- **Musiktheorie**: Skalenbibliothek (Dur/Moll, pentatonisch, Kirchentonarten), 12-TET Tonhöhe↔Frequenz, ein Zustandsautomat funktionaler Grammatik für Akkordfolgen + Stimmführung;
- **Sequencing**: Arpeggiator (Muster/Rate/Oktave), Step-Drum-Machine, Takt-/Beat-Raster-Partitur;
- **Material**: handgeschriebener SMF-Parser (MIDI-Import), SFZ-Sampler, Granularsynthese;
- **Sounddesign**: physikalische Modellierung, räumliches Panning, FX-Kette;
- **Mixing**: linearer Mehrspur-Mix + Fades + Peak-Begrenzung;
- **Analyse & Kopplung**: Beat-/Onset-Erkennung, FFT-Spektrum, Hüllkurvenextraktion, AudioReactive-Backen (Frame N liest ein festes Fenster; reine Funktion pro Frame, seek-äquivalent), Loudness-Normalisierung.

> Grenze: AAC/mp3 werden nicht selbst entwickelt (Patenthistorie + Aufwand); allgemeines Audio läuft über externes ffmpeg.

### Definierbare Kernel-Bibliothek (Registry der Speziallösungen)

Jeder Knotentyp ist eine **Speziallösung** eines Tripels „Kernel + Gitter + Dualität". Dritte (oder Sie) können mit `registry!` neue Lösungen registrieren, ohne den Engine-Kern zu ändern:

- Statische Einsammlung (das Makro expandiert im definierenden Crate; kein `inventory`/Proc-Macro, null Abhängigkeiten);
- **Drei-Fragen-Prüfung** bei der Registrierung (Kernform / Gitter / Dualität): unvollständige Antworten werden mit einem dreiteiligen Fehler abgelehnt;
- Offizielle und eigene Lösungen teilen dieselbe Registry — zur Laufzeit kein Unterschied;
- Mathematisch ist das die Operatorgleichung `f = g + Kf` mit sechs Inversionsstrategien — exakt / stochastisch / lokal / regularisiert / iterativ / spektral. Eine Fähigkeit hinzuzufügen heißt, ein Tripel zu deklarieren, nicht die Render-Schleife zu patchen.

### Infrastruktur für Determinismus und Reproduzierbarkeit

- **Signaturabschluss**: Knotengraphen / Cache-Schlüssel / Snapshot-Schlüssel / Frame-Signaturen sind ein „vierdimensionaler Konfigurations-Hash"; eine Parameteränderung ändert den Schlüssel, nichts wird falsch wiederverwendet; iterative Knoten schlüsseln auf `(ε, n)`, streamende Knoten auf die Frame-Nummer;
- **Byte-Fixpunkte**: USD Export→Import→Export ist byte-identisch; WAV 16/24/32 Bit läuft exakt zurück; EXR wird bitweise gegen die offizielle Implementierung verglichen;
- **Deterministisches Einzeldatei-Packaging**: derselbe Quellstand ergibt zweimal dieselbe SHA-256; der Entpack-Roundtrip vergleicht jede Datei Byte für Byte.

### GPU-Backends (ehrlich getrennt)

- **Vulkan**: Compute- und Raytracing-Szenen-Pipeline laufen wirklich (auf dem lavapipe-Software-ICD auch ohne GPU prüfbar); die `render_scene`-Ausgabe ist **byte-identisch zur CPU-Referenz** (0 abweichende Pixel); `is_available()` verlangt Compute- und Render-Selbsttest;
- **Metal**: isomorphe Portierung + Compile-Gating fertig, **nicht auf Hardware verifiziert, nicht aktiviert** (`is_available()` immer false);
- **WebGPU**: nur Machbarkeitsaussage (4–6 Personentage), kein Renderpfad.

### Anti-False-Green-Gates

Vier lokale Gate-Skripte (null Abhängigkeiten, nur `node`): Ghost-Write-Schutz (mit Gegenprobe), Open-Source-Hygiene-Scan, Compliance-Selbstprüfung (9/9, Gegenprobe 8/8 rot), Packaging-Roundtrip. Ein Gate kennt nur „bestanden" oder „rot" — „sieht gut aus" zählt nicht. **Falsches Grün ist gefährlicher als ein Fehlschlag.**

---

## Warum Single-File-Verteilung

- **Auditierbar**: eine Textdatei, für Menschen lesbar, `diff`-fähig, per Hash prüfbar — Veröffentlichung heißt Einfrieren.
- **Reproduzierbar**: Das Packen ist deterministisch — gleiche Eingabe, zweimal gepackt, ergibt exakt identische SHA-256.
- **Portabel**: kein Klonen eines großen Repositorys nötig, eine Datei lässt sich per E-Mail oder Chat übertragen und auch offline wiederherstellen und bauen.
- **Zum Trade-off**: Die Single-File-Verteilung tauscht den Komfort „Code direkt auf GitHub durchsuchen“ gegen die Determinismus-Garantie „eine Datei trägt alles, und zwar byteweise verifizierbar“. Nach dem Entpacken erhält man einen gewöhnlichen Cargo workspace, nicht anders als ein übliches Rust-Projekt.

---

## Schnellstart

Voraussetzung: [Rust](https://www.rust-lang.org/) (mit `cargo`) und [Node.js](https://nodejs.org/) 18+ installieren.

```bash
# 1. die prism-engine-v2.1-source.txt aus dem Repository herunterladen und als lokale Datei speichern
# 2. in einem Texteditor öffnen und das Node.js-Skript zwischen "UNPACK SCRIPT BEGIN" und "UNPACK SCRIPT END"
#    als unpack.cjs im selben Verzeichnis speichern
# 3. Entpacken ausführen (wenige Sekunden, stellt 642 Dateien wieder her)
node unpack.cjs
# 4. im wiederhergestellten Verzeichnis (also der workspace-Wurzel) die vollständigen Tests laufen lassen
cargo test --workspace --release
# Erwartet: 3929 passed / 0 failed
```

Nach dem Entpacken liegt ein standardmäßiger Cargo workspace vor (28 Crates unter `src/`); alle weiteren `cargo`-Befehle sind wie bei einem üblichen Rust-Projekt.

> wasm-Verifikationskette für Mobilgeräte: `rustup target add wasm32-unknown-unknown`, dann `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Repository-Struktur (nach dem Entpacken)

| Pfad | Beschreibung |
|------|------|
| `src/` | Quellcode der **28 Crates** (im Repository 29, ursprünglich `crates/`, bei der Verteilung zu `src/` umstrukturiert) |
| `Compliance/` | Compliance-Erklärungen: Verifikation der Abhängigkeitsfreiheit / Drittanbieter-Erklärungen / Asset-Register (im Repository ursprünglich `docs/合规/`) |
| `开发者指南.md` | In sich geschlossene Entwicklerdokumentation: Einstieg / Architektur / mathematischer Kern / Gates / Erweiterung (im Repository ursprünglich `docs/开发者指南.md`) |
| `tools/` | Gate-Skripte (Schutz vor Ghost-Writes, Hygiene-Scan, Compliance-Selbstprüfung) |
| `.github/workflows/` | CI-Konfiguration |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | zum Bauen erforderliche Teile (workspace members zeigen bereits auf `src/*`) |
| `examples/` | Beispiele für `prism-dsl`-Tests und Beispiel-Node-Pakete (per `include_str!` gelesen, fehlen sie, schlägt die Kompilierung fehl) |

---

## Modulübersicht

Die **28 Crates** im Distributionspaket, gruppiert nach Zuständigkeit:

| Kategorie | crate |
|------|-------|
| Kern | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Media-I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendering | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Generierung | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Text/UI | prism-text · prism-rt · prism-app |
| Fortgeschritten | prism-ai · prism-dsl · prism-infra · prism-author |
| Bypass (feature gate) | prism-ffi-hwenc |

---

## Bauen und Testen

```bash
# Smoke-Test über das ganze Repository (release, empfohlen)
cargo test --workspace --release
# gleichwertiger Alias
cargo xtest
# Schnellverifikation eines einzelnen crate
cargo test -p prism-core --release
```

- Vollständige Tests: **3929 passed / 0 failed / 0 ignored** (Release-Lauf im Repository: 179 Test-Suiten, 171,5 s auf allen Kernen; das Paket ist datei- und byte-identisch zum Repository — nur die Demo-Crate ist ausgenommen)
- CI: **6 jobs** — Schutz vor Ghost-Writes (ghost-guard) / Kompilieren + vollständige Tests (native) / wasm32-Build (wasm) /
  WGSL-Statikprüfung (wasm-verify) / Bitstrom-Abgleich über Implementierungsgrenzen mit dem offiziellen OpenEXR (interop) / Kreuzabgleich des node-Hosts (lint).
- Lokale Gate-Skripte: 3 (vollständig abhängigkeitsfrei, nur `node`, siehe unten) + 1 Packskript;
  davon Compliance-Selbstprüfung **9/9 bestanden**, Gegenselbstnachweis **8/8 rot bewertet**.

Lokale Ausführung der Gates (Node 18+ erforderlich):

```bash
node tools/ghost_guard.mjs --check        # Schutz vor Ghost-Writes: G1 Doppeldefinition/G2 echte Obermenge/G3 Dokumente am Zeilenanfang/G4 Kompilierung
node tools/hygiene_scan.mjs               # Open-Source-Hygiene: listet verbliebene interne Formulierungen einzeln auf (Datei:Zeilennummer), bei Treffer exit 1
node tools/compliance_check.mjs           # Compliance-Selbstprüfung: sollte 9/9 bestehen
node tools/compliance_check.mjs --self-test   # Gegenselbstnachweis: sollte rot bewerten und so die Trennschärfe der Kriterien belegen
```

> **Der Open-Source-Hygiene-Scan hat derzeit noch Altlast-Treffer**: Im Repository verbleiben frühe interne Formulierungen, wobei die
> **mit dem Paket ausgelieferte Dokumentationsebene bereits auf null steht** (README / Entwicklerhandbuch / Compliance); die übrigen liegen in **Quellcode-Kommentaren innerhalb des Distributionspakets**
> und in **nicht mit dem Paket ausgelieferten Repository-Dateien**, wurden nach Tatsachen klassifiziert und als eigene Aufräumposten geführt. Das Skript existiert genau dafür, die
> Reste **sichtbar und einzeln abarbeitbar** zu machen, statt Sauberkeit vorzutäuschen — bitte verstehen Sie „Trefferzahl ungleich 0“ nicht als Versagen des Gates.

---

## Compliance

Die Engine verspricht **null Abhängigkeiten von Drittanbieter-Code** und verifiziert dies durch Selbstprüfung:

- **C1 Null Abhängigkeiten von Drittanbieter-Code**: alle 130 Abhängigkeiten (110 Laufzeit + 20 Test) sind `path`-Abhängigkeiten dieses Repositorys; kein `build.rs`, keine nativen Link-Anweisungen, `Cargo.lock` sperrt 29 Pakete dieses Repositorys.
- **C2 vollständiges Asset-Register**: alle Hauptfelder der 5 Assets sind vollständig, der Stand der Eigenentwicklung liegt im kontrollierten Vokabular; der vollständige Lizenztext der Schriftlizenz OFL 1.1 liegt bei der Schrift unter `src/prism-text/font_data/LICENSE`.
- **C3 Drittanbieter-Attribution**: `Compliance/THIRD-PARTY-NOTICES.md` ist vorhanden und unterscheidet ausdrücklich „Datenassets ≠ Codeabhängigkeiten“.

> Hinweis: Dieses Repository verteilt **Quellcode**. Die Schrift-Assets (Prism Sans SC Subset) stehen unter SIL OFL 1.1, **der vollständige Lizenztext liegt bei der Schrift unter** `src/prism-text/font_data/LICENSE`; Asset-Herkunft und Lizenzregistrierung siehe `Compliance/assets/资产登记.md`.


---

## Lizenz

Dieses Projekt wird unter der **Apache License 2.0** veröffentlicht. Details siehe Datei [`LICENSE`](../LICENSE).

> Compliance-Hinweis: Apache-2.0 verlangt die Beibehaltung von Urheberrechts- und Lizenzhinweisen; bei einer Weitergabe auf Basis dieser Engine bitte die Attributionshinweise und das Asset-Register unter `Compliance/` beibehalten.

**Hinweis: Sollten andere Fassungen Übersetzungsfehler enthalten, gilt diese vereinfachte chinesische Fassung.**
