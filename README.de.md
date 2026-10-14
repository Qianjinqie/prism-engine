<p align="right">[English](README.en.md) | [简体中文](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch** | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)</p>

# Prism Engine

> Zero-Dependency- und deterministische Rust-Medien-Engine — verteilt als **einzelne TXT-Datei**: Lade eine 14-MB-Textdatei herunter, führe das Entpackskript aus und erhalte den vollständigen 26-Crate-Quellbaum — ganz ohne npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## Was ist das

Prism Engine ist eine in Rust geschriebene Medien-Engine ohne Drittanbieter-Codeabhängigkeiten, die Module für Bilder, Audio, Geometrie, Physik, Rendering, DSL und mehr umfasst. Alle 103 Abhängigkeiten sind `path`-Abhängigkeiten innerhalb dieses Repos — unabhängig vom Netzwerk, ohne Paketmanager, ohne Supply-Chain-Risiko.

Verwendet wird eine **Single-File-Source-Distribution**: Die gesamte Engine (481 Dateien, 26 Crates) ist in einer einzigen UTF-8-Textdatei `prism-engine-v1.0-source.txt` zusammengeführt. Wer diese Datei besitzt, kann lokal einen Verzeichnisbaum wiederherstellen, der exakt dem Repository entspricht.

## Warum Single-File-Distribution

- **Auditierbar**: Eine einzige Textdatei — mit bloßem Auge lesbar, per `diff` vergleichbar und per Hash verifizierbar; sie ist mit der Veröffentlichung eingefroren.
- **Reproduzierbar**: Die Paketierung ist deterministisch — zwei Paketierungen mit identischem Input ergeben exakt denselben SHA-256-Hash.
- **Portabel**: Kein Clone eines riesigen Repositorys nötig — eine einzige Datei lässt sich per E-Mail oder Chat übermitteln, und der Build lässt sich auch offline wiederherstellen.
- **Abwägung**: Die Single-File-Distribution nimmt den Verzicht auf den Komfort von „den Code direkt auf GitHub durchstöbern“ in Kauf und erhält im Gegenzug die Deterministik von „alles in einer einzigen Datei mitnehmen und Byte für Byte verifizieren können“. Nach dem Entpacken erhält man einen gewöhnlichen Cargo-Workspace, der sich nicht von einem üblichen Rust-Projekt unterscheidet.
- **Das Wichtigste**: Ich muss nur eine einzige Datei hochladen (doge

---

## Schnellstart

Voraussetzungen: Installiere [Rust](https://www.rust-lang.org/) (inklusive `cargo`) und [Node.js](https://nodejs.org/) 18+.

```bash
# 1. prism-engine-v1.0-source.txt aus dem Repository herunterladen und als lokale Datei speichern
# 2. Mit einem Texteditor öffnen und das Node.js-Skript zwischen "UNPACK SCRIPT BEGIN" und "UNPACK SCRIPT END"
#    als unpack.cjs im selben Verzeichnis speichern
# 3. Entpacken ausführen (dauert nur wenige Sekunden, stellt 481 Dateien wieder her)
node unpack.cjs
# 4. In den wiederhergestellten Quellbaum wechseln und die vollständige Testsuite ausführen
cd src/..
cargo test --workspace --release
# Erwartet: 3271 passed / 0 failed
```

Nach dem Entpacken erhält man einen Standard-Cargo-Workspace (26 Crates unter `src/`); alle weiteren `cargo`-Befehle sind dieselben wie bei einem üblichen Rust-Projekt.

> Wasm-Verifikationskette auf Mobilgeräten: `rustup target add wasm32-unknown-unknown`, danach `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Repository-Struktur (nach dem Entpacken)

| Pfad | Beschreibung |
|------|------|
| `src/` | Quellcode der 26 Crates (im Repository ursprünglich `crates/`, für die Distribution zu `src/` umgruppiert) |
| `Compliance/` | Compliance-Erklärungen: Zero-Dependency-Verifikation / Drittanbieter-Erklärung / Asset-Registrierung (im Repository ursprünglich `docs/合规/`) |
| `开发者指南.md` | In sich geschlossene Entwicklerdokumentation: Einstieg / Architektur / Mathematikkern / Gates / Erweiterungen (im Repository ursprünglich `docs/开发者指南.md`) |
| `tools/` | Gate-Skripte (Geister-Schreibschutz, Hygiene-Scan, Compliance-Selbsttest) |
| `.github/workflows/` | CI-Konfiguration |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Für den Build erforderliche Dateien (workspace members zeigen bereits auf `src/*`) |
| `examples/` | Testbeispiele für `prism-dsl` (per `include_str!` eingelesen; ohne sie schlägt die Kompilierung fehl) |

> Die maßgeblichen Design-Dokumente (`docs/设计/`) werden **nicht mit ausgeliefert** und verbleiben ausschließlich im vollständigen Repository des Herausgebers.

---

## Modulübersicht

Die 26 Crates sind nach Zuständigkeitsbereichen gruppiert:

| Kategorie | crate |
|------|-------|
| Kern | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Medien-I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendering | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Generierung | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Text/UI | prism-text · prism-rt · prism-app |
| Erweitert | prism-ai · prism-dsl · prism-infra |

---

## Build und Tests

```bash
# Smoke-Test über das gesamte Repository (release, empfohlen)
cargo test --workspace --release
# Äquivalenter Alias
cargo xtest
# Schnellverifikation eines einzelnen Crates
cargo test -p prism-core --release
```

- Vollständige Testsuite: **3271 passed / 0 failed** (auf einem frisch entpackten Baum bereits real ausgeführt und verifiziert).
- Gates: 14, darunter Geister-Schreibschutz (G1~G4), Open-Source-Hygiene-Scan (0 interne Spuren), Compliance-Selbsttest (9/9), Compliance-Gegenprobe (8/8, Verstöße schlagen garantiert an) und die wasm32-Build-Kette.

Gates lokal ausführen (erfordert Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Geister-Schreibschutz: G1 doppelte Definitionen/G2 strikte Obermenge/G3 Kopfdokumentation/G4 Kompilierung
node tools/hygiene_scan.mjs               # Open-Source-Hygiene: sollte „sauber (0 interne Spuren)“ melden
node tools/compliance_check.mjs           # Compliance-Selbsttest: sollte 9/9 bestehen
node tools/compliance_check.mjs --self-test   # Gegenprobe: sollte rot schlagen und damit beweisen, dass die Kriterien Diskriminierungskraft besitzen
```

---

## Compliance

Die Engine garantiert **null Drittanbieter-Codeabhängigkeiten** und hat dies durch Selbsttests verifiziert:

- **C1 Null Drittanbieter-Codeabhängigkeiten**: Alle 103 Abhängigkeiten sind `path`-Abhängigkeiten innerhalb dieses Repos; kein `build.rs`, keine nativen Linker-Anweisungen, und `Cargo.lock` fixiert die 26 Pakete dieses Repos.
- **C2 Vollständige Asset-Registrierung**: Die 5 Einträge der Asset-Haupttabelle weisen vollständig ausgefüllte Felder auf, der In-house-Entwicklungsstatus ist auf ein kontrolliertes Vokabular abgebildet; der vollständige Text der OFL-1.1-Fontlizenz liegt bei der Schrift unter `src/prism-text/font_data/LICENSE`.
- **C3 Drittanbieter-Zuschreibungserklärung**: `Compliance/THIRD-PARTY-NOTICES.md` ist vorhanden und unterscheidet ausdrücklich zwischen „Daten-Assets ≠ Codeabhängigkeiten“.

> Hinweis: Dieses Repository verteilt **Quellcode**. Die Font-Assets (Prism Sans SC Subset) stehen unter der SIL-OFL-1.1-Lizenz, **der vollständige Lizenztext liegt bei der Schrift unter** `src/prism-text/font_data/LICENSE`; Asset-Herkunft und Lizenzregistrierung siehe `Compliance/assets/资产登记.md`.

---

## Lizenz

Dieses Projekt wird unter der **Apache License 2.0** veröffentlicht. Details siehe Datei [`LICENSE`](LICENSE).

> Compliance-Hinweis: Apache-2.0 verlangt, Copyright- und Lizenzangaben zu bewahren; bei einer erneuten Distribution auf Basis dieser Engine bitte die Zuschreibungserklärungen und die Asset-Registrierung unter `Compliance/` beibehalten.
