[English](README.en.md) | [简体中文](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> Moteur multimédia Rust déterministe et sans dépendance — distribué sous forme de **fichier TXT unique** : téléchargez un texte de 14 Mo, exécutez le script de décompression, et vous obtenez l'arborescence source complète des 26 crates, sans npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## De quoi s'agit-il

Prism Engine est un moteur multimédia écrit en Rust et dépourvu de toute dépendance de code tiers, couvrant des modules tels que les images, l'audio, la géométrie, la physique, le rendu et le DSL. Les 103 dépendances sont toutes des dépendances `path` de ce dépôt — pas de réseau, pas de gestionnaire de paquets, aucun risque de supply chain.

La distribution s'effectue sous forme de **code source en fichier unique** : l'ensemble du moteur (481 fichiers, 26 crates) est fusionné dans un texte UTF-8 unique, `prism-engine-v1.0-source.txt`. Il suffit de détenir ce fichier pour reconstituer localement une arborescence identique à celle du dépôt.

## Pourquoi une distribution en fichier unique

- **Auditable** : un fichier texte, lisible à l'œil nu, comparable avec `diff`, vérifiable par hachage ; figé dès la publication.
- **Reproductible** : l'empaquetage est déterministe — le SHA-256 de deux empaquetages effectués avec les mêmes entrées est strictement identique.
- **Portable** : inutile de cloner un dépôt volumineux ; un seul fichier se transmet par e-mail ou par messagerie, et la construction peut être reconstituée hors ligne.
- **Compromis** : la distribution en fichier unique sacrifie la commodité de « parcourir directement le code sur GitHub » en échange de la garantie déterministe de « tout emporter dans un seul fichier, vérifiable octet par octet ». Après décompression, on obtient un Cargo workspace ordinaire, identique à tout projet Rust classique.
- **Le plus important** : il me suffit de téléverser un seul fichier et le tour est joué (doge

---

## Démarrage rapide

Prérequis : installer [Rust](https://www.rust-lang.org/) (avec `cargo`) et [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Téléchargez le fichier prism-engine-v1.0-source.txt présent dans le dépôt et enregistrez-le en local
# 2. Ouvrez-le avec un éditeur de texte, puis enregistrez le script Node.js situé entre
#    "UNPACK SCRIPT BEGIN" et "UNPACK SCRIPT END" dans un fichier unpack.cjs du même répertoire
# 3. Exécutez la décompression (quelques secondes, restaure 481 fichiers)
node unpack.cjs
# 4. Placez-vous dans l'arborescence source restaurée et lancez la suite complète de tests
cd src/..
cargo test --workspace --release
# Résultat attendu : 3271 passed / 0 failed
```

Après décompression, vous obtenez un Cargo workspace standard (26 crates sous `src/`), et les commandes `cargo` ultérieures sont les mêmes que pour un projet Rust classique.

> Chaîne de vérification wasm sur mobile : `rustup target add wasm32-unknown-unknown`, puis `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Structure du dépôt (après décompression)

| Chemin | Description |
|------|------|
| `src/` | Code source des 26 crates (nommé à l'origine `crates/` dans le dépôt, réorganisé en `src/` lors de la distribution) |
| `Compliance/` | Déclarations de conformité : vérification zéro dépendance / déclarations tiers / registre des actifs (nommé à l'origine `docs/合规/` dans le dépôt) |
| `开发者指南.md` | Documentation développeur autonome : prise en main / architecture / noyau mathématique / gates / extensions (nommé à l'origine `docs/开发者指南.md` dans le dépôt) |
| `tools/` | Scripts de gate (défense anti-écriture fantôme, scan d'hygiène, autocontrôle de conformité) |
| `.github/workflows/` | Configuration CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Éléments indispensables à la construction (les workspace members pointent déjà vers `src/*`) |
| `examples/` | Exemples utilisés par les tests de `prism-dsl` (lus via `include_str!`, la compilation échoue s'ils manquent) |

> La documentation de référence du design (`docs/设计/`) **n'est pas incluse dans la distribution** ; elle demeure dans le dépôt complet du diffuseur.

---

## Aperçu des modules

Les 26 crates sont regroupés par responsabilité :

| Catégorie | crate |
|------|-------|
| Noyau | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Média I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendu | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Génération | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texte/UI | prism-text · prism-rt · prism-app |
| Avancé | prism-ai · prism-dsl · prism-infra |

---

## Construction et tests

```bash
# Smoke test de tout le dépôt (release, recommandé)
cargo test --workspace --release
# Alias équivalent
cargo xtest
# Vérification rapide d'un seul crate
cargo test -p prism-core --release
```

- Suite complète de tests : **3271 passed / 0 failed** (validée par exécution réelle sur une arborescence issue d'une décompression fraîche).
- Gates : 14 au total, comprenant la défense anti-écriture fantôme (G1~G4), le scan d'hygiène open source (0 trace interne), l'autocontrôle de conformité (9/9), l'auto-vérification inversée de conformité (8/8, toute infraction passe au rouge) et la chaîne de construction wasm32.

Exécution locale des gates (Node 18+ requis) :

```bash
node tools/ghost_guard.mjs --check        # Défense anti-écriture fantôme : G1 définitions dupliquées / G2 sur-ensemble strict / G3 documentation en tête de fichier / G4 compilation
node tools/hygiene_scan.mjs               # Hygiène open source : doit rapporter « propre (0 trace interne) »
node tools/compliance_check.mjs           # Autocontrôle de conformité : doit passer 9/9
node tools/compliance_check.mjs --self-test   # Auto-vérification inversée : doit être jugée rouge, ce qui prouve que les critères sont discriminants
```

---

## Conformité

Le moteur garantit le **zéro dépendance de code tiers**, vérifié par autocontrôle :

- **C1 Zéro dépendance de code tiers** : les 103 dépendances sont toutes des dépendances `path` de ce dépôt ; pas de `build.rs`, pas d'instruction de liaison native, `Cargo.lock` verrouille les 26 paquets de ce dépôt.
- **C2 Registre des actifs complet** : les 5 entrées du tableau principal des actifs ont tous les champs renseignés, et les statuts de développement interne respectent un vocabulaire contrôlé ; le texte intégral de la licence de police OFL 1.1 est placé avec la police dans `src/prism-text/font_data/LICENSE`.
- **C3 Déclarations d'attribution tierces** : `Compliance/THIRD-PARTY-NOTICES.md` est présent et distingue explicitement « actifs de données ≠ dépendances de code ».

> Remarque : ce dépôt distribue du **code source**. Les actifs de polices (sous-ensemble Prism Sans SC) sont sous licence SIL OFL 1.1, **le texte intégral de la licence étant placé avec la police** dans `src/prism-text/font_data/LICENSE` ; les sources des actifs et le registre des licences figurent dans `Compliance/assets/资产登记.md`.

---

## Licence

Ce projet est publié sous **Apache License 2.0**. Voir le fichier [`LICENSE`](LICENSE).

> Rappel de conformité : Apache-2.0 impose de conserver les mentions de copyright et de licence ; en cas de redistribution basée sur ce moteur, veuillez conserver les déclarations d'attribution et le registre des actifs sous `Compliance/`.
