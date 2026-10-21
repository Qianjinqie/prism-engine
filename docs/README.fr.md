[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> Un moteur multimédia Rust déterministe et sans dépendances — distribué sous forme de **fichier TXT unique** : téléchargez un texte de 17 Mo, exécutez le script de décompression et vous obtenez l'arborescence source complète (28 crates), sans npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

(Les badges sont des badges factuels statiques, la définition des chiffres est donnée plus bas ; `crates-28` = le nombre de crates **contenus dans le paquet de distribution**, le dépôt contient en outre 1 crate de démonstration non inclus dans le paquet, soit 29 crates dans le dépôt. Ils pourront être remplacés par des badges dynamiques une fois la CI branchée.)

---

## De quoi s'agit-il

Prism Engine est un moteur multimédia écrit en Rust, sans aucune dépendance de code tierce : lire des images, composer, dessiner des images, monter des timelines, produire un film — tout s'accomplit dans un seul moteur déterministe. Les 110 dépendances d'exécution (plus 20 dépendances de test) sont toutes des dépendances `path` de ce dépôt — pas de réseau, pas de gestionnaire de paquets, aucun risque de chaîne d'approvisionnement.

La distribution se fait sous forme de **code source en fichier unique** : l'ensemble du moteur (622 fichiers, 28 crates) est fusionné dans un fichier texte UTF-8, `prism-engine-v2.0-source.txt`. Il suffit de posséder ce fichier pour reconstruire localement une arborescence identique à celle du dépôt.

---

## Ce que le moteur sait faire

En une phrase : **une image est l'apparence de la scène à un instant donné, une vidéo est la suite de ces apparences sur une durée** — à partir d'une même description de scène, figer un instant donne une image unique, avancer en continu et enregistrer donne une vidéo ; l'image fixe et l'image animée partagent un seul jeu de capacités.
La suite est organisée en cinq blocs selon « ce que vous voulez faire » : **importer → agencer → dessiner → livrer → intégrer dans votre propre chaîne d'outils**.

### Ce que le moteur lit (entrées)

- **Façade de décodage unifiée** : un unique point d'entrée `decode_image(bytes, ext)` aiguille les images fixes — PNG (non entrelacé 8/16 bits), JPEG baseline (IDCT en virgule fixe, identique bit à bit d'une plateforme à l'autre), **priorité au magic number, nom d'extension en secours** (une extension erronée n'empêche pas le décodage) ; tout format hors matrice ou nécessitant une coopération externe produit une erreur en trois parties (où / quoi / comment corriger) assortie d'une suggestion d'usage de ffmpeg, **jamais d'échec silencieux**.
- **Images animées / audio / conteneurs** : décodage natif image par image de GIF et APNG (`prism-media`) ; décodage WAV intégré, mp3/aac/FLAC via le chemin de détection ffmpeg (`prism-audio`) ; relecture MP4-MJPEG (`decode_mp4_video`).
- **Suite EXR complète** : lecture, écriture, compression, DWA, avec une **comparaison de flux binaire entre implémentations** face à l'OpenEXR officiel (et non une lecture de ce que l'on a soi-même écrit).
- **RAW d'appareil photo** : décodage RAW de base. **Limite** : `Compression=7` (lossless JPEG) des DNG réels n'est pas implémenté, les DNG produits par de vrais appareils ne sont pas lisibles — c'est consigné dans les « limites assumées », rien n'est dissimulé.
- **Assets 3D** : import glTF / OBJ, analyse USD (`.usda`) et paquets de paramètres MaterialX ; les mesh peuvent entrer dans la scène de rendu.
- **Polices externes** : `load_font_file` charge directement des TTF/OTF tiers, la chaîne de repli est résolue niveau par niveau, un glyphe manquant donne un diagnostic explicite plutôt qu'un carré vide.

### Ce que le moteur agence (grammaire de création)

- **Grammaire complète des blocs DSL v2** : un fichier `.prism` décrit « ce que l'on veut, comment le placer, comment l'animer, à quoi il ressemble, en quoi l'exporter » ; le produit de compilation est un graphe de nœuds + des canaux de propriétés + une timeline.
- **Cinq combinateurs** : `title_card` / `lower_third` / `split_screen` / `ken_burns` / `morph_between` se déploient en graphes de nœuds ; « un film de 90 secondes » passe du code de scène écrit à la main à un simple assemblage.
- **Transitions et morph** : Fade / Wipe / Dissolve / Push entièrement paramétrables (durée, easing, direction), morph par déformation de polyligne, séquences de scènes échantillonnables à tout instant.
- **Composition de formules** : `math_ast` + `math_layout` composent exposants / indices / fractions / racines / grands opérateurs / matrices, le bloc DSL `formula` compose directement, sur la même ligne de base que le texte courant.

### Ce que le moteur dessine (rendu et simulation)

- **Rendu hors ligne photoréaliste** : path tracing + BSDF physiquement corrects (diffus / spéculaire / verre / métal), verre et métal par inversion stochastique (série de Neumann en Monte-Carlo), prise en charge des médias volumétriques et de la diffusion sous-surface.
- **Quatre familles de primitives courbes** : cylindre / cône / tore / surface de révolution, toutes à intersection analytique (équation du quatrième degré résolue par la forme close de Ferrari, racines exactes) ; mesh triangulaire quelconque + SAH-BVH, comparaison de parité en double chemin.
- **Compute GPU** : couche sémantique compute indépendante du périphérique + dispatch Vulkan + SPIR-V minimal écrit à la main ; sous l'ICD logiciel Mesa lavapipe, l'ensemble fonctionne sur une plateforme sans GPU (vérifiable en headless).
- **Physique et particules** : solveur unifié XPBD (corps rigides / corps mous / tissu / articulations / fluides / fracturation), fumée FLIP, système de particules (AudioReactive couplé à l'enveloppe audio).
- **Art génératif et post-traitement** : champs de flux / L-system / automates cellulaires / réaction-diffusion / fractales / Voronoi ; blur / bloom / halation / grain de film / quatre styles NPR (toon / watercolor / halftone / oil) / débruitage trois-en-un.
- **Étalonnage et vectoriel** : étalonnage secondaire (sélection HSL + courbes RGB), préréglages de style cinematic / film / anime et trois niveaux de qualité brouillon / standard / cinéma (la qualité est un paramètre, pas une incantation) ; rastérisation de chemins vectoriels, draw-on et morph de chemins.
- **Traitement audio** : normalisation de sonie (loudnorm), détection de tempo, analyse spectrale, extraction d'enveloppe — le tout couplable à l'image (les particules AudioReactive suivent l'enveloppe audio).
- **Le backend de référence CPU est un citoyen de première classe** : une machine sans GPU produit malgré tout des images correctes, et il sert de référence de justesse pour le GPU.

### Ce que le moteur produit (livraison)

- **`render_movie` direct** : le produit de compilation DSL v2 couvre en un seul appel « évaluation image par image → rendu par tile → livraison » ; une seule commande `.prism` produit le film (`prism-app render demo.prism`) ; les nœuds non couverts signalent une erreur explicite, sans approximation silencieuse.
- **Boucle de prévisualisation** : `prism-app preview` lance le service de prévisualisation ; l'éditeur propose deux niveaux, Draft (motifs géométriques, fréquence d'images préservée) / Real (rendu réel à faible spp, rafraîchissement asynchrone).
- **Encodeur maison + contrôle de débit** : encodeur temporel dans le domaine JPEG à trames I/P dans un conteneur privé `PRMV` (deux générations de schéma, v2/v3), contrôle de débit par water-filling + estimation de mouvement sous-pixel qpel (mesuré : PSNR +2,577 dB, volume −21,6 %).
- **Chaîne de coopération ffmpeg** : avec ffmpeg → conversion en H.264/HEVC via un pipe rawvideo sur stdin ; sans ffmpeg → repli automatique sur MJPEG, **le moteur produit toujours un film** ; aucun binaire embarqué.
- **Livraison de séquences EXR** : les séquences d'images sont écrites directement en EXR, vers le pipeline de post-production.
- **Ferme distribuée** : répartition au niveau des trames + workers à N processus + fusion déterministe — MD5 parallèle == MD5 séquentiel, reprise après interruption prise en charge.

### Comment l'intégrer (écosystème)

- **Surface d'outils MCP** : trois outils JSON-RPC sur stdio — `scene_check` (compilation + rendu d'essai + estimation du nombre de trames), `render_submit` (job id déterministe + empreinte MD5 par trame), `diagnose` (localisation d'erreur en trois parties).
- **SDK de nœuds** : négociation de version sémantique (avec cas particulier 0.x), paquet de nœuds d'exemple fourni avec le paquet (`examples/sdk-nodes/cinematic-basic/`, compilable / chargeable / négociable).
- **Squelette de détection d'encodage matériel** : `prism-ffi-hwenc` sur trois plateformes (VideoToolbox / NVENC / VAAPI), chaîne de détection + matrice de feature gates ; une plateforme non conforme signale explicitement l'absence de prise en charge.

> Les limites détaillées de capacités (ce qui **n'est pas** réalisable) figurent dans la section « limites assumées » du `开发者指南.md` après décompression — on y consigne l'état réel actuel (les primitives courbes passent uniquement par le CPU, le gain d'accélération GPU doit être mesuré sur le matériel cible, l'encodage matériel est intégré selon la plateforme cible…). Ce document ne donne que la vue d'ensemble, sans exagération.

---

## Pourquoi une distribution en fichier unique

- **Auditable** : un seul fichier texte, lisible à l'œil nu, `diff`able, vérifiable par hachage ; publié, il est figé.
- **Reproductible** : l'empaquetage est déterministe — deux empaquetages des mêmes entrées donnent un SHA-256 strictement identique.
- **Portable** : pas besoin de cloner un dépôt volumineux, un fichier se transmet par e-mail ou messagerie, et la reconstruction du build reste possible hors ligne.
- **Compromis assumé** : la distribution en fichier unique sacrifie la commodité de « parcourir le code directement sur GitHub » en échange de la garantie déterministe « tout est emporté dans un seul fichier, vérifiable octet par octet ». Une fois décompressé, on obtient un Cargo workspace ordinaire, indiscernable d'un projet Rust classique.

---

## Démarrage rapide

Prérequis : installer [Rust](https://www.rust-lang.org/) (avec `cargo`) et [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Téléchargez prism-engine-v2.0-source.txt depuis le dépôt et enregistrez-le comme fichier local
# 2. Ouvrez-le dans un éditeur de texte et enregistrez le script Node.js situé
#    entre "UNPACK SCRIPT BEGIN" et "UNPACK SCRIPT END" sous unpack.cjs dans le même répertoire
# 3. Lancez la décompression (quelques secondes, restaure 622 fichiers)
node unpack.cjs
# 4. Dans le répertoire restauré (racine du workspace), lancez la suite de tests complète
cargo test --workspace --release
# Attendu : 3714 passed / 0 failed
```

Après décompression, on obtient un Cargo workspace standard (28 crates sous `src/`), les commandes `cargo` suivantes sont identiques à celles d'un projet Rust classique.

> Chaîne de vérification wasm côté mobile : `rustup target add wasm32-unknown-unknown`, puis `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Structure du dépôt (après décompression)

| Chemin | Description |
|------|------|
| `src/` | code source des **28 crates** (29 dans le dépôt, nom d'origine `crates/`, réorganisé en `src/` lors de la distribution ; le crate de démonstration `prism-anim-demo` n'est pas dans le paquet) |
| `Compliance/` | déclarations de conformité : vérification du zéro dépendance / mentions tierces / registre des assets (nom d'origine dans le dépôt : `docs/合规/`) |
| `开发者指南.md` | documentation développeur autonome : prise en main / architecture / noyau mathématique / garde-fous / extension (nom d'origine dans le dépôt : `docs/开发者指南.md`) |
| `tools/` | scripts de garde-fou (défense contre les écritures fantômes, scan d'hygiène, auto-contrôle de conformité) |
| `.github/workflows/` | configuration CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | éléments requis pour le build (les workspace members pointent déjà vers `src/*`) |
| `examples/` | exemples pour les tests de `prism-dsl` et paquets de nœuds d'exemple (lus par `include_str!`, la compilation échoue s'ils manquent) |

> La documentation de référence de conception (`docs/设计/`) **n'est pas incluse dans la distribution**, elle ne reste que dans le dépôt complet de l'éditeur.

---

## Aperçu des modules

Les **28 crates** du paquet de distribution sont regroupés par responsabilité (le dépôt contient en outre le crate de démonstration `prism-anim-demo`, présent uniquement comme exemple reproductible et absent du paquet) :

| Catégorie | crate |
|------|-------|
| Noyau | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S média | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendu | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation / génération | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texte / UI | prism-text · prism-rt · prism-app |
| Avancé | prism-ai · prism-dsl · prism-infra · prism-author |
| Contournement (feature gate) | prism-ffi-hwenc |

---

## Build et tests

```bash
# Test de fumée sur tout le dépôt (release, recommandé)
cargo test --workspace --release
# Alias équivalent
cargo xtest
# Vérification rapide d'un seul crate
cargo test -p prism-core --release
```

- Suite de tests complète : **3714 passed / 0 failed / 0 ignored** (cette version a réellement été exécutée sur l'**arborescence décompressée** : 166 blocs de tests ; le dépôt en compte 167, la différence d'un bloc correspond au bloc de tests vide du crate de démonstration, le nombre de cas est identique).
- CI : **6 jobs** — défense contre les écritures fantômes (ghost-guard) / compilation + tests complets (native) / build wasm32 (wasm) / vérification statique WGSL (wasm-verify) / comparaison de flux binaire entre implémentations face à l'OpenEXR officiel (interop) / contre-vérification croisée sur hôte node (lint).
- 3 scripts de garde-fou locaux (zéro dépendance, uniquement `node`, ci-dessous) + 1 script d'empaquetage ; parmi eux, l'auto-contrôle de conformité passe **9/9** et l'auto-preuve inversée **8/8 échoue comme attendu**.

Exécution locale des garde-fous (Node 18+ requis) :

```bash
node tools/ghost_guard.mjs --check        # Défense contre les écritures fantômes : G1 définitions en double / G2 sur-ensemble strict / G3 documentation en début de ligne / G4 compilation
node tools/hygiene_scan.mjs               # Hygiène open source : liste une à une les tournures internes résiduelles (fichier:ligne), toute occurrence provoque exit 1
node tools/compliance_check.mjs           # Auto-contrôle de conformité : doit passer 9/9
node tools/compliance_check.mjs --self-test   # Auto-preuve inversée : doit échouer, prouvant que les critères sont discriminants
```

> **Le scan d'hygiène open source renvoie encore des occurrences existantes** : des tournures internes anciennes subsistent dans le dépôt ; parmi elles, la **couche documentaire incluse dans le paquet est déjà à zéro** (README / guide du développeur / Compliance), les autres se trouvent dans les **commentaires du code source du paquet** et dans les **fichiers du dépôt non distribués**, classées telles quelles et inscrites comme éléments de nettoyage distincts. Ce script existe précisément pour que les résidus restent **visibles et nettoyables un par un**, plutôt que de faire semblant d'être propre — ne considérez pas « un nombre d'occurrences non nul » comme une défaillance du garde-fou.

---

## Conformité

Le moteur garantit **zéro dépendance de code tierce**, vérifié par auto-contrôle :

- **C1 Zéro dépendance de code tierce** : les 130 dépendances (110 d'exécution + 20 de test) sont toutes des dépendances `path` de ce dépôt ; aucun `build.rs`, aucune instruction de liaison native, `Cargo.lock` verrouille 29 paquets du dépôt.
- **C2 Registre d'assets complet** : les champs du tableau principal des 5 assets sont tous renseignés, l'état d'internalisation relève d'un vocabulaire contrôlé ; le texte intégral de la licence de police OFL 1.1 accompagne la police dans `src/prism-text/font_data/LICENSE`.
- **C3 Mentions d'attribution tierces** : `Compliance/THIRD-PARTY-NOTICES.md` est présent et distingue explicitement « asset de données ≠ dépendance de code ».

> Remarque : ce dépôt distribue du **code source**. L'asset de police (sous-ensemble Prism Sans SC) est sous licence SIL OFL 1.1, **le texte intégral de la licence accompagne la police** dans `src/prism-text/font_data/LICENSE` ; l'origine des assets et leur enregistrement de licence sont décrits dans `Compliance/assets/资产登记.md`.

---

## Empaquetage déterministe (éditeur)

Pour régénérer le TXT de distribution depuis le dépôt complet (le numéro de version peut être passé, par défaut la version courante v2.0) :

```bash
node tools/pack_source_txt.mjs          # Par défaut → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # Écriture équivalente (numéro de version explicite)
node tools/pack_source_txt.mjs 1.0      # Revient au nom de produit de l'époque v1.0 (exécuté sur les sources v1.0, reproduit l'ancien paquet octet par octet)
node tools/pack_source_txt.mjs --verify # Décompression et vérification en boucle juste après l'empaquetage (comparaison fichier par fichier, octet par octet, avec l'arborescence source)
# Enchaînez deux exécutions : un SHA-256 identique prouve que l'empaquetage déterministe tient
```

Le périmètre d'empaquetage est strictement limité à : `crates/` (→ `src/`), `docs/合规/` (→ `Compliance/`), `docs/开发者指南.md`, `tools/`, `.github/`, `.cargo/`, `examples/`, `Cargo.toml`, `Cargo.lock`. Sont délibérément **exclus** `docs/设计/`, `target/`, `node_modules/`, `dist/` lui-même et tout produit intermédiaire.

---

## Contribution

Les Issues et Pull Requests sont bienvenues. Avant de soumettre, veuillez conserver :

- `cargo test --workspace --release` entièrement vert ;
- `node tools/hygiene_scan.mjs` **sans nouvelle** tournure interne (le script liste toutes les occurrences, y compris les résidus existants) ;
- `node tools/compliance_check.mjs` passant 9/9.

Le guide d'architecture et d'extension plus détaillé se trouve dans `开发者指南.md` après décompression.

---

## Licence

Ce projet est publié sous **Apache License 2.0**. Voir le fichier [`LICENSE`](../LICENSE).

> Rappel de conformité : Apache-2.0 exige la conservation des mentions de copyright et de licence ; en cas de redistribution fondée sur ce moteur, veuillez conserver les mentions d'attribution et le registre d'assets sous `Compliance/`.
