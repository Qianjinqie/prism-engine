[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)


# Prism Engine

> Un moteur multimédia Rust déterministe et sans aucune dépendance — distribué sous forme d'**un seul fichier TXT** : téléchargez un texte de 18 MB, exécutez le script de dépaquetage et obtenez l'arborescence complète des sources (28 crates), sans npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3929%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## Qu'est-ce que c'est

Prism Engine est un moteur multimédia écrit en Rust, sans aucune dépendance à du code tiers :
lire des images, composer une mise en page, dessiner des plans, monter une timeline, produire
un film final — tout se fait dans un seul moteur déterministe. Ses 110 dépendances d'exécution
(plus 20 dépendances de test) sont toutes des dépendances `path` de ce dépôt — aucun accès
réseau, aucun gestionnaire de paquets, aucun risque de chaîne d'approvisionnement.

La distribution se fait sous forme de **sources en un seul fichier** : l'ensemble du moteur
(642 fichiers, 28 crates) est fusionné dans un texte UTF-8, `prism-engine-v2.1-source.txt`.
Il suffit de posséder ce fichier pour reconstruire localement une arborescence identique
à celle du dépôt.

---

## Ce que le moteur sait faire

En une phrase : **une image est un plan figé à un instant donné, une vidéo est une suite de
plans dans le temps** — à partir de la même description de scène, figer un instant donne une
image fixe, avancer en continu et enregistrer donne une vidéo ; l'image fixe et l'animation
partagent le même jeu de capacités. La suite s'organise en cinq blocs selon « ce que vous
voulez faire » : **importer → composer → dessiner → produire le film → intégrer à votre
propre chaîne d'outils**.

### Ce qu'on lit (entrées)

- **Façade de décodage unifiée** : un seul point d'entrée `decode_image(bytes, ext)` distribue les
  images fixes — PNG (non entrelacé 8/16 bits), JPEG baseline (IDCT en virgule fixe, déterminisme
  bit à bit d'une plateforme à l'autre), **nombre magique prioritaire, extension en repli** (une
  extension erronée n'empêche pas le décodage) ; les formats hors matrice ou nécessitant une
  coopération externe renvoient toujours une erreur en trois parties (où / quoi / comment corriger)
  avec une suggestion de coopération via ffmpeg — **aucun échec silencieux**.
- **Animation / audio / conteneurs** : décodage image par image natif pour GIF et APNG
  (`prism-media`) ; WAV décodé en interne, mp3/aac/FLAC via le chemin de détection ffmpeg
  (`prism-audio`) ; relecture MP4-MJPEG (`decode_mp4_video`).
- **Suite EXR complète** : lecture, écriture, compression, DWA, avec **comparaison de flux binaires
  entre implémentations** face à l'OpenEXR officiel (et non une simple écriture/lecture maison).
- **RAW d'appareil photo** : décodage RAW de base. **Limite** : `Compression=7` (lossless JPEG) des
  vrais DNG n'est pas implémenté, donc les DNG produits par un appareil photo réel ne sont pas
  lisibles — c'est consigné dans « les limites honnêtes », rien n'est dissimulé.
- **Assets 3D** : import glTF / OBJ, analyse d'USD (`.usda`) et packs de paramètres MaterialX ;
  les mesh peuvent entrer dans une scène de rendu.
- **Polices externes** : `load_font_file` charge directement des TTF/OTF tiers, la chaîne de repli
  se résout niveau par niveau, et un glyphe manquant produit un diagnostic explicite plutôt
  qu'un carré vide.

### Ce qu'on compose (grammaire de création)

- **Syntaxe complète des blocs DSL v2** : un seul `.prism` décrit « quoi, comment le disposer,
  comment l'animer, à quoi ça ressemble, vers quoi l'exporter » ; le produit de compilation est
  un graphe de nœuds + des canaux de propriétés + une timeline.
- **Cinq combinateurs** : `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between`, des combinateurs de haut niveau qui se déploient en graphe de nœuds ;
  « un film de 90 secondes » passe d'un code de scène écrit à la main à un simple assemblage.
- **Transitions et morph** : Fade / Wipe / Dissolve / Push entièrement paramétrés (durée, easing,
  direction), morph de polylignes, et une séquence de scènes échantillonnable à n'importe
  quel instant.
- **Composition de formules** : `math_ast` + `math_layout` composent exposants et indices /
  fractions / radicaux / grands opérateurs / matrices ; le bloc `formula` du DSL compose
  directement, sur la même ligne de base que le texte courant.

### Ce qu'on dessine (rendu et simulation)

- **Rendu hors ligne photoréaliste** : path tracing + BSDF physiquement corrects (diffus /
  spéculaire / verre / métal), verre et métal résolus par inversion stochastique (Monte-Carlo
  sur série de Neumann), prise en charge des milieux volumiques et de la diffusion sous-surface.
- **Quatre familles de primitives courbes** : cylindre / cône / tore / surface de révolution,
  toutes en intersection analytique (les équations du quatrième degré passent par la solution
  fermée de Ferrari, racines exactes) ; mesh triangulaire quelconque + SAH-BVH, avec contrôle
  croisé de parité entre deux chemins.
- **GPU compute** : couche sémantique compute indépendante du périphérique + dispatch Vulkan +
  SPIR-V minimal écrit à la main ; sous l'ICD logiciel Mesa lavapipe, tout tourne sur une
  plateforme sans GPU (vérifiable en headless).
- **Physique et particules** : solveur unifié XPBD (corps rigides / corps mous / tissu /
  articulations / fluides / fracture), fumée FLIP, systèmes de particules (AudioReactive
  synchronisé sur l'enveloppe audio).
- **Art génératif et post-traitement** : champs de flux / systèmes L / automates cellulaires /
  réaction-diffusion / fractales / Voronoi ; blur / bloom / halation / grain de film / quatre
  styles NPR (toon / watercolor / halftone / oil) / dénoising trois-en-un.
- **Étalonnage et vectoriel** : étalonnage secondaire (sélections HSL + courbes RGB), préréglages
  de style cinematic / film / anime et trois niveaux de qualité brouillon / standard / cinéma
  (la qualité est un paramètre, pas une incantation) ; rastérisation de tracés vectoriels,
  draw-on et morph de tracés.
- **Traitement audio** : normalisation de sonie (loudnorm), détection de tempo, analyse
  spectrale, extraction d'enveloppe — le tout synchronisable avec l'image (les particules
  AudioReactive suivent l'enveloppe audio).
- **Le backend de référence CPU est un citoyen de première classe** : une machine sans GPU
  produit malgré tout des images correctes, et il sert de référence de validation pour le GPU.

### Ce qu'on produit (livraison)

- **`render_movie` en direct** : le produit de compilation DSL v2 fait tout en un appel —
  « évaluation par image → rendu par tile → livraison » — et un `.prism` sort un film en une
  seule commande (`prism-app render demo.prism`) ; les nœuds non couverts signalent honnêtement
  une erreur, sans approximation silencieuse.
- **Boucle de prévisualisation** : `prism-app preview` lance un service de prévisualisation ;
  l'éditeur propose deux modes — Draft (motifs géométriques pour préserver la fréquence d'images) /
  Real (vrai rendu à faible spp, rafraîchissement asynchrone).
- **Encodeur maison + contrôle de débit** : un encodeur temporel dans le domaine JPEG à trames
  I/P, dans un conteneur privé `PRMV` (deux générations de schema, v2/v3), contrôle de débit par
  water-filling + estimation de mouvement subpixel qpel (mesuré : PSNR +2.577 dB,
  taille −21.6%).
- **Chaîne de coopération ffmpeg** : ffmpeg présent → conversion en H.264/HEVC via un pipe
  rawvideo sur stdin ; ffmpeg absent → repli automatique sur MJPEG, **le moteur produit toujours
  un film** ; aucun binaire n'est embarqué.
- **Livraison de séquences EXR** : la séquence d'images s'écrit directement en EXR et entre
  dans le pipeline de post-production.
- **Ferme distribuée** : distribution au niveau de l'image + N workers processus + fusion
  déterministe — MD5 en parallèle == MD5 en série, avec reprise du calcul après interruption.

### Comment l'intégrer (écosystème)

- **Surface d'outils MCP** : trois outils JSON-RPC sur stdio — `scene_check` (compilation + rendu
  d'essai + estimation du nombre d'images), `render_submit` (job id déterministe + empreinte MD5
  par image), `diagnose` (localisation d'erreur en trois parties).
- **SDK de nœuds** : négociation de version sémantique (y compris le cas particulier 0.x), avec un
  paquet de nœuds d'exemple fourni (`examples/sdk-nodes/cinematic-basic/`, compilable / chargeable /
  négociable).
- **Squelette de détection d'encodage matériel** : `prism-ffi-hwenc`, chaîne de détection sur trois
  plateformes (VideoToolbox / NVENC / VAAPI) + matrice de feature gates ; une plateforme non
  conforme renvoie explicitement « non pris en charge ».

> Les limites détaillées de ces capacités (ce qui **n'est pas** faisable) figurent dans la section
> « les limites honnêtes » du `开发者指南.md` livré dans l'archive — on y trouve l'état réel actuel
> (les primitives courbes ne passent que par le CPU, le gain du GPU doit être mesuré sur le
> matériel cible, l'encodage matériel s'intègre selon la plateforme cible…). Ce document ne donne
> que la vue d'ensemble, sans promesse excessive.

---

---

## Capacités internes (au-delà de la liste des fonctionnalités)

Certaines capacités ne rentrent pas dans la liste « lire / composer / dessiner / livrer / intégrer » — ce sont les actifs profonds du moteur. Ce chapitre garde la liste principale lisible.

### Audio : un noyau de DAW intégré

`prism-audio` n'est pas un module périphérique « qui sait lire du WAV », mais une chaîne complète de la composition au mixage ; chaque fonction est une fonction pure de ses paramètres (même entrée ⇒ bits identiques d'une plateforme à l'autre) :

- **Synthèse** : famille d'oscillateurs PolyBLEP sans repliement + enveloppes ADSR en forme fermée + 4 préréglages d'ambiance ;
- **Théorie musicale** : bibliothèque de gammes (majeur/mineur, pentatonique, modes), 12-TET hauteur↔fréquence, automate grammatical des progressions d'accords + conduite des voix ;
- **Séquencement** : arpégiateur (motif/vitesse/octave), boîte à rythmes par pas, partition sur grille de mesures/temps ;
- **Matériau** : analyseur SMF écrit à la main (import MIDI), échantillonneur SFZ, synthèse granulaire ;
- **Design sonore** : modélisation physique, panoramique spatial, chaîne d'effets ;
- **Mixage** : mixage linéaire multipiste + fondus + limitation des crêtes ;
- **Analyse et couplage** : détection de tempo/onset, spectre FFT, extraction d'enveloppes, cuisson AudioReactive (l'image N lit une fenêtre fixe ; fonction pure par image, équivalente au seek), normalisation de sonie.

> Limite : AAC/mp3 ne sont pas développés ici (historique des brevets + charge) ; l'audio général passe par ffmpeg externe.

### Bibliothèque de noyaux définissables (registre de solutions spéciales)

Chaque type de nœud est une **solution spéciale** d'un triplet « noyau + treillis + dualité ». Des tiers (ou vous) peuvent enregistrer de nouvelles solutions avec `registry!`, sans toucher au cœur :

- Collecte statique (la macro s'étend dans le crate qui définit ; pas d'`inventory`/proc-macro, zéro dépendance) ;
- **Validation en trois questions** à l'enregistrement (forme du noyau / treillis / dualité) : une réponse incomplète est refusée avec une erreur en trois parties ;
- Les solutions officielles et personnalisées partagent le même registre ; aucune différence à l'exécution ;
- Mathématiquement, c'est l'équation d'opérateur `f = g + Kf` avec six stratégies d'inversion — exacte / stochastique / locale / régularisée / itérative / spectrale. Ajouter une capacité, c'est déclarer un triplet, pas rafistoler la boucle de rendu.

### Infrastructure de déterminisme et de reproductibilité

- **Clôture de signature** : graphes de nœuds / clés de cache / clés d'instantané / signatures d'image se réduisent à un « hachage de configuration en quatre dimensions » ; changer un paramètre change la clé, donc rien n'est réutilisé à tort ; les nœuds itératifs utilisent `(ε, n)`, les nœuds de flux le numéro d'image ;
- **Points fixes d'octets** : USD export→import→export est identique octet par octet ; WAV 16/24/32 bits fait l'aller-retour exactement ; EXR est comparé bit à bit à l'implémentation officielle ;
- **Empaquetage déterministe en un seul fichier** : le même code produit deux fois le même SHA-256 ; le round-trip de dépaquetage compare chaque fichier octet par octet.

### Backends GPU (énoncés honnêtement)

- **Vulkan** : les pipelines de calcul et de scène en lancer de rayons tournent réellement (vérifiable sans GPU sur l'ICD logiciel lavapipe) ; la sortie de `render_scene` est **identique octet par octet à la référence CPU** (0 pixel différent) ; `is_available()` exige l'autotest calcul ET rendu ;
- **Metal** : port isomorphe + compilation conditionnelle prêts, **non vérifié sur matériel, non activé** (`is_available()` toujours false) ;
- **WebGPU** : conclusion de faisabilité seulement (4–6 jours-personne), pas de chemin de rendu.

### Garde-fous anti-faux-vert

Quatre scripts locaux (zéro dépendance, `node` uniquement) : défense contre les écritures fantômes (avec auto-preuve inverse), analyse d'hygiène open source, auto-contrôle de conformité (9/9, auto-preuve inverse 8/8 en rouge), vérification du round-trip d'empaquetage. Un garde-fou ne rend que « passe » ou « rouge » ; « ça a l'air bon » ne compte pas. **Le faux vert est plus dangereux qu'un échec.**

---

## Pourquoi une distribution en un seul fichier

- **Auditable** : un fichier texte, lisible à l'œil nu, `diff`-able, vérifiable par hachage ; une fois publié, il est figé.
- **Reproductible** : l'empaquetage est déterministe — deux empaquetages des mêmes entrées donnent un SHA-256 strictement identique.
- **Portable** : inutile de cloner un dépôt volumineux ; un seul fichier s'envoie par courriel ou par messagerie, et la reconstruction fonctionne même hors ligne.
- **Compromis assumé** : la distribution en un seul fichier sacrifie la commodité de « parcourir le code directement sur GitHub » en échange de la certitude de « tout emporter dans un fichier, vérifiable octet par octet ». Une fois dépaqueté, on obtient un Cargo workspace ordinaire, identique à tout projet Rust classique.

---

## Démarrage rapide

Prérequis : installer [Rust](https://www.rust-lang.org/) (avec `cargo`) et [Node.js](https://nodejs.org/) 18+.

```bash
# 1. téléchargez prism-engine-v2.1-source.txt depuis le dépôt et enregistrez-le localement
# 2. ouvrez-le dans un éditeur de texte et enregistrez le script Node.js situé entre
#    "UNPACK SCRIPT BEGIN" et "UNPACK SCRIPT END" sous le nom unpack.cjs, dans le même répertoire
# 3. lancez le dépaquetage (quelques secondes, restaure 642 fichiers)
node unpack.cjs
# 4. exécutez la suite de tests complète dans le répertoire restauré (racine du workspace)
cargo test --workspace --release
# attendu : 3929 passed / 0 failed
```

Après dépaquetage, vous obtenez un Cargo workspace standard (28 crates sous `src/`) ; les commandes `cargo` suivantes sont identiques à celles d'un projet Rust classique.

> Chaîne de validation wasm côté mobile : `rustup target add wasm32-unknown-unknown`, puis `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Structure du dépôt (après dépaquetage)

| Chemin | Description |
|------|------|
| `src/` | Sources des **28 crates** (29 dans le dépôt, initialement `crates/`, réorganisées en `src/` lors de la distribution) |
| `Compliance/` | Déclarations de conformité : vérification de l'absence de dépendances / mentions tierces / registre des assets (initialement `docs/合规/` dans le dépôt) |
| `开发者指南.md` | Documentation développeur autonome : prise en main / architecture / noyau mathématique / gates / extensions (initialement `docs/开发者指南.md` dans le dépôt) |
| `tools/` | Scripts de gate (défense contre les écritures fantômes, scan d'hygiène, auto-vérification de conformité) |
| `.github/workflows/` | Configuration CI |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Éléments requis pour la compilation (les workspace members pointent déjà vers `src/*`) |
| `examples/` | Exemples pour les tests `prism-dsl` et paquets de nœuds d'exemple (lus par `include_str!`, le build échoue s'ils manquent) |

---

## Aperçu des modules

Les **28 crates** de l'archive de distribution sont regroupées par responsabilité :

| Catégorie | crate |
|------|-------|
| Noyau | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| E/S médias | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendu | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation / génératif | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Texte / UI | prism-text · prism-rt · prism-app |
| Avancé | prism-ai · prism-dsl · prism-infra · prism-author |
| Voie latérale (feature gate) | prism-ffi-hwenc |

---

## Compilation et tests

```bash
# smoke test sur tout le dépôt (release, recommandé)
cargo test --workspace --release
# alias équivalent
cargo xtest
# vérification rapide d'une seule crate
cargo test -p prism-core --release
```

- Tests complets : **3929 passed / 0 failed / 0 ignored** (exécuté en release dans le dépôt : 179 suites, 171,5 s tous cœurs ; le paquet est identique octet par octet au dépôt — seul le crate de démonstration est exclu)
- CI : **6 jobs** — défense contre les écritures fantômes (ghost-guard) / compilation + tests complets (native) / build wasm32 (wasm) /
  vérification statique WGSL (wasm-verify) / comparaison de flux binaires entre implémentations avec l'OpenEXR officiel (interop) / validation croisée sur l'hôte node (lint).
- 3 scripts de gate locaux (zéro dépendance, `node` uniquement, ci-dessous) + 1 script d'empaquetage ;
  l'auto-vérification de conformité passe **9/9**, la preuve par l'inverse échoue volontairement **8/8**.

Exécution locale des gates (Node 18+ requis) :

```bash
node tools/ghost_guard.mjs --check        # défense contre les écritures fantômes : G1 redéfinition / G2 sur-ensemble strict / G3 doc en début de ligne / G4 compilation
node tools/hygiene_scan.mjs               # hygiène open source : liste les formulations internes résiduelles une par une (fichier:ligne), exit 1 dès un hit
node tools/compliance_check.mjs           # auto-vérification de conformité : doit passer 9/9
node tools/compliance_check.mjs --self-test   # preuve par l'inverse : doit échouer, ce qui démontre le pouvoir discriminant des critères
```

> **Le scan d'hygiène open source connaît encore des occurrences résiduelles** : des formulations
> internes anciennes subsistent dans le dépôt ; la **couche documentaire distribuée avec l'archive
> est déjà à zéro** (README / guide du développeur / Compliance), le reste se trouve dans les
> **commentaires de source à l'intérieur de l'archive** et dans les **fichiers du dépôt non
> distribués**, où il a été classé tel quel et inscrit comme poste de nettoyage distinct. Le script
> existe précisément pour rendre ces résidus **visibles et nettoyables un par un**, plutôt que de
> faire semblant d'être propre — ne comprenez pas « le nombre de hits n'est pas 0 » comme une
> défaillance du gate.

---

## Conformité

Le moteur garantit **zéro dépendance à du code tiers**, vérifié par auto-contrôle :

- **C1 Zéro dépendance à du code tiers** : les 130 dépendances (110 d'exécution + 20 de test) sont toutes des dépendances `path` de ce dépôt ; aucun `build.rs`, aucune directive de link natif, `Cargo.lock` verrouille 29 paquets du dépôt.
- **C2 Registre des assets complet** : les champs de la table principale des 5 assets sont tous renseignés, le statut d'internalisation relève d'un vocabulaire contrôlé ; le texte intégral de la licence de police OFL 1.1 accompagne la police dans `src/prism-text/font_data/LICENSE`.
- **C3 Mentions d'attribution tierces** : `Compliance/THIRD-PARTY-NOTICES.md` est présent et distingue explicitement « asset de données ≠ dépendance de code ».

> Remarque : ce dépôt distribue des **sources**. L'asset de police (sous-ensemble Prism Sans SC) est sous licence SIL OFL 1.1, **le texte intégral de la licence accompagne la police dans** `src/prism-text/font_data/LICENSE` ; l'origine des assets et l'enregistrement des licences figurent dans `Compliance/assets/资产登记.md`.


---

## Licence

Ce projet est publié sous **Apache License 2.0**. Voir le fichier [`LICENSE`](../LICENSE).

> Rappel de conformité : Apache-2.0 exige de conserver les mentions de copyright et de licence ; en cas de redistribution basée sur ce moteur, veuillez conserver les mentions d'attribution et le registre des assets sous `Compliance/`.

**Remarque : en cas d'erreurs de traduction dans les autres versions, c'est la version chinoise simplifiée (the Simplified Chinese version) qui fait foi.**
