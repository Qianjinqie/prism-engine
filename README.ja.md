[English](README.en.md) | [简体中文](README.md) | **日本語** | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> 依存ゼロ、決定論的な Rust メディアエンジン —— **単一ファイル TXT 配布**：14 MB のテキストを 1 つダウンロードし、展開スクリプトを実行するだけで完全な 26-crate のソースツリーが得られます。npm / pip / ffmpeg / OpenEXR は不要です。

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## これは何

Prism Engine は、Rust で書かれたサードパーティコード依存ゼロのメディアエンジンで、画像・オーディオ・ジオメトリ・物理・レンダリング・DSL などのモジュールをカバーします。全 103 件の依存はいずれも本リポジトリ内の `path` 依存——ネットワークに依存せず、パッケージマネージャーを必要とせず、サプライチェーンリスクもありません。

**単一ファイルソース配布**を採用：エンジン全体（481 ファイル・26 crate）を 1 つの UTF-8 テキスト `prism-engine-v1.0-source.txt` に統合しています。このファイルさえあれば、ローカルでリポジトリと完全に一致するディレクトリツリーを復元できます。

## なぜ単一ファイル配布なのか

- **監査可能**：1 つのテキストファイルで、人間がそのまま読め、`diff` もハッシュ検証もでき、リリースと同時に凍結されます。
- **再現可能**：パッキングは決定論的——同じ入力から 2 回パックした結果の SHA-256 は完全に一致します。
- **持ち運び可能**：巨大なリポジトリを clone する必要はなく、1 つのファイルをメールやチャットで送るだけでよく、オフラインでもビルドを復元できます。
- **トレードオフの説明**：単一ファイル配布は「GitHub 上で直接コードを閲覧できる」利便性を譲る代わりに、「1 つのファイルですべてを持ち運べ、バイト単位で検証できる」という決定論性を得るものです。展開して得られるのは通常の Cargo workspace であり、一般的な Rust プロジェクトと何ら変わりありません。
- **最も重要なのは**:ファイルを 1 つアップロードするだけで済むのです(doge

---

## クイックスタート

前提条件：[Rust](https://www.rust-lang.org/)（`cargo` を含む）と [Node.js](https://nodejs.org/) 18+ をインストールしてください。

```bash
# 1. リポジトリ内の prism-engine-v1.0-source.txt をダウンロードし、ローカルファイルとして保存
# 2. テキストエディタで開き、"UNPACK SCRIPT BEGIN" と "UNPACK SCRIPT END"
#    の間にある Node.js スクリプトを同じディレクトリの unpack.cjs として保存
# 3. 展開を実行（数秒程度で 481 ファイルを復元）
node unpack.cjs
# 4. 復元されたソースツリーに入り、全テストを実行
cd src/..
cargo test --workspace --release
# 期待結果：3271 passed / 0 failed
```

展開後は標準的な Cargo workspace（`src/` 直下に 26 crate）が得られ、以降の `cargo` コマンドは通常の Rust プロジェクトと同じように使えます。

> モバイル端末での wasm 検証フロー：`rustup target add wasm32-unknown-unknown` を実行してから、`cargo build -p prism-core --target wasm32-unknown-unknown` を実行します。

---

## リポジトリ構成（展開後）

| パス | 説明 |
|------|------|
| `src/` | 26 crate のソースコード（リポジトリ内での元名は `crates/`、配布時に `src/` へ再構成） |
| `Compliance/` | コンプライアンス宣言：ゼロ依存検証 / サードパーティ宣言 / 資産登録（リポジトリ内での元名は `docs/合规/`） |
| `开发者指南.md` | 自己完結型の開発者ドキュメント：導入 / アーキテクチャ / 数学カーネル / ゲート / 拡張（リポジトリ内での元名は `docs/开发者指南.md`） |
| `tools/` | ゲートスクリプト（ゴースト書き込み防御、衛生スキャン、コンプライアンス自己チェック） |
| `.github/workflows/` | CI 設定 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | ビルドに必要なファイル一式（workspace members は既に `src/*` を指しています） |
| `examples/` | `prism-dsl` のテスト用サンプル（`include_str!` で読み込まれるため、欠けているとコンパイルできません） |

> 設計に関する権威あるドキュメント（`docs/设计/`）は**配布には含まれず**、公開元の完全なリポジトリにのみ残されています。

---

## モジュール概要

26 crate は責務ごとにグループ化されています：

| カテゴリ | crate |
|------|-------|
| カーネル | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| メディア I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| レンダリング | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| アニメーション/生成 | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| テキスト/UI | prism-text · prism-rt · prism-app |
| 応用 | prism-ai · prism-dsl · prism-infra |

---

## ビルドとテスト

```bash
# リポジトリ全体のスモークテスト（release、推奨）
cargo test --workspace --release
# 等価なエイリアス
cargo xtest
# 単一 crate のクイック検証
cargo test -p prism-core --release
```

- 全量テスト：**3271 passed / 0 failed**（まったく新しく展開したツリー上で実際に実行して検証済み）。
- ゲート：14 項目。ゴースト書き込み防御（G1〜G4）、オープンソース衛生スキャン（内部痕跡 0 件）、コンプライアンス自己チェック（9/9）、コンプライアンス逆方向自己証明（8/8、違反時は必ず赤）、wasm32 ビルドチェーンを含みます。

ゲートをローカルで実行（Node 18+ が必要）：

```bash
node tools/ghost_guard.mjs --check        # ゴースト書き込み防御：G1 重複定義/G2 厳密スーパーセット/G3 冒頭ドキュメント/G4 コンパイル
node tools/hygiene_scan.mjs               # オープンソース衛生：「クリーン（内部痕跡 0 件）」と報告されるはず
node tools/compliance_check.mjs           # コンプライアンス自己チェック：9/9 通過であるはず
node tools/compliance_check.mjs --self-test   # 逆方向自己証明：赤と判定されるはずで、判定基準に判別力があることを証明
```

---

## コンプライアンス

エンジンは**サードパーティコード依存ゼロ**を公約しており、自己チェックで検証済みです：

- **C1 サードパーティコード依存ゼロ**：全 103 件の依存はいずれも本リポジトリの `path` 依存。`build.rs` なし、ネイティブリンク指示なし、`Cargo.lock` は本リポジトリの 26 パッケージをロックしています。
- **C2 資産登録の完備**：5 項目の資産はメインテーブルのフィールドがすべて揃い、内製化ステータスは管理対象の語彙に収められています。OFL 1.1 フォントライセンスの全文はフォントとともに `src/prism-text/font_data/LICENSE` に置かれています。
- **C3 サードパーティ帰属宣言**：`Compliance/THIRD-PARTY-NOTICES.md` が存在し、「データ資産 ≠ コード依存」を明確に区別しています。

> 注：本リポジトリが配布しているのは**ソースコード**です。フォント資産（Prism Sans SC サブセット）は SIL OFL 1.1 でライセンスされ、**ライセンス全文はフォントとともに** `src/prism-text/font_data/LICENSE` に置かれています。資産の出所とライセンス登録は `Compliance/assets/资产登记.md` を参照してください。

---

## ライセンス

本プロジェクトは **Apache License 2.0** の下で公開されています。詳細は [`LICENSE`](LICENSE) ファイルを参照してください。

> コンプライアンスに関する注意：Apache-2.0 は著作権およびライセンス表示の保持を求めます。本エンジンを基に再配布する場合は、`Compliance/` 配下の帰属宣言と資産登録を保持してください。
