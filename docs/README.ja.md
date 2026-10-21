[English](README.en.md) | [简体中文](../README.md) | **日本語** | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> 依存ゼロ、決定論的な Rust メディアエンジン —— **単一ファイル TXT 配布**：17 MB のテキストをダウンロードし、展開スクリプトを実行すれば完全なソースツリー（28 個の crate）が手に入る。npm / pip / ffmpeg / OpenEXR は一切不要。

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

（バッジは静的な事実バッジであり、数値の定義は後述する。`crates-28` = **配布パッケージ内**の crate 数。
リポジトリには別にパッケージへ入らないデモ crate が 1 個あるため、リポジトリ上の数は 29 となる。CI に接続すれば動的バッジに置き換え可能。）

---

## これは何か

Prism Engine は Rust で書かれたサードパーティコード依存ゼロのメディアエンジンである。画像の読み込み、レイアウト、
画面の描画、タイムラインの編集、最終出力まで、すべてを 1 つの決定論的エンジン内で完結させる。110 個の実行時依存
（ほかにテスト依存 20 個）はすべて本リポジトリの `path` 依存であり、ネットワークに依存せず、パッケージマネージャも
不要で、サプライチェーンリスクもない。

**単一ファイルソース配布**を採用している。エンジン全体（622 ファイル、28 個の crate）を 1 つの UTF-8
テキスト `prism-engine-v2.0-source.txt` に統合。このファイルを保有していれば、リポジトリと完全に同一の
ディレクトリツリーをローカルで復元できる。

---

## エンジンにできること

ひと言でまとめると：**画像はある瞬間の画面の姿であり、動画はある期間における画面の連なりの姿である** ——
同じシーン記述に対して、ある一瞬で止めれば単一フレーム、連続的に進めて記録すれば動画となり、静止画と動画は同じ能力体系を共有する。
以下では「あなたがやりたいこと」別に 5 つのブロックに分ける：**読み込む → 組み上げる → 描く → 完成品を出す → 自分のツールチェーンに接続する**。

### 何を読むか（入力）

- **統一デコードファサード**：単一の `decode_image(bytes, ext)` 入口が静止画を振り分ける——PNG（非インターレース 8/16bit）、
  ベースライン JPEG（固定小数点 IDCT、クロスプラットフォームでビット単位に決定論的）、**マジックナンバー優先・拡張子はフォールバック**（拡張子が間違っていてもデコード可能）。
  対応マトリクス外、または外部協調を要するフォーマットは一律三段式のエラー（どこで / 何が / どう直す）とし、ffmpeg との協調案を提示する。
  **サイレントに失敗することは決してない**。
- **アニメーション画像 / 音声 / コンテナ**：GIF、APNG はネイティブなフレーム単位デコード（`prism-media`）。WAV は内蔵デコード、mp3/aac/FLAC
  は ffmpeg 検出経路（`prism-audio`）。MP4-MJPEG の読み戻し（`decode_mp4_video`）。
- **EXR フルセット**：読み書き、圧縮、DWA に対応し、公式 OpenEXR と**実装間ビットストリーム突き合わせ**を実施（自前実装の自己読み書きではない）。
- **カメラ RAW**：ベースライン RAW デコード。**境界**：実在する DNG の `Compression=7`（lossless JPEG）は未実装であり、
  実機カメラが出力した DNG は読めない——「誠実な境界」に明記しており、隠していない。
- **3D アセット**：glTF / OBJ のインポート、USD（`.usda`）のパースと MaterialX パラメータパッケージ。mesh はレンダリングシーンに持ち込める。
- **外部フォント**：`load_font_file` でサードパーティ製 TTF/OTF を直接ロードし、フォールバックチェーンを段階的に解決する。
  グリフ欠落時は豆腐（tofu）ではなく明確な診断を返す。

### 何を組むか（創作文法）

- **DSL v2 全ブロック文法**：1 つの `.prism` で「何が欲しいか、どう配置するか、どう動くか、どう見えるか、
  何として書き出すか」を記述する。コンパイル産物はノードグラフ + プロパティチャネル + タイムライン。
- **5 つのコンビネータ**：`title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` の高水準コンビネータがノードグラフへ展開され、「90 秒の完成品」は手書きのシーンコードから組み立てへと変わる。
- **トランジションと morph**：Fade / Wipe / Dissolve / Push はすべてパラメータ化（長さ、イージング、方向）。
  Morph は折れ線変形、シーケンスは任意の時刻でサンプリングできる。
- **数式組版**：`math_ast` + `math_layout` で上下付き文字 / 分数 / 根号 / 大型演算子 /
  行列を組む。DSL の `formula` ブロックで直接組版でき、本文と同じベースラインに揃う。

### 何を描くか（レンダリングとシミュレーション）

- **オフラインフォトリアリスティックレンダリング**：パストレーシング + 物理的に正しい BSDF（拡散反射 / 鏡面 / ガラス / 金属）。
  ガラスと金属は確率的逆算（Neumann 級数のモンテカルロ）で扱い、ボリューム媒質とサブサーフェス散乱にも対応する。
- **曲線プリミティブ 4 種**：円柱 / 円錐 / トーラス / 回転曲面。すべて解析的に交差判定（四次方程式は
  Ferrari の閉形式解で厳密根）。任意の三角形 mesh + SAH-BVH に対応し、2 経路の parity 突き合わせを実施。
- **GPU compute**：デバイス非依存の compute セマンティクス層 + Vulkan dispatch + 手書きの最小 SPIR-V。
  Mesa lavapipe のソフトウェア ICD 下では GPU のないプラットフォームでも完全に動作する（headless で検証可能）。
- **物理とパーティクル**：XPBD 統一ソルバ（剛体 / 軟体 / クロス / ジョイント / 流体 / 破壊）、
  煙の FLIP、パーティクルシステム（AudioReactive が音声エンベロープと連動）。
- **ジェネラティブアートとポストエフェクト**：流れ場 / L-system / セルオートマトン / 反応拡散 / フラクタル / Voronoi。
  blur / bloom / halation / フィルムグレイン / NPR 4 スタイル（toon / watercolor / halftone / oil）/ 3 種統合ノイズリダクション。
- **カラーグレーディングとベクター**：セカンダリカラー補正（HSL セレクション + RGB カーブ）、cinematic / film / anime スタイルプリセット、
  およびドラフト / 標準 / シネマの 3 段階画質（品質はパラメータであり、占いではない）。ベクターパスのラスタライズ、パスの draw-on と morph。
- **音声処理**：ラウドネス正規化（loudnorm）、ビート検出、スペクトル分析、エンベロープ抽出——
  映像と連動できる（パーティクルの AudioReactive は音声エンベロープに追従）。
- **CPU リファレンスバックエンドはファーストクラス**：GPU のないマシンでも正しい画像を出力でき、同時に GPU の正解判定基準ともなる。

### 何を出すか（デリバリー）

- **`render_movie` で直行**：DSL v2 のコンパイル産物に対し 1 回の呼び出しで「フレーム単位の評価 → tile レンダリング →
  デリバリー」までを通し、`.prism` の単一コマンドで完成品を出力する（`prism-app render demo.prism`）。未対応のノードは正直にエラーを返し、サイレントに近似しない。
- **プレビューの閉ループ**：`prism-app preview` でプレビューサービスを起動。エディタは Draft モード（幾何パターンでフレームレートを維持）/
  Real モード（低 spp の実レンダリング、非同期リフレッシュ）の 2 モードプレビュー。
- **自作エンコーダ + レート制御**：I/P フレームの JPEG 領域時間方向エンコーダが独自の `PRMV` コンテナ（v2/v3 の 2 世代
  schema）を使用。water-filling レート制御 + qpel サブピクセル動き推定（実測で PSNR +2.577 dB、
  サイズ −21.6%）。
- **ffmpeg 協調チェーン**：ffmpeg があれば stdin の rawvideo パイプ経由で H.264/HEVC へ変換。
  ffmpeg がなければ自動的に MJPEG へ降格し、**エンジンは常に出力できる**。バイナリは一切同梱しない。
- **EXR シーケンス納品**：フレームシーケンスをそのまま EXR で書き出し、ポストプロダクションパイプラインへ渡す。
- **分散ファーム**：フレーム単位の配布 + N プロセスの worker + 決定論的マージ——並列 MD5 == 直列 MD5 であり、
  中断からの再開計算にも対応。

### どう統合するか（エコシステム）

- **MCP ツール面**：stdio JSON-RPC の 3 ツール——`scene_check`（コンパイル + 試しレンダリング + フレーム数見積）、
  `render_submit`（決定論的 job id + フレーム単位 MD5 ダイジェスト）、`diagnose`（三段式のエラー特定）。
- **ノード SDK**：セマンティックバージョニングのネゴシエーション（0.x の特例を含む）。パッケージにはサンプルノードパッケージを同梱
  （`examples/sdk-nodes/cinematic-basic/`、コンパイル可能 / ロード可能 / ネゴシエーション可能）。
- **ハードウェアエンコード検出の骨組み**：`prism-ffi-hwenc` の 3 プラットフォーム（VideoToolbox / NVENC / VAAPI）
  検出チェーン + feature gate マトリクス。プラットフォームが合わない場合は明確に非対応を返す。

> 詳細な能力の境界（何が**できないか**）は、パッケージを展開した後の `开发者指南.md` の「誠実な境界」節を参照——
> そこには現時点の実状が記されている（曲線プリミティブは CPU のみ、GPU 加速比は対象ハードウェアでの実測が必要、
> ハードウェアエンコードは対象プラットフォームごとに接続…）。本書は全体像を示すだけで、誇張はしない。

---

## なぜ単一ファイル配布なのか

- **監査可能**：1 つのテキストファイルであり、人の目で読め、`diff` が取れ、ハッシュ検証できる。リリースした時点で凍結される。
- **再現可能**：パッケージングは決定論的——同じ入力に対して 2 回パッケージしても SHA-256 が完全に一致する。
- **可搬**：巨大なリポジトリを clone する必要はない。1 つのファイルをメールやチャットで送るだけでよく、オフラインでも復元・ビルドできる。
- **トレードオフの説明**：単一ファイル配布は「GitHub 上で直接コードを閲覧する」利便性を手放す代わりに、「1 つのファイルで全部を持ち帰り、バイト単位で検証できる」決定論を手に入れる。展開後に得られるのは通常の Cargo workspace であり、一般的な Rust プロジェクトと変わらない。

---

## クイックスタート

前提：[Rust](https://www.rust-lang.org/)（`cargo` を含む）と [Node.js](https://nodejs.org/) 18+ をインストールしておく。

```bash
# 1. リポジトリ内の prism-engine-v2.0-source.txt をダウンロードし、ローカルファイルとして保存する
# 2. テキストエディタで開き、"UNPACK SCRIPT BEGIN" と "UNPACK SCRIPT END" の
#    間にある Node.js スクリプトを、同じディレクトリに unpack.cjs として保存する
# 3. 展開を実行する（数秒程度、622 ファイルを復元）
node unpack.cjs
# 4. 復元されたディレクトリ（workspace のルート）で全テストを実行する
cargo test --workspace --release
# 期待値：3714 passed / 0 failed
```

展開後は標準的な Cargo workspace（`src/` 配下に 28 個の crate）が得られ、以降の `cargo` コマンドは一般的な Rust プロジェクトと同じである。

> モバイル端での wasm 検証経路：`rustup target add wasm32-unknown-unknown`、その後 `cargo build -p prism-core --target wasm32-unknown-unknown`。

---

## リポジトリ構造（展開後）

| パス | 説明 |
|------|------|
| `src/` | **28 個の crate** のソース（リポジトリ内は 29 個。元の名前は `crates/` で、配布時に `src/` へ再構成。デモ crate `prism-anim-demo` はパッケージに入らない） |
| `Compliance/` | コンプライアンス宣言：ゼロ依存の検証 / サードパーティ声明 / アセット登録（リポジトリ内の元の名前は `docs/合规/`） |
| `开发者指南.md` | 自己完結型の開発者ドキュメント：導入 / アーキテクチャ / 数学カーネル / ゲート / 拡張（リポジトリ内の元の名前は `docs/开发者指南.md`） |
| `tools/` | ゲートスクリプト（ゴースト書き込み防御、衛生スキャン、コンプライアンス自己検査） |
| `.github/workflows/` | CI 設定 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | ビルドに必須のもの（workspace members は既に `src/*` を指している） |
| `examples/` | `prism-dsl` のテスト用サンプルとサンプルノードパッケージ（`include_str!` で読み込むため、欠けるとコンパイルが通らない） |

> 設計の権威ドキュメント（`docs/设计/`）は**配布物に含まれない**。リリース側の完全なリポジトリにのみ残る。

---

## モジュール概要

配布パッケージ内の **28 個の crate** を責務別にグループ化する（リポジトリには別にデモ crate `prism-anim-demo` があり、
リポジトリ内でのみ再現可能なサンプルとして存在し、パッケージには入らない）：

| カテゴリ | crate |
|------|-------|
| カーネル | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| メディア I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| レンダリング | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| アニメーション/生成 | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| テキスト/UI | prism-text · prism-rt · prism-app |
| 高度 | prism-ai · prism-dsl · prism-infra · prism-author |
| バイパス（feature gate） | prism-ffi-hwenc |

---

## ビルドとテスト

```bash
# 全リポジトリのスモークテスト（release、推奨）
cargo test --workspace --release
# 等価なエイリアス
cargo xtest
# 単一 crate の簡易検証
cargo test -p prism-core --release
```

- 全テスト：**3714 passed / 0 failed / 0 ignored**（本バージョンは**展開ツリー**上で実走済み：テストブロック 166 個。
  リポジトリ内は 167 ブロックで、差分の 1 ブロックはデモ crate の空のテストブロックであり、ケース数は同じ）。
- CI：**6 個の job**——ゴースト書き込み防御（ghost-guard）/ コンパイル + 全テスト（native）/ wasm32 ビルド（wasm）/
  WGSL 静的検査（wasm-verify）/ 公式 OpenEXR との実装間ビットストリーム突き合わせ（interop）/ node ホストでのクロス判定（lint）。
- ローカルゲートスクリプト 3 個（すべて依存ゼロ、`node` のみを使用、下記）+ パッケージングスクリプト 1 個。
  うちコンプライアンス自己検査は **9/9 通過**、逆方向の自己証明は **8/8 で赤判定**。

ローカルでゲートを実行する（Node 18+ が必要）：

```bash
node tools/ghost_guard.mjs --check        # ゴースト書き込み防御：G1 重複定義/G2 厳密な上位集合/G3 行頭ドキュメント/G4 コンパイル
node tools/hygiene_scan.mjs               # オープンソース衛生：残留する内部用語を 1 件ずつ列挙（ファイル:行番号）、ヒットで exit 1
node tools/compliance_check.mjs           # コンプライアンス自己検査：9/9 通過が期待値
node tools/compliance_check.mjs --self-test   # 逆方向の自己証明：赤判定が期待値。判定基準に識別力があることを示す
```

> **オープンソース衛生スキャンには現在も既存のヒットが残っている**：リポジトリ内に初期の内部用語が残留しており、
> うち**パッケージとして配布されるドキュメント層はすでにゼロ**（README / 開発者ガイド / Compliance）。残りは
> **配布パッケージ内のソースコメント**と**配布されないリポジトリファイル**にあり、事実に基づいて分類のうえ
> 独立したクリーンアップ項目としている。スクリプトの存在は、残留を**可視化し、1 件ずつ消せるようにする**ためであり、
> きれいなふりをするためではない——「ヒット数が 0 でない」ことをゲートの機能不全と解釈しないでほしい。

---

## コンプライアンス

エンジンは**サードパーティコード依存ゼロ**を約束し、自己検査で検証している：

- **C1 サードパーティコード依存ゼロ**：全 130 個の依存（実行時 110 個 + テスト 20 個）はすべて本リポジトリの `path` 依存。`build.rs` なし、ネイティブリンク指示なし、`Cargo.lock` は本リポジトリの 29 パッケージをロックしている。
- **C2 アセット登録が完備**：5 項目のアセット主表のフィールドが揃い、自社開発化の状態は統制語彙に収まっている。OFL 1.1 フォントライセンスの全文はフォントとともに `src/prism-text/font_data/LICENSE` に置かれている。
- **C3 サードパーティ帰属声明**：`Compliance/THIRD-PARTY-NOTICES.md` が存在し、「データアセット ≠ コード依存」を明確に区別している。

> 注：本リポジトリが配布するのは**ソース**である。フォントアセット（Prism Sans SC サブセット）は SIL OFL 1.1 でライセンスされ、**ライセンス全文はフォントとともに** `src/prism-text/font_data/LICENSE` に置かれている。アセットの出所とライセンス登録は `Compliance/assets/资产登记.md` を参照。

---

## 決定論的パッケージング（リリース側）

完全なリポジトリから配布 TXT を再生成する場合（バージョン番号は引数で渡せる。既定では現在のバージョン v2.0 を出力）：

```bash
node tools/pack_source_txt.mjs          # 既定 → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # 等価な書き方（バージョン番号を明示的に渡す）
node tools/pack_source_txt.mjs 1.0      # v1.0 時代の成果物名に戻す（v1.0 のソース上で実行すれば旧パッケージをバイト単位で再現）
node tools/pack_source_txt.mjs --verify # パッケージ直後に展開してループバック検証（ファイル単位・バイト単位でソースツリーを比較）
# 2 回連続で実行し、出力 SHA-256 が一致すれば決定論的パッケージングが成立した証明になる
```

パッケージ対象は厳密に次に限定される：`crates/`（→ `src/`）、`docs/合规/`（→ `Compliance/`）、`docs/开发者指南.md`、`tools/`、`.github/`、`.cargo/`、`examples/`、`Cargo.toml`、`Cargo.lock`。`docs/设计/`、`target/`、`node_modules/`、`dist/` 自体、および一切の中間生成物は意図的に**除外**する。

---

## コントリビュート

Issue と Pull Request を歓迎する。提出前に以下を維持すること：

- `cargo test --workspace --release` がオールグリーンであること；
- `node tools/hygiene_scan.mjs` で内部用語を**新たに増やさない**こと（スクリプトは既存分を含む全ヒットを列挙する）；
- `node tools/compliance_check.mjs` が 9/9 通過すること。

より詳細なアーキテクチャと拡張のガイドは、パッケージを展開した後の `开发者指南.md` を参照。

---

## ライセンス

本プロジェクトは **Apache License 2.0** でリリースされている。詳細は [`LICENSE`](../LICENSE) ファイルを参照。

> コンプライアンス上の注意：Apache-2.0 は著作権表示とライセンス表示の保持を要求する。本エンジンを基に再配布する場合は、`Compliance/` 配下の帰属声明とアセット登録を保持すること。
