[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | **한국어** | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> 제로 의존성, 결정론적 Rust 미디어 엔진 — **단일 파일 TXT 배포**: 17 MB짜리 텍스트 파일 하나를 내려받아 언팩 스크립트를 실행하면 완전한 소스 트리(28개 crate)를 얻을 수 있으며, npm / pip / ffmpeg / OpenEXR이 필요 없습니다.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

(배지는 정적 사실 배지이며, 수치 기준은 아래와 같습니다. `crates-28` = **배포 패키지 내**의 crate 수이고, 저장소에는 패키지에 들어가지 않는 데모 crate가 1개 더 있으므로 저장소 기준 수치는 29입니다. CI에 연결하면 동적 배지로 교체할 수 있습니다.)

---

## 무엇인가

Prism Engine은 Rust로 작성된, 제3자 코드 의존성이 없는 미디어 엔진입니다. 이미지 읽기, 레이아웃, 화면 그리기, 타임라인 편집, 최종 결과물 출력까지 전부 하나의 결정론적 엔진 안에서 완료됩니다. 전체 110개 런타임 의존성(추가로 20개 테스트 의존성)은 모두 이 저장소의 `path` 의존성입니다 — 네트워크에 의존하지 않고, 패키지 관리자가 필요 없으며, 공급망 리스크가 없습니다.

**단일 파일 소스 배포** 방식을 채택합니다: 엔진 전체(622개 파일, 28개 crate)가 하나의 UTF-8 텍스트 파일 `prism-engine-v2.0-source.txt`로 병합됩니다. 이 파일만 있으면 로컬에서 저장소와 완전히 동일한 디렉터리 트리를 복원할 수 있습니다.

---

## 엔진이 할 수 있는 것

한 문장 요약: **이미지는 어떤 시점의 화면이고, 영상은 한 구간 동안 이어진 화면들의 연속입니다** — 동일한 장면 서술에서 특정 순간을 고정하면 단일 프레임이 되고, 연속으로 진행시켜 기록하면 영상이 되며, 정지 이미지와 동적 이미지가 하나의 능력 집합을 공유합니다. 아래는 "당신이 하려는 일"을 기준으로 한 다섯 블록입니다: **읽어들이기 → 배치하기 → 그리기 → 결과물 출력 → 자신의 툴체인에 연결하기**.

### 무엇을 읽는가 (입력)

- **통합 디코딩 파사드**: `decode_image(bytes, ext)` 단일 진입점이 정적 이미지를 분배합니다 — PNG(비인터레이스 8/16bit), 베이스라인 JPEG(고정소수점 IDCT, 플랫폼 간 비트 단위 결정론), **매직 넘버 우선, 확장자 폴백**(확장자를 잘못 적어도 디코딩 가능); 매트릭스 밖이거나 외부 협조가 필요한 포맷은 일관되게 3단계 오류(어디서 / 무엇이 / 어떻게 고칠지)로 보고하고 ffmpeg 협조 방안을 제시하며, **절대 조용히 실패하지 않습니다**.
- **동영상 / 오디오 / 컨테이너**: GIF, APNG 네이티브 프레임별 디코딩(`prism-media`); WAV 내장 디코딩, mp3/aac/FLAC은 ffmpeg 탐지 경로(`prism-audio`); MP4-MJPEG 재읽기(`decode_mp4_video`).
- **EXR 풀 세트**: 읽기·쓰기, 압축, DWA, 그리고 공식 OpenEXR과의 **교차 구현 비트스트림 대조**(자체 작성·자체 판독이 아님).
- **카메라 RAW**: 베이스라인 RAW 디코딩. **경계**: 실제 DNG의 `Compression=7`(lossless JPEG)은 구현되지 않아, 실제 카메라가 촬영한 DNG는 읽을 수 없습니다 — 「정직한 경계」에 적어 두었고, 숨기지 않습니다.
- **3D 에셋**: glTF / OBJ 가져오기, USD(`.usda`) 파싱과 MaterialX 파라미터 패키지, mesh를 렌더링 장면에 투입 가능.
- **외부 폰트**: `load_font_file`로 서드파티 TTF/OTF를 직접 로드하고, 폴백 체인이 단계별로 해석하며, 글리프가 빠지면 두부 블록 대신 명확한 진단을 제공합니다.

### 무엇을 배치하는가 (창작 문법)

- **DSL v2 전체 블록 문법**: 하나의 `.prism`으로 "무엇을, 어떻게 배치하고, 어떻게 움직이며, 어떻게 보이고, 무엇으로 내보낼지"를 서술하며, 컴파일 산출물은 노드 그래프 + 속성 채널 + 타임라인입니다.
- **다섯 가지 컴비네이터**: `title_card` / `lower_third` / `split_screen` / `ken_burns` / `morph_between` 고수준 컴비네이터가 노드 그래프로 전개되어, "90초 완성 영상"을 직접 장면 코드를 쓰는 일에서 조립하는 일로 끌어내립니다.
- **전환과 morph**: Fade / Wipe / Dissolve / Push 모두 파라미터화(길이, 이징, 방향), Morph 폴리라인 변형, 장면 시퀀스는 임의 시점에서 샘플링 가능.
- **수식 조판**: `math_ast` + `math_layout`이 위첨자·아래첨자 / 분수 / 근호 / 대형 연산자 / 행렬을 조판하고, DSL `formula` 블록으로 직접 조판하며 본문과 같은 베이스라인을 씁니다.

### 무엇을 그리는가 (렌더링과 시뮬레이션)

- **오프라인 포토리얼 렌더링**: 경로 추적 + 물리적으로 올바른 BSDF(확산 / 거울 / 유리 / 금속), 유리와 금속은 확률적 역방향(Neumann 급수 몬테카를로), 볼류메트릭 매질과 서브서피스 산란 지원.
- **곡선 프리미티브 네 종**: 원기둥 / 원뿔 / 토러스 / 회전 곡면, 전부 해석적 교차(4차 방정식은 Ferrari 폐형해로 정확한 근); 임의 삼각 mesh + SAH-BVH, 이중 경로 parity 대조.
- **GPU compute**: 디바이스 독립 compute 시맨틱 계층 + Vulkan dispatch + 직접 작성한 최소 SPIR-V; Mesa lavapipe 소프트웨어 ICD 환경에서 GPU가 없는 플랫폼에서도 완전히 실행 가능(headless 검증 가능).
- **물리와 파티클**: XPBD 통합 솔버(강체 / 연체 / 천 / 조인트 / 유체 / 파괴), 연기 FLIP, 파티클 시스템(AudioReactive가 오디오 엔벨로프와 연동).
- **제너레이티브 아트와 후처리**: 유동장 / L-system / 세포 자동자 / 반응 확산 / 프랙탈 / Voronoi; blur / bloom / halation / 필름 그레인 / NPR 4스타일(toon / watercolor / halftone / oil) / 3-in-1 노이즈 리덕션.
- **컬러 그레이딩과 벡터**: 2차 컬러 그레이딩(HSL 선택 영역 + RGB 커브), cinematic / film / anime 스타일 프리셋과 드래프트/표준/시네마 3단계 품질(품질은 파라미터이지 미신이 아님); 벡터 경로 래스터화, 경로 draw-on과 morph.
- **오디오 처리**: 라우드니스 정규화(loudnorm), 비트 검출, 스펙트럼 분석, 엔벨로프 추출 — 화면과 연동 가능(파티클 AudioReactive가 오디오 엔벨로프를 따름).
- **CPU 레퍼런스 백엔드는 일급 시민**: GPU가 없는 머신에서도 올바른 이미지를 출력하며, 동시에 GPU의 정답 판정 기준입니다.

### 무엇을 출력하는가 (전달)

- **`render_movie` 직행**: DSL v2 컴파일 산출물을 한 번의 호출로 "프레임별 평가 → tile 렌더링 → 전달"까지 완료하고, `.prism` 단일 명령으로 완성 영상을 출력합니다(`prism-app render demo.prism`); 지원하지 않는 노드는 조용히 근사하지 않고 그대로 오류를 보고합니다.
- **프리뷰 루프**: `prism-app preview`로 프리뷰 서버를 기동; 에디터 Draft 단계(기하 패턴으로 프레임레이트 유지) / Real 단계(낮은 spp 실제 렌더링, 비동기 갱신)의 두 단계 프리뷰.
- **자체 인코더 + 레이트 제어**: I/P 프레임 JPEG 도메인 시간축 인코더가 자체 `PRMV` 컨테이너를 사용(v2/v3 두 세대 schema), water-filling 레이트 제어 + qpel 서브픽셀 움직임 추정(실측 PSNR +2.577 dB, 용량 −21.6%).
- **ffmpeg 협력 체인**: ffmpeg가 있으면 → stdin rawvideo 파이프로 H.264/HEVC 변환; ffmpeg가 없으면 → 자동으로 MJPEG로 저하, **엔진은 언제나 영상을 출력할 수 있음**; 어떤 바이너리도 번들하지 않음.
- **EXR 시퀀스 전달**: 프레임 시퀀스를 EXR로 직접 기록해 후반 파이프라인에 투입.
- **분산 팜**: 프레임 단위 분배 + N 프로세스 worker + 결정론적 병합 — 병렬 MD5 == 직렬 MD5, 중단 지점부터 재개 지원.

### 어떻게 통합하는가 (생태계)

- **MCP 도구 인터페이스**: stdio JSON-RPC 세 가지 도구 — `scene_check`(컴파일 + 시험 렌더 + 프레임 수 예측), `render_submit`(결정론적 job id + 프레임별 MD5 다이제스트), `diagnose`(3단계 오류 위치 파악).
- **노드 SDK**: 시맨틱 버전 협상(0.x 특례 포함), 패키지에 예제 노드 패키지 동봉(`examples/sdk-nodes/cinematic-basic/`, 컴파일 / 로드 / 협상 가능).
- **하드웨어 인코딩 탐지 골격**: `prism-ffi-hwenc` 세 플랫폼(VideoToolbox / NVENC / VAAPI) 탐지 체인 + feature gate 매트릭스, 플랫폼이 맞지 않으면 명확히 미지원으로 보고.

> 상세한 능력 경계(**무엇이 불가능한지**)는 패키지 내 `开发者指南.md`의 「정직한 경계」 절을 참조하십시오 — 그곳에는 현재의 실제 상태(곡선 프리미티브는 CPU만 사용, GPU 가속비는 대상 하드웨어에서 실측 필요, 하드웨어 인코딩은 대상 플랫폼별 연동…)가 기록되어 있습니다. 이 문서는 전체 조망만 제공하며, 과장하지 않습니다.

---

## 왜 단일 파일 배포인가

- **감사 가능**: 텍스트 파일 하나이므로 사람이 읽을 수 있고, `diff`할 수 있으며, 해시로 검증할 수 있고, 배포되는 순간 고정됩니다.
- **재현 가능**: 패키징이 결정론적이어서 동일한 입력으로 두 번 패키징하면 SHA-256이 완전히 일치합니다.
- **휴대 가능**: 거대한 저장소를 clone할 필요 없이 파일 하나를 메일이나 채팅으로 전달할 수 있고, 오프라인에서도 복원·빌드가 가능합니다.
- **트레이드오프 설명**: 단일 파일 배포는 "GitHub에서 코드를 직접 열람하는" 편의를 내주는 대신, "파일 하나로 전부 가져가고 바이트 단위로 검증 가능하다"는 결정성을 얻습니다. 언팩 후에는 일반 Cargo workspace가 나오며, 일반적인 Rust 프로젝트와 다를 바 없습니다.

---

## 빠른 시작

사전 준비: [Rust](https://www.rust-lang.org/)(`cargo` 포함)와 [Node.js](https://nodejs.org/) 18+ 설치.

```bash
# 1. 저장소의 prism-engine-v2.0-source.txt를 내려받아 로컬 파일로 저장
# 2. 텍스트 편집기로 열어 "UNPACK SCRIPT BEGIN"과 "UNPACK SCRIPT END"
#    사이의 Node.js 스크립트를 같은 디렉터리에 unpack.cjs로 따로 저장
# 3. 언팩 실행(수 초 소요, 622개 파일 복원)
node unpack.cjs
# 4. 복원된 디렉터리(즉 workspace 루트)에서 전체 테스트 실행
cargo test --workspace --release
# 예상: 3714 passed / 0 failed
```

언팩 후에는 표준 Cargo workspace(`src/` 아래 28개 crate)를 얻으며, 이후 `cargo` 명령은 일반 Rust 프로젝트와 같습니다.

> 모바일 wasm 검증 경로: `rustup target add wasm32-unknown-unknown`, 이어서 `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## 저장소 구조 (언팩 후)

| 경로 | 설명 |
|------|------|
| `src/` | **28개 crate**의 소스(저장소 내에는 29개, 원래 이름 `crates/`, 배포 시 `src/`로 재구성; 데모 crate `prism-anim-demo`는 패키지에 포함되지 않음) |
| `Compliance/` | 규정 준수 선언: 제로 의존성 검증 / 서드파티 선언 / 자산 등록(저장소 내 원래 이름 `docs/合规/`) |
| `开发者指南.md` | 자체 완결형 개발자 문서: 시작하기 / 아키텍처 / 수학 커널 / 게이트 / 확장(저장소 내 원래 이름 `docs/开发者指南.md`) |
| `tools/` | 게이트 스크립트(유령 쓰기 방어, 위생 스캔, 규정 준수 자체 점검) |
| `.github/workflows/` | CI 설정 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | 빌드 필수 요소(workspace members는 이미 `src/*`를 가리킴) |
| `examples/` | `prism-dsl` 테스트용 샘플과 예제 노드 패키지(`include_str!`로 읽으므로 없으면 컴파일되지 않음) |

> 설계 권위 문서(`docs/设计/`)는 **배포에 포함되지 않으며**, 배포자의 전체 저장소에만 남습니다.

---

## 모듈 개요

배포 패키지 내 **28개 crate**를 책임별로 묶었습니다(저장소에는 데모 crate `prism-anim-demo`가 하나 더 있으며, 재현 가능한 예제로서 저장소에만 존재하고 패키지에는 들어가지 않습니다):

| 분류 | crate |
|------|-------|
| 커널 | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| 미디어 I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| 렌더링 | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| 애니메이션/생성 | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| 텍스트/UI | prism-text · prism-rt · prism-app |
| 고급 | prism-ai · prism-dsl · prism-infra · prism-author |
| 우회(feature gate) | prism-ffi-hwenc |

---

## 빌드와 테스트

```bash
# 전 저장소 스모크(release, 권장)
cargo test --workspace --release
# 동일한 별칭
cargo xtest
# 단일 crate 빠른 검증
cargo test -p prism-core --release
```

- 전체 테스트: **3714 passed / 0 failed / 0 ignored**(이번 버전은 **언팩된 트리**에서 실제로 실행: 테스트 블록 166개; 저장소 내에는 167개이며, 차이가 나는 1개는 데모 crate의 빈 테스트 블록으로 케이스 수는 동일).
- CI: **6개 job** — 유령 쓰기 방어(ghost-guard) / 컴파일 + 전체 테스트(native) / wasm32 빌드(wasm) / WGSL 정적 검사(wasm-verify) / 공식 OpenEXR과 교차 구현 비트스트림 대조(interop) / node 호스트 교차 판정(lint).
- 로컬 게이트 스크립트 3개(완전한 제로 의존성, `node`만 사용, 아래) + 패키징 스크립트 1개; 이 가운데 규정 준수 자체 점검은 **9/9 통과**, 역방향 자기 증명은 **8/8 red 판정**.

로컬에서 게이트 실행(Node 18+ 필요):

```bash
node tools/ghost_guard.mjs --check        # 유령 쓰기 방어: G1 중복 정의/G2 엄격 초집합/G3 최상단 문서/G4 컴파일
node tools/hygiene_scan.mjs               # 오픈소스 위생: 잔존 내부 용어를 항목별로 나열(파일:행번호), 적중 시 exit 1
node tools/compliance_check.mjs           # 규정 준수 자체 점검: 9/9 통과해야 함
node tools/compliance_check.mjs --self-test   # 역방향 자기 증명: red로 판정되어야 하며, 판정 기준의 판별력을 입증
```

> **오픈소스 위생 스캔에는 현재도 기존 적중이 남아 있습니다**: 저장소에 초기 내부 용어가 남아 있으며, 그중 **패키지로 배포되는 문서 계층은 이미 0건**(README / 개발자 가이드 / Compliance)이고, 나머지는 **배포 패키지 내 소스 주석**과 **패키지에 포함되지 않는 저장소 파일**에 있습니다. 사실에 따라 분류해 별도의 정리 항목으로 등록했습니다. 스크립트의 존재 목적은 잔존 항목을 **가시적이고 항목별로 정리 가능하게** 만드는 것이지, 깨끗한 척하는 것이 아닙니다 — "적중 수가 0이 아님"을 게이트 실패로 받아들이지 마십시오.

---

## 규정 준수

엔진은 **제3자 코드 의존성 제로**를 약속하며, 자체 점검으로 검증했습니다:

- **C1 제3자 코드 의존성 제로**: 전체 130개 의존성(런타임 110개 + 테스트 20개)이 모두 이 저장소의 `path` 의존성; `build.rs` 없음, 네이티브 링크 지시자 없음, `Cargo.lock`이 저장소 내 29개 패키지를 고정.
- **C2 자산 등록 완비**: 5개 자산 마스터 테이블의 필드가 모두 갖추어지고, 자체 개발화 상태가 통제 어휘 안에 있음; OFL 1.1 폰트 라이선스 전문이 폰트와 함께 `src/prism-text/font_data/LICENSE`에 위치.
- **C3 서드파티 귀속 선언**: `Compliance/THIRD-PARTY-NOTICES.md`가 존재하며, "데이터 자산 ≠ 코드 의존성"을 명확히 구분.

> 참고: 이 저장소가 배포하는 것은 **소스**입니다. 폰트 자산(Prism Sans SC 서브셋)은 SIL OFL 1.1로 라이선스되며, **라이선스 전문은 폰트와 함께** `src/prism-text/font_data/LICENSE`에 위치합니다. 자산 출처와 라이선스 등록은 `Compliance/assets/资产登记.md`를 참조하십시오.

---

## 결정론적 패키징 (배포자)

전체 저장소에서 배포용 TXT를 다시 생성해야 할 경우(버전 번호를 넘길 수 있고, 기본값은 현재 출판 버전 v2.0):

```bash
node tools/pack_source_txt.mjs          # 기본 → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # 동일한 표기(버전 번호를 명시적으로 전달)
node tools/pack_source_txt.mjs 1.0      # v1.0 시절 산출물 이름으로 복귀(v1.0 소스에서 실행하면 기존 패키지를 바이트 단위로 재현)
node tools/pack_source_txt.mjs --verify # 패키징 직후 언팩 후 루프백 검증(소스 트리를 파일별·바이트별로 비교)
# 두 번 연속 실행해 출력 SHA-256이 일치하면 결정론적 패키징이 성립함을 증명
```

패키징 범위는 엄격히 다음으로 한정됩니다: `crates/`(→ `src/`), `docs/合规/`(→ `Compliance/`), `docs/开发者指南.md`, `tools/`, `.github/`, `.cargo/`, `examples/`, `Cargo.toml`, `Cargo.lock`. `docs/设计/`, `target/`, `node_modules/`, `dist/` 자체 및 모든 중간 산출물은 의도적으로 **제외**합니다.

---

## 기여

Issue와 Pull Request를 환영합니다. 제출 전에 다음을 유지해 주십시오:

- `cargo test --workspace --release` 전체 통과;
- `node tools/hygiene_scan.mjs`에서 내부 용어를 **새로 추가하지 않을 것**(스크립트가 기존 잔존 항목을 포함한 모든 적중을 나열합니다);
- `node tools/compliance_check.mjs` 9/9 통과.

더 자세한 아키텍처와 확장 가이드는 패키지 내 `开发者指南.md`를 참조하십시오.

---

## 라이선스

이 프로젝트는 **Apache License 2.0**으로 배포됩니다. 자세한 내용은 [`LICENSE`](../LICENSE) 파일을 참조하십시오.

> 규정 준수 안내: Apache-2.0은 저작권과 라이선스 고지를 보존할 것을 요구합니다. 이 엔진을 기반으로 재배포할 때는 `Compliance/` 아래의 귀속 선언과 자산 등록을 보존해 주십시오.
