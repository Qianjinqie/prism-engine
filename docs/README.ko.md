[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | **한국어** | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> 의존성 제로, 결정론적 Rust 미디어 엔진 —— **단일 파일 TXT 배포**: 14 MB 텍스트 파일 하나를 내려받아 언팩 스크립트를 실행하면 완전한 26-crate 소스 트리를 얻을 수 있으며, npm / pip / ffmpeg / OpenEXR가 필요 없습니다.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## 이것은 무엇인가

Prism Engine은 Rust로 작성된 서드파티 코드 의존성이 전혀 없는 미디어 엔진으로, 이미지, 오디오, 지오메트리, 물리, 렌더링, DSL 등의 모듈을 다룹니다. 전체 103개 의존성은 모두 본 저장소의 `path` 의존성입니다——네트워크에 의존하지 않고, 패키지 관리자가 필요 없으며, 공급망 리스크도 없습니다.

이 엔진은 **단일 파일 소스 배포**를 채택합니다: 엔진 전체(481개 파일, 26개 crate)를 하나의 UTF-8 텍스트 `prism-engine-v1.0-source.txt`로 병합했습니다. 이 파일만 있으면 저장소와 완전히 동일한 디렉터리 트리를 로컬에서 복원할 수 있습니다.

## 왜 단일 파일 배포인가

- **감사 가능**: 하나의 텍스트 파일로, 육안으로 읽을 수 있고 `diff`할 수 있으며 해시 검증도 가능합니다. 배포 즉시 동결됩니다.
- **재현 가능**: 패키징은 결정론적입니다——동일한 입력으로 두 번 패키징하면 SHA-256이 완전히 일치합니다.
- **휴대 가능**: 방대한 저장소를 clone할 필요 없이, 파일 하나를 메일이나 채팅으로 전달할 수 있고 오프라인에서도 빌드를 복원할 수 있습니다.
- **트레이드오프 설명**: 단일 파일 배포는 "GitHub에서 코드를 직접 열람할 수 있는" 편의를 내어놓는 대신, "파일 하나로 전부를 가져가고 바이트 단위로 검증 가능한" 결정성을 얻습니다. 언팩 후 얻어지는 것은 평범한 Cargo workspace로, 일반적인 Rust 프로젝트와 다름이 없습니다.
- **가장 중요한 것**: 파일 하나만 업로드하면 그만입니다(doge

---

## 빠른 시작

사전 준비: [Rust](https://www.rust-lang.org/)(`cargo` 포함)와 [Node.js](https://nodejs.org/) 18+를 설치합니다.

```bash
# 1. 저장소의 prism-engine-v1.0-source.txt를 내려받아 로컬 파일로 저장합니다
# 2. 텍스트 편집기로 열어 "UNPACK SCRIPT BEGIN"과 "UNPACK SCRIPT END"
#    사이의 Node.js 스크립트를 같은 디렉터리의 unpack.cjs로 저장합니다
# 3. 언팩을 실행합니다(수 초 소요, 481개 파일 복원)
node unpack.cjs
# 4. 복원된 소스 트리로 이동해 전체 테스트를 실행합니다
cd src/..
cargo test --workspace --release
# 예상 결과: 3271 passed / 0 failed
```

언팩 후에는 표준 Cargo workspace(`src/` 아래 26개 crate)를 얻으며, 이후의 `cargo` 명령은 일반적인 Rust 프로젝트와 동일합니다.

> 모바일에서의 wasm 검증 체인: `rustup target add wasm32-unknown-unknown`을 실행한 뒤 `cargo build -p prism-core --target wasm32-unknown-unknown`을 실행합니다.

---

## 저장소 구조(언팩 후)

| 경로 | 설명 |
|------|------|
| `src/` | 26개 crate의 소스(저장소 내 원래 이름은 `crates/`, 배포 시 `src/`로 재구성) |
| `Compliance/` | 컴플라이언스 선언: 제로 의존성 검증 / 서드파티 선언 / 자산 등록(저장소 내 원래 이름은 `docs/合规/`) |
| `开发者指南.md` | 셀프컨테인드 개발자 문서: 시작하기 / 아키텍처 / 수학 커널 / 게이트 / 확장(저장소 내 원래 이름은 `docs/开发者指南.md`) |
| `tools/` | 게이트 스크립트(고스트 쓰기 방어, 위생 스캔, 컴플라이언스 셀프 체크) |
| `.github/workflows/` | CI 설정 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | 빌드에 필요한 파일(workspace members는 이미 `src/*`를 가리킴) |
| `examples/` | `prism-dsl` 테스트용 샘플(`include_str!`로 읽어 들이며, 없으면 컴파일되지 않음) |

> 설계 권위 문서(`docs/设计/`)는 **배포에 포함되지 않으며**, 배포 측의 완전한 저장소에만 남아 있습니다.

---

## 모듈 개요

26개 crate를 역할별로 그룹화한 목록입니다:

| 카테고리 | crate |
|------|-------|
| 커널 | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| 미디어 I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| 렌더링 | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| 애니메이션/생성 | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| 텍스트/UI | prism-text · prism-rt · prism-app |
| 고급 | prism-ai · prism-dsl · prism-infra |

---

## 빌드와 테스트

```bash
# 전체 저장소 스모크 테스트(release, 권장)
cargo test --workspace --release
# 동등한 별칭
cargo xtest
# 단일 crate 빠른 검증
cargo test -p prism-core --release
```

- 전체 테스트: **3271 passed / 0 failed**(새로 언팩한 트리에서 실제로 실행해 검증했습니다).
- 게이트: 총 14개로, 고스트 쓰기 방어(G1~G4), 오픈소스 위생 스캔(내부 흔적 0건), 컴플라이언스 셀프 체크(9/9), 컴플라이언스 역방향 자기 증명(8/8, 위반 시 반드시 실패), wasm32 빌드 체인이 포함됩니다.

로컬에서 게이트를 실행하려면(Node 18+ 필요):

```bash
node tools/ghost_guard.mjs --check        # 고스트 쓰기 방어: G1 중복 정의/G2 엄격 상위 집합/G3 최상단 문서/G4 컴파일
node tools/hygiene_scan.mjs               # 오픈소스 위생: “干净（0 处内部痕迹）”라고 보고되어야 함
node tools/compliance_check.mjs           # 컴플라이언스 셀프 체크: 9/9 통과해야 함
node tools/compliance_check.mjs --self-test   # 역방향 자기 증명: 적색 판정이어야 하며, 판정 기준의 판별력을 증명
```

---

## 컴플라이언스

엔진은 **서드파티 코드 의존성 제로**를 약속하며, 셀프 체크로 검증했습니다:

- **C1 서드파티 코드 의존성 제로**: 전체 103개 의존성이 모두 본 저장소의 `path` 의존성입니다. `build.rs` 없음, 네이티브 링크 지시문 없음, `Cargo.lock`에는 26개 본 저장소 패키지가 고정되어 있습니다.
- **C2 자산 등록 완비**: 5개 자산의 마스터 테이블 필드가 모두 갖춰져 있고, 자체 개발화 상태는 통제 어휘집에 속합니다. OFL 1.1 폰트 라이선스 전문은 폰트와 함께 `src/prism-text/font_data/LICENSE`에 있습니다.
- **C3 서드파티 귀속 선언**: `Compliance/THIRD-PARTY-NOTICES.md`가 존재하며, "데이터 자산 ≠ 코드 의존성"을 명확히 구분합니다.

> 참고: 본 저장소가 배포하는 것은 **소스 코드**입니다. 폰트 자산(Prism Sans SC 서브셋)은 SIL OFL 1.1로 라이선스되며, **라이선스 전문은 폰트와 함께** `src/prism-text/font_data/LICENSE`에 있습니다. 자산 출처와 라이선스 등록은 `Compliance/assets/资产登记.md`를 참고하세요.

---

## 라이선스

본 프로젝트는 **Apache License 2.0**으로 배포됩니다. 자세한 내용은 [`LICENSE`](../LICENSE) 파일을 참고하세요.

> 컴플라이언스 안내: Apache-2.0은 저작권 및 라이선스 고지의 보존을 요구합니다. 본 엔진을 기반으로 재배포할 때는 `Compliance/` 아래의 귀속 선언과 자산 등록을 보존해 주세요.
