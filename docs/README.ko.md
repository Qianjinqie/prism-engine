[English](README.en.md) | [简体中文](../README.md) | [日本語](README.ja.md) | **한국어** | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> 의존성 제로, 결정론적 Rust 미디어 엔진 — **단일 파일 TXT로 배포**: 18 MB짜리 텍스트 하나를 내려받아 압축 해제 스크립트를 실행하면 완전한 소스 트리(28개 crate)를 얻는다. npm / pip / ffmpeg / OpenEXR이 필요 없다.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3929%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## 이것은 무엇인가

Prism Engine은 Rust로 작성된 서드파티 코드 의존성 제로의 미디어 엔진이다. 이미지 읽기, 조판,
화면 그리기, 타임라인 편집, 최종 영상 출력까지 전부 하나의 결정론적 엔진 안에서 완료된다.
전체 110개의 런타임 의존성(추가로 20개의 테스트 의존성)이 모두 이 저장소의 `path` 의존성이다 —
네트워크에 의존하지 않고, 패키지 관리자가 필요 없으며, 공급망 위험이 없다.

**단일 파일 소스 배포** 방식을 쓴다. 엔진 전체(642개 파일, 28개 crate)가 하나의 UTF-8
텍스트 `prism-engine-v2.1-source.txt`로 병합된다. 이 파일 하나만 있으면 저장소와 완전히 동일한
디렉터리 트리를 로컬에서 복원할 수 있다.

---

## 엔진이 할 수 있는 것

한 문장으로 요약하면 이렇다. **이미지는 화면이 어느 한 시점에 보이는 모습이고, 영상은 화면이
일정 시간 동안 이어지는 연속된 모습이다** — 동일한 장면 기술에서 어느 한 순간을 고정하면 단일
프레임이 되고, 연속적으로 진행하며 기록하면 영상이 된다. 정지 이미지와 동영상은 하나의 능력
집합을 공유한다. 아래는 "당신이 하려는 일"을 기준으로 다섯 블록으로 나눈 것이다.
**읽어 들이기 → 배치하기 → 그리기 → 최종 영상 출력 → 당신의 툴체인에 연결하기**.

### 무엇을 읽는가 (입력)

- **통합 디코딩 파사드**: `decode_image(bytes, ext)` 진입점 하나가 정지 이미지를 분기 처리한다 —
  PNG(비인터레이스 8/16bit), 베이스라인 JPEG(고정소수점 IDCT, 크로스 플랫폼 비트 단위 결정론),
  **매직 넘버 우선, 확장자는 폴백**(확장자를 잘못 써도 디코딩 가능). 매트릭스 외이거나 외부
  협력이 필요한 포맷은 일괄 3단계 오류 보고(어디서 / 무엇이 / 어떻게 고칠지)와 함께 ffmpeg
  협력 제안을 내놓으며, **결코 조용히 실패하지 않는다**.
- **애니메이션 / 오디오 / 컨테이너**: GIF, APNG 네이티브 프레임 단위 디코딩(`prism-media`); WAV
  내장 디코딩, mp3/aac/FLAC은 ffmpeg 프로브 경로(`prism-audio`); MP4-MJPEG 리드백
  (`decode_mp4_video`).
- **EXR 풀 세트**: 읽기/쓰기, 압축, DWA를 지원하며, 공식 OpenEXR과 **교차 구현 비트스트림
  대조**를 수행한다(자체 작성-자체 판독이 아니다).
- **카메라 RAW**: 베이스라인 RAW 디코딩. **경계**: 실제 DNG의 `Compression=7`(lossless JPEG)은
  구현되지 않아 실제 카메라로 촬영한 DNG는 읽을 수 없다 — 「정직한 경계」에 적어 두었으며
  숨기지 않는다.
- **3D 에셋**: glTF / OBJ 가져오기, USD(`.usda`) 파싱과 MaterialX 파라미터 패키지, mesh는
  렌더링 장면에 투입할 수 있다.
- **외부 폰트**: `load_font_file`로 서드파티 TTF/OTF를 직접 로드하고, 폴백 체인이 단계별로
  해석하며, 글리프가 없으면 두부 블록 대신 명확한 진단을 제공한다.

### 무엇을 배치하는가 (창작 문법)

- **DSL v2 전체 블록 문법**: 하나의 `.prism` 파일이 "무엇을 원하는지, 어떻게 배치하는지, 어떻게
  움직이는지, 어떻게 보이는지, 무엇으로 내보내는지"를 기술하며, 컴파일 산출물은 노드 그래프 +
  속성 채널 + 타임라인이다.
- **다섯 가지 컴비네이터**: `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` 고수준 컴비네이터가 노드 그래프로 전개되어, "90초 완성 영상"이 수작업 장면
  코드에서 조립 작업으로 내려온다.
- **전환과 morph**: Fade / Wipe / Dissolve / Push 완전 파라미터화(길이, 이징, 방향),
  Morph 폴리라인 변형, 장면 시퀀스는 임의 시점에서 샘플링할 수 있다.
- **수식 조판**: `math_ast` + `math_layout`이 위첨자/아래첨자 / 분수 / 근호 / 대형 연산자 /
  행렬을 조판하고, DSL `formula` 블록으로 바로 조판되며 본문과 같은 베이스라인을 쓴다.

### 무엇을 그리는가 (렌더링과 시뮬레이션)

- **오프라인 포토리얼 렌더링**: 경로 추적 + 물리적으로 올바른 BSDF(난반사 / 정반사 / 유리 / 금속),
  유리와 금속은 확률적 역변환(Neumann 급수 몬테카를로)을 사용하며, 볼류메트릭 매질과 서브서피스
  산란을 지원한다.
- **곡선 프리미티브 4종**: 원기둥 / 원뿔 / 토러스 / 회전 곡면, 전부 해석적 교차 계산(4차 방정식은
  Ferrari 폐형해로 정확한 근); 임의 삼각형 mesh + SAH-BVH, 이중 경로 parity 대조.
- **GPU compute**: 장치 독립적 compute 시맨틱 계층 + Vulkan dispatch + 직접 작성한 최소 SPIR-V;
  Mesa lavapipe 소프트웨어 ICD 아래에서 GPU 없는 플랫폼도 완전히 실행된다(headless 검증 가능).
- **물리와 파티클**: XPBD 통합 솔버(강체 / 연체 / 천 / 조인트 / 유체 / 파괴), 연기 FLIP,
  파티클 시스템(AudioReactive가 오디오 엔벨로프와 연동).
- **제너러티브 아트와 후처리**: 흐름장 / L 시스템 / 세포 자동자 / 반응-확산 / 프랙털 / Voronoi;
  blur / bloom / halation / 필름 그레인 / NPR 4스타일(toon / watercolor / halftone / oil) /
  3-in-1 노이즈 제거.
- **컬러 그레이딩과 벡터**: 2차 컬러 그레이딩(HSL 셀렉션 + RGB 커브), cinematic / film / anime
  스타일 프리셋과 드래프트/표준/시네마 3단계 화질(품질은 파라미터이지 미신이 아니다);
  벡터 패스 래스터화, 패스 draw-on과 morph.
- **오디오 처리**: 라우드니스 정규화(loudnorm), 비트 검출, 스펙트럼 분석, 엔벨로프 추출 —
  화면과 연동할 수 있다(파티클 AudioReactive가 오디오 엔벨로프를 따라간다).
- **CPU 참조 백엔드는 일등 시민**: GPU가 없는 기기에서도 올바른 이미지를 내놓으며, 동시에
  GPU의 정답 판정 기준이 된다.

### 무엇을 내놓는가 (전달)

- **`render_movie` 직행**: DSL v2 컴파일 산출물이 한 번의 호출로 "프레임별 평가 → tile 렌더링 →
  전달"까지 끝낸다. `.prism` 파일 하나로 단일 명령 완성 영상(`prism-app render demo.prism`);
  커버되지 않는 노드는 조용히 근사하지 않고 사실대로 오류를 보고한다.
- **프리뷰 폐루프**: `prism-app preview`로 프리뷰 서비스를 띄운다; 에디터의 Draft 단계(기하 패턴으로
  프레임레이트 유지) / Real 단계(낮은 spp의 실제 렌더링, 비동기 갱신) 두 단계 프리뷰.
- **자체 인코더 + 레이트 컨트롤**: I/P 프레임 JPEG 도메인 시계열 인코더가 사설 `PRMV` 컨테이너
  (v2/v3 두 세대 schema)를 사용하고, water-filling 레이트 컨트롤 + qpel 서브픽셀 움직임 추정
  (실측 PSNR +2.577 dB, 용량 −21.6%).
- **ffmpeg 협력 체인**: ffmpeg이 있으면 → stdin rawvideo 파이프로 H.264/HEVC 변환;
  ffmpeg이 없으면 → 자동으로 MJPEG로 강등, **엔진은 언제나 영상을 출력할 수 있다**;
  어떤 바이너리도 번들하지 않는다.
- **EXR 시퀀스 전달**: 프레임 시퀀스를 EXR로 직접 기록해 후반 파이프라인에 투입한다.
- **분산 팜**: 프레임 단위 분배 + N 프로세스 worker + 결정론적 병합 — 병렬 MD5 == 직렬 MD5,
  중단 지점부터 이어서 계산하는 것을 지원한다.

### 어떻게 통합하는가 (생태계)

- **MCP 도구 표면**: stdio JSON-RPC 세 가지 도구 — `scene_check`(컴파일 + 시험 렌더 + 프레임 수
  예측), `render_submit`(결정론적 job id + 프레임별 MD5 요약), `diagnose`(3단계 오류 위치 파악).
- **노드 SDK**: 시맨틱 버전 협상(0.x 특례 포함), 패키지에 예제 노드 패키지 동봉
  (`examples/sdk-nodes/cinematic-basic/`, 컴파일 가능 / 로드 가능 / 협상 가능).
- **하드웨어 인코딩 탐지 골격**: `prism-ffi-hwenc` 3개 플랫폼(VideoToolbox / NVENC / VAAPI)
  탐지 체인 + feature gate 매트릭스, 플랫폼이 맞지 않으면 명확히 미지원을 보고한다.

> 자세한 능력 경계(무엇이 **불가능한지**)는 패키지 압축을 풀고 `开发者指南.md`의 「정직한 경계」
> 절을 참고하라 — 그곳에는 현재의 실제 상태(곡선 프리미티브는 CPU만 사용, GPU 가속비는 대상
> 하드웨어에서 실측 필요, 하드웨어 인코딩은 대상 플랫폼별 연결…)가 기록되어 있다. 이 문서는
> 전체 조감도만 제시하며 과장하지 않는다.

---

---

## 엔진 내부 특수 능력 (기능 목록 바깥)

"읽기 / 배치 / 그리기 / 출력 / 통합" 목록에 담기지 않는 능력들 — 엔진의 심층 자산입니다. 이 장에서 본 목록을 깔끔하게 유지합니다.

### 오디오: 내장 DAW 커널

`prism-audio`는 "WAV를 읽을 줄 아는" 주변 모듈이 아니라 작곡부터 믹싱까지 전 과정을 갖춘 **결정론적 DAW 커널**입니다. 모든 함수는 매개변수의 순수 함수(같은 입력이면 플랫폼 간 비트 동일)입니다.

- **합성**: PolyBLEP 앤티앨리어싱 오실레이터군 + 폐형식 ADSR 엔벨로프 + 4가지 무드 프리셋;
- **음악 이론**: 음계 라이브러리(장단조/5음계/교회 선법), 12-TET 음높이↔주파수, 화성 진행의 **기능 문법 상태 기계** + 성부 진행;
- **시퀀싱**: 아르페지에이터(패턴/속도/옥타브), 스텝 드럼 머신, 마디·박자 그리드 스코어 편곡;
- **소재**: 직접 작성한 SMF 파서(MIDI 가져오기), SFZ 샘플러, 그래뉼러 합성;
- **사운드 디자인**: 물리 모델링, 공간 패닝, 이펙트 체인;
- **믹싱**: 멀티트랙 선형 믹스 + 페이드 + 피크 리미팅;
- **분석과 연동**: 비트/온셋 검출, FFT 스펙트럼, 엔벨로프 추출, AudioReactive 베이킹(N번째 프레임은 고정 창을 읽음; 프레임 단위 순수 함수, seek 등가), 라우드니스 정규화.

> 경계: AAC/mp3는 자체 개발하지 않습니다(특허 이력 + 공수). 일반 오디오는 외부 ffmpeg를 경유합니다.

### 정의 가능한 커널 라이브러리 (특해 레지스트리)

모든 노드 유형은 "커널 + 격자 + 쌍대" 삼중항의 **특해**입니다. 제3자(또는 당신)는 `registry!`로 새 특해를 등록할 수 있고, 엔진 코어를 고칠 필요가 없습니다.

- 정적 일괄 등록(매크로는 정의한 crate 안에서 전개; `inventory`/proc-macro 금지, 의존성 제로);
- 등록 시 **세 가지 질문 검증**(커널 형식/격자/쌍대). 답이 빠지면 3단 오류로 거부;
- 공식 특해와 사용자 특해는 같은 레지스트리를 공유하며 런타임 차이가 없습니다;
- 수학적으로는 연산자 방정식 `f = g + Kf`의 여섯 가지 역전 전략 — 정확/확률/국소/정규화/반복/스펙트럼. 능력 추가는 삼중항 선언이지, 렌더 루프에 덧대는 일이 아닙니다.

### 결정론과 재현성 기반

- **서명 폐포**: 노드 그래프/캐시 키/스냅샷 키/프레임 서명을 "4차원 구성 해시"로 통일. 매개변수를 바꾸면 키가 바뀌어 잘못된 재사용이 없습니다. 반복 노드는 `(ε, n)`, 스트림 노드는 프레임 번호를 키에 포함;
- **바이트 고정점**: USD는 export→import→export가 바이트 동일. WAV 16/24/32비트 왕복 일치. EXR은 공식 구현과 비트 단위 대조;
- **단일 파일 배포 결정성**: 같은 소스는 두 번 패키징해도 SHA-256이 같습니다. 압축 해제 왕복은 모든 파일을 바이트 단위로 비교합니다.

### GPU 백엔드 (정직한 분리 표기)

- **Vulkan**: compute와 레이트레이싱 장면 파이프라인이 실제 동작(lavapipe 소프트웨어 ICD에서 GPU 없이 검증 가능). `render_scene` 출력은 CPU 참조와 **바이트 동일(차이 픽셀 0)**. `is_available()`은 compute ∧ render 이중 자체 검사;
- **Metal**: 동형 이식 + 컴파일 게이트만 완료. **실기 검증 없음, 점등 없음**(`is_available()` 항상 false);
- **WebGPU**: 타당성 결론만(4–6인일). 렌더 경로 미구현.

### 가짜 초록 방지 게이트

로컬 게이트 4개(의존성 제로, `node`만 사용): 고스트 쓰기 방어(역방향 자기 증명 포함), 오픈소스 위생 스캔, 컴플라이언스 자체 점검(9/9, 역방향 자기 증명 8/8 레드), 패키징 왕복 검증. 게이트는 "통과" 또는 "레드"만 인정합니다. "괜찮아 보인다"는 안 됩니다. **가짜 초록은 실패보다 위험합니다.**

---

## 왜 단일 파일 배포인가

- **감사 가능**: 텍스트 파일 하나로, 사람이 읽을 수 있고, `diff` 가능하며, 해시 검증이 가능하다.
  배포 시점에 동결된다.
- **재현 가능**: 패키징이 결정론적이어서 같은 입력으로 두 번 패키징하면 SHA-256이 완전히 일치한다.
- **휴대 가능**: 거대한 저장소를 clone할 필요 없이 파일 하나를 메일이나 채팅으로 전송하면 되고,
  오프라인에서도 복원과 빌드가 가능하다.
- **트레이드오프 설명**: 단일 파일 배포는 "GitHub에서 코드를 직접 열람"하는 편의를 양보하는 대신
  "파일 하나로 전부를 가져가고 바이트 단위로 검증 가능"이라는 결정성을 얻는다. 압축을 풀면
  평범한 Cargo workspace가 나오며, 일반 Rust 프로젝트와 다를 바 없다.

---

## 빠른 시작

사전 준비: [Rust](https://www.rust-lang.org/)(`cargo` 포함)와 [Node.js](https://nodejs.org/) 18+를 설치한다.

```bash
# 1. 저장소의 prism-engine-v2.1-source.txt를 내려받아 로컬 파일로 저장한다
# 2. 텍스트 편집기로 열어 "UNPACK SCRIPT BEGIN"과 "UNPACK SCRIPT END"
#    사이의 Node.js 스크립트를 같은 디렉터리의 unpack.cjs로 따로 저장한다
# 3. 압축 해제 실행 (수 초 소요, 642개 파일 복원)
node unpack.cjs
# 4. 복원된 디렉터리(즉 workspace 루트)에서 전체 테스트 실행
cargo test --workspace --release
# 예상: 3929 passed / 0 failed
```

압축을 풀면 표준 Cargo workspace(`src/` 아래 28개 crate)가 나오며, 이후의 `cargo` 명령은 일반
Rust 프로젝트와 같다.

> 모바일 측 wasm 검증 경로: `rustup target add wasm32-unknown-unknown`, 이어서 `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## 저장소 구조 (압축 해제 후)

| 경로 | 설명 |
|------|------|
| `src/` | **28개 crate**의 소스(저장소 내 29개, 원래 이름 `crates/`, 배포 시 `src/`로 재구성) |
| `Compliance/` | 컴플라이언스 선언: 제로 의존성 검증 / 서드파티 고지 / 에셋 등록(저장소 내 원래 이름 `docs/合规/`) |
| `开发者指南.md` | 자체 완결형 개발자 문서: 시작하기 / 아키텍처 / 수학 커널 / 게이트 / 확장(저장소 내 원래 이름 `docs/开发者指南.md`) |
| `tools/` | 게이트 스크립트(고스트 쓰기 방어, 위생 스캔, 컴플라이언스 자체 점검) |
| `.github/workflows/` | CI 구성 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | 빌드 필수 요소(workspace members는 이미 `src/*`를 가리킴) |
| `examples/` | `prism-dsl` 테스트용 샘플과 예제 노드 패키지(`include_str!`로 읽으므로 없으면 컴파일되지 않음) |

---

## 모듈 개요

배포 패키지 내 **28개 crate**를 역할별로 묶으면 다음과 같다:

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
# 전체 저장소 스모크 테스트 (release, 권장)
cargo test --workspace --release
# 동등한 별칭
cargo xtest
# 단일 crate 빠른 검증
cargo test -p prism-core --release
```

- 전체 테스트: **3929 passed / 0 failed / 0 ignored** (저장소에서 release 실주행: 179개 테스트 스위트, 전 코어 병렬 171.5초. 패키지는 저장소와 바이트 단위로 동일하며 데모 crate만 제외)
- CI: **6개 job** — 고스트 쓰기 방어(ghost-guard) / 컴파일 + 전체 테스트(native) / wasm32 빌드(wasm) /
  WGSL 정적 검사(wasm-verify) / 공식 OpenEXR과 교차 구현 비트스트림 대조(interop) / node 호스트 교차 판정(lint).
- 로컬 게이트 스크립트 3개(전부 제로 의존성, `node`만 사용, 아래) + 패키징 스크립트 1개;
  그중 컴플라이언스 자체 점검은 **9/9 통과**, 역방향 자기 증명은 **8/8 판정 red**.

로컬 게이트 실행(Node 18+ 필요):

```bash
node tools/ghost_guard.mjs --check        # 고스트 쓰기 방어: G1 중복 정의/G2 엄격 초집합/G3 최상단 문서/G4 컴파일
node tools/hygiene_scan.mjs               # 오픈소스 위생: 잔존 내부 용어를 항목별로 나열(파일:줄번호), 적중 시 exit 1
node tools/compliance_check.mjs           # 컴플라이언스 자체 점검: 9/9 통과해야 함
node tools/compliance_check.mjs --self-test   # 역방향 자기 증명: red로 판정되어야 하며, 판정 기준의 판별력을 증명한다
```

> **오픈소스 위생 스캔에는 현재도 잔존 적중이 있다**: 저장소 안에 초기의 내부 용어가 남아 있으며,
> 그중 **패키지와 함께 배포되는 문서 계층은 이미 0건**이다(README / 개발자 가이드 / Compliance).
> 나머지는 **배포 패키지 내부의 소스 주석**과 **패키지와 함께 배포되지 않는 저장소 파일**에 있고,
> 사실에 따라 분류하여 별도의 정리 항목으로 등록했다. 스크립트가 존재하는 이유는 잔존물을
> **가시화하고 항목별로 정리**할 수 있게 하기 위함이지, 깨끗한 척하기 위함이 아니다 —
> "적중 수가 0이 아님"을 게이트 실패로 해석하지 말라.

---

## 컴플라이언스

엔진은 **서드파티 코드 의존성 제로**를 약속하며, 자체 점검으로 이를 검증했다:

- **C1 서드파티 코드 의존성 제로**: 전체 130개 의존성(110개 런타임 + 20개 테스트)이 모두 이 저장소의 `path` 의존성이다; `build.rs` 없음, 네이티브 링크 지시자 없음, `Cargo.lock`이 저장소 내 29개 패키지를 고정한다.
- **C2 에셋 등록 완비**: 5개 에셋 주표의 필드가 모두 채워져 있고, 자체 개발 전환 상태가 통제 어휘집 안에 있다; OFL 1.1 폰트 라이선스 전문이 폰트와 함께 `src/prism-text/font_data/LICENSE`에 위치한다.
- **C3 서드파티 귀속 고지**: `Compliance/THIRD-PARTY-NOTICES.md`가 존재하며, "데이터 에셋 ≠ 코드 의존성"을 명확히 구분한다.

> 참고: 이 저장소가 배포하는 것은 **소스**다. 폰트 에셋(Prism Sans SC 서브셋)은 SIL OFL 1.1로 라이선스되며, **라이선스 전문은 폰트와 함께** `src/prism-text/font_data/LICENSE`에 위치한다; 에셋 출처와 라이선스 등록은 `Compliance/assets/资产登记.md`를 참고하라.


---

## 라이선스

이 프로젝트는 **Apache License 2.0**으로 배포된다. 자세한 내용은 [`LICENSE`](../LICENSE) 파일을 참고하라.

> 컴플라이언스 안내: Apache-2.0은 저작권과 라이선스 고지를 유지할 것을 요구한다; 이 엔진을 기반으로 재배포할 때에는 `Compliance/` 아래의 귀속 고지와 에셋 등록을 유지하라.

**주: 다른 버전에 번역 오류가 있다면, 이 버전의 중국어 간체(the Simplified Chinese version)를 기준으로 하십시오.**
