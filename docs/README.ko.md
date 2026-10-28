<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Qianjinqie/prism-engine@main/docs/assets/logo.png" width="132" alt="Prism Engine">
</p>

<h1 align="center">Prism Engine</h1>

<h3 align="center">미디어 파이프라인 전체를 하나의 소스 트리에</h3>

<p align="center">
  <sub>디코드 · 조판 · 패스트레이싱 · 물리 시뮬레이션 · 오디오 합성 · 인코딩 | 의존성 제로 · 바이트 단위 결정적</sub>
</p>

<p align="center">
  <a href="../README.md">中文</a>
  · <a href="README.en.md">English</a>
  · <a href="README.de.md">Deutsch</a>
  · <a href="README.es.md">Español</a>
  · <a href="README.fr.md">Français</a>
  · <a href="README.ja.md">日本語</a>
  · <b><a href="README.ko.md">한국어</a></b>
  · <a href="README.pt-BR.md">Português</a>
  · <a href="README.ru.md">Русский</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-3B82F6?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/tests-3929%20passed-22C55E?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/dependencies-zero-16A34A?style=for-the-badge" alt="Dependencies">
  <img src="https://img.shields.io/badge/crates-28-8B5CF6?style=for-the-badge" alt="Crates">
  <img src="https://img.shields.io/badge/size-18.9%20MB-64748B?style=for-the-badge" alt="Size">
  <a href="https://github.com/Qianjinqie/prism-engine/releases"><img src="https://img.shields.io/badge/release-v2.1-6366F1?style=for-the-badge" alt="Release"></a>
</p>

<p align="center">
  <a href="https://github.com/Qianjinqie/prism-engine/releases/latest"><img src="https://img.shields.io/badge/DOWNLOAD-v2.1-6C5CE7?style=for-the-badge" alt="Download"></a>
  <a href="#3분-만에-실행하기"><img src="https://img.shields.io/badge/QUICK_START-3%20minutes-0EA5E9?style=for-the-badge" alt="Quick start"></a>
  <a href="#아키텍처"><img src="https://img.shields.io/badge/ARCHITECTURE-28%20crates-334155?style=for-the-badge" alt="Architecture"></a>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Qianjinqie/prism-engine@main/docs/assets/still-life.png" width="880" alt="Prism Engine이 렌더링한 정물">
</p>

<p align="center">
  <sub>목업이 아니라 엔진 출력입니다: 1200×675 · 256 spp · 패스 트레이싱 + 물리 기반 BSDF + ACES 톤 매핑 · 4 스레드 158초. 장면 소스는 <a href="gallery/gallery.rs"><code>docs/gallery/gallery.rs</code></a>.</sub>
</p>

<p align="center">
  <sub><b>642</b> 파일 · <b>28</b> crate · <b>26.1만</b> 줄의 Rust · <b>3929</b> 테스트 · 서드파티 의존성 <b>0</b> · <b>18.9 MB</b> 단일 파일</sub>
</p>

## 이 엔진이 존재하는 이유

일반적인 미디어 파이프라인은 조립품입니다. ffmpeg가 디코딩하고, Blender가 렌더링하고, ImageMagick이 정지 이미지를 쓰고, DAW가 소리를 맡고, 스크립트가 임시 파일로 이들을 붙입니다. 각각 훌륭한 도구지만, 이음새에서 두 가지가 사라집니다——**결정성**과 **감사 가능성**입니다. 기계를 바꾸면 부동소수점 경로가 달라지고, 버전을 바꾸면 기본값이 움직이며, 문제가 생기면 다섯 개의 블랙박스와 중간 파일 더미가 남습니다.

| | 일반적인 파이프라인 | Prism Engine |
| --- | --- | --- |
| 체인 | 대여섯 개 도구 + 접착 스크립트 + 임시 파일 | 하나의 소스 트리, 한 번의 빌드 |
| 다른 머신 | 부동소수점 경로가 바뀌면 픽셀도 바뀜 | 같은 입력이면 같은 바이트 |
| 문제 발생 | 다섯 개의 블랙박스와 중간 파일 더미 | 모든 픽셀이 소스 라인까지 추적됨 |
| 의존성 | 수십 개 네이티브 라이브러리, 각자의 버전 매트릭스 | 0 |

Prism Engine은 그 사슬 전체를 하나의 소스 트리로 끌어들입니다. 같은 장면 서술, 같은 파라미터라면 어느 기계에서든, 몇 번째 실행이든 픽셀과 바이트가 완전히 같습니다. 그것은 약속이 아니라 3929개의 테스트가 하나씩 못 박은 사실입니다.

ffmpeg나 Blender를 대체하려는 것이 아닙니다. 대체하는 것은 이들을 붙이는 접착층과, 그 접착층이 조용히 버리던 결정성입니다.

대상 독자는 의도적으로 좁습니다. "같은 입력, 어느 기계에서든, 같은 출력"이 필요한 사람——배치 렌더링 파이프라인, 재현 가능한 실험을 하는 그래픽스 연구, 렌더링을 CI에 넣으려는 팀.

### 세 가지 단단한 제약

| 제약 | 구현 방식 | 치르는 비용 |
| --- | --- | --- |
| **서드파티 코드 의존성 제로** | 런타임 110 + 테스트 20 의존성이 모두 저장소 내 `path` 의존성. `build.rs` 없음, 네이티브 링크 없음, 빌드 시 네트워크 불필요 | 모든 포맷, 알고리즘, 폰트, 코덱을 직접 작성 |
| **바이트 단위 결정성** | 고정소수점 IDCT, 고정 샘플링 시퀀스, 4차원 설정 해시. 같은 입력이면 플랫폼을 넘어 비트 단위로 일치 | 부동소수점 고속 경로를 포기하고, 순서 없는 컨테이너의 반복 순서에 의존하지 않음 |
| **단일 파일 배포** | 642개 파일·28개 crate를 18.9 MB UTF-8 텍스트 하나로 합치고 `node unpack.cjs`로 복원 | GitHub에서 소스를 바로 열어 보는 편의를 포기 |

## 3분 만에 실행하기

```bash
# 1. 단일 파일 소스 패키지를 받고 SHA-256 검증
curl -L -O https://github.com/Qianjinqie/prism-engine/releases/download/v2.1/prism-engine-v2.1-source.txt
curl -L -O https://github.com/Qianjinqie/prism-engine/releases/download/v2.1/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt

# 2. 패키지에 들어 있는 언팩 스크립트 추출 (Node 18+ 만 필요, npm install 불필요)
awk '/^UNPACK SCRIPT BEGIN$/{f=1;next} /^UNPACK SCRIPT END$/{f=0} f' \
  prism-engine-v2.1-source.txt > unpack.cjs
node unpack.cjs                    # 642개 파일 복원

# 3. 빌드하고 전체 테스트 실행
cargo test --workspace --release     # 3929 passed / 0 failed

# 4. 아무것도 쓰지 말고, 먼저 영상을 봅니다
cargo run -p prism-app --release     # 내장 데모 장면
```

복원 결과는 평범한 Cargo workspace(`src/` 아래 28개 crate)이며, 이후 `cargo` 명령은 일반 Rust 프로젝트와 같습니다. 첫 빌드에는 몇 분이 걸립니다.

## 한 장면을, 한 편의 텍스트로

`.prism`은 엔진의 장면 서술 언어입니다. 아래는 완전히 렌더링 가능한 3초 장면입니다——구체, 재질, 궤도 카메라, 방향광, 길이, 그레이딩, 출력 사양까지 전부:

```prism
scene "spin_red_ball" {
    sphere ball {
        radius 1.0
        material {
            albedo #c0392b
            roughness 0.3
            metalness 0.05
        }
    }

    camera main {
        orbit {
            center 0,0,0
            radius 8.0
        }
        fov 45
    }

    light sun {
        direction (1.0, -1.0, -0.5)
        color #fff4e6
        intensity 3.0
    }
}

timeline {
    duration 3s
    fps 12
}

aesthetic {
    teal-orange
    contrast 1.12
    filmic
}

output {
    resolution 480x270
    format frames
}
```

```bash
cargo run -p prism-app --release -- examples/spin_red_ball.prism          # 프레임 시퀀스
cargo run -p prism-app --release -- render examples/w2d_showcase.prism \
    --out out/demo.mp4                                                     # 한 번에 완성 영상
```

`render`는 v2 직행 경로입니다. 노드 그래프 컴파일 → 프레임별 평가 → 컨테이너 인코딩까지, 중간 프레임을 디스크에 남기지 않습니다. 지원하지 않는 노드는 조용히 건너뛰지 않고 명시적인 `R5xxx` 오류를 냅니다.

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Qianjinqie/prism-engine@main/docs/assets/dsl-title.png" width="760" alt="DSL로 렌더링한 타이틀 프레임">
  <br>
  <sub>`examples/w2d_showcase.prism`의 10번째 프레임: 3D 구체, 텍스트 레이어, 그레이딩, 페이드 트랙이 하나의 타임라인을 공유합니다.</sub>
</p>

## Rust API만 쓰기

DSL을 쓰지 않고 라이브러리를 직접 호출할 수 있습니다. 아래는 히어로 이미지의 골격입니다(전체 장면은 190줄, [`docs/gallery/gallery.rs`](gallery/gallery.rs)):

```rust
use prism_core::node::EvalCtx;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, MaterialKind, RenderParams, Scene, Sphere};

let scene = Scene {
    spheres: vec![(
        Sphere { center: Vec3::new(0.0, 0.6, 0.0), radius: 0.6 },
        MaterialKind::Lambert { albedo: Vec3::new(0.8, 0.2, 0.15) },
    )],
    ..Default::default()
};
let cam = Camera::new(
    Vec3::new(0.0, 1.2, 4.8),   // 시점
    Vec3::new(0.0, 0.72, 0.0),  // 바라보는 점
    Vec3::new(0.0, 1.0, 0.0),   // 위쪽
    30.0, 0.05, 4.85, 0.0, 1.0 / 48.0,
);
let ctx = EvalCtx::new(20261028, 0, 0);
let params = RenderParams::new(256, 10);  // 픽셀당 256 샘플, 최대 10회 바운스

// 단일 픽셀: 장면, 카메라, 샘플링 컨텍스트, 픽셀 좌표, 해상도, 파라미터
let color = render_pixel(&scene, &cam, &ctx, (640, 360), (1280, 720), &params);
```

`render_pixel`은 순수 함수입니다. 장면, 카메라, 샘플링 컨텍스트, 픽셀 좌표를 받아 선형 RGB를 돌려줍니다. 같은 파라미터라면 어느 스레드에서, 어떤 순서로 호출해도 결과 비트가 같습니다——그래서 병렬 렌더링에 동기화가 전혀 필요 없습니다.

## 기능 전경

| 층 | 할 수 있는 일 |
| --- | --- |
| **읽기** | PNG(8/16 bit), 베이스라인 JPEG(고정소수점 IDCT), GIF/APNG 프레임 단위, EXR(DWA 포함), 베이스라인 RAW, WAV. glTF / OBJ / USDA 3D 에셋, MaterialX 파라미터, 외부 TTF/OTF 폰트 |
| **짜기** | DSL v1/v2 컴파일, 다섯 가지 고수준 컴비네이터(타이틀 카드 / 로어 서드 / 분할 화면 / 켄 번스 / 장면 간 모프), 네 가지 전환 + 모프, 수식 조판(위·아래첨자 / 분수 / 근호 / 행렬 / 대형 연산자), 타임라인 임의 시각 샘플링 |
| **그리기** | 패스 트레이싱 + 물리 기반 BSDF(확산 / 경면 / 유리 / 금속 / 표면하 산란), Woodcock 이질 매질, 원기둥 / 원뿔 / 토러스 / 회전면 해석적 교차(4차 방정식을 Ferrari 폐형해로), SAH-BVH 삼각 메시, CPU 참조 백엔드, Vulkan compute + 레이트레이싱 이중 파이프라인 |
| **움직이기** | 통합 XPBD 솔버(강체 / 연체 / 천 / 관절 / 유체 / 파괴), FLIP 연기, 파티클(오디오 엔벨로프 연동 가능), 유동장 / L 시스템 / 세포 오토마타 / 반응 확산 / 프랙털 / Voronoi |
| **다듬기** | 세컨더리 그레이딩(HSL 선택 + RGB 커브), cinematic / film / anime 프리셋, blur / bloom / halation / 필름 그레인, NPR 네 스타일(toon / watercolor / halftone / oil), 삼중 디노이즈, 벡터 패스 래스터화와 draw-on |
| **소리** | PolyBLEP 신시사이저, ADSR, 음계와 코드 진행 상태 기계, 아르페지에이터, 스텝 드럼, SMF 파싱, SFZ 샘플러, 그래뉼러 합성, 멀티트랙 믹싱, 라우드니스 정규화, 비트 / onset / FFT 분석 |
| **출력** | 자체 `PRMV` 컨테이너(v2/v3) + I/P 프레임 JPEG 도메인 부호화 + water-filling 레이트 제어 + qpel 움직임 추정, ffmpeg 파이프라인(있으면 H.264/HEVC, 없으면 자동 MJPEG 강등), EXR 시퀀스, 멀티프로세스 프레임 팜(이어서 계산, 병렬 MD5 == 직렬 MD5) |
| **연결** | MCP stdio 세 도구(`scene_check` / `render_submit` / `diagnose`), 노드 SDK(시맨틱 버전 협상), HTTP/1.1 렌더 서비스, batch manifest 직곱 배치 렌더링, 세 플랫폼 하드웨어 인코더 탐색 골격 |
| **부수 경로** | 생성 예술: 유동장 / L 시스템 / 세포 오토마타 / 반응-확산 / Mandelbrot / Voronoi, 벡터 draw-on과 차트. AI 보조: 결정론적 mock 추론 계층(스토리보드 / 자막 / 색상 힌트), 모델 가중치는 데이터로 읽고 곱셈-덧셈은 직접 작성, ML 프레임워크 없음 |

> 표는 능력만 나열합니다. 각 능력이 못 하는 것은 [정직한 경계](#정직한-경계)에 있으며 표 안에 숨기지 않았습니다.

## 목록에 담기지 않는 내부 능력

위 표에 들어가지 않는 것들이 있습니다. 그것들은 "포맷이 하나 더 늘었다"는 이야기가 아니라, 엔진이 스스로를 위해 키운 기반입니다.

### 오디오: 내장 DAW 커널

`prism-audio`는 "WAV를 읽을 수 있다" 수준의 주변 모듈이 아닙니다. 작곡에서 믹싱까지 하나의 사슬이며, 모든 함수가 파라미터의 순수 함수입니다(같은 입력이면 플랫폼을 넘어 비트 단위로 일치):

- **합성**: PolyBLEP 안티에일리어싱 오실레이터군, 폐형 ADSR 엔벨로프, 4가지 무드 프리셋;
- **음악 이론**: 음계 라이브러리(장단조 / 5음 음계 / 교회 선법), 12-TET 음급↔주파수, 코드 진행의 기능 문법 상태 기계 + 성부 진행;
- **편곡**: 아르페지에이터(패턴 / 속도 / 옥타브), 스텝 드럼, 마디와 박자 그리드;
- **소재**: 직접 작성한 SMF 파싱(MIDI 가져오기), SFZ 샘플러, 그래뉼러 합성;
- **사운드 디자인**: 물리 모델링, 공간 패닝, 이펙트 체인;
- **믹싱**: 멀티트랙 선형 믹스, 페이드, 피크 클리핑 방지;
- **분석과 연동**: 비트 / onset 검출, FFT 스펙트럼, 엔벨로프 추출, 라우드니스 정규화. AudioReactive 베이킹은 프레임 N에서 고정 창을 읽어 프레임의 순수 함수가 되고 seek과 동등합니다.

> 경계: AAC / mp3 인코딩은 직접 구현하지 않습니다(특허 역사적 부담 + 공수). 외부 ffmpeg와 협력합니다.

### 정의 가능한 커널 레지스트리

엔진의 각 노드 타입은 "커널 + 격자 + 쌍대" 삼중항의 **특수해**입니다. 서드파티는 `registry!`로 새 특수해를 정적으로 등록할 수 있고, **엔진 코어를 건드리지 않습니다**:

- 정적 포함: 매크로가 정의 crate 안에서 전개되며, `inventory`와 proc-macro를 금지해 의존성 제로를 유지;
- 등록 시 **세 가지 질문**(커널 형식 / 격자 / 쌍대)을 검증하고, 답이 빠지면 세 단락 오류로 거부;
- 공식 특수해와 사용자 특수해가 같은 레지스트리를 지나며 런타임에 구분되지 않음;
- 수학적으로는 작용소 방정식 `f = g + Kf`의 여섯 가지 역해법(정확 / 확률 / 국소 / 정규화 / 반복 / 스펙트럼)에 대응하며, 새 기능 추가는 삼중항 선언이지 렌더 루프에 대한 패치가 아닙니다.

### GPU: 세 경로, 세 상태

| 백엔드 | 상태 | 근거 |
| --- | --- | --- |
| **Vulkan** | 실제 동작 | compute + 레이트레이싱 이중 파이프라인. Mesa lavapipe 소프트웨어 ICD에서 GPU 없이 검증 가능. `render_scene` 출력이 CPU 참조와 **바이트 단위 0 차이**. `is_available()`은 compute ∧ render 이중 자체 점검 |
| **Metal** | 컴파일 게이트 완료, 미점등 | 동형 이식 완료, 실기 검증 없음, `is_available()` 항상 false |
| **WebGPU** | 결론만 | 타당성 평가 완료(이식량 4–6 인일), 렌더 경로 미구현 |

### 가짜 초록 방지 게이트

의존성 제로 게이트 스크립트 네 개(Node만 사용): 유령 쓰기 방어(역 자체 시험 포함), 오픈소스 위생 스캔, 컴플라이언스 자체 점검(9/9 통과 + 8/8 반증 판정), 패키지 왕복 검증. 게이트 결과는 "통과"와 "실패" 둘뿐이며, "아마 괜찮을 것"은 인정하지 않습니다——**가짜 초록은 실패보다 위험합니다**.

## 결정성은 구호가 아니다

파라미터에서 바이트까지, 네 층의 장치:

1. **서명 폐포**: 노드 그래프, 캐시 키, 스냅샷 키, 프레임 서명을 4차원 설정 해시로 통일. 파라미터 하나를 바꾸면 키가 반드시 바뀌어 잘못된 재사용이 없습니다. 반복 노드의 키에는 `(ε, n)`이, 스트림 노드의 키에는 프레임 번호가 들어가며, `None`일 때는 이전 동작으로 바이트 단위로 퇴화합니다.
2. **바이트 고정점**: USD 내보내기→가져오기→내보내기가 바이트 단위로 일치. WAV 16/24/32 bit 왕복 일치. EXR은 공식 OpenEXR과 비트스트림 단위로 대조.
3. **패키징 결정성**: 같은 소스를 두 번 패키징하면 SHA-256이 완전히 일치. 언팩 왕복에서 모든 파일을 바이트 단위로 비교.
4. **테스트가 판정 기준**: 3929개 테스트 중 다수가 황금값을 고정(허용 오차 1e-2에서 1e-6으로, 실측 잔차 4.53e-7). 형상 충실도는 Hausdorff 거리로 억제하며 "대충 맞는다"로 끝내지 않습니다.

## 아키텍처

```mermaid
flowchart LR
    IN["입력<br/>PNG · JPEG · EXR · RAW<br/>GIF · APNG · WAV<br/>glTF · OBJ · USDA"] --> DSL["편성<br/>prism-dsl<br/>노드 그래프 · 채널 · 타임라인"]
    DSL --> RT["평가<br/>2D 래스터 · 3D 패스 트레이싱<br/>물리 · 파티클 · 제너러티브"]
    DSL --> AU["오디오<br/>prism-audio<br/>합성 · 편곡 · 믹싱"]
    RT --> FX["후처리<br/>그레이딩 · bloom · 디노이즈 · NPR"]
    FX --> OUT["출력<br/>PRMV · ffmpeg · EXR 시퀀스"]
    AU --> OUT
```

28개 crate를 책임별로:

| 분류 | crate |
| --- | --- |
| 코어 | `prism-core` · `prism-math` · `prism-bytes` · `prism-geometry` · `prism-noise` |
| 미디어 I/O | `prism-io` · `prism-media` · `prism-mux` · `prism-audio` · `prism-texture` |
| 렌더링 | `prism-render` · `prism-raster` · `prism-gpu` · `prism-postfx` · `prism-vector` · `prism-particles` |
| 애니메이션 / 생성 | `prism-anim` · `prism-genart` · `prism-aesthetic` · `prism-physics` |
| 텍스트 / UI | `prism-text` · `prism-rt` · `prism-app` |
| 고급 | `prism-ai` · `prism-dsl` · `prism-infra` · `prism-author` |
| 사이드카(feature gate) | `prism-ffi-hwenc` |

언팩 후 디렉터리 구조:

| 경로 | 내용 |
| --- | --- |
| `src/` | 28개 crate 소스(저장소에서는 `crates/`, 배포 시 재구성) |
| `Compliance/` | 의존성 제로 검증, 서드파티 고지, 에셋 등록 |
| `开发者指南.md` | 자기 완결형 개발자 가이드: 시작 / 아키텍처 / 수학 커널 / 게이트 / 확장 |
| `tools/` | 게이트와 패키징 스크립트 |
| `examples/` | `.prism` 예제와 노드 패키지(`include_str!`로 읽으므로 없으면 빌드 실패) |
| `.github/workflows/` | CI 설정 |

> 설계 정본(`docs/设计/`)은 배포자의 전체 저장소에만 있으며 배포에는 포함되지 않습니다. 데모 crate `prism-anim-demo`도 마찬가지입니다.

## 빌드, 테스트, 게이트

```bash
cargo test --workspace --release      # 3929 passed / 0 failed / 0 ignored
cargo xtest                           # 동등한 별칭
cargo test -p prism-core --release    # 단일 crate 스모크 테스트

node tools/ghost_guard.mjs --check    # 유령 쓰기 방어: G1 중복 정의 / G2 엄격한 상위 집합 / G3 최상위 문서 / G4 빌드
node tools/hygiene_scan.mjs           # 위생: 남은 내부 용어를 file:line으로 나열, 적중 시 exit 1
node tools/compliance_check.mjs       # 컴플라이언스 자체 점검: 9/9 통과해야 함
node tools/compliance_check.mjs --self-test  # 반증: 실패해야 하며, 판정 기준의 판별력을 보여줌
node tools/pack_source_txt.mjs --verify      # 패키징과 언팩 왕복을 바이트 단위로 검증
```

- 전체: **3929 passed / 0 failed / 0 ignored**, 179개 테스트 구간, 전 코어 병렬 171.5초(release).
- CI는 여섯 job: 유령 쓰기 방어, 빌드 + 전체 테스트, wasm32 빌드, WGSL 정적 검사, 공식 OpenEXR과 비트스트림 상호 검증, Node 호스트 교차 판정.
- 위생 스캔에는 여전히 적중이 남아 있습니다. 배포 문서 층은 정리되었고, 나머지는 소스 주석과 배포 대상이 아닌 저장소 파일에 있습니다. 스크립트의 존재는 잔여물을 **보이게 두고 하나씩 정리**하기 위함이며, 깨끗한 척하기 위함이 아닙니다.

## 정직한 경계

| 경계 | 현황 | 출로 |
| --- | --- | --- |
| DNG `Compression=7`(lossless JPEG) | 미구현. 실제 카메라 DNG는 읽을 수 없음 | 16 bit TIFF / PNG로 변환 후 사용 |
| mp3 / AAC / FLAC | 디코딩은 외부 ffmpeg 탐색 경로, 인코딩은 직접 구현하지 않음 | ffmpeg 설치, 또는 WAV만 사용 |
| 곡선 프리미티브 GPU | CPU 경로만 | 삼각화 후 SAH-BVH로 |
| Metal | 게이트 준비 완료, 실기 미검증, 미점등 | Apple 기기가 생기면 연결 |
| WebGPU | 타당성 결론만, 렌더 미구현 | 4–6 인일 |
| 하드웨어 인코딩 | 세 플랫폼 탐색 골격만 | 대상 플랫폼별로 구현 |
| GPU 가속비 | lavapipe에서 정확성만 검증, 실제 GPU 미측정 | 대상 하드웨어에서 측정 필요 |
| 위생 스캔 | 소스 주석에 기존 적중 | 하나씩 정리, 스크립트로 추적 |

이것들은 "나중에"가 아니라 현재 명확히 지원하지 않는 항목입니다. 이슈에 숨기는 것보다 여기 적는 편이 낫다고 판단했습니다.

## 컴플라이언스

엔진은 서드파티 코드 의존성 제로를 표방하며 자체 점검으로 검증합니다:

- **C1 서드파티 코드 의존성 제로**: 130개 의존성(런타임 110 + 테스트 20)이 모두 저장소 내 `path` 의존성. `build.rs` 없음, 네이티브 링크 지시 없음. `Cargo.lock`이 저장소 내 29개 패키지를 고정.
- **C2 에셋 등록 완비**: 5개 에셋의 주표 항목이 모두 채워져 있습니다. 폰트는 SIL OFL 1.1로 배포하며 라이선스 전문을 `src/prism-text/font_data/LICENSE`에 동봉.
- **C3 서드파티 귀속 고지**: `Compliance/THIRD-PARTY-NOTICES.md`가 "데이터 자산 ≠ 코드 의존성"을 명확히 구분.

## 라이선스

Apache License 2.0, [`LICENSE`](../LICENSE)를 참고하세요.

재배포 시 `Compliance/` 아래의 귀속 고지와 에셋 등록을 유지해 주세요.

<sub>히어로 이미지와 DSL 스크린샷은 모두 엔진 자체가 렌더링한 것입니다. 파라미터는 그림 설명과 [`docs/gallery/gallery.rs`](gallery/gallery.rs)에 있습니다.</sub>
