---
record_id: D-20260728-001
type: develop
method: RHO_GS
created_at: 2026-07-28T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/develop/2026/D-20260728-001-rho-gs-namespace-separation.md
retrospective: true
---

# D-20260728-001 — RHO_GS namespace 및 CUDA backend 분리

## 요약

| 항목 | 내용 |
|---|---|
| 구현 목적 | Faster-GS와 독립된 RHO_GS 연구 공간 구성 |
| 초기 의도 | FasterGSTestbed 구조를 유지하면서 Python·CUDA namespace 충돌 제거 |
| 기준 branch | `rho-gs/testbed` |
| 현재 상태 | 구현 및 import 검증 완료, commit 미생성 |
| 관련 Work | 아직 없음 |

## 범위

- Method 경로를 `Methods.RHO_GS`로 분리
- Model, Renderer, Trainer, Loss 이름을 RHO_GS 기준으로 변경
- Method entry의 `MODEL`, `RENDERER`, `TRAINING_INSTANCE` 갱신
- CUDA backend를 `RHOGSCudaBackend`로 분리
- config의 `METHOD_TYPE`, model name, W&B project를 RHO_GS로 변경
- RHO_GS 연구 설명과 NeRFICG 개발 가이드 추가

연구 알고리즘 자체의 Hessian 계산이나 optimizer 변경은 이 작업 범위에 포함되지 않는다.

## 기준 저장소 상태

아래 값은 2026-07-28 문서화 시점 기준이다.

| 저장소 | commit | branch | dirty | remote |
|---|---|---|---|---|
| NeRFICG | `c7437127af1681f565a57fcd9d7819fde1adc0a7` | `main` | Method 디렉터리 미추적 | `https://github.com/nerficg-project/nerficg.git` |
| RHO_GS | `751ec974c7ab2389c9bc789866ebec2fd6d8d54a` | `rho-gs/testbed` | rename·수정·신규 docs 존재 | `https://github.com/moot1801/RHO-GS` |

## Events

### 2026-07-28 — branch-selected

- upstream `FasterGSTestbed` 계열을 RHO_GS 연구 시작점으로 선택했다.
- RHO_GS 내부 branch를 `rho-gs/testbed`로 사용한다.

### 2026-07-28 — namespace-separated

- Python import 경로를 `Methods.RHO_GS`로 변경했다.
- `RHOGSModel`, `RHOGSRenderer`, `RHOGSTrainer`, `RHOGSLoss` 명칭을 적용했다.
- Method entry가 RHO_GS 클래스를 공개하도록 변경했다.

### 2026-07-28 — cuda-backend-separated

- `FasterGSTestbedCudaBackend` 경로와 package 이름을 `RHOGSCudaBackend`로 변경했다.
- Python binding과 C++ binding import 이름을 갱신했다.
- GPU-visible 환경에서 Method import를 검증했다.

### 2026-07-28 — documentation-added

- `docs/README.md`에 NeRFICG 구조, Method entry, 설정·콜백·Dataset 정책을 문서화했다.
- rasterizer의 `@once_differentiable` 제약과 초기 Hessian 실험 주의사항을 기록했다.

## 주요 변경 경로

| 경로 | 변경 목적 |
|---|---|
| `__init__.py` | RHO_GS Method entry 공개 |
| `Model.py` | RHO_GS Model과 CUDA optimizer import |
| `Renderer.py` | RHO_GS Renderer와 rasterizer import |
| `Trainer.py` | RHO_GS Trainer와 Loss 연결 |
| `Loss.py` | RHO_GS Loss 명칭 분리 |
| `utils.py` | RHO_GS backend 경로 사용 |
| `rho_gs_garden.yaml` | RHO_GS 실행 설정 |
| `RHOGSCudaBackend/` | 독립 CUDA extension package |
| `docs/README.md` | NeRFICG·RHO_GS 개발 가이드 |

## 검증

| 검증 | 결과 | 근거 |
|---|---|---|
| RHO_GS branch | 확인 | `rho-gs/testbed` |
| Method entry | 확인 | `MODEL`, `RENDERER`, `TRAINING_INSTANCE` |
| Python import | 성공 | GPU-visible 환경의 이전 Codex 세션 |
| CUDA backend import | 성공 | 설치된 `RHOGSCudaBackend` 사용 |
| commit | 미생성 | 현재 worktree에 staged/unstaged 변경 존재 |

## 위험 및 주의점

| 위험 | 영향 | 대응 |
|---|---|---|
| nested repository | root commit만으로 RHO_GS 변경 재현 불가 | RHO_GS 내부에서 별도 commit 관리 |
| backend rename 변경 미커밋 | worktree 손실 시 복구 어려움 | 검토 후 RHO_GS commit 생성 필요 |
| CUDA extension stale build | CUDA 변경이 실행 binary에 반영되지 않을 수 있음 | 변경 후 backend 재빌드 및 import 검증 |
| double backward 미지원 | PyTorch autograd Hessian 직접 계산 불가 | finite difference·근사 또는 CUDA 확장 |
| topology 변경 | Gaussian Hessian 시간축 비교 불안정 | stable ID 또는 densification 비활성화 |

## 커밋

| commit | 내용 | 상태 |
|---|---|---|
| 미생성 | RHO_GS namespace 및 backend 분리 | uncommitted |

## 한계 및 후속 작업

- 이 기록은 이전 Codex 작업을 사후 정리한 소급 기록이다.
- 세부 실행 command 전체는 미수집이다.
- 현재 변경을 검토한 뒤 RHO_GS 저장소에 독립 commit이 필요하다.
- Hessian 실험 구현은 별도 Develop ID로 관리한다.

### 2026-07-30 — namespace-cleanup-verified

- 상태: `commit-ready`
- namespace와 coupling 실험 프레임워크만 유지하기 위해 미커밋 tile Gaussian 통계 및 fixed-view 통계 runtime·CUDA 코드를 제거했다.
- rasterizer API는 Faster-GS 기준 동작으로 복원하고 coupling runtime에서 제거된 optional 인자 전달을 정리했다.
- active code의 tile 통계 참조가 없음을 확인했다.
- coupling framework 표준 `unittest` 16개와 변경 Python 파일 구문 검사, staged·unstaged whitespace 검사가 통과했다.
- 전체 Method import는 현재 Codex 실행 환경에 CUDA GPU가 노출되지 않아 GUI 초기화 단계에서 재검증하지 못했다.
- CUDA backend 소스가 복원됐으므로 실제 GPU 실행 전 현재 소스 기준 재빌드가 필요하다.
