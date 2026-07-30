---
record_id: D-20260729-001
type: develop
method: RHO_GS
created_at: 2026-07-29T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/develop/2026/D-20260729-001-coupling-experiment-framework.md
retrospective: false
---

# D-20260729-001 — Inter-Gaussian Coupling 실험 프레임워크

## 요약

| 항목 | 내용 |
|---|---|
| 구현 목적 | Grouping·attribute·residual·Jacobian·solver·aggregation·acceptance를 config로 조합하는 second-order reference testbed 구축 |
| 초기 의도 | position Hessian/Gauss-Newton 비대각 block의 의미와 grouping별 coupling capture를 정확성 우선으로 분석 |
| 구현 위치 | `src/Methods/RHO_GS/experiments/` 독립 패키지 |
| 기존 경로 영향 | RHO_GS Model/Renderer/Trainer 및 CUDA kernel 수정 없음 |
| 상태 | CPU 수치 검증 완료, 실제 Lego CUDA integration 미검증 |
| 관련 Work | 실제 analyzer/one-step 실행 시 신규 Work 발급 예정 |

## 기준 저장소 상태

| 저장소 | commit | branch | dirty | 비고 |
|---|---|---|---|---|
| NeRFICG | `c7437127af1681f565a57fcd9d7819fde1adc0a7` | `main` | 기존 Method·문서 변경 존재 | 작업 전 확인값 |
| RHO_GS | `751ec974c7ab2389c9bc789866ebec2fd6d8d54a` | `rho-gs/testbed` | namespace·tile stats 변경 및 본 구현 존재 | 기존 변경 보존 |

## 모듈 경계

| 모듈 | 책임 |
|---|---|
| `registry.py`, `protocols.py`, `types.py` | 전략 등록, 확장 protocol, typed record |
| `config.py`, `configs/` | YAML fragment composition과 dotted override |
| `coupling/grouping/` | independent, kNN, visible kNN, visible-overlap kNN |
| `coupling/parameter_blocks/` | PositionBlock과 추가 attribute extension point |
| `coupling/residuals/` | RGB L2 전체·sampled residual |
| `coupling/jacobian/` | repeated VJP reference와 finite difference validator |
| `coupling/curvature/` | dense sampled `J^T J`, normalized coupling, capture ratio |
| `coupling/solvers/` | per-Gaussian/group GN·LM, Adam state adapter |
| `coupling/aggregation/` | anchor-only, predicted-reduction weighted average |
| `coupling/acceptance/` | always, primary loss, LM gain-ratio acceptance |
| `state.py`, `stateful_trainer.py` | optimizer/RNG/sampler 포함 portable state |
| `evaluation/`, `result_logger.py` | audit view, metric, CUDA timing, memory, artifact schema |
| `runners/` | checkpoint analyzer, one-step, fixed-topology rollout, summary, stateful training |

## 핵심 수치 정책

### Residual과 curvature

- 최초 residual은 sampled RGB L2이며 objective는 `0.5 * sum(r²)`이다.
- `J^T J`와 `J^T r`는 같은 residual 정의를 사용한다.
- 기존 RHO_GS의 `0.8 L1 + 0.2 DSSIM`은 remaining-attribute Adam step의 native loss로 별도 사용한다.
- position-only predicted/actual reduction과 hybrid audit 결과를 구분한다.

### Jacobian

- custom rasterizer의 double backward를 요구하지 않도록 sampled output scalar별 first-order VJP를 반복한다.
- finite difference는 중앙 차분이며 CPU synthetic test에서 analytic VJP와 비교한다.
- full scene용이 아니라 0.25× image, top tile 2개, tile당 16 pixel, anchor 8개의 reference budget이 기본이다.

### Overlapping group

- proposal별 predicted reduction을 합하지 않는다.
- aggregation된 최종 update를 union parameter vector에 배치한 뒤 union Hessian quadratic model에서 한 번 계산한다.
- `anchor_only`는 group coupling을 모두 계산하되 anchor update만 적용한다.

### Checkpoint

- 기존 `final.pt`는 model-only이므로 LM coupling 분석에는 사용 가능하지만 Adam state 비교에는 사용하지 않는다.
- `StatefulRHOGSTrainer`는 기존 Trainer를 수정하지 않는 subclass이며 1K/15K/30K portable snapshot을 저장한다.
- portable state는 parameter, PyTorch Adam, RNG, camera sampler, SH degree, iteration, snapshot-local ID를 포함한다.

## Sampling과 grouping

| 항목 | 구현 |
|---|---|
| visibility | camera frustum과 projected footprint의 image 교차, optional opacity threshold |
| footprint | `3 × max(scale) × focal / depth` isotropic 근사 |
| top tile | rectangle-difference 누적 후 등록 수 top-K |
| anchor | sampled tile과 겹치는 visible Gaussian의 seed 고정 uniform subset |
| overlap | tile 또는 projected pixel bbox intersection, configurable threshold |
| disjoint | `overlapping=false`에서 앞 group에 사용한 Gaussian 제외 |

실제 CUDA rasterizer의 instance list와 동일하다고 간주하지 않는다. 근사 종류는 render/group metadata에 기록한다.

## Events

### 2026-07-29 — planned

- 상태: `planned`
- Faster-GS 원본을 보존하기 위해 구현 대상을 RHO_GS testbed의 독립 `experiments/` package로 결정했다.
- 첫 milestone을 checkpoint analyzer와 one-step reference로 두고 CUDA 최적화는 제외했다.
- 사용자는 residual provider의 독립 모듈화, 0.25×/top2 tile/16 pixel/8 anchor budget, train51 primary view를 선택했다.

### 2026-07-29 — reference-framework-implemented

- 상태: `implemented-uncommitted`
- registry/protocol 기반 모듈과 계층 config composer를 추가했다.
- PositionBlock, 4개 grouping, RGB L2 residual, repeated VJP/finite difference, dense GN curvature를 구현했다.
- per/group GN·LM, Adam adapter, aggregation, acceptance를 구현했다.
- analyzer, one-step counterfactual, fixed-topology rollout runner와 machine-readable logger를 추가했다.
- 기존 RHO_GS core Python/CUDA 파일은 수정하지 않았다.

### 2026-07-29 — overlap-prediction-corrected

- 상태: `implemented-uncommitted`
- overlapping proposal의 predicted reduction 단순 합이 residual을 중복 계산하는 문제를 구현 중 검토에서 발견했다.
- aggregation 이후 union Hessian에서 하나의 quadratic reduction을 계산하도록 수정하고 단위 테스트를 추가했다.

### 2026-07-29 — cpu-reference-verified

- 상태: `implemented-cpu-verified-gpu-unverified`
- Python 문법 검사와 `git diff --check`를 통과했다.
- CPU reference unit test 13개가 모두 통과했다.
- 실제 Lego checkpoint analyzer/one-step/rollout은 장시간 GPU 실행 승인을 받지 않았으므로 미실행이다.

## 변경 파일

| 경로 | 변경 목적 |
|---|---|
| `experiments/` | coupling reference framework 전체 신규 추가 |
| `rho_gs_lego_coupling_states.yaml` | 1K/15K/30K optimizer-state baseline 설정 신규 추가 |
| `docs/develop/2026/D-20260729-001-...md` | 본 구현 source of truth |
| root `docs/DEV.md` | active develop와 critical risk 인덱스 갱신 |

## 검증

| 검증 | 결과 | 비고 |
|---|---|---|
| Python syntax | 통과 | 신규 Python 전체 `py_compile` |
| config/registry | 통과 | fragment, override, duplicate protection, stub failure |
| grouping | 통과 | seed determinism, visible overlap filtering |
| parameter | 통과 | snapshot/update/restore, norm clipping |
| Jacobian | 통과 | central finite difference 상대 오차 기준 통과 |
| gradient identity | 통과 | `J^T r`와 autograd backward 일치 |
| curvature | 통과 | symmetry와 PSD tolerance, diagonal block 일치 |
| solver | 통과 | LM solution, block/group consistency, predicted reduction |
| aggregation | 통과 | anchor/weighted overlap과 union prediction |
| state/profiling | 통과 | portable state·RNG roundtrip, CPU timing/memory smoke |
| test 합계 | `13/13` 통과 | 실제 CUDA scene test 제외 |
| whitespace | 통과 | RHO_GS nested repo `git diff --check` |

## 관련 명령

```bash
export PYTHONPATH=src:src/Methods/RHO_GS
```

```bash
python -m experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  checkpoint=<checkpoint-path> \
  grouping=visible_overlap_knn \
  grouping.group_size=8
```

```bash
python -m experiments.runners.one_step_benchmark \
  experiment=lego_position_reference \
  checkpoint=<checkpoint-path> \
  solver=group_lm
```

```bash
python -m experiments.runners.rollout_benchmark \
  experiment=lego_position_reference \
  checkpoint=<portable-state-path> \
  benchmark.iterations=100
```

```bash
PYTHONPATH=src/Methods/RHO_GS python -m unittest discover \
  -s src/Methods/RHO_GS/experiments/tests -v
```

## 커밋

| commit | 내용 | 상태 |
|---|---|---|
| 미생성 | Coupling reference framework | uncommitted |

## 한계 및 후속 작업

- 실제 Lego CUDA end-to-end smoke가 아직 필요하다.
- visible-overlap은 CUDA instance identity가 아닌 footprint 근사다.
- LPIPS는 암묵적 network download를 피하기 위해 reference 결과에서 `null`이다.
- oracle JᵀJ grouping, 추가 attribute blocks, robust residual, CUDA optimized Jacobian은 명시적 미구현이다.
- model-only late checkpoint는 Adam 비교에 부적합하다. portable early/middle/late state baseline을 먼저 생성해야 한다.
- GPU 결과가 reference tolerance를 통과하기 전 CUDA kernel 최적화를 시작하지 않는다.

### 2026-07-29 — validation-expanded

- 상태: `implemented-cpu-verified-gpu-unverified`
- 앞선 `cpu-reference-verified` Event의 13개 검사 이후 baseline opt-in 격리와 synthetic one-step loss 감소 검사를 추가했다.
- 최종 CPU test 결과는 `15/15` 통과다.
- summarizer plot과 기본 dry-run sweep runner도 추가했으며 실제 GPU sweep은 수행하지 않았다.

### 2026-07-29 — reproducibility-and-adam-guard

- 상태: `implemented-cpu-verified-gpu-unverified`
- experiment YAML과 portable-state training YAML이 repository의 `*.yaml` ignore 규칙에 묻히지 않도록 좁은 예외를 `.gitignore`에 추가했다.
- root NeRFICG와 nested RHO_GS의 commit·branch·dirty 상태를 `environment.json`에 분리 기록하도록 수정했다.
- 비어 있는 Adam moment state는 기본 거부하고 명시적 `allow_cold_start=true`에서만 허용하도록 강화했다.
- 거절된 position proposal은 hybrid remaining-attribute Adam 전에 복원하도록 수정했다.
- 최종 CPU test 결과는 `16/16` 통과다.

### 2026-07-29 — work-registered

- 상태: `implemented-cpu-verified-gpu-unverified`
- portable-state baseline을 [[src/Methods/RHO_GS/docs/work/2026/W-20260729-001-rho-gs-lego-portable-state-baseline\|W-20260729-001]]로 등록했다.
- analyzer·one-step·rollout을 [[src/Methods/RHO_GS/docs/work/2026/W-20260729-002-lego-position-coupling-reference\|W-20260729-002]]로 등록했다.
- 두 Work 모두 실제 실행 전이므로 `awaiting-run` 상태다.

### 2026-07-29 — commit-scope-correction

- 상태: `commit-prepared-gpu-unverified`
- 사용자는 Git commit을 RHO_GS nested repository로만 제한했다.
- 기존 `.gitignore`의 Obsidian 사용자 변경과 섞이지 않도록 coupling YAML 예외 규칙은 제거했다.
- 신규 experiment YAML과 portable-state YAML은 이번 RHO_GS commit에서 명시적으로 force-stage한다.
- NeRFICG root의 DEV/WORKS/MAIN index와 정책은 로컬 문서로만 유지하고 commit하지 않는다.

### 2026-07-30 — gpu-smoke-blocked

- 상태: `committed-gpu-blocked`
- commit `5f1ef86`과 namespace commit `d51aa85` 상태에서 기존 Lego 30K checkpoint를 사용한 최소 analyzer GPU smoke를 실행했다.
- 조건은 independent group 1개, Gaussian 1개, top tile 1개, sampled pixel 1개, image scale 0.125였다.
- runtime setup, dataset/checkpoint load, RHO_GS model·renderer 생성까지 성공했으나 pixel sampling에서 중단됐다.
- `experiments/sampling.py`가 CPU generator를 생성한 뒤 Framework의 CUDA default device를 따르는 `torch.randperm()`에 전달해 `Expected a 'cuda' device type for generator but found 'cpu'`가 발생했다.
- failure artifact는 `output/RHO_GS/experiments/d20260729_001_gpu_smoke_analyze-20260730T020430173365Z-1a3b62fa`에 저장됐다.
- 기존 16개 CPU `unittest`는 모두 통과했으므로 현재 blocker는 GPU default-device 조건을 다루지 못한 sampling 경로로 한정된다.
- 문서화된 `PYTHONPATH=src:src/Methods/RHO_GS`는 source `RHOGSCudaBackend`가 설치 extension namespace를 가리는 별도 충돌을 만든다. `PYTHONPATH=src`와 `Methods.RHO_GS.experiments...` package entry로 실행하면 이 충돌을 피할 수 있다.
- `rho_gs_lego_coupling_states.yaml`에는 퇴역한 `TILE_GAUSSIAN_STATS.ACTIVE` 항목이 남아 있어 후속 정리 대상이다.
- 이번 검증에서는 원인 확인과 artifact 기록만 수행했고 source 수정은 하지 않았다.

### 2026-07-30 — gpu-reference-verified

- 상태: `gpu-smoke-verified-fix-uncommitted`
- sampling, grouping anchor 및 rollout camera sequence의 CPU generator가 CUDA default device와 충돌하지 않도록 `torch.randperm(..., device="cpu")`를 명시했다.
- CUDA default-device에서 sampling과 grouping determinism을 확인하는 regression test를 추가했고 GPU 전체 `unittest` 17개가 통과했다.
- GPU runner 명령을 `PYTHONPATH=src`와 `Methods.RHO_GS.experiments.runners.<runner>` package entry로 정정하고 portable-state config의 퇴역 tile 통계 항목을 제거했다.
- 최소 independent analyzer는 `d20260729_001_gpu_smoke_analyze_fixed2-20260730T021437670040Z-158c0bd4`에서 완료됐고 failure record는 0개였다.
- visible-overlap K=2 analyzer도 전체 경로와 pairwise CSV 생성을 완료했으나 random anchor가 sampled residual에 기여하지 않아 Hessian과 coupling이 0이었다.
- 동일 full tile loss의 전체 position gradient는 14,279개 Gaussian에서 nonzero였으므로 rasterizer backward 또는 residual sampling 전체가 0인 문제는 아니었다.
- gradient norm 상위 Gaussian 49,945와 49,873의 repeated-VJP reference는 Jacobian norm 43.9866, raw off-diagonal Frobenius norm 642.5876, normalized spectral coupling 0.9690, Frobenius coupling 1.3670, symmetry relative error 0을 기록했다.
- 이 결과는 실제 기여 pair의 GPU Jacobian·Gauss-Newton·normalized coupling 경로가 동작함을 검증하며, tile-overlap random anchor가 실제 contributor를 보장하지 않는 별도 sampling 한계를 드러낸다.
- 공식 W-20260729-002/R01 실행 전 contributor-aware anchor 또는 분석 전용 fixed/oracle anchor 전략을 추가하는 것이 필요하다.
