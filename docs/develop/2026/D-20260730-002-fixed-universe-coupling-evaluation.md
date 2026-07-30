---
record_id: D-20260730-002
type: develop
method: RHO_GS
created_at: 2026-07-30T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/develop/2026/D-20260730-002-fixed-universe-coupling-evaluation.md
retrospective: false
---

# D-20260730-002 — Fixed-universe Coupling 평가

## 요약

| 항목 | 내용 |
|---|---|
| 구현 목적 | grouping으로 평가 모집단을 만든 기존 induced-union selection bias 분리 |
| 범위 | universe registry, fixed cube, anchor/group 제한, 명시적 edge capture, random baseline, oracle 상한 |
| 기존 경로 영향 | `universe=disabled`가 기본이며 기존 analyzer·one-step·rollout 동작 보존 |
| 관련 위험 | root `RISK-RHO-015` |
| 관련 Work | 실제 fixed-region GPU benchmark 등록 예정 |
| 상태 | `cpu-verified-gpu-pending` |

## 문제와 근거

기존 checkpoint analyzer는 anchor별 group을 먼저 만들고 모든 group member의 합집합에서만 exact Jacobian과 Hessian을 계산했다. 같은 induced union의 전체 pair energy를 분모로 사용했으므로, grouping이 선택하지 않은 Gaussian과 강한 coupling을 가졌는지는 평가할 수 없었다.

따라서 기존 약 99% capture 결과는 선택된 부분집합 내부 응집도에는 유효하지만 고정된 공간 모집단에서 중요한 coupling을 99% 찾았다는 의미가 아니다.

## 구현 정책

| 구성 | 정책 |
|---|---|
| 실행 순서 | residual/pixel → universe → anchor → group → universe Jacobian/Hessian |
| universe | world-space axis-aligned fixed cube |
| 중심 | explicit xyz 또는 checkpoint-local Gaussian ID |
| 경계 | 각 축에서 `abs(x-center) <= half_extent` |
| overflow | 기본 `error`; 명시적 `nearest_center`만 truncation 허용 |
| anchor | universe와 geometry/contributor 조건의 교집합 |
| member | 모든 grouping에서 universe 밖 ID 금지 |
| unconditional | `contributor_only=False`, `filter_group_members=False` |
| conditional | contributor-only universe 또는 member filter를 결과에 명시 |
| 기존 호환 | `universe=disabled`에서 group-induced union 사용 |

## 지표 정의

| 지표 | 분모 | 분자 |
|---|---|---|
| `induced_union_capture` | group union 내부 모든 pair | group 내부 pair |
| `fixed_universe_group_capture` | fixed universe 모든 pair | group 내부 pair |
| `fixed_universe_anchor_capture` | anchor-to-universe 모든 pair | 선택 anchor-member pair |
| `edge_budget_oracle_capture` | anchor-to-universe 모든 pair | 같은 global edge 수의 raw energy top pair |
| `oracle_relative_capture` | oracle capture ratio | anchor capture ratio |

모든 capture는 position Gauss-Newton off-diagonal block의 squared Frobenius norm을 energy로 사용한다. Edge는 unordered unique pair로 중복 제거한다.

`edge_budget_oracle_capture`는 per-anchor K와 실제 group 구성 가능성을 강제하지 않는 느슨한 상한이다. 분석 전용 `oracle_jtj_topk` grouping은 여전히 미구현이다.

## 변경 파일

| 경로 | 내용 |
|---|---|
| `experiments/coupling/universe/` | disabled/fixed-cube 전략과 universe ID hash |
| `experiments/types.py`, `protocols.py`, `registry.py` | `GaussianUniverse`와 extension point |
| `experiments/coupling/anchor_selection/strategies.py` | anchor 후보를 universe로 제한 |
| `experiments/coupling/grouping/strategies.py` | member 제한과 `random_in_universe` 기준선 |
| `experiments/coupling/curvature/assembler.py` | explicit edge capture와 equal-budget oracle |
| `experiments/runners/common.py` | universe-first orchestration |
| `experiments/runners/analyze_checkpoint.py` | fixed/induced 지표와 pair classification |
| `experiments/configs/` | universe fragment와 Lego fixed-region reference |
| `experiments/tests/test_universe_capture.py` | cube, overflow, 경계, capture 회귀 테스트 |

## Artifact

| 파일 | 추가 정보 |
|---|---|
| `universe.json` | 전체 ID, ID hash, 중심, half extent, filter, overflow 정책 |
| `coupling_metrics.csv` | eligible/selected anchor edge, group edge, induced-union pair 여부 |
| `summary.json` | capture 정의, fixed/induced/oracle 지표와 edge 수·energy |
| `timing.csv` | `universe_selection` phase |

## Events

### 2026-07-30 — implementation-and-cpu-validation

- 상태: `cpu-verified-gpu-pending`
- RHO_GS core와 CUDA는 변경하지 않고 `experiments/` reference framework만 확장했다.
- 전체 unit test 30개 중 CPU 및 일반 test 28개가 통과했다.
- 현재 sandbox에서 CUDA가 노출되지 않아 CUDA default-device test 2개는 skip됐다.
- 기존 기본 config는 `universe=disabled`이며 legacy `coupling_capture_ratio` 회귀 테스트가 통과했다.
- 신규 YAML은 repository 전역 `*.yaml` ignore 대상이므로 commit 시 신규 config 3개를 명시적으로 force-add해야 한다.

## 한계 및 주의

- `visible_only=True`는 현재 sampled tile로 제한된 근사 render-state visibility를 사용한다.
- arbitrary 3D cube는 같은 sampled pixel contributor를 보장하지 않는다. zero pair는 unconditional sparsity 결과로 보존한다.
- center Gaussian ID는 checkpoint-local이며 다른 topology/checkpoint에서 같은 identity를 뜻하지 않는다.
- fixed-universe exact repeated VJP는 universe Gaussian 수에 비례하므로 초기 최대 크기는 64개다.
- GPU analyzer end-to-end와 실제 cube 중심·크기 선정은 아직 검증되지 않았다.

### 2026-07-30 — gpu-unit-regression-verified

- 상태: `gpu-unit-verified-analyzer-pending`
- 샌드박스 밖 RTX 4070 Ti 환경에서 CUDA default-device sampling과 randomized contribution probe를 포함한 전체 unit/regression test `30/30`이 통과했다.
- fixed-cube GPU analyzer end-to-end와 실제 scene region benchmark는 아직 수행하지 않았다.

### 2026-07-30 — validation-count-correction

- 기존 값: GPU unit/regression `30/30`
- 수정 값: 최종 GPU unit/regression `32/32`
- 사유: random grouping의 contributor member filter와 visible-universe filter 회귀 테스트 2개를 구현 후 추가했다.
- 기존 30개 테스트가 실패한 것은 아니며 최종 test suite 크기 증가를 반영한 정정이다.

### 2026-07-30 — fixed-cube-gpu-analyzer-verified

- 상태: `gpu-analyzer-verified-uncommitted`
- 선택 규칙은 기존 공식 seed 0 contributor sample의 첫 anchor ID `200811`을 cube 중심으로 고정하는 것이며, center xyz는 `(-0.0877858, -0.0023533, 0.0639722)`다.
- half extent 0.01 strict cube는 45개 Gaussian을 포함했고 universe hash는 `47acd5cebb73d6c9`였다. visibility/contributor universe filter와 overflow truncation은 적용하지 않았다.
- 두 run 모두 anchor 후보 36개, candidate hash `7d0fd13147843670`, 동일 anchor 8개를 사용했다.
- R01 visible-overlap artifact: `output/RHO_GS/experiments/d20260730_002_r01_fixed_cube_visible_overlap-20260730T055012012433Z-8c49981e`
- R02 random-visible artifact: `output/RHO_GS/experiments/d20260730_002_r02_fixed_cube_random_visible-20260730T055050474386Z-ece33bf7`

| 지표 | visible-overlap K=8 | random-visible K=8 |
|---|---:|---:|
| unique selected anchor edge | 53 | 55 |
| fixed anchor capture | 20.56% | 11.06% |
| equal-budget oracle capture | 97.30% | 97.49% |
| oracle relative capture | 21.13% | 11.35% |
| fixed universe group capture | 63.54% | 24.81% |
| induced-union capture | 81.69% | 55.47% |
| induced union Gaussian | 35 | 36 |
| valid pair ratio | 81.70% | 66.07% |
| zero group | 0/8 | 0/8 |
| group construction | 11.78 ms | 1.31 ms |
| Jacobian + curvature | 408.66 ms | 419.79 ms |
| peak allocated GPU memory | 321,540,096 bytes | 321,540,096 bytes |

- 고정 universe 전체 990 pair의 raw-zero 비율은 35.86%, anchor-to-universe 324 pair는 22.22%였다.
- visible-overlap selected anchor edge의 raw-zero 비율은 13.21%, random-visible은 21.82%였다.
- visible-overlap은 거의 같은 unique edge budget에서 random보다 raw anchor energy를 약 1.86배 포착했지만, oracle 상한의 21.13%에 그쳤다.
- 기존처럼 grouping 이후 union을 분모로 두면 visible-overlap capture는 81.69%지만 고정 anchor 분모에서는 20.56%다. 이는 induced-union selection이 grouping 성능을 낙관적으로 보일 수 있다는 사용자 가설을 실제 artifact에서 확인한다.
- selected anchor `c_ij` 중앙값은 visible-overlap 0.8710, random-visible 0.9070으로 raw capture 순서와 반대였다. normalized pair 강도와 전체 raw energy capture는 별도 지표로 해석한다.
- 두 artifact 모두 `completed`, zero group 0/8, failure file 0 byte였다.

### 2026-07-30 — legacy-passthrough-exact-verified

- 상태: `gpu-analyzer-verified-legacy-exact-uncommitted`
- 최종 GPU unit/regression test suite `33/33`이 통과했다. float32 capture division passthrough test를 추가한 결과다.
- 첫 legacy regression R03에서 모든 pairwise coupling은 동일했지만 새 explicit edge helper가 Python float division을 사용해 기존 capture와 `2.95e-8` 차이가 발생했다.
- ratio를 기존과 동일하게 Hessian tensor dtype에서 계산하도록 수정하고 regression test를 추가했다.
- 최종 artifact: `output/RHO_GS/experiments/d20260730_002_r04_legacy_exact_regression-20260730T060152664176Z-5321eda9`
- 기존 공식 W-20260729-002/R01과 anchor ID, candidate hash, sampled Gaussian 62개, 1,891개 pair key와 `c_ij`/`c_ij_fro`/raw block norm, capture, `epsilon_G`, condition number, group diagnostics가 정확히 일치했다.
- 기존과 최종 capture는 모두 `0.9894756078720093`, 모든 pairwise 수치의 최대 절대 차이는 0이었다.
- R04는 `completed`, failure file 0 byte였다. R03은 중간 regression artifact이며 최종 legacy 근거로 사용하지 않는다.

### 2026-07-30 — yaml-count-correction

- 기존 값: commit 시 force-add가 필요한 신규 YAML 3개
- 수정 값: 신규 YAML 4개
- 사유: `universe/disabled.yaml`, `universe/fixed_cube.yaml`, `grouping/random_in_universe.yaml`, `experiment/lego_position_fixed_region.yaml`을 모두 포함한 실제 파일 수로 정정했다.
- repository의 `*.yaml` ignore 정책은 변경하지 않았으며 향후 RHO_GS commit 시 네 파일만 명시적으로 force-add한다.
