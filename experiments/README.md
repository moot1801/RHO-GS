# RHO_GS Inter-Gaussian Coupling 실험 프레임워크

## 목적과 범위

이 패키지는 기존 RHO_GS 학습·렌더링 경로를 바꾸지 않고 sampled Gaussian/pixel subset에서 position coupling을 분석한다. 최초 milestone은 정확성 검증용 dense reference 구현이며 CUDA 최적화가 아니다.

| 완료 기능 | 상태 |
|---|---|
| Registry 기반 universe/anchor/grouping/parameter/residual/Jacobian/curvature/solver/aggregation/acceptance 교체 | 완료 |
| Position parameter block | 완료 |
| independent, kNN 3D, visible kNN, visible-overlap kNN | 완료 |
| RGB L2 전체·sampled residual | 완료 |
| 반복 VJP 및 finite-difference Jacobian | 완료 |
| sampled dense `J^T J`, normalized coupling, capture ratio | 완료 |
| per-Gaussian/group GN·LM, Adam state adapter | 완료 |
| one-step counterfactual, fixed-topology rollout | reference 구현 완료 |
| optimizer/RNG/sampler portable checkpoint | 완료 |
| contributor-aware anchor와 zero-group 진단 | 완료 |
| fixed-cube evaluation universe, random-in-universe, fixed-universe capture | 완료 |
| oracle JᵀJ grouping, 추가 attribute, CUDA 최적화 | 미구현 |

## 실행 환경

NeRFICG root에서 다음 path를 사용한다.

```bash
export PYTHONPATH=src
```

모든 runner는 `key=value` 인자를 받는다. `grouping=knn_3d` 같은 값은 `configs/grouping/knn_3d.yaml` fragment를 선택하며 `grouping.group_size=8`은 최종 값을 덮어쓴다. 소문자 `true`/`false`도 실제 boolean으로 해석된다.

### 1. 현재 late checkpoint 분석

```bash
python -m Methods.RHO_GS.experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r01_late_checkpoint_analysis \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  anchor_selection=random_contributor \
  grouping=visible_overlap_knn \
  grouping.group_size=8 \
  attributes=position
```

### 2. One-step counterfactual

```bash
python -m Methods.RHO_GS.experiments.runners.one_step_benchmark \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r02_late_one_step \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  solver=group_lm \
  aggregation=anchor_only
```

현재 `final.pt`는 optimizer state가 없는 model-only checkpoint다. LM 분석은 가능하지만 `solver=adam`은 기본적으로 실패시켜 불공정 비교를 막는다. `solver.allow_cold_start=true`는 탐색용 cold-start임을 결과에 기록할 때만 사용한다.

### Fixed-region checkpoint 분석

기존 analyzer는 group을 먼저 만든 뒤 group member의 합집합을 분모로 사용했다. 이 `induced_union_capture_ratio`는 선택된 부분집합 내부 응집도이며, grouping이 고정 모집단의 coupling을 얼마나 찾았는지는 나타내지 않는다. 공정한 grouping 비교에는 grouping 전에 cube universe를 고정한다.

```bash
python -m Methods.RHO_GS.experiments.runners.analyze_checkpoint \
  experiment=lego_position_fixed_region \
  experiment_name=<work-run-label> \
  checkpoint=<checkpoint-path> \
  universe=fixed_cube \
  universe.center_gaussian_id=<checkpoint-local-id> \
  universe.half_extent=<world-space-half-extent> \
  universe.maximum_gaussians=64 \
  anchor_selection=random_contributor \
  anchor_selection.filter_group_members=False \
  grouping=visible_overlap_knn \
  grouping.group_size=8
```

처리 순서는 `sampled residual → fixed universe → anchor → group → universe Jacobian/Hessian`이다. `overflow_policy=error`가 기본이므로 cube가 최대 크기를 넘으면 영역을 암묵적으로 잘라내지 않고 실패한다. `nearest_center`는 명시한 경우에만 결정론적 진단용 truncation으로 사용한다.

같은 universe·anchor·K에서 `grouping=random_in_universe`를 실행하면 공간 grouping의 random 기준선을 얻는다. `edge_budget_oracle_capture_ratio`는 실제 grouping이 아니라 동일한 전체 edge 수에서 raw off-diagonal energy가 큰 edge를 고른 느슨한 상한이다.

### 3. early/middle/late portable state 학습

```bash
python -m Methods.RHO_GS.experiments.runners.train_with_portable_states \
  -c src/Methods/RHO_GS/rho_gs_lego_coupling_states.yaml
```

| iteration | artifact |
|---:|---|
| 1,000 | `portable_states/iteration_001000.rho_state.pt` |
| 15,000 | `portable_states/iteration_015000.rho_state.pt` |
| 30,000 | `portable_states/iteration_030000.rho_state.pt` |

각 artifact는 Gaussian parameter, PyTorch Adam state, RNG, camera sampler, SH degree, iteration, snapshot-local Gaussian identity를 포함한다. Densification이 끝나기 전 snapshot 사이에서 같은 integer ID가 같은 Gaussian이라고 가정하면 안 된다.

### 4. Fixed-topology rollout

```bash
python -m Methods.RHO_GS.experiments.runners.rollout_benchmark \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r03_portable_rollout \
  checkpoint=<portable-state-path> \
  grouping=visible_overlap_knn \
  grouping.group_size=8 \
  solver=group_lm \
  benchmark.iterations=100
```

rollout은 densify/clone/split/prune/opacity reset/SH 증가 callback을 호출하지 않는다. position은 선택한 coupled solver로 갱신하고, position gradient를 제거한 뒤 나머지 속성만 복원된 Adam으로 갱신한다. 매 step Gaussian 수와 NaN/Inf를 검사한다.

실행 후 최소 보고 형식은 `Work/Run ID`, `완료·실패·중단`, `output 경로 또는 오류 로그`다. 제공 명령을 바꿨거나 artifact를 이동·삭제한 경우도 함께 전달한다.

## 기본 sampling

| 항목 | 기본값 | 의미 |
|---|---:|---|
| image scale | 0.25× | dataset 로딩 시 camera/image를 함께 축소 |
| tile | top 2 | 근사 footprint 등록 수가 큰 tile |
| pixel | tile당 16 | seed 고정 uniform sample |
| anchor | 8 | sampled pixel contributor 모집단에서 seed 고정 uniform 선택 |
| footprint | isotropic 3σ | `max(scale) × focal / depth` 보수적 근사 |

`random_contributor`는 Rademacher VJP로 Gaussian별 sampled position Jacobian energy `||J_i||_F²`를 추정하고, threshold를 넘는 contributor 중 uniform random anchor를 고른다. `filter_group_members=true`이면 group member도 같은 contributor 모집단으로 제한한다. 정확한 group Jacobian과 Hessian은 이후 기존 reference provider로 다시 계산하므로 probe 점수 자체를 coupling 값으로 사용하지 않는다.

| anchor strategy | 용도 |
|---|---|
| `random_eligible` | 기존 geometry-visible 모집단 기준선 |
| `fixed` | checkpoint-local ID 재현·진단 |
| `random_contributor` | 편향을 줄인 기본 coupling 분석 |
| `contribution_topk` | 상위 기여 Gaussian 분석 전용; `analysis_only` |

visible-overlap grouping의 공간 overlap은 여전히 rasterizer instance ID가 아니라 projected footprint 근사를 사용한다. anchor의 sampled residual 기여 여부와 공간 overlap 근사는 서로 다른 조건이며 각각 `anchors.jsonl`과 `groups.jsonl`에 기록된다.

### Evaluation universe

| universe | 의미 |
|---|---|
| `disabled` | 기존 group-induced union 평가 보존 |
| `fixed_cube` | grouping 전에 world-space axis-aligned cube로 모집단 고정 |

`fixed_cube.visible_only=true`는 현재 sampled tile로 제한된 render-state visibility와 cube를 교집합한다. `contributor_only=false`는 zero pair를 포함하는 unconditional 모집단이며, anchor만 contributor 중 선택할 수 있다. `contributor_only=true`는 active subset 조건부 분석이므로 두 결과를 혼동하지 않는다.

### Zero group 해석

`group_diagnostics.csv`는 zero group도 삭제하지 않고 다음 상태로 분류한다.

| 분류 | 의미 |
|---|---|
| `anchor_zero` | exact sampled Jacobian에서 anchor block이 0 |
| `member_zero` | anchor는 active지만 비교할 active member가 없음 |
| `no_shared_support` | 양쪽 block은 active지만 같은 sampled pixel에 기여하지 않음 |
| `small_offdiagonal` | shared support는 있으나 설정 threshold보다 작은 off-diagonal |
| `valid` | shared support와 유효 off-diagonal pair 존재 |

전체 group 수, zero/valid group 비율, valid pair 비율은 `summary.json`에 함께 저장한다. zero group을 누락한 조건부 결과만으로 grouping을 평가하면 selection bias가 생길 수 있으므로 전체 통계를 기본 보고값으로 사용한다.

## 결과 구조

```text
output/RHO_GS/experiments/<experiment-id>/
├── resolved_config.yaml
├── environment.json
├── checkpoint_metadata.json
├── universe.json
├── anchors.jsonl
├── groups.jsonl
├── group_diagnostics.csv
├── coupling_metrics.csv
├── one_step_results.csv
├── rollout_metrics.csv
├── timing.csv
├── memory.csv
├── failures.jsonl
├── summary.json
└── plots/
```

실패·NaN은 누락하지 않고 `failures.jsonl`에 남는다. GPU 구간은 CUDA event와 synchronization으로 측정한다. LPIPS는 weight 다운로드를 암묵적으로 발생시키지 않기 위해 reference runner에서는 `null`이며 기존 NeRFICG 평가 경로로 별도 측정한다.

### 선택적 sweep과 plot

```bash
python -m Methods.RHO_GS.experiments.runners.sweep \
  runner=analyze_checkpoint \
  matrix=src/Methods/RHO_GS/experiments/configs/sweep/initial_position.yaml \
  dry_run=true
```

기본값은 실행하지 않고 `sweep_manifest.json`만 만든다. matrix의 `<checkpoint-path>`를 바꾼 뒤 `dry_run=false`로 명시한 조합만 실행한다. 결과 plot은 다음처럼 생성한다.

```bash
python -m Methods.RHO_GS.experiments.runners.summarize_results \
  output/RHO_GS/experiments/<experiment-id>
```

## 새 전략 등록

1. 대응 protocol을 구현한다: `experiments/protocols.py`.
2. `REGISTRIES["<kind>"].register("name")` decorator를 붙인다.
3. 해당 package `__init__.py`에서 모듈을 import한다.
4. config fragment를 추가한다.
5. CPU 수치 test와 가능하면 finite-difference 비교를 추가한다.

예시:

```python
@REGISTRIES["grouping"].register("my_grouping")
class MyGrouping:
    name = "my_grouping"

    def build_groups(self, gaussian_state, render_state, view, config):
        return GroupSet(...)
```

핵심 runner에 새 `if/elif`를 추가할 필요가 없다.

Anchor 전략은 `REGISTRIES["anchor_selection"]`에 등록한다. contributor score가 필요한 전략은 `requires_contribution_scores=True`를 선언하고 `select(..., contribution_scores)`에서 명시적 candidate population을 반환한다. `AnchorSet` artifact에는 선택 ID/점수, seed, candidate count/hash가 저장되며 전체 candidate ID는 메모리에서 grouping filter에만 사용한다.

Universe 전략은 `REGISTRIES["universe"]`에 등록하며 `build(...)`가 grouping 전에 완전한 Gaussian ID 집합과 hash를 반환한다. anchor와 group strategy는 전달된 universe 밖 ID를 선택하면 안 된다.

## 수치 검증

```bash
(cd src/Methods/RHO_GS && python -m unittest discover -s experiments/tests -v)
```

검증 항목은 config/registry, grouping determinism, snapshot/restore, step clipping, finite difference Jacobian, `J^T r` gradient identity, `J^T J` symmetry·PSD, block consistency, GN/LM solve, predicted reduction, overlapping aggregation, portable state, timing/memory smoke다.

## 알려진 제한

- 반복 VJP는 sampled reference이며 full image/full scene 계산용이 아니다.
- custom rasterizer는 double backward를 지원하지 않으므로 exact Newton은 범위 밖이다.
- scale+rotation, opacity, DC color, opacity+DC, SH, composite block은 registry에 존재하지만 `NotImplementedError`를 발생시킨다.
- `sqrt_l1_dssim`, `mse_dssim_diagonal`, `robust_rgb_irls` residual도 명시적 skeleton이다.
- `oracle_jtj_topk`와 optimized CUDA 경로는 Priority 4로 남겨 두었다.
- `edge_budget_oracle_capture_ratio`는 per-anchor K와 group realizability를 강제하지 않는 global edge-budget 상한이며 `oracle_jtj_topk` grouping을 대체하지 않는다.
- fixed universe의 exact repeated-VJP 비용은 Gaussian 수에 선형으로 증가하므로 reference config는 최대 64개를 기본으로 한다.
- one-step/rollout GPU 장면 통합은 실제 실행 전까지 `미확인`이다. 근거 구현은 `experiments/runners/`이며 CPU 수치 test만 완료됐다.
- output의 Gaussian ID는 해당 checkpoint/snapshot 내부 index다. topology가 다른 checkpoint끼리 직접 identity로 대응시키지 않는다.
- randomized VJP score는 유한 probe 추정치이며 exact per-Gaussian Jacobian norm이 아니다. probe count·seed·threshold를 비교 실험에서 고정한다.
- `contribution_topk`는 기여도가 큰 표본으로 분포를 편향시키므로 upper-bound/진단 결과로만 해석한다.
