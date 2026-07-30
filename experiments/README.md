# RHO_GS Inter-Gaussian Coupling 실험 프레임워크

## 목적과 범위

이 패키지는 기존 RHO_GS 학습·렌더링 경로를 바꾸지 않고 sampled Gaussian/pixel subset에서 position coupling을 분석한다. 최초 milestone은 정확성 검증용 dense reference 구현이며 CUDA 최적화가 아니다.

| 완료 기능 | 상태 |
|---|---|
| Registry 기반 grouping/parameter/residual/Jacobian/curvature/solver/aggregation/acceptance 교체 | 완료 |
| Position parameter block | 완료 |
| independent, kNN 3D, visible kNN, visible-overlap kNN | 완료 |
| RGB L2 전체·sampled residual | 완료 |
| 반복 VJP 및 finite-difference Jacobian | 완료 |
| sampled dense `J^T J`, normalized coupling, capture ratio | 완료 |
| per-Gaussian/group GN·LM, Adam state adapter | 완료 |
| one-step counterfactual, fixed-topology rollout | reference 구현 완료 |
| optimizer/RNG/sampler portable checkpoint | 완료 |
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
| anchor | 8 | sampled tile과 겹치는 visible Gaussian에서 seed 고정 선택 |
| footprint | isotropic 3σ | `max(scale) × focal / depth` 보수적 근사 |

visible-overlap grouping은 현재 rasterizer가 실제 생성한 Gaussian ID 목록이 아니라 위 projected footprint 근사를 사용한다. 이 차이는 `groups.jsonl` metadata에 기록된다.

## 결과 구조

```text
output/RHO_GS/experiments/<experiment-id>/
├── resolved_config.yaml
├── environment.json
├── checkpoint_metadata.json
├── groups.jsonl
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
- one-step/rollout GPU 장면 통합은 실제 실행 전까지 `미확인`이다. 근거 구현은 `experiments/runners/`이며 CPU 수치 test만 완료됐다.
- output의 Gaussian ID는 해당 checkpoint/snapshot 내부 index다. topology가 다른 checkpoint끼리 직접 identity로 대응시키지 않는다.
- tile/footprint overlap 기반 random anchor는 실제 sampled residual 기여를 보장하지 않으므로 zero Jacobian group이 생성될 수 있다. zero group을 실패로 누락하지 말고 기록하며 contributor-aware 또는 oracle anchor는 별도 전략으로 추가해야 한다.
