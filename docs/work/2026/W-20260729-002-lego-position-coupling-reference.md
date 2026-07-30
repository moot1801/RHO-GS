---
record_id: W-20260729-002
type: work
method: RHO_GS
created_at: 2026-07-29T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/work/2026/W-20260729-002-lego-position-coupling-reference.md
retrospective: false
---

# W-20260729-002 — Lego Position Coupling Reference

## 요약

| 항목 | 내용 |
|---|---|
| 목적 | sampled position `J^T J` coupling, one-step LM, fixed-topology rollout 검증 |
| scene | NeRF Synthetic Lego |
| 관련 Develop | [[src/Methods/RHO_GS/docs/develop/2026/D-20260729-001-coupling-experiment-framework\|D-20260729-001]] |
| 상태 | `awaiting-run` |
| 기본 budget | 0.25×, top 2 tile, 16 pixel/tile, 8 anchor, train view 51 |

## Runs

### R01 — Late model-only checkpoint analyzer

| 항목 | 값 |
|---|---|
| 상태 | `awaiting-run` |
| checkpoint | W-20260728-003/R01 `final.pt`, iteration 30K |
| optimizer state | 없음; coupling/LM 분석만 수행 |
| grouping | visible-overlap kNN, K=8 |

```bash
PYTHONPATH=src:src/Methods/RHO_GS \
python -m experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r01_late_checkpoint_analysis \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  grouping=visible_overlap_knn \
  grouping.group_size=8
```

### R02 — Late model-only checkpoint one-step Group LM

| 항목 | 값 |
|---|---|
| 상태 | `awaiting-run` |
| 선행 조건 | R01 analyzer 수치 검증 |
| optimizer state | 없음; remaining-attribute Adam은 cold-state로 결과에 표시됨 |

```bash
PYTHONPATH=src:src/Methods/RHO_GS \
python -m experiments.runners.one_step_benchmark \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r02_late_one_step \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  solver=group_lm \
  aggregation=anchor_only
```

### R03 — Portable-state fixed-topology rollout

| 항목 | 값 |
|---|---|
| 상태 | `awaiting-run` |
| 선행 조건 | W-20260729-001/R01 portable state와 R01/R02 검증 |
| iterations | 최초 smoke 후 100 |

```bash
PYTHONPATH=src:src/Methods/RHO_GS \
python -m experiments.runners.rollout_benchmark \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r03_portable_rollout \
  checkpoint=<portable-state-path> \
  benchmark.iterations=100
```

## Events

### 2026-07-29 — planned

- 상태: `awaiting-run`
- runner와 CPU reference test만 준비됐으며 실제 Lego CUDA 실행 결과는 없다.
- R01부터 순서대로 실행하고 각 output directory와 failure record를 보고한다.

### 2026-07-30 — command-correction

- 기존 `PYTHONPATH=src:src/Methods/RHO_GS`는 source `RHOGSCudaBackend`가 설치 extension namespace를 가리므로 GPU runner 명령에서 사용하지 않는다.
- analyzer, one-step, rollout은 `PYTHONPATH=src`와 `Methods.RHO_GS.experiments.runners.<runner>` package entry를 사용한다.

```bash
PYTHONPATH=src \
python -m Methods.RHO_GS.experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r01_late_checkpoint_analysis \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  grouping=visible_overlap_knn \
  grouping.group_size=8
```

### 2026-07-30 — gpu-smoke-evidence

- 상태: `awaiting-run`
- Develop 검증용 최소 analyzer는 GPU에서 완료됐지만 공식 R01 조건을 실행한 것은 아니므로 Run 상태는 변경하지 않는다.
- random tile-overlap anchor는 실제 sampled residual contributor를 보장하지 않아 K=2 pair가 zero Hessian을 만들 수 있음을 확인했다.
- 동일 tile의 실제 gradient 상위 Gaussian pair에서는 raw off-diagonal Frobenius norm 642.5876과 normalized spectral coupling 0.9690을 확인했다.
- 공식 R01 전에 contributor-aware 또는 fixed/oracle anchor 전략을 추가해 zero group 비율과 유효 group 수를 함께 기록해야 한다.

### 2026-07-30 — contributor-aware-anchor-implemented

- 상태: `awaiting-run`
- [[src/Methods/RHO_GS/docs/develop/2026/D-20260730-001-contributor-aware-anchor-diagnostics\|D-20260730-001]]에서 randomized VJP contributor population과 seed 고정 random anchor를 구현했다.
- 공식 R01 preset은 `anchor_selection=random_contributor`, contributor member filter, zero-group 보존·분류를 사용한다.
- `anchors.jsonl`, `group_diagnostics.csv`, checkpoint SHA-256을 추가 재현 artifact로 정의했다.
- 구현 검증용 GPU smoke는 공식 R01과 구분하며, R01 상태는 실제 명령 실행 전까지 `awaiting-run`을 유지한다.

### 2026-07-30 — contributor-aware-gpu-smoke

- 상태: `awaiting-run`
- 공식 R01 전 Develop 검증으로 late checkpoint의 K=2 최소 analyzer를 실행했다.
- artifact: `output/RHO_GS/experiments/d20260730_001_gpu_contributor_k2_smoke_r02-20260730T033253118076Z-ebee2fbe`
- contributor 후보 12,284개에서 seed 0으로 anchor 66,981을 선택했고 근접 contributor 66,980과 group을 구성했다.
- exact 결과는 zero group 0/1, valid pair 1/1, shared sampled pixel 1, normalized spectral coupling 0.138625, capture ratio 1.0, failure 0이었다.
- 같은 조건의 반복 실행에서 candidate hash와 선택 ID가 일치해 sampling 재현성을 확인했다.
- 본 실행은 anchor 1개인 구현 smoke이므로 K=8·anchor 8개의 공식 R01 완료로 간주하지 않는다.

### 2026-07-30 — gpu-smoke-artifact-superseded

- 상태: `awaiting-run`
- 최종 metadata schema 검증 artifact는 `output/RHO_GS/experiments/d20260730_001_gpu_contributor_k2_smoke_r03-20260730T033633597845Z-24edc684`다.
- 앞선 R02 수치는 유지하되, 재현 시에는 sampled pixel ID와 footprint/group metadata까지 포함한 R03을 우선 참조한다.
- 공식 R01 상태와 조건은 변경하지 않는다.
