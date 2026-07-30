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
