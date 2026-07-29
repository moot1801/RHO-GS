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
