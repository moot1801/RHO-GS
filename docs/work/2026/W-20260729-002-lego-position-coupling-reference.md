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
conda run -n nerficg env PYTHONPATH=src \
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

### 2026-07-30 — R01-completed

- 상태: `R01-completed-R02-R03-awaiting-run`
- clean RHO_GS commit `a6d60bdcb6aa7ae3be94776c6d5542e386dedd7d`에서 공식 late checkpoint analyzer를 완료했다.
- output: `output/RHO_GS/experiments/w20260729_002_r01_late_checkpoint_analysis-20260730T044854068438Z-15eeac28`
- checkpoint SHA-256: `5c4982037959f05e5dcf8069525fb1c5f4901ee30c8cdf8454887bf09875f320`
- 환경: RTX 4070 Ti, PyTorch 2.8.0+cu128, RHO_GS branch `rho-gs/testbed`, 실행 시 method dirty `false`.

```bash
conda run -n nerficg env PYTHONPATH=src \
  python -m Methods.RHO_GS.experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  experiment_name=w20260729_002_r01_late_checkpoint_analysis \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  seed=0 \
  anchor_selection=random_contributor \
  anchor_selection.sampled_anchor_count=8 \
  anchor_selection.maximum_anchors=8 \
  anchor_selection.probe_count=4 \
  anchor_selection.probe_seed=0 \
  anchor_selection.relative_score_threshold=1.0e-8 \
  anchor_selection.filter_group_members=True \
  grouping=visible_overlap_knn \
  grouping.group_size=8 \
  grouping.candidate_pool_size=32 \
  grouping.sampled_anchor_count=8 \
  grouping.maximum_groups=8 \
  evaluation.primary_view_id=51 \
  evaluation.image_scale_factor=0.25 \
  evaluation.top_tiles=2 \
  evaluation.pixels_per_tile=16
```

#### R01 sampling과 group

| 항목 | 값 |
|---|---:|
| view / scale | train 51 / 0.25× |
| tile / pixel | top 2 / tile당 16, RGB residual scalar 96개 |
| contributor 후보 | 1,563 |
| candidate hash | `ea870dedaa8e058f` |
| anchor | 8 |
| group | 8개, 모두 K=8 |
| sampled Gaussian union | 62 |
| zero / valid group | 0 / 8 |
| valid pair | 202 / 224, 90.18% |

Anchor ID는 `200811, 200910, 206993, 90329, 89363, 200835, 97420, 205619`이며 checkpoint-local index다.

#### R01 coupling

| 범위 | pair | zero | median `c_ij` | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| sampled union 전체 | 1,891 | 1,598 | 0 | 0.6336 | 0.9173 | 0.99995 |
| group 내부 unique | 223 | 22 | 0.8167 | 0.9908 | 0.9949 | 0.99995 |

| 요약 지표 | 값 |
|---|---:|
| capture ratio | 0.989476 |
| union normalized off-diagonal `epsilon_G` | 9.36249 |
| damped condition number | 599,554.25 |
| minimum / maximum eigenvalue | -1.41e-5 / 597.183 |
| symmetry relative error | 0 |

최소 eigenvalue의 작은 음수는 최대 eigenvalue 대비 약 2.4e-8이며 float32 sampled `J^T J`의 수치 오차 범위로 해석한다. condition number와 `epsilon_G`가 커서 향후 group solve에서는 damping과 step acceptance가 중요하다.

#### R01 비용과 상태

| phase | 시간 |
|---|---:|
| contribution probe | 35.12 ms |
| anchor selection | 6.82 ms |
| group construction | 11.82 ms |
| exact Jacobian + curvature | 411.10 ms |
| peak allocated / reserved | 321,540,096 / 350,224,384 bytes |
| failure record | 0 byte |

#### R01 해석

- contributor-conditioned group 내부에는 큰 normalized coupling이 집중됐으며 sampled union의 off-diagonal energy 98.95%가 group 내부 pair에 포함됐다.
- sampled union 전체 pair의 84.5%는 coupling 0이고 median도 0이므로 현재 표본에서는 group 밖 관계가 대체로 희소하다.
- 이는 visible-overlap kNN의 국소성 가설을 지지하는 초기 증거지만, 단일 view·seed와 32 sampled pixel에 조건화된 결과다.
- contributor threshold와 member filter가 0 pair를 줄이는 selection이므로 전체 visible Gaussian의 unconditional coupling 분포로 일반화하지 않는다.
- 공식 R01만 완료했으며 R02 one-step과 R03 rollout은 실행하지 않았다.
