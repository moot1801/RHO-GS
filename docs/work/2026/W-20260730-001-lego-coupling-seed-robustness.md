---
record_id: W-20260730-001
type: work
method: RHO_GS
created_at: 2026-07-30T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/work/2026/W-20260730-001-lego-coupling-seed-robustness.md
retrospective: false
---

# W-20260730-001 — Lego Coupling Seed Robustness

## 요약

| 항목 | 내용 |
|---|---|
| 목적 | contributor-aware visible-overlap K=8 coupling 결과의 sampling seed 안정성 확인 |
| 비교 기준 | [[src/Methods/RHO_GS/docs/work/2026/W-20260729-002-lego-position-coupling-reference\|W-20260729-002/R01 seed 0]] |
| 추가 Run | R01 seed 1, R02 seed 2 |
| 상태 | `completed` |
| Dataset / checkpoint | Lego late 30K model-only `final.pt` |
| checkpoint SHA-256 | `5c4982037959f05e5dcf8069525fb1c5f4901ee30c8cdf8454887bf09875f320` |
| RHO_GS commit | `6f8b295cd603dd07465cda553ea5b96ceb6f75a7` |

## 공통 조건

| 항목 | 값 |
|---|---:|
| primary view | train 51 |
| image scale | 0.25× |
| tile / pixel | top 2 / tile당 16 |
| residual scalar | 96 |
| anchor | random contributor 8개 |
| contribution probe | seed별 Rademacher 4회 |
| grouping | visible-overlap 3D kNN |
| group size / candidate pool | 8 / 32 |
| member filter | contributor-only |
| damping | 0.001 |

Top-level seed와 probe seed를 같은 값으로 설정했다. 따라서 각 반복은 pixel sampling, randomized contribution probe, anchor selection을 함께 변경하는 전체 sampling robustness 실험이다.

## Runs

### R01 — seed 1

```bash
conda run -n nerficg env PYTHONPATH=src \
  python -m Methods.RHO_GS.experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  experiment_name=w20260730_001_r01_seed1 \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  seed=1 \
  anchor_selection=random_contributor \
  anchor_selection.sampled_anchor_count=8 \
  anchor_selection.maximum_anchors=8 \
  anchor_selection.probe_count=4 \
  anchor_selection.probe_seed=1 \
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

- output: `output/RHO_GS/experiments/w20260730_001_r01_seed1-20260730T051647073037Z-1d451258`
- resolved seed / probe seed: 1 / 1
- candidate hash: `6dd5e1eaee671d4b`
- 상태: `completed`, failure 0 byte

### R02 — seed 2

```bash
conda run -n nerficg env PYTHONPATH=src \
  python -m Methods.RHO_GS.experiments.runners.analyze_checkpoint \
  experiment=lego_position_reference \
  experiment_name=w20260730_001_r02_seed2 \
  checkpoint=output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt \
  seed=2 \
  anchor_selection=random_contributor \
  anchor_selection.sampled_anchor_count=8 \
  anchor_selection.maximum_anchors=8 \
  anchor_selection.probe_count=4 \
  anchor_selection.probe_seed=2 \
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

- output: `output/RHO_GS/experiments/w20260730_001_r02_seed2-20260730T051719280183Z-615073d1`
- resolved seed / probe seed: 2 / 2
- candidate hash: `e2d2c8f8bda4037d`
- 상태: `completed`, failure 0 byte

## Seed별 결과

Seed 0은 W-20260729-002/R01 artifact에서 동일 계산식으로 재집계했다. Seed 0 실행 commit은 `a6d60bd`, seed 1·2는 `6f8b295`이며 두 commit 사이 coupling 코드는 동일하고 문서 commit만 추가됐다. 세 실행 모두 method dirty는 `false`다.

| 지표 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| contributor 후보 | 1,563 | 1,352 | 1,328 |
| sampled Gaussian union | 62 | 64 | 64 |
| zero group | 0/8 | 0/8 | 0/8 |
| valid pair ratio | 90.18% | 92.86% | 96.43% |
| union zero-pair ratio | 84.51% | 84.57% | 83.73% |
| group 내부 unique pair | 223 | 224 | 224 |
| group 내부 zero-pair ratio | 9.87% | 7.14% | 3.57% |
| group 내부 median `c_ij` | 0.8167 | 0.5533 | 0.8196 |
| group 내부 p95 `c_ij` | 0.9949 | 0.9967 | 0.9943 |
| group 내부 max `c_ij` | 0.99995 | 0.99987 | 0.99997 |
| group 내부 raw median | 0.1041 | 0.0361 | 0.0838 |
| capture ratio | 0.98948 | 0.98372 | 0.99989 |
| `epsilon_G` | 9.362 | 8.660 | 11.586 |
| condition number | 599,554 | 97,681 | 533,745 |

## 3-seed 집계

표준편차는 세 표본의 sample standard deviation이다.

| 지표 | 평균 ± 표준편차 | 범위 |
|---|---:|---:|
| contributor 후보 | 1,414.3 ± 129.3 | 1,328–1,563 |
| sampled Gaussian union | 63.33 ± 1.15 | 62–64 |
| valid pair ratio | 93.15% ± 3.14%p | 90.18–96.43% |
| union zero-pair ratio | 84.27% ± 0.47%p | 83.73–84.57% |
| group 내부 zero-pair ratio | 6.86% ± 3.16%p | 3.57–9.87% |
| group 내부 median `c_ij` | 0.7299 ± 0.1529 | 0.5533–0.8196 |
| group 내부 p95 `c_ij` | 0.99531 ± 0.00128 | 0.99430–0.99675 |
| capture ratio | 0.99103 ± 0.00820 | 0.98372–0.99989 |
| `epsilon_G` | 9.870 ± 1.527 | 8.660–11.586 |
| condition number | 410,327 ± 272,752 | 97,681–599,554 |

## 비용 안정성

| 지표 | 평균 ± 표준편차 | 범위 |
|---|---:|---:|
| contribution probe | 34.07 ± 1.23 ms | 32.72–35.12 ms |
| group construction | 11.95 ± 0.21 ms | 11.82–12.19 ms |
| exact Jacobian + curvature | 405.72 ± 4.73 ms | 402.23–411.10 ms |
| peak allocated GPU memory | 321,540,096 ± 0 bytes | 동일 |

## 해석

- 세 seed 모두 zero group이 없고 valid pair ratio가 90% 이상이어서 contributor-aware grouping의 관측 가능성은 안정적으로 유지됐다.
- union 전체 pair의 약 84%가 반복해서 0인 반면 group 내부 zero 비율은 3.6–9.9%였다. coupling의 희소성과 국소 집중 패턴이 seed 변경에도 유지됐다.
- capture ratio는 모든 seed에서 98.37% 이상이고 평균 99.10%였다. visible-overlap kNN이 현재 sampled union의 주요 raw off-diagonal energy를 일관되게 포착했다.
- group 내부 median coupling은 seed 1에서 0.553으로 낮아져 중간 강도 분포에는 sampling 변동이 존재한다. 반면 p95는 세 seed 모두 약 0.995로 강한 tail은 안정적이었다.
- condition number는 약 6.1배 범위로 크게 달라졌다. solve 안정성은 coupling 존재 여부보다 pixel/anchor sample에 더 민감할 수 있으므로 damping과 acceptance 비교가 필요하다.
- probe, grouping, Jacobian 시간과 peak memory는 seed 간 안정적이었다.

## 한계와 후속 작업

- 표본 수 3은 초기 안정성 확인용이며 신뢰구간이나 전체 분포 결론에 충분하지 않다.
- pixel, probe, anchor seed를 함께 바꿨으므로 관측된 변동을 각 random source로 분해할 수 없다.
- 동일 view와 top tile 2개에 한정되며 다른 camera/view locality는 아직 미확인이다.
- contributor threshold와 member filter에 조건화된 결과이므로 geometry-visible 전체 모집단의 unconditional coupling이 아니다.
- 다음 실험은 같은 seed별 pixel 조건에서 `random_eligible`과 `random_contributor`를 비교해 zero-group 감소와 selection bias를 분리하는 것이 적합하다.

## Events

### 2026-07-30 — completed

- 상태: `completed`
- R01 seed 1과 R02 seed 2 analyzer가 failure 없이 완료됐다.
- W-20260729-002/R01 seed 0과 동일한 방식으로 union 및 group 내부 pair 통계를 재집계했다.
- 세 seed의 핵심 locality와 capture 패턴은 안정적이지만 condition number와 group median coupling 변동을 확인했다.
