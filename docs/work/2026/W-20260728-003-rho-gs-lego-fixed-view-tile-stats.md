---
record_id: W-20260728-003
type: work
method: RHO_GS
created_at: 2026-07-28T19:21:37+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/work/2026/W-20260728-003-rho-gs-lego-fixed-view-tile-stats.md
retrospective: false
---

# W-20260728-003 — RHO_GS Lego 고정 뷰 Tile 통계

## 요약

| 항목 | 내용 |
|---|---|
| 목적 | 동일 training view에서 tile별 등록·처리 Gaussian 후보 수의 학습 시간축 변화 측정 |
| 가설 | view를 고정하면 densification 이후 tile 부하의 공간 분포와 안정화 과정을 직접 비교할 수 있음 |
| Dataset | `dataset/nerf_synthetic/lego` |
| 상태 | `completed` |
| 관련 Develop | [[src/Methods/RHO_GS/docs/develop/2026/D-20260728-003-fixed-view-tile-stats\|D-20260728-003]] |
| 비교 Work | [[src/Methods/RHO_GS/docs/work/2026/W-20260728-002-rho-gs-lego-tile-stats\|W-20260728-002]] |
| output | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37` |
| artifact 상태 | `local-only` |

## 성공 조건

- [x] 30,000 iteration 학습 정상 종료
- [x] train view 51을 1,000 iteration 간격으로 30회 고정 수집
- [x] raw NPZ, summary CSV, snapshot heatmap 생성
- [x] temporal plot과 직전·최초 대비 difference heatmap 생성
- [x] NPZ metadata, shape, iteration, 값 범위 및 instance 합계 불변식 검증
- [x] final checkpoint와 test 200-view 평가 생성

## 재현 정보

### 저장소

실행 당시 RHO_GS는 namespace 분리와 D-20260728-002·003 구현을 포함한 dirty worktree였다. commit만으로 현재 결과를 재현할 수 없다.

| 저장소 | commit | branch | dirty | remote |
|---|---|---|---|---|
| NeRFICG | `c7437127af1681f565a57fcd9d7819fde1adc0a7` | `main` | Method 및 문서 변경 존재 | `https://github.com/nerficg-project/nerficg.git` |
| RHO_GS | `751ec974c7ab2389c9bc789866ebec2fd6d8d54a` | `rho-gs/testbed` | rename·수정·신규 파일 존재 | `https://github.com/moot1801/RHO-GS` |

### 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti, 12,282 MiB |
| Driver | `580.173.02` |
| Python | `3.11.15` |
| PyTorch | `2.8.0+cu128` |
| CUDA runtime | `12.8` |
| CUDA toolkit | 미수집 |
| cuDNN | 미수집 |
| Conda environment | 미수집 |

### Dataset과 설정

| 항목 | 값 |
|---|---|
| Dataset path | `dataset/nerf_synthetic/lego` |
| split/이미지 수 | NeRF Synthetic 기본 train, test 200-view 평가 |
| image scale | `null` — 원본 800×800 |
| random seed | `0` |
| 초기 random points | `100,000` |
| iterations | `30,000` |
| tile | 16×16 pixel, 50×50 tile grid |
| 수집 간격 | 1,000 iteration |
| 수집 view | fixed train view 51 |
| 수집 phase | random training sample 선택 및 optimizer step 전 `pre_optimizer` 재렌더링 |
| config | `src/Methods/RHO_GS/rho_gs_lego_fixed_view_tile_stats.yaml` |

## Runs

### R01

#### Actual command

```bash
python scripts/train.py -c src/Methods/RHO_GS/rho_gs_lego_fixed_view_tile_stats.yaml
```

#### Reproduction command

dirty source 상태를 동일하게 복원한 뒤 저장된 설정으로 실행한다.

```bash
python scripts/train.py \
  -c output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/training_config.yaml
```

### 사전 smoke 및 회귀 확인

| 구분 | output | 결과 |
|---|---|---|
| 2K fixed stats | `rho_gs_lego_fixed_view_tile_stats_smoke_on_2026-07-28-19-08-51` | view 51 snapshot 2개와 시각화 생성 |
| 잘못된 off | `rho_gs_lego_fixed_view_tile_stats_smoke_off_2026-07-28-19-10-07` | 소문자 `false`가 문자열 참값이어서 stats가 계속 활성화됨; 비교에서 제외 |
| 2-iteration on | `rho_gs_fixed_view_equiv_on_2026-07-28-19-12-31` | snapshot 2개 생성 |
| 2-iteration off | `rho_gs_fixed_view_equiv_off_2026-07-28-19-12-50` | `ACTIVE=False`, stats 디렉토리 미생성 |
| off 반복 | `rho_gs_fixed_view_equiv_off_repeat_2026-07-28-19-20-04` | CUDA 반복 실행 자연 변동 확인 |

## Events

### 2026-07-28T19:21:37+09:00 — started

- 상태: `running`
- fixed train view 51, 1K interval로 30K 학습을 시작했다.
- raw·snapshot·temporal·difference 시각화와 final test 평가를 활성화했다.

### 2026-07-28 — tile-stats-collected

- iteration 1,000부터 30,000까지 1,000 간격으로 동일 train view 51을 30회 수집했다.
- 모든 NPZ metadata가 `view_mode=fixed`, `view_id=51`, `capture_phase=pre_optimizer`임을 확인했다.

### 2026-07-28 — training-completed

- 상태: `evaluation-pending`
- framework time: 약 339.36초, `training_iteration` 약 5분 35초, 11.18 ms/iteration
- peak VRAM: 0.47 GiB allocated, 0.63 GiB reserved
- 30K snapshot의 pre-cleanup Gaussian 수: 316,869
- `training_cleanup` 후 최종 Gaussian 수: 316,096
- tile stats 시각화 생성 시간: 약 17.75초
- final checkpoint 저장 시간: 약 35.00초

### 2026-07-28 — evaluation-completed

- 상태: `completed`
- test RGB와 GT를 각각 200장 생성했다.
- 저장 후 재로딩한 8-bit PNG 기준 결과는 PSNR 36.08 dB, SSIM 0.983, LPIPS-VGG 0.020이다.

### 2026-07-28 — artifact-verified

- NPZ 30개, snapshot heatmap 30개, temporal plot 1개, difference heatmap 29개를 확인했다.
- test RGB/GT 각 200장을 확인했다.
- 모든 NPZ에서 `0 <= processed_mean <= processed_max <= registered`를 확인했다.
- 모든 NPZ에서 `registered.sum()`과 metadata의 `n_instances`가 일치했다.
- temporal plot과 2K·15K·30K difference heatmap을 시각적으로 확인했다.
- checkpoint, config, summary 및 metrics의 SHA-256을 기록했다.

## 결과

### 학습 및 평가

| 항목 | 값 |
|---|---:|
| 학습 iteration | 30,000 |
| framework time | 약 339.36초 |
| training iteration 평균 | 11.18 ms |
| 최종 Gaussian | 316,096 |
| peak allocated VRAM | 0.47 GiB |
| peak reserved VRAM | 0.63 GiB |
| PSNR mean / median | 36.08 / 36.85 dB |
| SSIM mean / median | 0.983 / 0.984 |
| LPIPS-VGG mean / median | 0.020 / 0.019 |

### 고정 뷰 Tile 통계

`registered`는 tile range에 등록된 후보 수이며 `processed`는 valid pixel이 alpha 조기 종료 전까지 평가한 후보 수이다.

| iteration | pre-cleanup Gaussian | registered mean / max | processed mean mean / max max | processing ratio mean |
|---:|---:|---:|---:|---:|
| 1,000 | 17,473 | 129.1 / 1,086 | 105.1 / 925 | 0.423 |
| 5,000 | 211,918 | 552.1 / 5,051 | 367.3 / 5,051 | 0.407 |
| 10,000 | 294,038 | 799.6 / 7,424 | 563.5 / 6,907 | 0.441 |
| 15,000 | 316,869 | 883.1 / 7,862 | 601.8 / 7,617 | 0.481 |
| 20,000 | 316,869 | 864.0 / 7,681 | 566.9 / 7,435 | 0.556 |
| 25,000 | 316,869 | 864.1 / 7,624 | 555.3 / 7,332 | 0.595 |
| 30,000 | 316,869 | 869.9 / 7,580 | 552.0 / 7,364 | 0.638 |

| 정점 | iteration | 값 |
|---|---:|---:|
| registered mean | 13,000 | 884.628 |
| registered max | 13,000 | 7,903 |
| processed mean의 tile 평균 | 13,000 | 628.084 |
| processed max의 전체 최대 | 15,000 | 7,617 |

## 초기 해석

- 동일 view에서 Gaussian topology와 등록·처리량은 densification 구간 동안 빠르게 증가하고 13K~15K에서 정점에 도달한다.
- 15K 이후 pre-cleanup Gaussian 수는 고정되며 registered 평균도 약 864~870으로 안정화된다.
- 같은 구간에서 processed 평균은 601.8에서 552.0으로 감소하고 평균 처리 비율은 0.481에서 0.638로 증가한다. 서로 다른 집계량이므로 ratio 증가를 단순한 총 처리량 증가로 해석하면 안 된다.
- 직전 snapshot 대비 공간 변화는 후기 iteration에서 매우 작지만, 최초 1K 대비 증가는 Lego 실루엣과 경계 tile을 따라 유지된다.
- 고정 뷰 결과는 candidate 부하가 image plane의 국소 영역과 강하게 대응함을 보여준다. 그러나 이는 Hessian 비대각 성분 또는 Gaussian 간 gradient 관계가 국소적이라는 직접 증거는 아니다.
- W-20260728-002와 달리 view 선택 효과는 제거됐지만 densification·pruning과 파라미터 변화는 여전히 시간축에 함께 포함된다.

## Artifact

| 종류 | 경로 | SHA-256/개수 | 상태 |
|---|---|---|---|
| checkpoint | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/checkpoints/final.pt` | `5c4982037959f05e5dcf8069525fb1c5f4901ee30c8cdf8454887bf09875f320` | local-only |
| config | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/training_config.yaml` | `4566359ee0695e36e534fa96e80f73b936f5b2195d2ce0eeac68608aff24b437` | local-only |
| summary | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/tile_gaussian_stats/summary.csv` | `7a2ab884d225cc9a0865b5e0c89da5b120c8215563b92b8c069c4859263ff0a9` | local-only |
| raw tile stats | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/tile_gaussian_stats/*.npz` | 30개 | local-only |
| snapshot heatmap | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/tile_gaussian_stats/iteration_*.png` | 30개 | local-only |
| temporal plot | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/tile_gaussian_stats/temporal_*.png` | 1개 | local-only |
| difference heatmap | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/tile_gaussian_stats/difference_*.png` | 29개 | local-only |
| metrics | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/test_30000/metrics_8bit.txt` | `57eb998f5bbf13b16d25101e32ce2c6314f54e2c3e83d7918af6a6b10a36a5e5` | local-only |
| test render | `output/RHO_GS/rho_gs_lego_fixed_view_tile_stats_w20260728_003_r01_2026-07-28-19-21-37/test_30000` | RGB 200장, GT 200장 | local-only |
| raw console log | 미저장 | 미수집 | missing |

## 한계 및 후속 작업

- RHO_GS dirty worktree를 commit하기 전에는 commit ID만으로 재현할 수 없다.
- fixed render가 model·densification state를 직접 갱신하지는 않지만 별도 CUDA 실행에 따른 미세한 trajectory 변화 가능성은 배제하지 않는다.
- 현재 통계는 tile aggregate이므로 높은 count를 유발한 Gaussian identity와 attribute를 알 수 없다.
- `processed`는 실제 backward gradient 또는 color 기여 Gaussian 수가 아니다.
- Hessian 실험에서는 Gaussian topology 변화와 identity를 통제하고 tile 후보 집합과 attribute pair 정의를 별도로 저장해야 한다.

### 2026-07-30 — implementation-retired

- 상태: `completed`
- 이 Work의 학습·평가 결과와 local artifact는 완료된 실험 증거로 유지한다.
- 결과를 생성한 fixed-view tile 통계 구현은 coupling 실험 환경을 단순화하기 위해 commit 전에 현재 working tree에서 제거됐다.
- 현재 RHO_GS HEAD에는 전용 fixed-view 통계 runner가 없으므로 당시 명령을 재실행하려면 구현 복원 또는 coupling evaluator 기반 재구현이 필요하다.
