---
record_id: W-20260728-002
type: work
method: RHO_GS
created_at: 2026-07-28T18:42:29+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/work/2026/W-20260728-002-rho-gs-lego-tile-stats.md
retrospective: false
---

# W-20260728-002 — RHO_GS Lego tile Gaussian 통계

## 요약

| 항목 | 내용 |
|---|---|
| 목적 | 실제 학습 렌더링에서 tile별 등록 Gaussian 후보 수와 pixel traversal 처리량의 분포 및 학습 중 변화를 수집 |
| 가설 | 장면과 view에 따라 tile별 후보 수가 불균일하며, alpha blending 조기 종료로 등록 후보와 실제 순회량 사이에 차이가 발생 |
| Dataset | `dataset/nerf_synthetic/lego` |
| 상태 | `completed` |
| 관련 Develop | [[src/Methods/RHO_GS/docs/develop/2026/D-20260728-002-tile-gaussian-stats\|D-20260728-002]] |
| output | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29` |
| artifact 상태 | `local-only` |

## 성공 조건

- [x] 30,000 iteration 학습 정상 종료
- [x] 실제 sampled training view를 1,000 iteration 간격으로 30회 수집
- [x] 각 시점의 raw NPZ, summary CSV, heatmap 생성
- [x] NPZ shape, iteration 간격, 범위 및 instance 합계 불변식 검증
- [x] final checkpoint와 test 200-view 평가 생성

## 재현 정보

### 저장소

실행 당시 RHO_GS는 `D-20260728-001` namespace 분리와 `D-20260728-002` 구현을 포함한 dirty worktree였다. commit만으로 현재 결과를 재현할 수 없다.

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
| 수집 view | 해당 iteration에 실제 sampled training view |
| 수집 phase | `pre_optimizer` forward |
| config | `src/Methods/RHO_GS/rho_gs_lego_tile_stats.yaml` |

## Runs

### R01

#### Actual command

```bash
python scripts/train.py -c src/Methods/RHO_GS/rho_gs_lego_tile_stats.yaml
```

#### Reproduction command

dirty source 상태를 동일하게 복원한 뒤 저장된 설정으로 실행한다.

```bash
python scripts/train.py \
  -c output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/training_config.yaml
```

## Events

### 2026-07-28T18:42:29+09:00 — started

- 상태: `running`
- `rho_gs_lego_tile_stats_w20260728_002_r01` model name으로 30K 학습을 시작했다.
- GUI와 W&B는 비활성화하고 timing, VRAM, final checkpoint, test rendering을 활성화했다.

### 2026-07-28 — tile-stats-collected

- 상태: `running`
- iteration 1,000부터 30,000까지 1,000 간격의 실제 sampled view 30개를 수집했다.
- 각 수집 시점에 `registered`, `processed_mean`, `processed_max` 50×50 배열과 JSON metadata를 NPZ로 저장했다.
- summary CSV는 각 배열의 min, mean, median, p95, max 및 평균 처리 비율을 기록했다.

### 2026-07-28 — training-completed

- 상태: `evaluation-pending`
- training loop: 약 5분 38초, `training_iteration` timing 337초, 11.24 ms/iteration
- peak VRAM: 0.47 GiB allocated, 0.55 GiB reserved
- 통계 수집 후 pre-optimizer Gaussian 수: 314,451
- post-training 정리 후 최종 Gaussian 수: 313,817
- 30개 heatmap 생성 시간: 약 8.74초
- final checkpoint 저장 시간: 약 36.40초

### 2026-07-28 — evaluation-completed

- 상태: `completed`
- test RGB와 GT를 각각 200장 생성했다.
- 저장 후 재로딩한 8-bit PNG를 기준으로 PSNR, SSIM, LPIPS-VGG를 계산했다.
- 결과: PSNR 36.05 dB, SSIM 0.983, LPIPS-VGG 0.020.

### 2026-07-28 — artifact-verified

- 30개 NPZ와 30개 heatmap, test RGB 200장, GT 200장을 확인했다.
- NPZ iteration은 1,000부터 30,000까지 1,000 간격이며 모두 50×50 shape이다.
- 모든 원소에서 `0 <= processed_mean <= processed_max <= registered`를 확인했다.
- 모든 NPZ에서 `registered.sum()`과 metadata의 `n_instances`가 일치했다.
- checkpoint, config, summary 및 metrics의 SHA-256을 기록했다.

## 결과

### 학습 및 평가

| 항목 | 값 |
|---|---:|
| 학습 iteration | 30,000 |
| 학습 loop | 약 5분 38초 |
| 최종 Gaussian | 313,817 |
| peak allocated VRAM | 0.47 GiB |
| peak reserved VRAM | 0.55 GiB |
| PSNR mean / median | 36.05 / 36.82 dB |
| SSIM mean / median | 0.983 / 0.984 |
| LPIPS-VGG mean / median | 0.020 / 0.019 |

### Tile 통계 표본

`registered`는 tile range에 등록된 후보 수이고, `processed`는 valid pixel이 transmittance 조기 종료 전까지 exponent/alpha 평가를 수행한 후보 수이다.

| iteration | view | Gaussian | registered mean / max | processed mean mean / max max | processing ratio mean |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 51 | 17,402 | 128.2 / 1,068 | 104.2 / 941 | 0.436 |
| 10,000 | 48 | 292,560 | 987.3 / 7,061 | 788.4 / 6,646 | 0.548 |
| 20,000 | 74 | 314,451 | 925.1 / 7,725 | 656.3 / 7,638 | 0.615 |
| 30,000 | 11 | 314,451 | 1,002.9 / 8,123 | 719.1 / 8,123 | 0.746 |

| 집계 | 값 |
|---|---:|
| registered mean 범위 | 128.2–1,078.2 |
| registered max 범위 | 1,068–12,186 |
| processed mean의 tile 평균 범위 | 104.2–814.2 |
| processed max의 전체 최대 범위 | 941–12,126 |
| processing ratio mean 범위 | 0.396–0.793 |
| 초기 10개 표본 processing ratio 평균 | 0.4665 |
| 마지막 10개 표본 processing ratio 평균 | 0.7127 |

## 초기 해석

- tile별 등록 후보와 처리량의 분산이 크며, 일부 tile의 후보 수가 10,000개를 넘는다.
- 대부분 시점에서 등록 후보 수보다 실제 순회 수가 작아 alpha blending 조기 종료의 연산 절감이 관측된다.
- 후기 표본의 평균 처리 비율이 초기 표본보다 높지만, 매 시점의 view가 다르고 densification으로 Gaussian 수도 변하므로 학습 진행에 따른 인과적 증가로 결론 내릴 수 없다.
- 현재 산출물은 tile aggregate만 포함하므로 높은 count를 유발한 Gaussian의 identity나 attribute별 관계는 판별할 수 없다.
- 다음 시간축 분석에서는 고정 view 재렌더링 또는 동일 view별 그룹화를 추가해야 view 선택 효과와 학습 효과를 분리할 수 있다.

## Artifact

| 종류 | 경로 | SHA-256/개수 | 상태 |
|---|---|---|---|
| checkpoint | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/checkpoints/final.pt` | `3abbe792f7c71c75f7dbf34e1fd02564902f9d0bb0ebcddbf83d810b0e30cc28` | local-only |
| config | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/training_config.yaml` | `94455a36f3ce442338214a429fb18bf9a62db02dd678827581059a1965096102` | local-only |
| summary | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/tile_gaussian_stats/summary.csv` | `5407fdc597e4588b390cab2d288d5e41a33e5196590046877df43d2ec46c93cc` | local-only |
| raw tile stats | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/tile_gaussian_stats/*.npz` | 30개 | local-only |
| heatmap | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/tile_gaussian_stats/*.png` | 30개 | local-only |
| metrics | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/test_30000/metrics_8bit.txt` | `4ffe7252c7e3d58f23ae528a11f813ec34a37990065b893bea65962bc1ec4ab2` | local-only |
| test render | `output/RHO_GS/rho_gs_lego_tile_stats_w20260728_002_r01_2026-07-28-18-42-29/test_30000` | RGB 200장, GT 200장 | local-only |
| raw console log | 미저장 | 미수집 | missing |

## 한계 및 후속 작업

- RHO_GS dirty worktree를 commit하기 전에는 commit ID만으로 재현할 수 없다.
- 실제 sampled view가 시점마다 달라 현재 summary만으로 순수한 시간축 변화를 비교할 수 없다.
- `processed`는 실제 color/alpha 기여 Gaussian 수가 아니라 조기 종료 전 평가 후보 수이다.
- 별도 통계 커널이 수집 iteration의 sorted candidate를 다시 순회하므로 측정 오버헤드가 포함된다.
- Gaussian identity, attribute, tile 간 관계 및 Hessian 값은 아직 수집하지 않는다.
- `tile_gaussian_stats/.~lock.summary.csv#` 외부 편집기 잠금 파일이 발견됐으며 사용자 파일 가능성을 고려해 보존했다.

### 2026-07-30 — implementation-retired

- 상태: `completed`
- 이 Work의 학습·평가 결과와 local artifact는 완료된 실험 증거로 유지한다.
- 결과를 생성한 tile 통계 구현은 coupling 실험 환경을 단순화하기 위해 commit 전에 현재 working tree에서 제거됐다.
- 동일 기능의 재현에는 당시 구현 복원 또는 coupling framework 기반 재구현이 필요하며, 현재 RHO_GS HEAD에서 이 문서의 실행 명령을 그대로 사용할 수 없다.
