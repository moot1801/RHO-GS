---
record_id: D-20260728-003
type: develop
method: RHO_GS
created_at: 2026-07-28T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/develop/2026/D-20260728-003-fixed-view-tile-stats.md
retrospective: false
---

# D-20260728-003 — 고정 뷰 Tile 통계와 시간축 시각화

## 요약

| 항목 | 내용 |
|---|---|
| 구현 목적 | 동일 training view의 tile Gaussian 통계를 반복 측정해 view 선택 효과와 학습 시간축 변화를 분리 |
| 초기 의도 | W-20260728-002의 sampled-view 교란을 제거하고 tile 부하의 공간·시간 변화 관찰 |
| 범위 | sampled/fixed view 모드, 고정 뷰 검증·캐시, 비학습 재렌더링, temporal plot, 직전·최초 대비 difference heatmap, Lego 실험 config |
| 제외 범위 | Hessian 계산, Gaussian identity·후보 목록 저장, attribute별 통계, optimizer 변경 |
| 상태 | 구현 및 검증 완료, commit 미생성 |
| 관련 Work | [[src/Methods/RHO_GS/docs/work/2026/W-20260728-003-rho-gs-lego-fixed-view-tile-stats\|W-20260728-003]] |

## 기준 저장소 상태

| 저장소 | commit | branch | dirty | remote |
|---|---|---|---|---|
| NeRFICG | `c7437127af1681f565a57fcd9d7819fde1adc0a7` | `main` | Method 및 문서 변경 존재 | `https://github.com/nerficg-project/nerficg.git` |
| RHO_GS | `751ec974c7ab2389c9bc789866ebec2fd6d8d54a` | `rho-gs/testbed` | namespace 분리와 tile stats 변경 존재 | `https://github.com/moot1801/RHO-GS` |

## 설계

### View 모드

| 설정 | 기본값 | 의미 |
|---|---|---|
| `TILE_GAUSSIAN_STATS.VIEW_MODE` | `sampled` | `sampled`는 기존 학습 forward, `fixed`는 지정 뷰 별도 재렌더링 |
| `TILE_GAUSSIAN_STATS.FIXED_VIEW_IDS` | `[]` | fixed 모드에서 반복 측정할 training sample ID 목록 |
| `TILE_GAUSSIAN_STATS.SAVE_TEMPORAL_PLOTS` | `False` | view별 시간축 plot과 difference heatmap 생성 |

- fixed view ID는 sampler 생성 시 training dataset에서 검증하고 캐시한다.
- 수집은 해당 iteration의 optimizer step과 random training sample 선택 전에 수행한다.
- `torch.no_grad()`와 `update_densification_info=False`를 사용해 autograd 및 densification 정보를 갱신하지 않는다.
- 배경색은 해당 camera의 기본 배경을 사용하므로 random background 설정과 분리된다.
- raw NPZ metadata에 `view_mode=fixed`, `view_id`, `capture_phase=pre_optimizer`를 기록한다.

### 시각화

| 산출물 | 내용 |
|---|---|
| `temporal_view_<id>.png` | Gaussian·tile instance topology, registered mean/p95/max, processed mean/p95/max, 처리 비율 mean/p95 |
| `difference_iteration_<iter>_view_<id>.png` | 직전 표본 및 최초 표본 대비 registered·processed mean 변화 |

- 시간축·difference 시각화에는 모든 snapshot raw array가 필요하므로 해당 기능 사용 시 raw 저장을 강제한다.
- 동일 metric의 difference map은 공통 대칭 색 범위를 사용해 iteration 간 강도를 비교할 수 있다.

## 위험 및 정책

| 위험 | 영향 | 대응 |
|---|---|---|
| 별도 fixed render | 수집 iteration 실행 시간과 CUDA allocator/kernel 실행 순서 변화 | interval 제한, `no_grad`, densification update 비활성화, 회귀 결과 별도 기록 |
| CUDA 반복 실행 비결정성 | 별도 프로세스 checkpoint의 strict equality 비교 실패 | stats-off 반복 실행을 대조군으로 두고 차이 규모를 함께 비교 |
| 종료 시 Gaussian 정리 | 마지막 snapshot Gaussian 수와 final checkpoint 수가 다름 | snapshot은 `pre_optimizer`, 최종 수는 `training_cleanup` 이후임을 구분 |
| CLI boolean 문자열 | 소문자 `false`가 문자열 참값으로 동작 | CLI override에는 Python 표기 `False`/`True`만 사용 |
| difference 공통 색 범위 | 후기의 작은 직전 변화가 육안상 거의 보이지 않을 수 있음 | raw NPZ/summary를 정량 분석의 source of truth로 사용 |

## Events

### 2026-07-28 — planned

- 상태: `planned`
- W-20260728-002의 서로 다른 sampled view 교란을 제거하기 위해 train view 51을 고정하기로 했다.
- 기존 sampled 모드는 호환성을 위해 기본값으로 유지했다.

### 2026-07-28 — fixed-view-implemented

- 상태: `implemented-uncommitted`
- Trainer에 `sampled`와 `fixed` view 모드, fixed ID 검증·캐시 및 별도 수집 경로를 추가했다.
- fixed render는 training sampler를 소비하지 않고 densification 정보를 갱신하지 않도록 구현했다.
- recorder summary와 metadata에 view mode를 추가했다.

### 2026-07-28 — temporal-visualization-implemented

- Gaussian topology와 registered·processed·ratio 시간축 plot을 추가했다.
- 직전 snapshot 및 최초 snapshot 대비 registered·processed difference heatmap을 추가했다.

### 2026-07-28 — smoke-verified

- 2K fixed-view smoke에서 iteration 1K와 2K가 모두 view 51로 저장되는 것을 확인했다.
- NPZ 2개, heatmap 2개, temporal plot 1개, difference heatmap 1개를 확인했다.
- temporal plot의 축·범례와 difference map의 공간 정렬을 시각적으로 확인했다.

### 2026-07-28 — cli-boolean-pitfall

- `TRAINING.BACKUP.RENDER_TESTSET=false`와 `TILE_GAUSSIAN_STATS.ACTIVE=false`는 boolean이 아니라 문자열로 해석돼 기능이 계속 활성화됐다.
- `src/Framework.py`가 `ast.literal_eval` 실패 시 값을 문자열로 유지하는 것이 원인이다.
- 이후 검증부터 `False`와 `True`만 사용했다.

### 2026-07-28 — regression-investigated

- 올바른 `False`를 사용한 2-iteration stats on/off 비교의 checkpoint 최대 절대 차이는 `6.768008e-04`였다.
- stats-off를 새 프로세스에서 반복한 두 checkpoint 간 최대 절대 차이도 `5.881060e-04`였다.
- on/off 차이가 stats-off 반복 실행의 자연 변동과 같은 규모이므로 fixed render가 모델 상태를 직접 변경한다는 증거는 확인되지 않았다.
- 단, 별도 CUDA render가 allocator 또는 실행 순서를 미세하게 바꿀 가능성까지 배제하지는 않는다.

### 2026-07-28 — full-run-verified

- [[src/Methods/RHO_GS/docs/work/2026/W-20260728-003-rho-gs-lego-fixed-view-tile-stats\|W-20260728-003/R01]] 30K 학습과 test 평가가 정상 종료됐다.
- NPZ 30개 모두 fixed train view 51, 1K 간격, 50×50 shape 및 수치 불변식을 만족했다.
- heatmap 30개, temporal plot 1개, difference heatmap 29개를 생성하고 대표 시각화를 확인했다.

## 변경 파일

| 경로 | 변경 목적 |
|---|---|
| `Trainer.py` | view 모드 설정, fixed ID 검증·캐시, 비학습 재렌더링 수집 |
| `TileStatsRecorder.py` | view mode metadata, temporal plot, difference heatmap |
| `rho_gs_lego_fixed_view_tile_stats.yaml` | Lego 30K fixed view 51 실험 설정 |

## 검증

| 검증 | 결과 | 근거 |
|---|---|---|
| Python syntax | 통과 | `Trainer.py`, `TileStatsRecorder.py` `py_compile` |
| whitespace | 통과 | `git diff --check` |
| fixed-view smoke | 통과 | 2K, view 51 snapshot 2개 |
| stats disabled smoke | 통과 | 올바른 `ACTIVE=False`에서 stats 디렉토리 미생성 |
| metadata | 통과 | `view_mode=fixed`, `view_id=51`, `capture_phase=pre_optimizer` |
| raw NPZ 불변식 | 통과 | 30개, shape·범위·instance 합계 일치 |
| 시각화 | 통과 | heatmap 30, temporal 1, difference 29 |
| full 30K | 통과 | W-20260728-003/R01, exit code 0 |
| test 평가 | 통과 | RGB/GT 각 200장, metric 파일 생성 |

## 오류 및 대응

| 오류 | 원인 | 대응 | 결과 |
|---|---|---|---|
| 소문자 `false` override 무효 | `ast.literal_eval` 실패 후 문자열로 유지 | Python literal `False` 사용 | stats 비활성화 및 test rendering 비활성화 확인 |
| on/off strict `1e-6` 비교 실패 | 별도 프로세스 CUDA 실행 자체의 비결정성 포함 | stats-off 반복 대조군 추가 | 차이 규모가 같은 order임을 확인 |
| 최종 Gaussian 수 불일치 | 30K snapshot 이후 `finalize()`의 `training_cleanup` 수행 | 두 phase의 수를 별도 기록 | 316,869 pre-cleanup, 316,096 final |

## 관련 명령

```bash
python scripts/train.py \
  -c src/Methods/RHO_GS/rho_gs_lego_fixed_view_tile_stats.yaml
```

CLI boolean override 예시:

```bash
python scripts/train.py \
  -c src/Methods/RHO_GS/rho_gs_lego_fixed_view_tile_stats.yaml \
  TRAINING.BACKUP.RENDER_TESTSET=False \
  TRAINING.TILE_GAUSSIAN_STATS.ACTIVE=False
```

## 커밋

| commit | 내용 | 상태 |
|---|---|---|
| 미생성 | 고정 뷰 tile 통계 및 시간축 시각화 | uncommitted |

## 한계 및 후속 작업

- fixed view는 view 선택 교란을 제거하지만 densification·pruning과 파라미터 학습 변화는 함께 포함한다.
- 별도 fixed render의 순수 overhead를 stats-off와 동일 프로세스 조건에서 정밀 benchmark하지 않았다.
- `processed`는 실제 gradient·color 기여 수가 아니라 alpha 조기 종료 전 평가 후보 수이다.
- tile aggregate만으로 Gaussian identity나 attribute별 관계, Hessian 비대각 성분은 알 수 없다.
- 다음 단계는 고정 topology 또는 stable ID 조건에서 tile별 Gaussian 후보 관계와 attribute별 Hessian 수집을 설계하는 것이다.

### 2026-07-30 — implementation-retired

- 상태: `retired-before-commit`
- coupling 실험 프레임워크와 RHO_GS namespace만 유지한다는 사용자 결정에 따라 fixed-view tile 통계 구현을 현재 working tree에서 제거했다.
- 별도 fixed-view render, temporal/difference 시각화 및 전용 YAML 설정은 활성 코드에 남지 않는다.
- W-20260728-003의 측정 결과와 artifact는 완료된 과거 실험 기록으로 보존한다.
- 향후 audit view 및 시간축 평가는 coupling framework의 evaluator/runner 경계에서 고정 topology 정책과 함께 구현한다.
