---
record_id: D-20260728-002
type: develop
method: RHO_GS
created_at: 2026-07-28T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/develop/2026/D-20260728-002-tile-gaussian-stats.md
retrospective: false
---

# D-20260728-002 — Tile별 Gaussian 등록 및 순회 통계

## 요약

| 항목 | 내용 |
|---|---|
| 구현 목적 | RHO_GS 학습 렌더링에서 tile별 Gaussian 후보 등록량과 pixel traversal 처리량 수집 |
| 초기 의도 | Hessian 구현 전에 tile-local 연산 밀도와 alpha blending 조기 종료 특성을 실제 장면에서 관찰 |
| 범위 | CUDA 통계 커널, optional binding/API, Trainer 수집, NPZ·CSV·heatmap recorder, Lego 실험 config |
| 제외 범위 | Hessian 계산, Gaussian identity 저장, attribute별 통계, optimizer 변경, 고정 view 시간축 수집 |
| 상태 | 구현 및 검증 완료, commit 미생성 |
| 관련 Work | [[src/Methods/RHO_GS/docs/work/2026/W-20260728-002-rho-gs-lego-tile-stats\|W-20260728-002]] |

## 기준 저장소 상태

| 저장소 | commit | branch | dirty | remote |
|---|---|---|---|---|
| NeRFICG | `c7437127af1681f565a57fcd9d7819fde1adc0a7` | `main` | Method 및 문서 변경 존재 | `https://github.com/nerficg-project/nerficg.git` |
| RHO_GS | `751ec974c7ab2389c9bc789866ebec2fd6d8d54a` | `rho-gs/testbed` | namespace 분리와 tile stats 변경 존재 | `https://github.com/moot1801/RHO-GS` |

## 설계 및 위험

### 수집 의미

| 출력 | 정의 |
|---|---|
| `registered` | sorting과 선택적 tile culling 이후 tile instance range에 등록된 Gaussian 후보 수 |
| `processed_mean` | valid pixel별로 transmittance 조기 종료 전 exponent/alpha 평가를 수행한 후보 수의 tile 평균 |
| `processed_max` | 같은 pixel별 처리 후보 수의 tile 최대 |

통계는 정상 blend kernel의 제어 흐름과 출력을 변경하지 않도록 별도 CUDA kernel로 계산한다. 수집이 꺼져 있으면 빈 tensor를 반환하고 별도 kernel을 실행하지 않는다. Python 공개 API는 기본적으로 기존 image tensor 반환을 유지하며 `collect_tile_stats=True`일 때만 image와 통계를 함께 반환한다.

| 위험 | 영향 | 대응 |
|---|---|---|
| 통계 커널의 후보 재순회 | 수집 iteration 시간 증가 | 기본 비활성화, 설정한 interval에서만 실행 |
| `processed` 의미 오해 | 실제 기여 Gaussian 수로 잘못 해석 | exponent/alpha 평가 후보 수임을 API·문서에 명시 |
| 랜덤 sampled view | 시간축 변화가 view 선택에 교란 | raw metadata에 view ID 저장, 후속 실험에서 fixed view 추가 |
| densification과 pruning | Gaussian 수와 identity 변화 | 매 snapshot에 Gaussian 수 저장, identity 비교는 별도 설계 |
| tile edge | 해상도가 16의 배수가 아니면 invalid pixel 존재 | valid pixel만 평균과 최대 계산 |
| autograd 출력 추가 | backward signature 및 기존 호출 호환성 위험 | 통계 tensor를 non-differentiable로 표시하고 기본 반환 계약 유지 |
| CUDA extension stale build | Python 수정과 binary 불일치 | wheel 재빌드·재설치 후 import 및 GPU smoke 검증 |

## Events

### 2026-07-28 — planned

- 상태: `planned`
- 16×16 tile별 `registered`, `processed_mean`, `processed_max`를 1,000 iteration 간격으로 수집하기로 했다.
- 기존 rasterization 경로의 출력 및 gradient를 유지하는 optional instrumentation으로 범위를 제한했다.

### 2026-07-28 — cuda-stats-implemented

- 상태: `implemented-uncommitted`
- 최종 `tile_instance_ranges`와 sorted primitive index를 읽는 별도 통계 kernel을 추가했다.
- 각 valid pixel에서 기존 blend와 같은 candidate traversal 및 transmittance early termination을 적용했다.
- forward API가 optional 통계 tensor 3개를 반환하도록 확장했다.

### 2026-07-28 — python-recorder-implemented

- `TileGaussianStats` binding type과 optional renderer 반환 경로를 추가했다.
- Trainer가 `(iteration + 1) % interval == 0`인 실제 sampled view의 pre-optimizer forward에서 통계를 요청하도록 연결했다.
- `TileStatsRecorder`가 raw NPZ, append summary CSV 및 post-training heatmap을 생성하도록 구현했다.
- 기본 설정은 비활성화하고 Lego 전용 30K config에서만 활성화했다.

### 2026-07-28 — backend-built

- editable install은 wheel build에 성공했지만 namespace package finder가 `RHOGSCudaBackend`를 노출하지 못해 smoke import가 실패했다.
- 일반 wheel 방식으로 강제 재설치한 뒤 site-packages의 `RHOGSCudaBackend` import를 확인했다.
- 설치 명령은 `--no-build-isolation --no-deps`를 사용했으며 dependency 다운로드는 수행하지 않았다.

```bash
pip install src/Methods/RHO_GS/RHOGSCudaBackend \
  --force-reinstall \
  --no-build-isolation \
  --no-deps
```

### 2026-07-28 — smoke-verified

- 기본 sorting, separate sorting, tile culling, per-Gaussian backward 조합에서 2-iteration GPU smoke를 완료했다.
- 기본 smoke의 50×50 NPZ shape와 `registered.sum() == n_instances`를 확인했다.
- stats on/off 2-iteration final checkpoint의 최대 tensor 절대 차이는 `5.180481821298599e-07`로 허용 기준 `1e-6` 이하였다.
- smoke output의 heatmap 생성도 확인했다.

### 2026-07-28 — full-run-verified

- [[src/Methods/RHO_GS/docs/work/2026/W-20260728-002-rho-gs-lego-tile-stats\|W-20260728-002/R01]] 30K 학습과 test 평가가 정상 종료됐다.
- NPZ 30개, heatmap 30개, test RGB/GT 각 200개를 확인했다.
- 30개 NPZ의 iteration, shape, 값 범위 및 instance 합계 불변식을 검증했다.

## 변경 파일

| 경로 | 변경 목적 |
|---|---|
| `Renderer.py` | 학습 renderer의 optional tile stats 반환 |
| `Trainer.py` | 설정, 수집 interval, sampled view ID 및 recorder lifecycle 연결 |
| `TileStatsRecorder.py` | NPZ·CSV·heatmap 저장 |
| `rho_gs_lego_tile_stats.yaml` | Lego 30K 수집 실험 설정 |
| `RHOGSCudaBackend/__init__.py` | tile stats type 공개 |
| `RHOGSCudaBackend/RHOGSCudaBackend/torch_bindings/__init__.py` | binding type 공개 |
| `RHOGSCudaBackend/RHOGSCudaBackend/torch_bindings/rasterization.py` | optional Python API와 autograd signature 확장 |
| `RHOGSCudaBackend/RHOGSCudaBackend/rasterization/include/kernels_forward.cuh` | tile stats CUDA kernel |
| `RHOGSCudaBackend/RHOGSCudaBackend/rasterization/include/forward.h` | optional output pointer 선언 |
| `RHOGSCudaBackend/RHOGSCudaBackend/rasterization/src/forward.cu` | 통계 kernel 조건부 launch |
| `RHOGSCudaBackend/RHOGSCudaBackend/rasterization/include/rasterization_api.h` | wrapper API 확장 |
| `RHOGSCudaBackend/RHOGSCudaBackend/rasterization/src/rasterization_api.cu` | 통계 tensor 할당 및 반환 |

## 검증

| 검증 | 결과 | 근거 |
|---|---|---|
| Python syntax | 통과 | 변경 Python 파일 `py_compile` |
| whitespace | 통과 | `git diff --check` |
| backend build/install | 통과 | wheel build 및 site-packages import |
| 기본 sorting smoke | 통과 | 2 iteration, NPZ 2개 |
| separate sorting smoke | 통과 | 2 iteration, 기본 경로와 동일 통계 확인 |
| tile culling smoke | 통과 | 2 iteration, culling 이후 등록량 감소 확인 |
| per-Gaussian backward smoke | 통과 | 2 iteration 정상 backward |
| stats on/off regression | 통과 | checkpoint 최대 절대 차이 `5.180481821298599e-07` |
| heatmap 생성 | 통과 | smoke 2개 및 full run 30개 |
| full 30K | 통과 | `W-20260728-002/R01`, exit code 0 |
| raw NPZ 불변식 | 통과 | 30개, 50×50, 범위 및 `n_instances` 합계 일치 |

## 오류 및 대응

| 오류 | 원인 | 대응 | 결과 |
|---|---|---|---|
| `div_round_up` compile error | `rasterization_api.cu` scope에서 helper가 노출되지 않음 | `(value + tile - 1) / tile` 정수 ceil division으로 교체 | build 성공 |
| editable install 후 import 실패 | editable finder가 중첩 namespace package를 노출하지 못함 | non-editable wheel 강제 재설치 | import 및 smoke 성공 |
| 첫 NPZ 사후 검증 실패 | 검증 명령에서 metadata key를 `metadata`로 추정 | 실제 key `metadata_json`으로 정정 후 재검증 | 30개 모두 통과 |

## 커밋

| commit | 내용 | 상태 |
|---|---|---|
| 미생성 | Tile별 Gaussian 등록 및 순회 통계 | uncommitted |

## 한계 및 후속 작업

- 현재 구현은 tile aggregate만 저장하며 Gaussian ID나 후보 목록은 저장하지 않는다.
- 서로 다른 sampled view의 통계를 직접 시간축 비교하면 view 선택 효과가 섞인다.
- `processed`는 nonzero alpha/color 기여 수나 backward 영향 수가 아니다.
- 수집 시 별도 kernel이 candidate range를 다시 순회하므로 정밀한 overhead benchmark가 필요하다.
- CUDA backend wheel은 현재 환경에 설치됐지만 소스 변경 commit과 portable 설치 절차가 아직 필요하다.
- 다음 단계는 fixed-view 추적, Gaussian identity 연결, attribute별 관계 및 Hessian 수집 설계다.

### 2026-07-30 — implementation-retired

- 상태: `retired-before-commit`
- coupling 실험 프레임워크와 RHO_GS namespace만 유지한다는 사용자 결정에 따라 tile 통계 runtime·CUDA 구현을 현재 working tree에서 제거했다.
- 제거 범위는 optional CUDA 통계 kernel, rasterizer 통계 반환 API, renderer/trainer 연결, `TileStatsRecorder.py`, 전용 YAML 설정이다.
- 기존 실험 결과와 artifact는 유효한 과거 기록이므로 이 문서와 W-20260728-002는 보존한다.
- 향후 overlap 정보가 필요하면 기존 학습 경로에 instrumentation을 복원하지 않고 coupling framework의 adapter 경계에서 최소 기능으로 재설계한다.
