---
record_id: D-20260730-001
type: develop
method: RHO_GS
created_at: 2026-07-30T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/develop/2026/D-20260730-001-contributor-aware-anchor-diagnostics.md
retrospective: false
---

# D-20260730-001 — Contributor-aware Anchor 및 Zero-group 진단

## 요약

| 항목 | 내용 |
|---|---|
| 구현 목적 | geometry-visible random anchor와 sampled residual 기여의 불일치 제거 |
| 범위 | anchor registry, randomized VJP probe, grouping 연동, zero-group 진단, artifact schema |
| 기존 경로 영향 | `experiments/` 내부만 변경; RHO_GS core/CUDA 변경 없음 |
| 관련 위험 | root `RISK-RHO-013` |
| 관련 Work | [[src/Methods/RHO_GS/docs/work/2026/W-20260729-002-lego-position-coupling-reference\|W-20260729-002]] |
| 상태 | `implemented-validation-pending` |

## 문제와 근거

- 기존 pipeline은 tile/pixel sampling과 anchor random sampling을 독립적으로 수행했다.
- visibility는 frustum·isotropic footprint 근사였으며 sampled residual에 대한 실제 alpha/raster contribution을 뜻하지 않았다.
- GPU 진단에서 random K=2 group은 zero Hessian이었지만 동일 tile의 position gradient nonzero Gaussian은 14,279개였다.
- gradient 상위 pair는 normalized spectral coupling 0.9690을 보여 Jacobian/Hessian 구현보다 anchor 관측 조건이 우선 문제임을 확인했다.

## 구현 정책

| 구성 | 정책 |
|---|---|
| anchor population | sampled residual에 대한 randomized VJP position Jacobian energy threshold 통과 Gaussian |
| selection | contributor population 내 seed 고정 uniform random |
| top-K | `contribution_topk` 분석 전용 metadata 명시 |
| member filter | config로 contributor population 제한 가능 |
| exact curvature | probe가 아니라 기존 repeated-VJP exact sampled group Jacobian 사용 |
| zero group | 제거하지 않고 원인별 분류·저장 |
| 재현 정보 | strategy, seed, selected ID/score, candidate count/hash, probe count/seed, threshold |

Randomized score는 Rademacher vector `z`에 대해 `mean ||J_i^T z||²`로 계산하며 기대값은 `||J_i||_F²`다. residual-weighted gradient `J_i^T r`는 상쇄될 수 있으므로 contributor 판정의 기본 score로 사용하지 않는다.

## 변경 파일

| 경로 | 내용 |
|---|---|
| `experiments/coupling/anchor_selection/` | 4개 selector와 randomized VJP probe |
| `experiments/coupling/diagnostics/` | exact Jacobian block activity·shared pixel support·zero group 분류 |
| `experiments/types.py`, `protocols.py`, `registry.py` | `AnchorSet`과 extension protocol/registry |
| `experiments/coupling/grouping/strategies.py` | 외부 anchor와 optional contributor member mask 수용 |
| `experiments/runners/` | residual 이후 anchor 구성, artifact·summary 기록 |
| `experiments/configs/anchor_selection/` | selector fragments |
| `experiments/tests/` | determinism, probe, filter, zero-group, CUDA default-device 검사 |

## Artifact

| 파일 | 핵심 정보 |
|---|---|
| `anchors.jsonl` | 선택 anchor ID/score, candidate count/hash, seed, probe metadata |
| `groups.jsonl` | group member와 anchor candidate hash 연결 |
| `group_diagnostics.csv` | block norm, shared support, valid pair, zero group, classification |
| `checkpoint_metadata.json` | checkpoint SHA-256, size, mtime |
| `summary.json` | zero/valid group와 valid pair 비율 |

## Events

### 2026-07-30 — implementation-started

- 상태: `implemented-validation-pending`
- 사용자는 contributor-aware random anchor, 재현 정보 보강, zero group 보존 방향을 승인했다.
- CUDA kernel과 RHO_GS core는 변경하지 않았다.

### 2026-07-30 — gpu-reference-verified

- 상태: `gpu-smoke-verified-uncommitted`
- 실제 RTX 4070 Ti 환경에서 전체 unit/regression test `24/24`가 통과했다. CUDA default-device sampling과 randomized VJP probe 검사도 포함한다.
- 이전 zero-group과 같은 Lego late checkpoint, train view 51, 0.125×, top tile 1개, tile 전체 256 pixel, visible-overlap K=2 조건으로 analyzer를 실행했다.
- 최종 artifact: `output/RHO_GS/experiments/d20260730_001_gpu_contributor_k2_smoke_r02-20260730T033253118076Z-ebee2fbe`
- checkpoint SHA-256: `5c4982037959f05e5dcf8069525fb1c5f4901ee30c8cdf8454887bf09875f320`
- contributor 후보는 12,284개, candidate hash는 `6b78c7fbc5bc3105`, 선택 anchor는 66,981, member는 66,980이었다.
- exact group Jacobian에서 active block 2개, shared sampled pixel 1개, valid pair 1개, zero-group ratio 0, normalized spectral coupling 0.138625를 기록했다.
- 동일 명령의 선행 smoke와 최종 smoke에서 candidate hash, anchor/member, coupling 값이 일치했다.
- failure record는 0 byte이며 symmetry relative error는 0이었다.

### 2026-07-30 — small-energy-capture-corrected

- 상태: `gpu-smoke-verified-uncommitted`
- 첫 GPU smoke에서 group이 유일한 off-diagonal pair를 모두 포함했는데도 capture ratio가 0.333693으로 계산되는 기존 수치 오류를 발견했다.
- 원인은 float32 off-diagonal energy가 machine epsilon보다 작을 때 분모를 epsilon으로 clamp한 것이었다.
- total energy가 정확히 0일 때만 0을 반환하고 그 외에는 `captured / total`을 그대로 계산하도록 수정했다.
- 작은 nonzero energy 회귀 테스트를 추가했고 최종 GPU artifact에서 capture ratio 1.0을 확인했다.

### 2026-07-30 — final-metadata-artifact-verified

- 상태: `gpu-smoke-verified-uncommitted`
- 재현성 metadata 보강 후 최종 권장 artifact를 `output/RHO_GS/experiments/d20260730_001_gpu_contributor_k2_smoke_r03-20260730T033633597845Z-24edc684`로 지정한다.
- 앞선 R02 artifact의 coupling 수치는 유효하지만 exact sampled pixel ID와 group 내부 render-state metadata가 없으므로 최종 schema 검증에는 R03을 사용한다.
- R03은 sampled pixel ID 256개, 실제 view ID 51, candidate hash, anchor/member ID, contributor filter, footprint 근사와 tile ID를 저장했다.
- candidate count 12,284, candidate hash `6b78c7fbc5bc3105`, anchor 66,981, member 66,980, normalized spectral coupling 0.138625가 앞선 반복과 동일했다.
- capture ratio 1.0, zero-group ratio 0, valid-pair ratio 1.0, failure record 0 byte를 확인했다.

## 한계 및 주의

- randomized score는 finite-probe 추정치이므로 probe count·seed가 다른 결과를 직접 비교하지 않는다.
- contributor threshold가 너무 높으면 모집단이 사라지거나 강한 Gaussian에 편향될 수 있다.
- `contribution_topk`는 sampling distribution 자체가 편향되므로 일반 random coupling 분포로 해석하지 않는다.
- shared support는 선택된 RGB residual pixel에서 exact position Jacobian block이 nonzero인지로 판정한다.
