---
record_id: W-20260729-001
type: work
method: RHO_GS
created_at: 2026-07-29T00:00:00+09:00
template_version: 1
canonical_path: src/Methods/RHO_GS/docs/work/2026/W-20260729-001-rho-gs-lego-portable-state-baseline.md
retrospective: false
---

# W-20260729-001 — RHO_GS Lego Portable-State Baseline

## 요약

| 항목 | 내용 |
|---|---|
| 목적 | Adam state가 포함된 early/middle/late 동일 학습 trajectory checkpoint 생성 |
| scene | NeRF Synthetic Lego |
| 관련 Develop | [[src/Methods/RHO_GS/docs/develop/2026/D-20260729-001-coupling-experiment-framework\|D-20260729-001]] |
| 상태 | `awaiting-run` |
| topology 정책 | 원본 30K 학습 정책 사용; snapshot ID는 각 시점 local identity |
| test 평가 | 비활성화; 암묵적 LPIPS weight download 방지 |

## Runs

### R01 — 1K/15K/30K portable state baseline

| 항목 | 값 |
|---|---|
| 상태 | `awaiting-run` |
| config | `src/Methods/RHO_GS/rho_gs_lego_coupling_states.yaml` |
| model label | `rho_gs_lego_coupling_states_w20260729_001_r01` |
| seed | 0 |
| environment | Conda `nerficg`, GPU는 실행 보고 시 수집 |
| 예상 state | `portable_states/iteration_{001000,015000,030000}.rho_state.pt` |

```bash
PYTHONPATH=src:src/Methods/RHO_GS \
python -m experiments.runners.train_with_portable_states \
  -c src/Methods/RHO_GS/rho_gs_lego_coupling_states.yaml
```

실행 후 `W-20260729-001/R01`, 상태, 생성된 output 경로 또는 오류 로그를 보고한다.

## Events

### 2026-07-29 — planned

- 상태: `awaiting-run`
- model-only final checkpoint가 Adam moment를 포함하지 않아 공정한 early/middle/late 비교용 baseline을 계획했다.
- 실제 실행과 결과는 아직 확인하지 않았다.
