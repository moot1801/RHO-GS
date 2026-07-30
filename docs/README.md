# NeRFICG 및 RHO_GS 개발 가이드

이 문서는 NeRFICG 위에서 RHO_GS를 실험·구현하기 위해 알아야 하는 공통 구조와 진입점, 설정 정책, 학습 생명주기 및 주의사항을 정리한다.

현재 RHO_GS는 Faster-GS의 `FasterGSTestbed` 계열을 기반으로 분리한 연구용 Method이다. 본 문서의 내용은 현재 저장소 코드를 기준으로 하며, 명령은 특별한 언급이 없다면 NeRFICG 최상위 디렉터리에서 실행하는 것을 전제로 한다.

## 1. 핵심 개념

NeRFICG는 특정 NeRF 또는 Gaussian Splatting 구현 하나가 아니라, 여러 Method가 학습·데이터셋·평가·GUI 체계를 공유하도록 만든 프레임워크이다.

일반적인 실행 흐름은 다음과 같다.

```text
YAML 설정
  ↓
Framework.setup()
  ↓
GLOBAL.METHOD_TYPE / GLOBAL.DATASET_TYPE 확인
  ↓
Methods.<METHOD_TYPE> 및 Datasets.<DATASET_TYPE> 동적 import
  ↓
MODEL + RENDERER + TRAINING_INSTANCE + Dataset 생성
  ↓
Trainer.run(dataset)
  ↓
콜백 기반 학습 → 체크포인트 → 렌더링/평가
```

핵심 설계 원칙은 다음과 같다.

| 원칙                | 의미                                                      |
| ----------------- | ------------------------------------------------------- |
| Method 디렉터리 기반 탐색 | 중앙 registry를 수정하지 않고 `src/Methods/<이름>`으로 Method를 추가한다. |
| 세 컴포넌트 분리         | Model, Renderer, Trainer가 상태·렌더링·학습 흐름을 각각 담당한다.        |
| 설정 기반 실행          | 클래스 기본값, YAML, CLI override를 계층적으로 결합한다.                |
| 콜백 기반 학습          | 전처리, 매 iteration 작업, 후처리를 우선순위로 조합한다.                   |
| 공통 Dataset/View   | 서로 다른 Method가 같은 카메라 및 영상 데이터 표현을 사용한다.                 |

주요 근거 파일은 [Framework.py](../../../Framework.py)와 [Implementations.py](../../../Implementations.py)이다.

## 2. 프로젝트 구조

| 경로 | 역할 |
|---|---|
| [`scripts/`](../../../../scripts) | 설정 생성, 학습, 추론, 벤치마크, GUI 등의 CLI 진입점 |
| [`src/Framework.py`](../../../Framework.py) | 설정 로딩, GPU 설정, 랜덤 시드, 공통 초기화 |
| [`src/Implementations.py`](../../../Implementations.py) | Method와 Dataset 자동 탐색 및 인스턴스 생성 |
| [`src/Methods/Base/`](../../Base) | Model, Renderer, Trainer 공통 추상 계층 |
| [`src/Methods/`](../..) | 실제 Method 구현 디렉터리 |
| [`src/Datasets/`](../../../Datasets) | 데이터셋 로더와 공통 `View` 표현 |
| [`src/Cameras/`](../../../Cameras) | 카메라 모델과 좌표 변환 |
| [`src/Optim/`](../../../Optim) | optimizer, scheduler, parameter 관리 유틸리티 |
| [`src/CudaUtils/`](../../../CudaUtils) | 공용 CUDA 확장 |
| [`src/Thirdparty/`](../../../Thirdparty) | 선택적 외부 확장 래퍼 |
| [`src/ICGui/`](../../../ICGui) | 렌더링 및 학습 GUI |
| [`configs/`](../../../../configs) | 일반적인 학습 YAML 저장 위치 |
| `output/` | 학습 결과, 체크포인트, 렌더링 및 메트릭 |
| `.cache/` | Torch extension과 기타 캐시 |

### 2.1 중첩 저장소 구조

현재 `src/Methods/RHO_GS`는 바깥 NeRFICG 저장소 안에 있지만 자체 `.git`을 가진 독립 저장소이다.

따라서 작업 위치를 다음과 같이 구분해야 한다.

| 작업 | 기준 위치 |
|---|---|
| 학습·추론·설정 생성 실행 | NeRFICG 최상위 디렉터리 |
| RHO_GS 브랜치·커밋·diff 확인 | `src/Methods/RHO_GS` |
| NeRFICG 프레임워크 변경 | NeRFICG 최상위 저장소 |

바깥 저장소에서 실행한 Git 명령만으로 RHO_GS 내부 변경이 함께 기록된다고 가정하면 안 된다.

## 3. Method 등록 및 접근 방법

### 3.1 Method 공개 계약

각 Method의 공식 진입점은 `src/Methods/<Method>/__init__.py`이다. NeRFICG는 해당 모듈에서 다음 세 객체를 찾는다.

```python
MODEL = ModelClass
RENDERER = RendererClass
TRAINING_INSTANCE = TrainerClass
```

현재 RHO_GS의 공개 계약은 다음과 같다.

```python
from Methods.RHO_GS.Model import RHOGSModel
from Methods.RHO_GS.Renderer import RHOGSRenderer
from Methods.RHO_GS.Trainer import RHOGSTrainer

MODEL = RHOGSModel
RENDERER = RHOGSRenderer
TRAINING_INSTANCE = RHOGSTrainer
```

근거: [RHO_GS `__init__.py`](../__init__.py)

`GLOBAL.METHOD_TYPE`에는 Method 디렉터리 이름을 대소문자까지 동일하게 지정한다.

```yaml
GLOBAL:
    METHOD_TYPE: RHO_GS
```

### 3.2 컴포넌트 책임

| 컴포넌트 | 공통 책임 | RHO_GS 구현 |
|---|---|---|
| Model | 학습 파라미터, 상태, 체크포인트, PLY 변환 | [`RHOGSModel`](../Model.py) |
| Renderer | Model과 View를 받아 미분 가능한 이미지 생성 | [`RHOGSRenderer`](../Renderer.py) |
| Trainer | loss, optimizer step, densification, 평가 및 백업 | [`RHOGSTrainer`](../Trainer.py) |
| Loss | 렌더링 결과와 GT의 목적함수 계산 | [`RHOGSLoss`](../Loss.py) |
| CUDA backend | rasterization, backward, 선택적 fused Adam | [`RHOGSCudaBackend`](../RHOGSCudaBackend) |

### 3.3 Method를 읽는 권장 순서

1. `__init__.py`: 실제로 공개되는 클래스 확인
2. `Trainer.py`: 전체 알고리즘과 콜백 순서 확인
3. `Model.py`: 파라미터 표현, optimizer param group, topology 변경 확인
4. `Renderer.py`: Python과 CUDA backend의 입력·출력 경계 확인
5. `Loss.py`: 실제 최적화 목적함수 확인
6. CUDA backend: rasterization과 gradient 계산 방식 확인

RHO_GS의 Hessian 연구에서는 `Trainer.training_iteration()`에서 시작해 `Renderer.render_image_training()`과 `RHOGSCudaBackend.diff_rasterize()`까지 내려가는 흐름이 핵심이다.

## 4. 실행 진입점

### 4.1 설정 생성

```bash
python scripts/create_config.py \
  -m RHO_GS \
  -d MipNeRF360 \
  -o rho_gs_garden
```

이 명령은 Method와 Dataset 클래스의 기본 설정을 수집해 `configs/rho_gs_garden.yaml`을 생성한다.

현재 RHO_GS에서 관리 중인 예제 설정은 루트 `configs/`가 아니라 [`rho_gs_garden.yaml`](../rho_gs_garden.yaml)에 있다. 새로 생성한 설정과 이 파일의 값이 다를 수 있으므로, 연구 기준 설정 파일을 명시적으로 선택해야 한다.

### 4.2 Method 확장 설치

```bash
python scripts/install.py -m RHO_GS
```

[`install.py`](../../../../scripts/install.py)는 Method를 import하면서 발생하는 `Framework.ExtensionError`를 확인하고 필요한 확장을 순차적으로 설치한다.

주의사항:

- CUDA toolkit 및 C++ compiler 조합에 영향을 받는다.
- C++/CUDA 소스를 변경하면 extension을 다시 빌드해야 한다.
- 설치 과정에서 패키지 다운로드가 발생할 수 있으므로 네트워크 정책을 확인해야 한다.
- Python import가 단순한 정적 확인이 아니라 CUDA extension 로딩까지 유발할 수 있다.

### 4.3 학습

```bash
python scripts/train.py \
  -c src/Methods/RHO_GS/rho_gs_garden.yaml
```

[`train.py`](../../../../scripts/train.py)의 실행 순서는 다음과 같다.

1. `Framework.setup(require_custom_config=True)`
2. `GLOBAL.METHOD_TYPE`으로 Trainer 생성
3. `GLOBAL.DATASET_TYPE`으로 Dataset 생성
4. `training_instance.run(dataset)` 실행
5. Framework 종료 처리

CLI에서 일부 설정만 덮어쓸 수도 있다.

```bash
python scripts/train.py \
  -c src/Methods/RHO_GS/rho_gs_garden.yaml \
  TRAINING.NUM_ITERATIONS=1000 \
  TRAINING.GUI.ACTIVATE=False
```

### 4.4 추론 및 평가

```bash
python scripts/inference.py \
  -d output/RHO_GS/<run-directory> \
  -s test \
  -m \
  --checkpoint final.pt
```

[`inference.py`](../../../../scripts/inference.py)는 출력 디렉터리의 `training_config.yaml`을 읽어 학습 당시 Method와 Dataset을 복원한다.

주요 옵션은 다음과 같다.

| 옵션 | 의미 |
|---|---|
| `-s test train val` | 선택한 subset 렌더링 |
| `-s all` | 모든 subset과 등록된 camera trajectory 렌더링 |
| `-m` | GT가 있는 경우 PSNR, SSIM, LPIPS 계산 |
| `-b` | 반복 렌더링으로 FPS 측정 |
| `--visualize_errors` | 예측과 GT의 오차 시각화 |
| `--closest_train` | 가장 가까운 학습 카메라 영상 추가 렌더링 |
| `--checkpoint` | 사용할 `.pt` 체크포인트 이름 |

### 4.5 전체 장면 벤치마크

```bash
python scripts/benchmark.py -c <config.yaml>
```

[`benchmark.py`](../../../../scripts/benchmark.py)는 `DATASET.PATH`의 부모 디렉터리 아래 모든 장면을 탐색해 순차 학습하고 평균 메트릭·시간·VRAM 통계를 만든다.

주의할 점은 완료된 각 run 디렉터리를 마지막에 하나의 summary 디렉터리로 이동한다는 것이다. 기존 자동화가 개별 run 경로를 참조한다면 경로 변경을 고려해야 한다.

### 4.6 GUI

```bash
python scripts/gui.py
```

학습 중 GUI는 다음 설정으로 제어한다.

```yaml
TRAINING:
    GUI:
        ACTIVATE: false
```

GUI는 별도 프로세스와 GPU 렌더링을 사용한다. 메모리·시간 측정이나 Hessian 수집이 목적인 실험에서는 우선 비활성화하는 편이 안전하다. GUI import에 실패하면 `GuiTrainer`는 환경에 따라 `BaseTrainer`로 대체될 수 있다.

## 5. 설정 시스템

### 5.1 설정 우선순위

```text
@Framework.Configurable.configure 기본값
                  ↓
              YAML 설정
                  ↓
         CLI SECTION.KEY=VALUE
```

각 컴포넌트는 담당 설정 영역을 가진다.

| 클래스 | 설정 영역 |
|---|---|
| `BaseModel` | `MODEL` |
| `BaseRenderer` | `RENDERER` |
| `BaseTrainer` | `TRAINING` |
| `BaseDataset` | `DATASET` |
| Framework | `GLOBAL` |

중첩 설정은 재귀적으로 병합되므로 YAML에 일부 항목만 지정하면 나머지는 클래스 기본값이 유지된다.

`GLOBAL`의 누락 항목은 Framework 기본값으로 보완될 수 있지만, 다른 영역에서 존재하지 않는 키를 접근하면 일반적으로 `AttributeError`가 발생한다. 설정 오타가 조용히 무시된다고 가정해서는 안 된다.

### 5.2 주요 `GLOBAL` 설정

| 설정 | 의미 |
|---|---|
| `METHOD_TYPE` | 사용할 Method 디렉터리 이름 |
| `DATASET_TYPE` | 사용할 Dataset 모듈 이름 |
| `GPU_INDICES` | GPU 목록, `null`이면 CPU 의도 |
| `RANDOM_SEED` | Python, NumPy, PyTorch seed |
| `LOG_LEVEL` | 0: silent, 1: normal, 2: verbose, 3: debug |
| `ANOMALY_DETECTION` | PyTorch autograd anomaly detection |
| `FILTER_WARNINGS` | 일반 warning 필터링 여부 |

Framework는 GPU 설정에 따라 기본 tensor type/device를 전역으로 변경한다. `torch.empty(0)`처럼 device를 명시하지 않은 텐서도 전역 설정의 영향을 받을 수 있으므로 CPU/GPU 경계를 확인해야 한다.

또한 seed를 설정하지만 cuDNN benchmark가 활성화되어 있으며 deterministic algorithm 강제가 적용되지 않는다. 같은 seed가 strict한 bitwise 재현성을 의미하지는 않는다.

## 6. Trainer와 콜백 생명주기

[`BaseTrainer`](../../Base/Trainer.py)는 메서드에 부착된 콜백 메타데이터를 수집하고 priority가 높은 순서로 실행한다.

```text
Pre-training callbacks
    ├─ Dataset preloading
    ├─ sampler 생성
    └─ Gaussian 초기화

Training callbacks, 매 iteration
    ├─ SH degree 증가
    ├─ densification
    ├─ Morton ordering
    ├─ opacity reset
    ├─ 실제 forward/loss/backward/optimizer step
    └─ logging/GUI

Post-training callbacks
    ├─ Gaussian 정리
    ├─ 최종 checkpoint
    └─ train/test/val 렌더링
```

콜백은 다음 decorator로 정의한다.

```python
@pre_training_callback(...)
@training_callback(...)
@post_training_callback(...)
```

공통 속성은 다음과 같다.

| 속성 | 의미 |
|---|---|
| `priority` | 같은 단계에서 큰 값이 먼저 실행됨 |
| `active` | bool 또는 Trainer 설정 경로 |
| `start_iteration` | 실행 시작 iteration |
| `end_iteration` | 현재 구현에서는 해당 iteration까지 포함 |
| `iteration_stride` | 실행 간격, 0 이하이면 비활성화 |

`active='USE_MORTON_ORDERING'` 또는 `iteration_stride='DENSIFICATION_INTERVAL'`처럼 문자열을 쓰면 Trainer 인스턴스의 설정값을 참조한다.

### 6.1 RHO_GS의 실제 optimizer step

현재 학습 단계는 다음 순서다.

```python
image = self.renderer.render_image_training(...)
loss = self.loss(image, rgb_gt)
loss.backward()
self.model.gaussians.optimizer.step()
self.model.gaussians.optimizer.zero_grad()
```

따라서 gradient나 Hessian을 관찰하는 첫 Python hook은 `loss.backward()` 전후이다.

- `backward()` 전: loss, 렌더링 결과, 선택된 View 기록
- `backward()` 직후: Gaussian별 gradient 기록
- `optimizer.step()` 직전: optimizer state와 gradient를 함께 분석
- `optimizer.step()` 직후: 실제 parameter update 기록

## 7. Model 및 Gaussian 파라미터 정책

RHO_GS의 Gaussian별 학습 속성은 optimizer param group으로 분리되어 있다.

| Param group | 내부 표현 | 형태 |
|---|---|---:|
| `means` | 3D 위치 | `N × 3` |
| `sh_coefficients_0` | view-independent SH | `N × 1 × 3` |
| `sh_coefficients_rest` | 고차 SH | `N × K × 3` |
| `opacities` | sigmoid 이전 logit | `N × 1` |
| `scales` | exp 이전 log scale | `N × 3` |
| `rotations` | 정규화 이전 quaternion | `N × 4` |

실제 optimizer는 활성화된 물리량이 아니라 raw parameter를 업데이트한다.

| 속성 | raw parameter | 렌더링에 사용되는 값 |
|---|---|---|
| scale | `_scales` | `exp(_scales)` |
| opacity | `_opacities` | `sigmoid(_opacities)` |
| rotation | `_rotations` | 정규화된 quaternion |

따라서 Hessian을 해석할 때 다음을 구분해야 한다.

- 물리적 scale 공간과 log-scale parameter 공간
- opacity 공간과 logit 공간
- 정규화된 rotation 공간과 unconstrained raw quaternion 공간
- 속성별 차원, 단위, learning rate 차이

서로 다른 속성 block의 Hessian 절댓값을 그대로 비교하면 parameterization과 단위 차이를 관계 강도로 오해할 수 있다. 최소한 parameter scale, gradient scale, learning rate 또는 block normalization을 함께 기록해야 한다.

## 8. Dataset과 View 정책

[`BaseDataset`](../../../Datasets/Base.py)은 `train`, `test`, `val` subset을 가지며 현재 선택된 subset을 mutable한 `dataset.mode`로 유지한다.

```python
dataset.train()
dataset.test()
dataset.eval()
```

이 호출들은 새 Dataset을 반환하지 않고 같은 객체의 mode를 변경한다. 여러 helper가 연속으로 mode를 바꾸는 경우 마지막 호출이 이후 iteration에 영향을 준다.

각 [`View`](../../../Datasets/utils.py)는 다음 정보를 포함할 수 있다.

- camera 및 `c2w`/`w2c`
- RGB, alpha, depth, segmentation
- frame index와 timestamp
- optical flow 및 기타 부가 데이터

이미지 텐서의 기본 채널 규약은 CHW이다.

| 데이터 | 채널 수 |
|---|---:|
| RGB | 3 |
| alpha | 1 |
| depth | 1 |
| segmentation | 1 |
| optical flow | 2 |

좌표계는 오른손 좌표계이며 카메라 기준 방향은 다음과 같다.

- x: 오른쪽
- y: 아래
- z: 앞쪽

### 8.1 데이터 preloading

| `PRELOADING_LEVEL` | 저장 위치 | 특성 |
|---:|---|---|
| 0 | 디스크 | RAM/VRAM 절약, 반복 I/O 증가 |
| 1 | RAM | 일반적인 분석 실험에 적합 |
| 2 | VRAM | 빠르지만 Hessian 버퍼와 VRAM 경쟁 가능 |

현재 RHO_GS 예제 설정은 level 2이다. Hessian이나 tile별 통계를 추가로 저장하는 초기 실험에서는 0 또는 1이 더 안전하다.

alpha가 존재할 때 Trainer는 GT RGB를 배경색과 합성한 후 loss를 계산한다. color/opacity 관계를 분석할 때는 배경색과 alpha 합성 정책도 실험 변수로 기록해야 한다.

## 9. Renderer 정책

[`BaseRenderer`](../../Base/Renderer.py)의 `render_image()`는 일반적으로 다음 형태의 dictionary를 반환한다.

```python
{
    "rgb": image,
    "alpha": alpha,
    "depth": depth,
}
```

Method가 모든 채널을 제공할 의무는 없다. 현재 RHO_GS는 최종적으로 RGB만 반환한다.

RHO_GS Renderer의 현재 제약은 다음과 같다.

| 제약 | 현재 동작 |
|---|---|
| CPU | 미지원 |
| multi-GPU | 첫 번째 GPU만 사용 |
| camera type | perspective camera만 지원 |
| distortion | 경고 후 무시 |
| training render | `render_image_training()` 직접 호출 필요 |

`BaseRenderer`의 model type 검사는 `isinstance`가 아니라 정확한 `type(model)` 기준이다. 기존 Model을 subclass한 실험 Model을 Renderer에 전달할 경우 valid model type 목록도 함께 수정해야 할 수 있다.

## 10. 체크포인트와 출력

| 확장자 | 저장 내용 | 주요 용도 |
|---|---|---|
| `.pt` | Model state, 메타데이터, Model 설정 | 추론·평가·모델 재사용 |
| `.train` | Trainer 객체와 optimizer를 포함한 학습 상태 | 완전한 학습 재개 |

일반적인 출력 구조는 다음과 같다.

```text
output/<METHOD_TYPE>/<MODEL_NAME>_<timestamp>/
├── training_config.yaml
├── checkpoints/
│   ├── final.pt
│   └── ...
├── test_<iteration>/
├── train_<iteration>/
├── timings.txt
├── vram_stats.txt
└── n_gaussians.txt
```

Trainer는 실행 시 사용한 설정을 출력 디렉터리의 `training_config.yaml`로 복사한다. 추론과 재현 시에는 이 파일이 기준이 된다.

체크포인트 로딩에는 Python 객체 역직렬화가 포함될 수 있으므로 신뢰할 수 없는 외부 `.pt` 또는 `.train` 파일을 로드하면 안 된다.

### 10.1 메트릭 정책

공통 Renderer는 비교 일관성을 위해 결과를 8-bit 이미지로 저장하고 다시 읽은 후 PSNR, SSIM, LPIPS를 계산한다. 따라서 메모리의 float 렌더링 결과에서 직접 계산한 수치와 미세하게 다를 수 있다.

## 11. RHO_GS Hessian 연구의 현재 제약

### 11.1 현재 rasterizer는 double backward를 지원하지 않음

[`torch_bindings/rasterization.py`](../RHOGSCudaBackend/RHOGSCudaBackend/torch_bindings/rasterization.py)의 backward에는 다음 decorator가 적용되어 있다.

```python
@once_differentiable
def backward(...):
    ...
```

따라서 현재 렌더링 loss에 대해 다음 방식으로 Hessian을 직접 계산할 수 없다.

```python
torch.autograd.grad(loss, parameters, create_graph=True)
torch.autograd.functional.hessian(...)
```

첫 번째 gradient는 custom CUDA backward가 계산하지만, 해당 gradient에 대한 두 번째 미분 그래프는 만들어지지 않는다.

`RENDERER.USE_PER_GAUSSIAN_BACKWARD`도 second-order 기능이 아니다. 이는 first-order backward를 계산하는 CUDA 실행 전략 중 하나이다.

### 11.2 Hessian block 정의

Gaussian `i`, 속성 `a`의 파라미터를 `θᵢᵃ`라고 하면 분석 대상은 다음 block으로 정리할 수 있다.

```text
Hᵢⱼᵃᵇ = ∂²L / (∂θᵢᵃ ∂θⱼᵇ)
```

| 구분 | 질문 |
|---|---|
| `i = j`, `a = b` | 한 Gaussian의 같은 속성 내부 curvature는 어떤가? |
| `i ≠ j`, `a = b` | 같은 속성에서 Gaussian 간 관계가 얼마나 강한가? |
| `i = j`, `a ≠ b` | 한 Gaussian 내부 속성 간 coupling은 어떤가? |
| `i ≠ j`, `a ≠ b` | 서로 다른 Gaussian과 속성 사이의 cross coupling은 어떤가? |

전체 global Hessian을 구성하기보다 관심 있는 tile, Gaussian 집합, 속성 block만 선택하는 것이 현실적이다.

### 11.3 초기 계산 후보

| 방법 | CUDA 변경 | 장점 | 한계 |
|---|---:|---|---|
| gradient finite difference | 불필요 | 현재 backend로 시작 가능 | 파라미터 수에 비례해 매우 느림 |
| loss finite difference | 불필요 | 구현 단순 | 수치 오차와 연산량이 큼 |
| Gauss–Newton/Fisher 근사 | 일부 필요 가능 | PSD 구조와 block 근사에 유리 | 실제 Hessian과 다름 |
| double backward 구현 | 필요 | PyTorch autograd 연계 가능 | custom backward 전체 미분 구현 필요 |
| 전용 local Hessian CUDA kernel | 필요 | tile/attribute 구조에 최적화 가능 | 구현 및 검증 비용이 큼 |

작은 장면의 가능성 확인 단계에서는 finite difference로 일부 Gaussian과 속성만 측정한 뒤, 유의미한 희소성과 locality가 확인되면 Gauss–Newton 또는 전용 CUDA 경로로 확장하는 순서가 적절하다.

## 12. Hessian 실험 설계 시 반드시 고정할 요소

### 12.1 Gaussian identity 변경

다음 연산은 Gaussian 개수 또는 배열 순서를 바꾼다.

| 연산 | 영향 |
|---|---|
| duplicate/split | 새로운 Gaussian 생성 |
| prune | Gaussian 제거 |
| Morton ordering | Gaussian 배열 순서 변경 |
| finalization | low-opacity pruning 후 Morton 정렬 |

학습 시간에 따라 같은 Gaussian의 Hessian을 추적하려면 stable ID를 별도로 부여하거나 초기 실험에서 topology 변경을 비활성화해야 한다.

### 12.2 비교 중 값을 변화시키는 이벤트

다음 이벤트도 Hessian 분포를 갑자기 바꿀 수 있다.

- opacity reset
- SH degree 증가
- learning-rate decay
- random background color
- 학습 View sampler의 변경
- densification 직후 optimizer state 재구성

Hessian snapshot에는 최소한 다음 메타데이터를 함께 저장하는 것이 좋다.

```text
iteration
view/frame ID
loss
Gaussian stable IDs
active SH degree
각 attribute learning rate
Gaussian count
tile ID와 tile bounds
parameterization(raw/activated)
```

### 12.3 tile 소속 정의

Gaussian은 화면에서 여러 tile에 걸칠 수 있다. 따라서 `per-tile Hessian`을 말할 때 소속 정책을 먼저 정해야 한다.

| 후보 정책 | 의미 |
|---|---|
| 중심점 기준 | projected mean이 들어간 tile 하나에만 할당 |
| footprint 교차 기준 | bounding box가 겹치는 모든 tile에 할당 |
| 실제 기여 기준 | alpha/radiance 기여가 발생한 tile에만 할당 |
| 대표 tile + overlap 기록 | 중심 tile에 소속시키고 다른 tile과의 overlap은 별도 edge로 기록 |

position·scale·rotation의 공간적 locality를 보려면 footprint 교차가 자연스럽고, color·opacity의 blending 관계를 보려면 실제 기여 기준이 더 직접적이다. 두 정의를 섞으면 속성별 비교 해석이 불명확해질 수 있다.

## 13. 초기 Hessian 실험용 권장 설정

Gaussian identity와 측정 환경을 단순화하려면 다음과 같은 override를 고려할 수 있다.

```bash
python scripts/train.py \
  -c src/Methods/RHO_GS/rho_gs_garden.yaml \
  TRAINING.NUM_ITERATIONS=1000 \
  TRAINING.GUI.ACTIVATE=False \
  TRAINING.DATA.PRELOADING_LEVEL=1 \
  TRAINING.DENSIFICATION_INTERVAL=-1 \
  TRAINING.USE_MORTON_ORDERING=False \
  TRAINING.OPACITY_RESET_INTERVAL=1000000 \
  TRAINING.EXTRA_OPACITY_RESET_ITERATION=1000000
```

이 설정의 의도는 다음과 같다.

| 설정 | 의도 |
|---|---|
| 짧은 iteration | 초기 분포와 변화 가능성 확인 |
| GUI 비활성화 | GPU 메모리와 timing 간섭 감소 |
| RAM preload | Hessian 버퍼를 위한 VRAM 확보 |
| densification 비활성화 | Gaussian 개수와 identity 고정 |
| Morton ordering 비활성화 | Gaussian index 유지 |
| opacity reset 지연 | 중간 discontinuity 제거 |

이는 품질 최적화 설정이 아니라 분석을 단순화하기 위한 실험 설정이다. 구조가 확인된 뒤 실제 Faster-GS 학습 정책을 단계적으로 다시 활성화해야 한다.

## 14. 구현 위치 선택 기준

| 구현 대상 | 권장 위치 |
|---|---|
| snapshot 주기와 실험 제어 | `Trainer.py`의 callback |
| gradient 및 optimizer update 기록 | `training_iteration()`의 backward/step 경계 |
| Gaussian별 stable ID | `Model.py`의 `Gaussians`와 densify/prune/sort 경로 |
| attribute block 선택 | `Model.py`의 optimizer param group 또는 별도 실험 모듈 |
| projected tile/Gaussian 관계 | `Renderer.py` 또는 CUDA forward buffer |
| 실제 blending 관계 | CUDA rasterization forward/backward |
| local Hessian/Gauss–Newton kernel | `RHOGSCudaBackend` |
| 시각화 및 통계 저장 | 별도 `experiments/` 또는 `analysis/` 모듈을 Trainer callback에서 호출 |

초기 연구 코드를 `Trainer.training_iteration()` 안에 모두 직접 넣으면 baseline 흐름과 실험 로직이 빠르게 뒤섞인다. 최소한 다음 책임은 분리하는 편이 좋다.

```text
Trainer callback
  ├─ snapshot 시점 결정
  ├─ 현재 View/Gaussian 선택
  └─ 분석 모듈 호출

Hessian 분석 모듈
  ├─ parameter block 선택
  ├─ 계산 또는 근사
  ├─ sparse/block 통계
  └─ 파일 저장 및 시각화용 데이터 생성
```

## 15. 공통 주의사항 체크리스트

### 실행 전

- [ ] NeRFICG 최상위 디렉터리에서 명령을 실행하는가?
- [ ] `GLOBAL.METHOD_TYPE`이 `RHO_GS`인가?
- [ ] 의도한 config 파일을 사용하고 있는가?
- [ ] Dataset 경로와 image scale이 맞는가?
- [ ] GPU, CUDA toolkit, compiler 조합이 맞는가?
- [ ] CUDA 변경 후 extension을 다시 빌드했는가?

### 실험 전

- [ ] GUI와 W&B가 측정에 불필요하게 개입하지 않는가?
- [ ] preload 수준이 Hessian 저장 VRAM과 충돌하지 않는가?
- [ ] Gaussian identity 정책을 정했는가?
- [ ] densification, prune, ordering, opacity reset을 기록하거나 고정했는가?
- [ ] raw/activated parameterization을 명시했는가?
- [ ] tile 소속 및 중복 Gaussian 처리 정책을 정했는가?
- [ ] fixed View와 random View 중 어떤 조건인지 기록했는가?

### 결과 해석 전

- [ ] 대각/비대각 block을 분리했는가?
- [ ] 속성 차원과 단위를 정규화했는가?
- [ ] learning rate 차이를 함께 고려했는가?
- [ ] 절댓값뿐 아니라 부호, 분포, sparsity threshold를 기록했는가?
- [ ] Gaussian 간 거리, footprint overlap, blending 순서와 연결해 분석했는가?
- [ ] topology 변경 전후 결과를 직접 같은 index로 비교하지 않았는가?

## 16. 확인된 구현상 주의점

| 항목 | 내용 |
|---|---|
| RHO_GS CPU 실행 | Renderer에서 명시적으로 거부됨 |
| multi-GPU | 첫 번째 GPU만 사용 |
| camera | perspective만 지원 |
| double backward | `@once_differentiable`로 차단됨 |
| strict 재현성 | seed만으로 보장되지 않음 |
| callback end | 현재 구현에서 end iteration 포함 |
| callback stride | 0 이하이면 비활성화 |
| Dataset mode | mutable하므로 helper 호출 순서에 영향받음 |
| Model type 검사 | exact type 기준이라 subclass가 거부될 수 있음 |
| checkpoint 보안 | 신뢰할 수 없는 파일 로드 금지 |
| benchmark 출력 | 완료 후 run 디렉터리를 summary 디렉터리로 이동 |
| 메트릭 | 저장·재로딩한 8-bit 이미지 기준 |

## 17. 주요 참고 파일

### NeRFICG 공통

- [루트 README](../../../../README.md)
- [Framework 설정 및 초기화](../../../Framework.py)
- [Method/Dataset 동적 로딩](../../../Implementations.py)
- [Base Model](../../Base/Model.py)
- [Base Renderer](../../Base/Renderer.py)
- [Base Trainer](../../Base/Trainer.py)
- [콜백 decorator](../../Base/utils.py)
- [GUI Trainer](../../Base/GuiTrainer.py)
- [Base Dataset](../../../Datasets/Base.py)
- [View 및 데이터 유틸리티](../../../Datasets/utils.py)

### RHO_GS

- [Method entry](../__init__.py)
- [Model](../Model.py)
- [Renderer](../Renderer.py)
- [Trainer](../Trainer.py)
- [Loss](../Loss.py)
- [예제 설정](../rho_gs_garden.yaml)
- [CUDA autograd binding](../RHOGSCudaBackend/RHOGSCudaBackend/torch_bindings/rasterization.py)
- [CUDA rasterization source](../RHOGSCudaBackend/RHOGSCudaBackend/rasterization)

---

이 문서는 현재 RHO_GS testbed 구조를 기준으로 한다. 이후 Hessian 수집 모듈, stable Gaussian ID, 실험 출력 형식이 추가되면 각각의 데이터 스키마와 재현 절차를 본 문서 또는 별도 실험 문서에 함께 기록한다.
