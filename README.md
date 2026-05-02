# CRMC-Docker: 분할 추론 정책 기반 분산 ResNet 추론 시스템

기존 CRMC 시뮬레이션에서 평가했던 **다중 디바이스 분할 추론 정책(Split-point Partition Policy)** 을, 실제로 **도커 컨테이너로 분리된 서버들**이 정책에 따라 ResNet34 레이어를 나눠 계산하도록 확장하는 프로젝트입니다.

---

## 1. 배경 (기존 CRMC 정리)

기존 CRMC 코드(`resnet34_TinyImageNet/`)는 ResNet34 / TinyImageNet 환경에서 다음 요소를 고려한 **분할 추론(Split Computing) 정책**을 그래프 기반으로 시뮬레이션합니다.

- **디바이스 3대** (`D1`, `D2`, `D3`)
  - `D_C`: 디바이스별 연산 능력 (FLOPS)
  - `D_tt`: 디바이스 간 전송 대역폭
  - `D_BER`: 디바이스 간 통신 채널 BER
- **ResNet34 레이어 분할 지점 (split point)** 을 노드로 갖는 가중치 그래프 위에서, 다음 정책들의 latency / energy / accuracy를 비교
  - `GSPDA` — 본 논문이 제안한 그래프 기반 최단경로 분할 (정확도 임계치 + 에너지 임계치 만족 하에 추론 시간 최소)
  - `low_BER` — 가장 BER이 낮은 디바이스로 오프로드
  - `com_best` — 연산 성능이 가장 좋은 디바이스에 전부 할당
  - `min_inter` — 중간 데이터(intermediate feature)가 가장 작은 분할점 사용

기존 코드는 어디까지나 **분석/시뮬레이션**이며, 실제 모델 가중치를 들고 디바이스들이 통신하면서 추론을 수행하지는 않습니다.

## 2. 본 프로젝트의 목표

기존 시뮬레이션 결과를 **실측**으로 검증할 수 있는 형태로 옮깁니다.

- ResNet34 모델을 분할 가능한 형태로 정의
- 정책이 출력한 split point와 디바이스 할당 결과를 입력으로 받아
- 각 도커 컨테이너(서버)가 자기 책임 구간만 forward 한 뒤
- 다음 컨테이너로 intermediate tensor를 전송하고
- 마지막 컨테이너가 최종 logits / classification 결과를 반환

이를 통해 다음을 측정합니다.

| 항목 | 시뮬레이션 값 | 실측 값 |
|---|---|---|
| End-to-end inference latency | ✅ | ✅ (본 프로젝트) |
| Per-device compute time | ✅ | ✅ |
| Intermediate tensor 전송 시간 | ✅ | ✅ |
| Top-1 / Top-5 accuracy | 예측치 | ✅ (실측 정답률) |

## 3. 아키텍처

```
                          ┌─────────────────────────┐
                          │   Orchestrator (Host)   │
                          │ - 정책 선택 (GSPDA 등)  │
                          │ - split-plan 생성       │
                          │ - 입력 이미지 송신      │
                          └───────────┬─────────────┘
                                      │ split-plan + image
                                      ▼
        ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
        │  device-1      │───▶│  device-2      │───▶│  device-3      │
        │  (container)   │    │  (container)   │    │  (container)   │
        │  layers[0..a]  │    │  layers[a..b]  │    │  layers[b..N]  │
        │  gRPC/HTTP     │    │  gRPC/HTTP     │    │  gRPC/HTTP     │
        └────────────────┘    └────────────────┘    └────────────────┘
                                                            │ logits
                                                            ▼
                                                       Orchestrator
```

- **`orchestrator/`** — 정책 모듈(기존 CRMC 코드 포팅)을 호출해 split-plan을 만들고, 각 컨테이너를 호출하여 추론을 트리거 / 결과 수집
- **`device/`** — 모든 디바이스가 공유하는 단일 추론 서버 이미지. 환경변수(`DEVICE_ID`, `LAYER_RANGE`, `NEXT_HOP_URL`)에 따라 자기 구간만 실행
- **`model/`** — 분할 가능한 ResNet34 정의 (`forward_partial(x, start, end)`)와 학습된 weight
- **`policy/`** — 기존 CRMC의 `GSPDA_resnet.py`, `scheme_*` 등을 포팅하여 `compute_split_plan(...)` 인터페이스로 노출
- **`docker-compose.yml`** — `device-1`, `device-2`, `device-3`, `orchestrator` 4개 서비스를 하나의 사용자 정의 네트워크에서 기동. 디바이스별 자원 제약(`cpus`, `mem_limit`)으로 `D_C` 차이를 모사하고, `tc qdisc`로 링크 latency / loss 를 모사하여 `D_tt`, `D_BER` 도 재현

## 4. 동작 흐름

1. `docker compose up` → 3개 device 컨테이너가 모델 weight 로드 후 대기
2. Orchestrator가 정책 모듈을 호출해 split plan 산출
   ```
   policy = "GSPDA"  # or "low_BER" | "com_best" | "min_inter"
   plan = compute_split_plan(policy, D_C, D_tt, D_BER, acc_thresh, energy_thresh)
   # plan 예시: [("device-1", 0, 7), ("device-3", 7, 18), ("device-2", 18, 34)]
   ```
3. Orchestrator가 첫 디바이스에 입력 이미지 + 전체 plan 전송
4. 각 디바이스는 자기 구간의 layer를 forward → 다음 hop으로 intermediate tensor 전송
5. 마지막 디바이스가 logits 반환 → Orchestrator가 latency / accuracy 기록

## 5. 기존 시뮬레이션과의 매핑

| 기존 변수 | 실측 시스템에서의 구현 |
|---|---|
| `D_C[i]` (FLOPS) | `--cpus`, `--memory` 제약으로 모사 |
| `D_tt[i]` (bandwidth) | `tc qdisc add ... rate ...` 로 인터페이스 대역폭 제한 |
| `D_BER[i]` | `tc qdisc add ... loss ...` 로 패킷 손실 / 에러 모사 |
| `SS_f[i]` (split별 FLOPs) | `model/resnet_split.py` 에서 실제 forward 시간 측정 |
| `SS_d[i]` (intermediate data size) | 실제 직렬화된 tensor 바이트 수 측정 |

## 6. 디렉터리 구조

```
CRMC/
├── README.md
├── resnet34_TinyImageNet/        # (기존) 시뮬레이션 / 정책 평가 코드
│   ├── GSPDA_resnet.py
│   ├── CRMC_eval_*.py
│   ├── scheme_*.py
│   └── ...
│
├── docker-compose.yml            # (예정) 3 device + orchestrator 구성
├── orchestrator/                 # (예정)
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── device/                       # (예정) 공통 추론 서버 이미지
│   ├── Dockerfile
│   ├── server.py
│   └── requirements.txt
├── model/                        # (예정)
│   ├── resnet_split.py
│   └── weights/                  # 학습된 ResNet34 weight (ignore)
├── policy/                       # (예정) 기존 CRMC 정책 코드 포팅
│   ├── gspda.py
│   ├── scheme_low_ber.py
│   ├── scheme_com_best.py
│   ├── scheme_min_inter.py
│   └── shortest_path_graph.py
└── results/                      # (예정) latency / accuracy 측정 결과 (ignore)
```

## 7. 향후 작업

- [ ] ResNet34 분할 가능한 모델 클래스 구현 (`forward_partial`)
- [ ] 단일 device 추론 서버 (FastAPI 또는 gRPC) 작성
- [ ] 기존 CRMC 정책 코드 `policy/` 로 포팅 및 인터페이스 통일
- [ ] `docker-compose.yml` 작성 + 자원/네트워크 제약 셋업
- [ ] Orchestrator → 정책 → 분산 추론 end-to-end 연결
- [ ] 4개 정책 × 다양한 채널 조건에서 실측 latency / accuracy 비교 리포트

## 8. 참고

- 모델: ResNet34
- 데이터셋: TinyImageNet
- 기존 시뮬레이션 코드: `resnet34_TinyImageNet/`
