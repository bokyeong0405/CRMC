# CRMC-Docker: 분할 추론 정책 기반 분산 ResNet 추론 시스템

[bokyeong0405/CRMC](https://github.com/bokyeong0405/CRMC) 의 GSPDA 시뮬레이션을 **실제 도커 컨테이너 5개로 구성된 분산 추론 시스템**으로 확장한 프로젝트입니다.

User 컨테이너가 TinyImageNet 검증 이미지를 보내면 → Orchestrator가 각 서버의 자원/링크 상태를 보고 GSPDA 정책으로 분할 계획을 산출 → 3개 서버가 ResNet34 레이어를 나눠 순차적으로 추론 → User에게 최종 결과 반환합니다.

---

## Quick Start

```bash
# 0. 사전 준비: parameter/, data/ 폴더에 weight + validation .npy 배치
#    (gitignore 처리되어 있어 별도로 옮겨와야 함, 아래 "데이터 준비" 참고)

# 1. 이미지 3개 빌드 (TF 다운로드 포함, 첫 빌드는 ~5분)
docker compose build

# 2. 서버 3개 + orchestrator 백그라운드 기동
docker compose up -d server-1 server-2 server-3 orchestrator

# 3. 헬스체크 (호스트에서 직접)
curl http://localhost:8080/healthz

# 4. user 클라이언트로 추론 요청 10건 발사
docker compose run --rm user

# 5. 정리
docker compose down
```

---

## 시스템 구성

```
┌─────────────────────────────────────────────────────────────────────┐
│ docker network: crmc (bridge)                                        │
│                                                                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐ │
│  │  server-1  │    │  server-2  │    │  server-3  │    │   user   │ │
│  │            │    │            │    │            │    │          │ │
│  │ POWER=3    │    │ POWER=25   │    │ POWER=50   │    │ /data    │ │
│  │ cpus=0.2   │    │ cpus=1.0   │    │ cpus=2.0   │    │  mount   │ │
│  │ TF+FastAPI │    │ TF+FastAPI │    │ TF+FastAPI │    │ httpx    │ │
│  └─────┬──────┘    └─────┬──────┘    └─────┬──────┘    └────┬─────┘ │
│        │                  │                  │              │       │
│        │  POST /run (forward chain, GSPDA-decided order)    │       │
│        │   ┌──────────────┴──────────────────┘              │       │
│        │   │                                                │       │
│        │   │   ┌────────────────┐                           │       │
│        └───┴───│  orchestrator  │◀──────POST /infer─────────┘       │
│                │  GSPDA + plan  │                                   │
│                │  /data/acc     │                                   │
│                │  config.json   │                                   │
│                └────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             │ port 8080 published to host (debug)
                             ▼
                          host machine
```

### 컴포넌트별 역할

| 컨테이너 | 이미지 | 역할 |
|---|---|---|
| `server-1` / `-2` / `-3` | `crmc/device` (TF + FastAPI) | 동일 image, env로 `DEVICE_ID` / `COMPUTE_POWER` 다르게 주입. `/run` 엔드포인트가 자기 slice만 forward → 다음 hop으로 텐서 전달 |
| `orchestrator` | `crmc/orchestrator` (numpy/scipy/sklearn/networkx, **TF 없음**) | 각 서버 `/capacity` 조회 → GSPDA로 split-plan 산출 → 첫 서버에 추론 위임 |
| `user` | `crmc/user` (numpy + httpx) | 검증 이미지 N장 → orchestrator `/infer` POST → 정확도/latency 통계 출력 |

### 데이터 흐름 (1 inference)

```
user                orchestrator             chain (예: server-1 → server-3 → server-2)
  │                       │                                  │
  │── POST /infer ────────▶ (1) GET /capacity × 3            │
  │   {tensor_b64,...}    │     (lazy cache)                 │
  │                       │ (2) compute_split_plan(D_C,..)   │
  │                       │ (3) POST server-1 /run ──────────▶ slice 0 forward
  │                       │     (multipart: meta+tensor)     │      │
  │                       │                                  │      ▼
  │                       │                       server-3 /run ──▶ slice 1
  │                       │                                  │      │
  │                       │                                  │      ▼
  │                       │                       server-2 /run ──▶ slice 2
  │                       │                                  │      │
  │                       │   logits (Response 체인 반대로)   │      │
  │                       │ ◀──────────────────────────────  │ ◀────┘
  │ ◀─ JSON {logits,plan} ┤                                  │
```

---

## 시뮬레이션 매핑 (어디서 어떻게 쓰이는가)

| 시뮬레이션 변수 | 출처 | 본 시스템에서의 구현 |
|---|---|---|
| `D_C[i]` | server `/capacity` 응답 | `COMPUTE_POWER` env 값. 동시에 cgroup `cpus:` 제한으로 실제 추론 속도도 함께 차이남 |
| `D_tt[i]` | `orchestrator/config.example.json` | 설정값으로만 GSPDA 입력에 들어감 (실제 링크 throttling은 미적용) |
| `D_BER[i]` | 같은 config | 같음 (실제 패킷 손실 시뮬은 안 함 — TCP 재전송 때문에 의미 없음) |
| `SS_f[i]` | `policy/constants.py` (정적) | ResNet34 split 별 FLOPs, 시뮬과 동일 |
| `SS_d[i]` | `policy/constants.py` (정적) | intermediate tensor 크기, 시뮬과 동일 |
| `acc_thresh` / `energy_thresh` | config | GSPDA의 후보 path 필터링 임계치 |

---

## 디렉터리 구조

```
CRMC/
├── README.md
├── docker-compose.yml             # 5개 서비스 + 네트워크 + 볼륨
├── .gitignore                     # parameter/, data/ 제외
│
├── model/                         # 분할 가능한 ResNet34 라이브러리
│   ├── __init__.py
│   └── resnet_split.py            # ResNet34, split_resnet, assign_weights, ...
│
├── policy/                        # GSPDA 정책 라이브러리
│   ├── __init__.py
│   ├── constants.py               # SS, SS_f, SS_d, start[], end[], END_TO_SPLIT_POINT
│   ├── graph.py                   # create_graph (시뮬과 동일)
│   ├── inference_time.py          # path → latency 추정
│   ├── set_ss_sb.py               # path → (split_points, BER) 디코더
│   ├── accuracy.py                # predict_accuracy_res, .npy lookup
│   ├── gspda.py                   # GSPDA 본체 알고리즘
│   └── plan.py                    # compute_split_plan() 공개 API + SplitPlan 타입
│
├── device/                        # 추론 서버
│   ├── Dockerfile                 # python:3.10-slim + libhdf5 + TF 2.15
│   ├── server.py                  # GET /capacity, POST /run, GET /healthz
│   └── requirements.txt
│
├── orchestrator/                  # 컨트롤 플레인
│   ├── Dockerfile                 # python:3.10-slim + scipy/sklearn (no TF)
│   ├── main.py                    # POST /infer, /refresh-capacity, /healthz
│   ├── config.example.json        # D_tt, D_BER, 임계치 등
│   └── requirements.txt
│
├── user/                          # 검증 클라이언트
│   ├── Dockerfile                 # python:3.10-slim + numpy + httpx
│   ├── client.py                  # CLI: --num-requests, --shuffle, ...
│   └── requirements.txt
│
├── parameter/                     # (gitignore) 학습 weight + 정확도 lookup
│   ├── ResNet_ImageNet_tf.data-00000-of-00001
│   ├── ResNet_ImageNet_tf.index
│   └── acc/
│       ├── acc_0.npy ~ acc_4.npy            # 1-split lookup
│       └── acc_0_1.npy ~ acc_3_4.npy        # 2-split lookup
│
├── data/                          # (gitignore) 검증 텐서 ~1GB
│   ├── X_val_s.npy                # (20000, 64, 64, 3) float32 [0,1]
│   └── y_val_encoded.npy          # (20000,) int32, 200 클래스
│
└── resnet34_TinyImageNet/         # 원본 시뮬레이션 코드 (참고용)
    ├── GSPDA_resnet.py
    ├── CRMC_eval_*.py
    ├── Resnet_SC.py
    └── ...
```

---

## API

### orchestrator (포트 8080)

- `GET /healthz` — `{ok, servers, D_tt, D_BER}` 반환
- `POST /infer` — 본문 `{tensor_b64, tensor_dtype, tensor_shape}`. 응답 `{predicted_class, logits, end_to_end_latency_ms, compute_ms_per_hop, plan}`
- `POST /refresh-capacity` — 캐시된 `D_C` 무효화 후 재조회. 실험 중 `COMPUTE_POWER`를 바꿀 때

### device (포트 8000, 내부 전용)

- `GET /capacity` — `{device_id, compute_power}` 반환
- `POST /run` — multipart (`metadata` JSON form + `tensor` binary file). plan에 따라 자기 slice 실행 후 다음 hop으로 forward, 마지막 hop은 logits을 raw bytes로 응답

---

## 설정

### `orchestrator/config.example.json`

```json
{
  "servers": ["server-1", "server-2", "server-3"],
  "D_tt": [1000, 1000, 1000],   // [s1↔s2, s2↔s3, s1↔s3]
  "D_BER": [10, 10, 10],         // 같은 ordering
  "acc_thresh": 0.49,
  "energy_thresh": 500,
  "server_port": 8000
}
```

### 컨테이너별 주요 env (compose에서 주입)

| 컨테이너 | env | 의미 |
|---|---|---|
| server-* | `DEVICE_ID` | `server-1` / `-2` / `-3` |
| server-* | `COMPUTE_POWER` | 시뮬 `D_C[i]` 값 (3 / 25 / 50) |
| server-* | `WEIGHT_PATH` | TF checkpoint base path (`/weights/ResNet_ImageNet_tf`) |
| server-* | `NUM_CLASSES`, `INPUT_SHAPE` | 모델 출력/입력 차원 (`200`, `64,64,3`) |
| orchestrator | `CONFIG_PATH` | `/app/config.json` |
| orchestrator | `CRMC_ACC_DATA_DIR` | 정확도 lookup `.npy` 디렉터리 (`/data/acc`) |
| user | `ORCHESTRATOR_URL` | `http://orchestrator:8080` |
| user | `NUM_REQUESTS`, `SHUFFLE`, `SEED` | 검증 루프 제어 |

---

## 데이터 준비 (gitignore된 파일)

레포에는 안 올라가지만, 컨테이너가 마운트 받아야 동작합니다.

### `parameter/` — 학습 weight + 정확도 lookup
- `ResNet_ImageNet_tf.data-00000-of-00001` (~85 MB)
- `ResNet_ImageNet_tf.index`
- `acc/acc_*.npy` × 16개

### `data/` — TinyImageNet 검증 텐서
- `X_val_s.npy`: (20000, 64, 64, 3) float32, 이미 [0,1] 정규화
- `y_val_encoded.npy`: (20000,) int32, 클래스 0..199

`docker-compose.yml`은 이 두 폴더를 그대로 read-only 볼륨 마운트합니다 — 폴더 위치만 맞으면 별도 설정 없이 바로 동작합니다.

---

## 향후 작업

- [ ] 정책 비교: `low_BER`, `com_best`, `min_inter` 도 `policy/` 로 포팅하여 GSPDA와 latency/accuracy 비교 리포트
- [ ] `D_tt` 실제 throttling: `tc qdisc rate` 적용해서 시뮬값과 실측 일치 검증
- [ ] 자동 sweep: `D_tt` / `COMPUTE_POWER` 격자 실험 + 결과 누적 → `results/` 에 CSV
- [ ] CIFAR-10 weight (`parameter/ResNet_tf.h5`) 도 옵션으로 지원 (소규모 빠른 실험용)

---

## 참고

- 시뮬레이션 원본: <https://github.com/bokyeong0405/CRMC> (`resnet34_TinyImageNet/`)
- 모델: ResNet34, 데이터셋: TinyImageNet (200 classes)
