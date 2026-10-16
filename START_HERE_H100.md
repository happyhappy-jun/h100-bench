# H100 8GPU 실행: 10 → 100 → 1000 steps

**이미 이전 명령으로 작업 중이라면 [H100_CONTINUE.md](docs/H100_CONTINUE.md)를 먼저 읽는다.** 현재 작업을 보존하면서 최신 코드·지침으로 전환하는 절차와 실행 명령이 있다.

기본 실행은 **한 노드 H100 8장으로 Llama-3.1-8B·Qwen3.8-27B의 SFT·GRPO 네 가지 실험을 수행**한다. 네 가지를 모두 10 steps로 실행한 뒤 100 steps, 마지막으로 1000 steps를 실행한다. **2시간이 넘어도 전체 단계 완료를 우선**한다.

사전 소형 모델 smoke, 1·2·4GPU 시험, scaling sweep, 반복 실험을 선행 조건으로 두지 않는다. 각 단계가 끝나면 보고서를 갱신하므로 먼저 끝난 결과부터 확인할 수 있다. 네 가지 모두 1000 steps를 완료하는 데 걸리는 시간은 아직 측정하지 않았다.

## 검증 범위와 장비

- A6000에서 실제 verl SFT·GRPO와 정확한 Meta-Llama-3.1-8B 실행 기록이 있다.
- H100용 pinned native 버전, Qwen3.8-27B, 새 캡처 훅의 H100 실행은 미검증이다.
- GRPO는 단일 턴 코드 출력값 예측 대체 과제다. 전체 저장소 수정·도구 호출 에이전트 비용을 측정하지 않는다.

| 항목 | 기준 |
| --- | --- |
| GPU | 한 노드 H100 80GB 8장, NVLink 권장 |
| OS / driver | Linux, NVIDIA driver 580.65.06 이상 |
| RAM / CPU | RAM 1TB 권장, Ray worker에 충분한 CPU 할당(예: 64 cores) |
| 환경 | Python 3.12, `uv`, git, 별도 학습·보고서 가상환경 |
| 모델 접근 | Meta-Llama-3.1-8B 접근 권한과 Qwen/Qwen3.8-27B 다운로드 가능 |
| 저장 공간 | 모델·환경 캐시 300GB 이상과 결과 공간. 아래 정책에 따라 quota와 여유 공간 확인 |

## 1. 환경과 모델을 먼저 준비

GPU 작업은 할당된 노드에서 실행한다. 로그인 노드에서 학습하지 않는다. Hugging Face 인증은 정상 credential store를 사용하고 토큰을 명령·로그·Git에 쓰지 않는다.

```bash
git clone --branch benchmark/framework-evidence https://github.com/happyhappy-jun/h100-bench.git
cd h100-bench
git rev-parse HEAD
nvidia-smi
df -h .

# uv 설치 안내: https://docs.astral.sh/uv/getting-started/installation/
bash scripts/setup_verl.sh h100
source .venv-verl-h100/bin/activate

# 고정 revision 모델을 실험 시작 전에 캐시한다.
python -m benchkit.campaign prepare --output output/h100-prepared.json
```

준비 결과의 revision·캐시 경로와 오류를 확인한다. 모델 준비 실패를 해결한 뒤 실험을 시작한다. 실행 프로필은 캐시만 사용하며 실험 중 모델을 새로 다운로드하지 않는다. 설치·다운로드·스케줄러 대기와 실제 실험 시간은 구분한다.

## 2. 전체 단계를 바로 실행

```bash
BENCH_PYTHON="$PWD/.venv-verl-h100/bin/python" \
  bash scripts/run_h100.sh results/h100-8gpu-001
```

동일한 기본 campaign을 Python으로 직접 실행할 수도 있다.

```bash
python -m benchkit.campaign run \
  --output results/h100-8gpu-002 \
  --steps 10 100 1000 \
  --case-timeout-hours 24
```

두 명령은 대안이다. 같은 실험을 중복 실행하지 않는다. 결과 폴더는 매 campaign마다 새 이름을 쓴다. 기본값에는 전역 2시간 제한이 없다. 개별 attempt의 24시간 watchdog은 무한 정지를 막는 상한이며 예상 실행 시간이 아니다.

| 단계 | 실행 순서 | GPU | warmup 제외 후 목표 관측 steps |
| --- | --- | --- | --- |
| 10 steps | Llama SFT → Llama GRPO → Qwen SFT → Qwen GRPO | 각 실행 8 | 8 |
| 100 steps | 동일한 네 가지 | 각 실행 8 | 98 |
| 1000 steps | 동일한 네 가지 | 각 실행 8 | 998 |

각 stage·model·mode 조합은 **고정된 원본 모델에서 시작하는 독립 attempt**다. 10-step checkpoint를 이어서 100 steps까지 학습하는 방식이 아니다. 성공 기준은 실제 관측 step 수와 종료 상태로 확인한다. 실패나 시간 초과를 완료로 표시하지 않는다.

| 프로필 | 길이 | batch | GRPO rollout |
| --- | --- | --- | --- |
| `h100_quick_llama8b_sft` | SFT 4,096 | global 8, micro 1/GPU | 해당 없음 |
| `h100_quick_llama8b_grpo` | prompt 2,048 + response 1,024 | global 8, micro 1/GPU | group 4, TP 1 |
| `h100_quick_qwen27b_sft` | SFT 4,096 | global 8, micro 1/GPU | 해당 없음 |
| `h100_quick_qwen27b_grpo` | prompt 2,048 + response 1,024 | global 8, micro 1/GPU | group 4, TP 4 |

파일명의 `quick`은 짧은 문맥의 workload를 식별한다. 전체 1000-step campaign이 빠르게 끝난다는 보장은 아니다. 제안서의 장문맥 전체 조건 재현과도 구분한다.

## 3. 캡처와 저장 정책

매 step batch payload와 중간 지표를 최대한 보존한다. **Full checkpoint는 각 attempt의 마지막 step에만 저장**한다. model·optimizer·extra를 요청하며 생성된 파일은 자동 삭제하지 않는다. 중간 가중치 전체를 저장하는 정책은 아니다.

계획상 full checkpoint당 16 bytes/parameter를 적용하면 Llama 약 128GB, Qwen 약 432GB다. 네 가지 실험의 한 단계는 약 1.12TB, 세 단계가 모두 끝나면 약 **3.36TB + 임시 파일·캡처·캐시·여유 공간**이 필요할 수 있다. 이는 저장 계획용 가정이며 실제 사용량이나 상한을 보장하지 않는다. 1000-step payload 누적량도 별도로 고려한다. 실패한 attempt에는 최종 checkpoint가 없을 수 있다.

기존 `configs/h100_llama8b*.json`, `configs/h100_qwen27b*.json`의 매 step full-checkpoint 프로필도 남아 있다. 이는 별도 실행이며 기본 staged campaign의 선행 조건이 아니다. 훨씬 큰 저장 공간이 필요할 수 있다.

SFT compute 구간, RL iteration, 전체 벽시계, capture·hash 시간을 구분한다. Native RL의 마지막 iteration은 checkpoint 저장 시간을 포함할 수 있으므로 평균을 compute-only 시간으로 표시하지 않는다.

## 4. 단계별 결과 확인

각 단계 종료 시 보고서를 갱신하고 전체 종료 시 최종 보고서를 남긴다. 완료·실패·시간 초과·미시작 상태와 사용 가능한 관측값을 함께 전달한다. 누락값을 0으로 채우지 않는다. 오류가 생기면 실패 원본을 유지하고 새 attempt로 수정·재시도한다.

Campaign 루트의 `campaign.json`에서 상태를, `latest_report.json`에서 최신 성공 보고서 위치를 확인한다. 단계 보고서는 `reports/steps-0010`, `reports/steps-0100`, `reports/steps-1000`에 있고 최종 보고서는 `report/report.md`다. 보고서 생성 자체가 실패하면 해당 로그와 남은 원시 증거를 전달한다. 이전 단계 실패로 후속 단계가 차단된 항목도 완료로 표시하지 않는다.

`report.md`, CSV, 그림, raw artifact index와 실행 원본을 확인한다. 필요하면 보고서만 별도로 생성할 수 있다.

```bash
.venv-report/bin/python -m benchkit.cli report \
  --input results/h100-8gpu-001 --output output/h100-report
```

별도로 시간제한 실행을 요청한 경우에만 `--budget-minutes 100`을 사용한다. 이 선택은 전체 1000-step 완료를 보장하지 않으며, 제한 시간의 부분 결과를 반환한다. 현재 기본 요청은 전체 단계 완료다.

오류 대응은 [에이전트 요청문](docs/H100_AGENT_PROMPT.md), [handoff](docs/AGENT_HANDOFF.md), [런타임 계약](docs/RUNTIME.md)을 따른다. 증거 범위는 [검증 기록](docs/VALIDATION.md), 방법·지표 관계는 [방법 대응표](docs/METHOD_MAPPING.md)에 있다.
