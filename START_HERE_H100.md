# H100 실행 패키지: 여기서 시작

이 저장소를 받은 에이전트는 [실행 요청문](docs/H100_AGENT_PROMPT.md)과 [handoff](docs/AGENT_HANDOFF.md)를 읽고 설치, 짧은 실행, 수정, 본 측정, 보고서 생성을 이어간다. 비공개 Slack이나 제안서 원본 없이 실행할 수 있다.

## 현재 상태

- 실제 verl SFT와 GRPO는 A6000에서 Qwen2.5-0.5B와 정확한 Meta-Llama-3.1-8B로 실행했다.
- H100, Qwen3.8-27B, 새 최대 중간 산출물 캡처의 GPU 실행은 미검증이다.
- 최대 캡처 코드는 CPU 계약 검사와 두 pinned verl 버전의 Hydra 구성 검사를 통과했다. GPU에서의 수정·재시도는 다음 에이전트가 수행한다.
- GRPO는 코드 출력값 예측이라는 단일 턴 대체 과제다. 전체 저장소 수정·도구 호출 에이전트의 비용 실측으로 보고하지 않는다.

## 준비할 서버

| 항목 | 기준 |
|---|---|
| GPU | 한 노드 H100 80GB SXM 8장, NVLink 권장 |
| OS / driver | Linux, NVIDIA driver 580.65.06 이상 |
| RAM / CPU | RAM 1TB 권장, CPU는 Ray worker에 충분히 할당(예: 64 cores) |
| 환경 | Python 3.12, `uv`, git; 설치 스크립트가 별도 가상환경 구성 |
| 모델 접근 | Hugging Face의 Meta-Llama-3.1-8B 접근 권한; Qwen/Qwen3.8-27B 다운로드 가능 |
| 저장 공간 | 모델·환경 캐시 300GB 이상 + 결과 공간. 아래 체크포인트 정책을 먼저 확인 |

기본 설정은 매 step 입력·출력·RL 중간 batch와 전체 model/optimizer/extra 체크포인트를 보존한다. 자동 삭제하지 않는다. 체크포인트당 16 bytes/parameter로 계획하면 Llama 8B 약 128GB, Qwen 27B 약 432GB다. 두 모델 × 두 모드 × 12 steps는 **약 13.4TB + 임시 파일·캡처·캐시·여유 공간**이 필요할 수 있다. 이는 저장 계획용 추정이며 상한 보장이 아니다. 실제 quota도 확인한다.

공간이 부족하면 조용히 저장을 끄지 않는다. 별도 실행 이름과 명시적인 `--set checkpoint_every_steps=12` 같은 정책으로 최종 체크포인트만 저장할 수 있다. 이 경우 매 step batch 캡처는 유지되지만 중간 가중치가 모두 남는 실행은 아니다. 변경 정책을 결과에 함께 기록한다.

## 설치와 첫 실행

GPU 작업은 할당된 노드에서 실행한다. Slurm 클러스터에서는 로그인 노드에서 학습하지 않는다.

```bash
git clone --branch benchmark/framework-evidence https://github.com/happyhappy-jun/h100-bench.git
cd h100-bench
git rev-parse HEAD
nvidia-smi
df -h .

# uv가 없다면 공식 설치 안내: https://docs.astral.sh/uv/getting-started/installation/
bash scripts/setup_verl.sh h100
source .venv-verl-h100/bin/activate
python -m benchkit.cli doctor --config configs/h100_llama8b.json

# 네트워크/GPU 학습 없이 명령 구성 확인
python -m benchkit.cli run --config configs/h100_llama8b.json \
  --output results/llama8b-plan --dry-run

# 짧은 Llama SFT: 2 optimizer steps 중 첫 step은 warmup
# 본 측정과 구분되는 짧은 workload이며 최대 캡처는 유지된다.
python -m benchkit.cli run --config configs/h100_llama8b.json \
  --set steps=2 --set warmup_steps=1 --set seq_length=512 \
  --output results/llama8b-sft-smoke-001

# 짧은 Llama GRPO
python -m benchkit.cli run --config configs/h100_llama8b_grpo.json \
  --set steps=2 --set warmup_steps=1 --set seq_length=512 \
  --set prompt_length=256 --set response_length=256 \
  --output results/llama8b-grpo-smoke-001
```

모델 다운로드는 최초 실제 실행 시 고정 revision으로 수행한다. Llama 인증은 정상 Hugging Face 인증 경로를 사용하고 토큰을 명령·로그·Git에 쓰지 않는다.

짧은 실행에서도 `manifest.json`, `metrics.jsonl`, `artifacts/manifest-*.jsonl`, payload, `checkpoints/`를 확인한다. payload 해시와 shape/dtype를 확인하고 신뢰한 이 실행의 payload만 decode한다. 실제 checkpoint 수와 내용을 확인한다. 모델 경로 symlink는 원본 모델 캐시를 가리키므로 결과 폴더만 옮기면 끊길 수 있다.

## 본 측정과 보고서

```bash
# Llama 8B / Qwen 27B 각각 SFT와 GRPO. 실패한 case도 보존한다.
BENCH_PYTHON="$PWD/.venv-verl-h100/bin/python" \
  bash scripts/run_h100.sh results/h100-full-001

.venv-report/bin/python -m benchkit.cli report \
  --input results/h100-full-001 --output output/h100-report
```

4개 case 각각 12 steps이며 첫 2 steps를 통계에서 제외한다. 충분한 반복과 GPU 수별 확장성 측정은 본 실행 성공 후 별도 attempt로 추가한다. 짧은 성능 실행을 수렴·문제 해결 성능 검증으로 해석하지 않는다.

Slurm에서는 사이트의 partition/account/QoS에 맞춰 `scripts/slurm_bench.sbatch`를 사용한다. `BENCH_PYTHON`은 H100 가상환경의 절대 경로로 지정한다. 다른 클러스터의 운영 경로나 QoS를 그대로 복사하지 않는다.

## 실패했을 때 읽을 문서

1. [AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md): 오류 분류, 수정·재시도 절차, 알려진 오류 대응표, 완료 조건.
2. [RUNTIME.md](docs/RUNTIME.md): pinned framework API, 계측 범위, 캡처 훅의 호환성 위험.
3. [VALIDATION.md](docs/VALIDATION.md): 실제로 검증한 범위와 아직 하지 않은 검증.
4. [METHOD_MAPPING.md](docs/METHOD_MAPPING.md): 제안서 방법과 측정 항목의 대응 및 대체 과제 한계.

새 폴더로 재시도하고 실패 원본을 유지한다. 모델·정밀도·배치·시퀀스·보상 변경은 별도 workload로 기록한다. native Ray/TransferQueue hook 문제는 먼저 pinned 소스의 함수 서명과 데이터 구조를 확인해 `benchkit`의 adapter를 수정한다. upstream 소스 자체를 바꾸면 현재 doctor가 거절하므로 조용히 bypass하지 않는다.
