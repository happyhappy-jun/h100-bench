# H100 SFT · verl GRPO benchmark

Llama-3.1-8B와 Qwen3.8-27B를 실제 학습 프레임워크에서 실행하고, 성능·확장성 검증자료에 필요한 표·그래프·원시 로그를 수집한다.

**H100 서버에서는 [START_HERE_H100.md](START_HERE_H100.md)부터 시작한다.** 후속 에이전트에게는 [실행 요청문](docs/H100_AGENT_PROMPT.md)과 [handoff](docs/AGENT_HANDOFF.md)를 함께 전달한다. H100 실측과 새 최대 캡처 기능의 GPU 검증은 아직 수행하지 않았다.

**이전 안내로 작업 중인 H100 에이전트는 [최신 전환 안내](docs/H100_CONTINUE.md)를 먼저 읽는다.** 기존 작업·수정 보존, 최신 checkout 준비, 재사용할 환경, 실행·재시도 명령을 한 문서에 정리했다.

- **SFT:** verl의 PyTorch FSDP2 trainer로 제공된 에이전트 궤적을 학습한다. 사전학습 가중치를 로드하고 실제 역전파·옵티마이저 업데이트를 수행한다.
- **GRPO:** verl + vLLM에서 응답 생성, 응답 기반 보상, 그룹 advantage, 참조 모델 KL, 정책 업데이트 및 rollout 가중치 전달을 수행한다.
- **RL 작업 범위:** 정답을 계산할 수 있는 Python 코드의 출력값 예측이다. 임의 advantage를 사용하지 않는다. 실제 저장소 수정·셸 도구·SWE-bench 채점이 포함된 전체 에이전트 실험은 아니다.
- **산출물:** 실행별 JSON/JSONL·설정·환경·GPU telemetry·로그, 집계 CSV·Markdown 보고서, PNG/SVG 그래프. 누락·실패 항목은 그대로 표시한다.

기존 루트의 `bench_train.py`, `bench_rollout.py`, `run_all.sh`, `setup.sh`, `report.py`는 ZIP 원본 계열의 **legacy** 코드다. 아래의 `benchkit` 명령이 새 프레임워크 벤치마크 진입점이다.

**기본 실행은 8 GPU 고정, 10 → 100 → 1,000 step 캠페인이다.** 각 단계에서 Llama/Qwen × SFT/GRPO 네 case를 실행한다. 2시간 제한은 두지 않고 완료를 우선한다. 매 step payload·지표는 보존하며 full checkpoint는 각 attempt 마지막에 저장한다.

[방법론·측정항목 대응표](docs/METHOD_MAPPING.md) · [후속 에이전트 handoff](docs/AGENT_HANDOFF.md)

## 모델과 환경

| 프로필 | 모델 | 모드 | 기본 GPU | 시퀀스 상한 |
|---|---|---|---|---|
| `configs/h100_quick_*_sft.json` | Llama 8B / Qwen27B | 기본 캠페인 SFT | H100 8장 | 4,096 |
| `configs/h100_quick_*_grpo.json` | 동일 두 모델 | 기본 캠페인 GRPO | H100 8장 | prompt 2,048 + response 1,024 |
| `configs/h100_llama8b.json` | Meta-Llama-3.1-8B | SFT | H100 8장 | 8,192 |
| `configs/h100_llama8b_grpo.json` | 동일 모델 | GRPO | H100 8장 | prompt 4,096 + response 4,096 |
| `configs/h100_qwen27b.json` | Qwen3.8-27B | SFT | H100 8장 | 16,384 |
| `configs/h100_qwen27b_grpo.json` | 동일 모델 | GRPO | H100 8장 | prompt 8,192 + response 8,192 |
| `configs/a6000_smoke.json` / `_grpo.json` | Qwen2.5-0.5B-Instruct | 개발용 SFT / GRPO | A6000 2장 | 512 |
| `configs/a6000_llama8b_sft.json` / `_grpo.json` | Meta-Llama-3.1-8B | 개발용 SFT / GRPO | A6000 4장 | 2,048 |

시퀀스 길이는 상한이다. 보고서의 처리량에는 설정값으로 채운 토큰 수가 아니라 프레임워크가 기록한 실제 토큰 수를 사용한다. Llama base 모델에는 이 벤치마크에서 명시한 [Llama 3 역할 템플릿](templates/llama3_role.jinja)을 적용한다. 이전 Llama 기록은 24 GPU DeepSpeed 계속 사전학습이며, 여기서 수행하는 SFT·GRPO와 별도 실험이다.

H100 기본 대상은 **80GB SXM × 8, NVLink, Linux 서버 1대**다. 모델·환경 캐시용 여유 디스크 300GB 이상을 별도로 확보하고, 호스트 RAM은 원본 키트 기준 1TB를 권장한다. 기본 캠페인의 최종 체크포인트 12개는 약 3.36TB + payload·임시 파일·여유 공간으로 계획한다(16 bytes/parameter/checkpoint 가정, 실측이나 상한 보장 아님). 기존 장문맥 full 프로필의 매-step 체크포인트 정책은 약 13.4TB + 여유 공간을 요구할 수 있다. 실제 요구량은 결과의 환경·메모리 기록으로 확인한다. 기본 실행은 단일 노드이며 대규모 다중 노드·MoE 검증은 포함하지 않는다.

## 설치

두 환경은 설치·실행 기록을 구분한다. H100의 기준 환경은 **verl v0.9.1의 공식 lock**이고, A6000 개발 환경은 기존 CUDA 12.8 드라이버에 맞춘 **verl v0.8.0 호환 환경**이다. A6000 검증을 H100 환경의 실행 성공으로 취급하지 않는다.

```bash
git clone --branch benchmark/framework-evidence https://github.com/happyhappy-jun/h100-bench.git
cd h100-bench

# uv가 설치되어 있어야 한다. H100 기준 환경은 CUDA 13 호환 드라이버가 필요하다.
bash scripts/setup_verl.sh h100
source .venv-verl-h100/bin/activate
```

설치 스크립트는 프레임워크 commit과 dependency lock을 고정한다. 그래프 생성용 패키지는 별도 `.venv-report`에 설치해 학습 환경의 NumPy·CUDA 패키지를 바꾸지 않는다. 기준 H100 환경의 드라이버 요구사항을 충족하지 못하면 중단한다. 설치·모델 접근 문제를 임의 버전 강제 설치나 모델 대체로 숨기지 않는다. Llama 가중치는 Hugging Face에서 해당 모델 접근 권한이 필요하다. 인증은 서버의 정상 Hugging Face 로그인/토큰 설정을 사용하며 로그에 토큰을 남기지 않는다.

A6000 개발 환경은 별도 설치한다. flash-attn 소스 빌드에는 해당 CUDA toolkit이 필요하며 `FLASH_ATTN_WHEEL`로 호환 wheel을 지정할 수 있다.

```bash
bash scripts/setup_verl.sh a6000-dev
source .venv-verl-a6000-dev/bin/activate
```

## 실행과 결과 확인

```bash
# 설치 후, 측정 시작 전에 고정 revision의 모델을 미리 받는다.
python -m benchkit.campaign prepare --output output/h100-prepared-001.json

# GPU 학습 없이 12개 case와 실행 순서 확인
python -m benchkit.campaign run --output results/h100-001 --dry-run

# 할당된 8 H100에서 10 → 100 → 1,000 steps를 끝까지 진행
bash scripts/run_h100.sh results/h100-001
```

순서는 10 step에서 두 모델 × 두 모드, 이어서 100 step의 네 case, 마지막으로 1,000 step의 네 case다. 각 attempt는 동일하게 고정한 원본 가중치에서 새로 시작하며 앞선 checkpoint를 이어 학습하지 않는다. 워밍업 2 steps를 제외하므로 측정 step 수는 8/98/998이다. 기본 전체 캠페인 시간 제한은 없다. 개별 case의 24시간 watchdog은 `--case-timeout-hours`로 바꿀 수 있다. GPU 수별 확장성 스윕이나 반복 측정을 선행 조건으로 요구하지 않는다.

```bash
# 코드 수정 뒤 실패했던 모델/모드/단계만 새 결과 폴더에서 재시도
bash scripts/run_h100.sh results/h100-retry-001 \
  --models qwen27b --modes grpo --steps 100 1000

# 이전 장문맥 12-step/매-step full checkpoint 프로필은 별도 진입점 유지
bash scripts/run_h100_full.sh results/h100-long-context-001
```

기본 캠페인은 global batch 8, SFT 4,096 tokens, GRPO prompt 2,048 + response 1,024 상한을 사용한다. 원래 8K/16K 장문맥 프로필과 같은 workload로 합치지 않는다. Qwen27B rollout은 TP4, Llama rollout은 TP1이다. 실제 token 수와 프레임워크 설정을 보고서에 기록한다.

case가 실패하면 해당 모델·모드의 더 긴 단계를 `blocked`로 기록하고 다른 case를 진행한다. 동일한 실패 명령을 무작정 반복하지 않는다. 원인을 수정하고 새 캠페인 경로로 재시도한다. 자신의 Ray·학습 자식 프로세스를 정리하지 못하면 다음 case를 시작하지 않는다.

주요 출력:

- `campaign.json`: 모든 계획·시도·완료·실패·차단 상태와 실제 경과 시간
- `reports/steps-0010/`, `steps-0100/`, `steps-1000/`: 단계별 보고서
- `report/report.md`: 최종 통합 보고서. `latest_report.json`은 가장 최근 성공한 보고서 위치
- 실행별 `metrics.jsonl`, `summary.json`, `framework-0.log`, `gpu.csv`, `host_process.jsonl`, `source/`, `packages.json`
- 실행별 `artifacts/`, `rollouts/`, `checkpoints/`, `artifact_inventory.json`: 중간 batch·응답·보상·advantage·최종 model/optimizer/extra와 파일 목록

원시 payload는 매 step 보존한다. 전체 체크포인트는 각 독립 attempt의 마지막 step에만 저장하며 삭제하지 않는다. 내부 activation 또는 모든 gradient tensor를 저장하는 것은 아니다. 큰 checkpoint 파일의 사후 전체 해시는 기본 캠페인에서 유보하고 크기·경로를 기록한다. payload 캡처 manifest의 SHA256는 별도 유지된다. 전체 파일 해시는 측정 후 아래처럼 추가한다.

```bash
python -m benchkit.run_evidence results/h100-001/steps-0010/llama8b-sft
.venv-report/bin/python -m benchkit.cli report \
  --input results/h100-001 --output output/h100-report
```

SFT의 CUDA-fenced compute 시간, RL의 native iteration 시간, 전체 case 벽시계, 캡처 및 checkpoint 시간을 구분한다. 마지막 GRPO step에는 native checkpoint 저장 시간이 포함될 수 있으므로 `timing_s/save_checkpoint`도 확인한다. 10/100/1000 단계는 실행 길이 비교이며 8 GPU 한 점으로 GPU 확장성 검증을 주장하지 않는다. 정답률 0% 또는 보상 변동 부족은 그대로 기록한다. 생성된 응답과 로그는 공개 저장소에 자동 업로드하지 않는다.

## Slurm 개발 검증

저장소 루트에서 제출한다. SSH 호스트·계정·QoS는 로컬 운영 설정에서 지정한다. 개발용 QoS는 사용자가 승인한 것을 사용하고 다른 작업을 직접 종료하지 않는다.

```bash
# 실제 클러스터의 ACCOUNT/QOS로 바꿔 사용
BENCH_PYTHON="$PWD/.venv-verl-a6000-dev/bin/python" \
  sbatch --partition=a6000 --account=ACCOUNT --qos=QOS \
  --gres=gpu:2 --cpus-per-task=16 --mem=96G --time=01:00:00 \
  scripts/slurm_bench.sbatch configs/a6000_smoke.json results/a6000-sft-001
```

GRPO는 `configs/a6000_smoke_grpo.json`을 사용한다. GPU 검증은 로그인 노드에서 실행하지 않는다. GPU 이상·스토리지 장애 노드는 운영 확인에 따라 `--exclude`로 제외한다. 각 attempt의 job ID와 실패 원인을 함께 보존한다.

## 확장성·반복 측정

같은 모델·가중치·환경·데이터·전역 배치·시퀀스·알고리즘을 유지하고 GPU 수만 바꿔 실행한다. 예를 들어 Llama SFT의 `global_batch_size=16`을 유지한 2/4/8 GPU 비교:

```bash
python -m benchkit.cli run --config configs/h100_llama8b.json \
  --set gpus=2 --output results/llama8b-sft-g2-r1
python -m benchkit.cli run --config configs/h100_llama8b.json \
  --set gpus=4 --output results/llama8b-sft-g4-r1
python -m benchkit.cli run --config configs/h100_llama8b.json \
  --set gpus=8 --output results/llama8b-sft-g8-r1
```

작은 GPU 구성에서 OOM이 나면 실패를 남긴다. 배치·시퀀스 축소는 별도 workload이며 기존 strong-scaling 곡선에 합치지 않는다. 여러 번 실행한 값은 모두 보존하며 최고값만 선택하지 않는다. 보고서는 비교 조건과 모델·프레임워크 revision이 일치한 실행만 scaling에 사용한다. 32K SFT는 `--set seq_length=32768`로 별도 실행한다.

## 개발 체크

```bash
python -m unittest discover -s tests -v
bash -n scripts/setup_verl.sh scripts/run_h100.sh scripts/slurm_bench.sbatch
```

CPU 테스트·dry-run 통과는 GPU 실행 성공을 뜻하지 않는다. 실제 검증 상태는 실행별 manifest와 [검증 기록](docs/VALIDATION.md)에 남긴다.

`data/sessions.jsonl`은 원본 키트에 포함된 공개 SWE-bench 과제 기반 에이전트 대화 80건이다. 과제 본문·인용 코드 등 제3자 자료의 권리는 각 원저작자에게 있다. 비공개 제안서·Slack 원문은 이 저장소에 포함하지 않는다.
