# H100 8-GPU 실측 결과 (node7, 2026-10-16)

한 노드 H100 80GB HBM3 8장(node7)에서 실행한 결과만 포함한다. 다른 노드에서 실행한 결과는 포함하지 않는다. 각 실험은 고정 revision 원본 가중치에서 새로 시작했고 첫 2 steps는 warmup으로 통계에서 제외했다.

## 환경

| 항목 | 값 |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 × 8 (NVLink NV18), 1 노드 |
| CPU / RAM | Intel Xeon Platinum 8468 2소켓 × 48코어, 2,015.5 GiB |
| Driver | 575.57.08 커널 드라이버 + CUDA 13.0 forward-compatibility libcuda 580.95.05 |
| Framework | verl 0.9.1 (`1876b06d`), torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.9.0, ray 2.55.1, flash-attn 2.8.3 |
| Benchmark 코드 | `e44796a0e6b8c82a36ade9e5b916d68dba18c2d3` (캠페인 `h100-8gpu-001`) |
| Workload | global batch 8, micro 1/GPU, BF16 FSDP2. SFT 길이 상한 4,096. GRPO prompt 2,048 + response 1,024, group 4, Llama rollout TP1 |
| 저장 정책 | 매 step batch 캡처, full checkpoint는 각 실험 마지막 step에만 저장 |

## 결과 요약

| 실험 | 상태 | 관측 steps | 처리량 tokens/s (전체 / GPU당) | step 시간 평균 / 중앙값 (s) | 최대 메모리 (GiB) |
| --- | --- | ---: | --- | --- | ---: |
| Llama-3.1-8B SFT 10 | 완료 | 8 | 46,080 / 5,760 | 0.703 / 0.704 | 10.73 |
| Llama-3.1-8B SFT 100 | 완료 | 98 | 43,987 / 5,498 | 0.733 / 0.693 | 10.73 |
| Llama-3.1-8B GRPO 10 | 완료 | 8 | 416 / 52 | 28.19 / 27.02 | 11.56 |
| Llama-3.1-8B GRPO 100 | **중단 (87/100 step)** | 85 | 482 / 60 (부분) | 25.58 / 25.28 | 11.57 |
| Qwen3.8-27B SFT 10 | 실패 | 0 | — | — | — |
| Qwen3.8-27B GRPO 10 | 실패 | 0 | — | — | — |

- SFT 처리량은 unpadded 입력 토큰(마스킹된 prompt 포함), GRPO는 prompt + 생성 응답 토큰 기준이다.
- SFT step 시간은 CUDA 동기화 구간(`timing_s/train_step`), GRPO는 native `timing_s/step`(rollout·보상·업데이트·가중치 동기화 포함)이다.
- 메모리는 PyTorch peak allocated(GRPO는 actor)이며 vLLM을 포함한 GPU 전체 사용량이 아니다.
- 1,000-step 단계는 실행하지 않았다.

### GRPO step 구성 (warmup 제외 평균, s)

| 실험 | 생성 | actor 업데이트 | old log-prob | ref |
| --- | ---: | ---: | ---: | ---: |
| Llama GRPO 10 | 17.75 | 3.08 | 1.35 | 1.16 |
| Llama GRPO 100 (부분) | 16.59 | 2.99 | 1.29 | 1.17 |

### GRPO 보상 (재계산값, `reward_audit_corrected.json`)

| 실험 | 응답 수 | 정답률 | 형식 통과 | 보상 차이가 있는 그룹 |
| --- | ---: | ---: | ---: | ---: |
| Llama GRPO 10 | 256 | 0.0 | 1.17% | 3 / 64 |
| Llama GRPO 100 (부분) | 2,720 | 0.0 | 2.13% | 56 / 680 |

원래 `summary.json`의 보상 감사는 응답 안의 U+0085 문자에서 JSONL 행을 잘못 나누는 결함이 있어 255개 응답·깨진 행 7개로 기록됐다. 원본은 수정하지 않았고, 줄바꿈("\n") 기준으로 다시 계산한 값을 별도 파일로 남겼다. GRPO는 단일 턴 코드 출력 예측 대체 과제이며 전체 코딩 에이전트 비용이나 수렴을 나타내지 않는다.

## 실패·중단 기록

- **Llama GRPO 100 중단**: 사용자는 1,000-step 단계만 실행하지 말라고 지시했으나, 에이전트가 캠페인 작업 전체(Slurm job 2983)를 즉시 종료해 87 step에서 중단됐다. 장비나 프레임워크 오류가 아니다. 최종 checkpoint와 `summary.json`이 없고 `manifest.json`은 `running`으로 남아 있다. 표의 값은 남은 `metrics.jsonl`에서 계산한 부분 결과다. 같은 조건의 재측정을 node7에 대기시켜 두었다.
- **Qwen SFT 10 실패**: pinned verl의 턴별 SFT 토큰화가 Qwen3.8 chat template과 맞지 않았다. system 턴 단독 렌더링이 `No user query found in messages`로 실패했다. 또 user 질의가 없는 턴은 기본 system 블록(42 tokens)이 두 번 제거돼 assistant 턴 앞부분이 잘렸다.
- **Qwen GRPO 10 실패**: rollout(TP4)·보상·old/ref log-prob·advantage까지 step 1이 진행된 뒤 actor backward에서 실패했다. fla gated-delta-rule의 tilelang 커널 JIT 컴파일이 원인이다. 잠금 환경의 nvcc 13.2와 CUDA runtime 13.0 헤더가 맞지 않아 CCCL이 `CUDA compiler and CUDA toolkit headers are incompatible`로 거절했다.
- 두 Qwen 오류의 수정(benchkit adapter, `CUDA_HOME=/usr/local/cuda-13.0`)을 적용한 node7 재측정은 이 커밋 시점에 진행 중이다.

## 파일

- `measurements-node7.csv`: 위 표의 원자료
- `campaign-h100-8gpu-001/campaign.json`: 캠페인 상태(종료 시점 기록)
- `campaign-h100-8gpu-001/reports/steps-0010/`: 10-step 단계 자동 보고서(표·그림·artifact 색인)
- `campaign-h100-8gpu-001/steps-*/<run>/`: 실행별 `summary.json`, `manifest.json`, `config.json`, native `metrics.jsonl`, 데이터·모델 provenance, artifact inventory
- `pre-campaign-smoke/llama8b-sft-smoke-001/`: 이전 지침의 Llama SFT 2-step smoke(seq 512, 매 step full checkpoint). 별도 workload이며 성능 결과가 아니다.

체크포인트, 캡처 payload, 생성 응답(rollouts), 원시 framework·Ray 로그는 용량과 검토 전 모델 출력 때문에 포함하지 않았다. 원본은 클러스터의 결과 폴더에 보존되어 있다.
