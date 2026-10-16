# H100 에이전트에게 전달할 요청문

아래 내용을 H100 서버에서 작업할 에이전트에게 전달한다.

---

한 노드 H100 8GPU에서 Llama-3.1-8B와 Qwen3.8-27B 각각의 SFT·GRPO를 실행해 줘. `START_HERE_H100.md`, `AGENTS.md`, `docs/AGENT_HANDOFF.md`, `docs/RUNTIME.md`, `docs/VALIDATION.md`를 먼저 읽어.

환경과 고정 revision 모델 캐시를 먼저 준비해. `python -m benchkit.campaign prepare --output output/h100-prepared.json`을 사용해. 준비가 끝나면 `bash scripts/run_h100.sh results/h100-8gpu-001`로 시작해. 새 campaign에는 새 폴더 이름을 써. 설치·모델 다운로드·스케줄러 대기와 실제 실험 시간을 구분해.

기본 순서는 10 steps에서 네 가지 모두 → 100 steps에서 네 가지 모두 → 1000 steps에서 네 가지 모두야. 각 attempt는 원본의 고정 가중치에서 독립적으로 시작해. 이전 단계 checkpoint를 이어 학습하는 실험으로 바꾸지 마. 첫 2 steps는 warmup이므로 완료 시 관측 대상은 각각 8·98·998 steps야.

전체 단계 완료가 우선이고 2시간이 넘어도 계속해. 기본 전역 2시간 제한은 없어. 각 attempt의 24시간 watchdog은 무한 정지를 막는 상한이야. 1000-step 전체가 2시간 안에 끝난다고 약속하지 마. 단계가 끝날 때마다 보고서를 갱신하고 마지막에 최종 보고서를 남겨. 사전 소형 모델 smoke, 1·2·4GPU 시험, scaling sweep, 반복 실험을 추가 선행 조건으로 만들지 마.

현재 H100용 native 버전, Qwen27B, 새 캡처 훅의 H100 실행은 미검증이야. A6000 성공을 H100 성공으로 취급하지 마. GPU 할당·driver·RAM·quota·모델 접근을 확인하고 승인된 자원만 사용해. 인증정보를 로그·Git에 남기거나 다른 작업을 종료하지 마.

기본 workload는 8GPU, global batch 8, microbatch 1이야. SFT 길이는 4,096이고 GRPO는 prompt 2,048·response 1,024·rollout group 4야. 매 step batch payload와 중간 지표는 최대한 보존하되 full model·optimizer·extra checkpoint는 각 attempt의 마지막 step에만 저장해. 세 단계를 모두 완료하면 model·mode 조합마다 세 개의 final checkpoint가 남아. 생성한 기록은 자동 삭제하지 마. 기존 매 step full-checkpoint 프로필은 별도 실행을 위한 것이야.

입력·mask·token IDs·응답·보상·log probabilities·advantage·업데이트 경계 자료, native metrics, resolved config, source snapshot, GPU/host telemetry, Ray 로그와 artifact manifest를 보존해. 없는 필드를 있다고 주장하지 마. 현재 훅은 내부 activation 전체나 gradient tensor 전체를 저장하지 않아. 캡처 오류와 학습 성공을 구분해. 최종 checkpoint가 없는 실패 실행도 그대로 남겨.

설치·tokenizer·FSDP·vLLM·Ray·TransferQueue·계측 오류는 최초 오류와 pinned 소스로 진단해. 필요한 최소 수정 뒤 새 attempt로 재시도하고, 변경 연결을 attempt ledger에 기록해. 실패·시간 초과·미시작을 완료로 표시하지 마. 모델·배치·길이·dtype·offload·보상·캡처 정책 변경은 별도 workload로 표시해. 손실·보상·시간을 가짜 값으로 채우지 마.

SFT의 CUDA-fenced compute 구간, RL iteration, 전체 벽시계, 캡처와 사후 해시 시간을 구분해. Native RL 최종 iteration에 checkpoint 저장이 포함될 수 있어. GRPO는 단일 턴 코드 출력 예측 대체 과제이므로 전체 에이전트 비용으로 보고하지 마. 세 step 수의 실행을 GPU scaling이나 독립 반복 실험으로 해석하지 마.

별도로 시간제한을 요청받았을 때만 `--budget-minutes 100` 같은 옵션을 사용해. 그 모드는 부분 결과를 반환하며 전체 1000-step 완료를 보장하지 않아. 현재 요청에는 이 제한을 적용하지 마.

마지막에는 실제 완료한 항목, 실패·시간 초과·미시작 항목, 원본·보고서 경로와 남은 문제를 전달해. 결과는 `report.md`, `measurements.csv`, `stages.csv`, `scaling.csv`, `artifacts.md`, `artifacts.csv`, 가능한 PNG/SVG로 남겨. 일부가 실패해도 사용 가능한 관측값과 원시 증거를 먼저 보여 줘.
