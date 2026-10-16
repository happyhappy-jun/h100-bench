# H100 에이전트에게 전달할 요청문

아래 내용을 H100 서버에서 작업할 에이전트에게 전달한다.

---

이 저장소의 H100 benchmark를 실행해 Llama-3.1-8B와 Qwen3.8-27B 각각의 실제 SFT·GRPO 성능 자료를 수집해 줘. 먼저 `START_HERE_H100.md`, `AGENTS.md`, `docs/AGENT_HANDOFF.md`, `docs/RUNTIME.md`, `docs/VALIDATION.md`를 읽어.

우선순위는 H100에서 실행 가능한 상태를 만들고 실제 측정 자료를 얻는 것이다. 설치·의존성·tokenizer·FSDP·vLLM·Ray·TransferQueue·계측 문제를 로그로 진단하고, 원인을 좁혀 코드를 수정한 뒤 새 attempt로 재시도해. 현재 H100 및 Qwen27B의 실제 실행과 새 최대 캡처 훅은 GPU 검증 전이다. A6000의 성공을 H100 성공으로 취급하지 마.

서버의 GPU 할당, driver, RAM, disk quota, 모델 접근 권한을 먼저 확인해. site에서 승인된 자원과 스케줄러를 사용하고 다른 작업을 종료하지 마. 짧은 Llama SFT·GRPO로 진입점을 확인한 뒤 Qwen27B와 본 프로필을 실행해. 인증정보가 로그·Git에 남지 않게 해.

중간 산출물은 최대한 보존해. 기본값은 매 step batch와 full checkpoint 보존이며 자동 삭제하지 않아. 입력·mask·token IDs·응답·보상·log probabilities·advantage·업데이트 경계 자료, native metric, resolved config, source snapshot, GPU/host telemetry, Ray/프로세스 로그, 체크포인트, artifact manifest와 SHA256를 확인해. 누락된 필드를 있다고 주장하지 말고 capture 오류를 실제 학습 성공과 별도로 표시해. 내부 activation 전체나 gradient tensor 전체까지 현재 hook이 저장하는 것은 아니야.

용량 부족이나 호환성 때문에 저장 정책·모델·배치·길이·dtype·offload·보상 등을 바꾸면 새 프로필과 변경 사유를 남겨. 손실·보상·시간을 가짜 값으로 채우거나 실패 원본을 덮어쓰지 마. capture와 checkpoint I/O가 포함된 실행 시간임을 기록하고 SFT의 CUDA-fenced compute 구간, RL iteration, 전체 작업 벽시계, 사후 파일 해시 시간을 구분해.

결과는 실행별 원시 증거와 `report.md`, `measurements.csv`, `stages.csv`, `scaling.csv`, `artifacts.md`, `artifacts.csv`, PNG/SVG로 남겨. H100 실측과 A6000 개발 결과·추산을 구분해. GRPO가 단일 턴 코드 출력 예측 대체 과제라는 한계를 유지해. 모든 실패와 수정·재실행의 연결을 attempt ledger에 기록하고 마지막에 실제 성공한 항목, 미완료 항목, 결과 경로를 알려 줘.
