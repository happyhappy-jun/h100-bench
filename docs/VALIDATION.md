# 검증 기록 — 2026-10-16

이 패키지는 H100 쪽 에이전트에게 먼저 인계하기 위해 전체 GPU 검증을 기다리지 않고 공개한다. 아래 실제 실행은 최대 중간 payload·매-step full checkpoint 보존 변경 **이전**이다. 새 캡처 설정의 처리량을 이 값으로 대신하지 않는다.

| 환경 / 모델 | 실제 실행 | 확인한 범위 |
|---|---|---|
| A6000 2장 / Qwen2.5-0.5B-Instruct | SFT·GRPO 각각 4 steps 완료 | 실제 pretrained 학습, gradient, vLLM generation, response reward, actor update, 반복별 weight sync |
| A6000 4장 / Meta-Llama-3.1-8B | SFT 4 steps 완료 | FSDP2 전체 가중치 학습, CUDA 동기화 계측 v2, 실제 token count |
| A6000 4장 / Meta-Llama-3.1-8B | GRPO 4 steps 완료 | base tokenizer template 수정 후 실제 generation/reward/advantage/update/weight sync, GPU telemetry |
| H100 / 두 모델 | 미실행 | 설치·설정·실행 코드와 handoff 제공 |
| 새 maximal payload / full checkpoint capture | GPU 미검증 | CPU 계약 검사와 pinned v0.8/v0.9.1 Hydra 구성 확인. 실제 capture와 checkpoint decode는 후속 에이전트가 확인 |

개발 시 관측값은 환경과 길이가 다른 참고 실측이다. H100 비용 추산의 검증 결과가 아니다.

| 실행 | 측정 step 수 | 평균 step 시간 | 추가 근거 |
|---|---:|---:|---|
| 소형 모델 SFT | 3 | 0.4425 s | aggregate 약 865.6 tokens/s |
| 소형 모델 GRPO | 3 | 6.6996 s | aggregate 약 540.3 tokens/s |
| Llama 8B SFT, seq 2048 | 3 | 7.3651 s | aggregate 약 2224.6 tokens/s, PyTorch peak 약 19.61 GiB |
| Llama 8B GRPO, prompt 1024 + response 1024 | 3 | 69.0908 s | actor PyTorch peak 약 19.36 GiB, vLLM 포함 GPU telemetry peak 약 42.93 GiB |

Llama GRPO는 측정 응답 96개, 완전한 그룹 24개 중 3개에서 보상 변동을 보였다. 정답률은 0%였다. 실제 학습 루프 실행과 일부 형식 보상 차이는 확인했지만 문제 해결 능력 개선이나 수렴을 입증하지 않는다.

실패 시도도 보존했다. Slurm visibility 충돌, base Llama tokenizer template, GPU UUID 정규화, native timer와 중첩 timing namespace 단위 문제를 수정했다. 과거 prefix caching 불일치 및 환경 변경이 섞인 시도는 review overlay로 비교에서 제외한다. 원본 summary를 고쳐서 성공 기록으로 바꾸지 않는다.

원시 실행 자료와 모델 출력은 운영 결과 디렉터리에 보존하고 공개 Git에는 포함하지 않는다. 후속 H100 에이전트는 자신이 생성한 원시 증거로 이 문서를 갱신해야 한다. 기존 A6000 수치만으로 H100 및 Qwen27B 성공을 선언하지 않는다.
