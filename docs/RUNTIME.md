# 프레임워크 실행 계약과 측정 범위

이 문서는 벤치마크 코드가 실제로 호출하는 API와 시간 측정 범위를 기록한다. H100 및 Qwen3.8-27B의 실행 성공은 해당 장비에서 생성한 결과로 확인해야 한다. A6000 소형 모델의 성공을 H100 검증으로 표시하지 않는다.

## 환경 고정

| 용도 | verl 고정 소스 | 주요 런타임 |
|---|---|---|
| H100 기준 환경 | v0.9.1, `1876b06d0a3e4e71e06230be10af14492ca8a75b` | 공식 `uv.lock`의 `fsdp` + `vllm`: PyTorch 2.11.0/CUDA 13, vLLM 0.24.0, Transformers 5.9.0, FlashAttention 2.8.3, flash-linear-attention 0.5.2 |
| A6000 개발 호환성 환경 | v0.8.0, `7aed6b230776f963fa09509c10d9c3a767d1102c` | PyTorch 2.8.0/CUDA 12.8, vLLM 0.10.2, Transformers 4.56.2, Ray 2.49.2, FlashAttention 2.8.3 |

v0.8.0 태그의 설치 메타데이터는 `0.8.0.dev`로 표시될 수 있으므로 커밋을 함께 확인한다. H100 환경은 공식 잠금 파일로만 설치한다. 설치 스크립트는 Linux 드라이버 `580.65.06` 이상을 요구하며, CUDA 13 환경을 CUDA 12.8 개발 장비에 강제로 설치하지 않는다. A6000 호환성 환경은 Qwen3.8 지원 환경이 아니다.

```bash
bash scripts/setup_verl.sh h100
# 또는 개발 장비에서:
bash scripts/setup_verl.sh a6000-dev
```

학습 환경에 보고서 패키지를 추가 설치하지 않는다. `setup_verl.sh`는 보고서용 `requirements-report.txt`를 별도 `.venv-report`에 설치한다. 위치는 `BENCH_REPORT_VENV`로 지정할 수 있으며, 학습 환경과 같은 디렉터리면 중단한다. 각각의 환경에 설치 목록을 저장한다. 보고서 생성은 `.venv-report/bin/python`을 사용한다. 보고서 패키지 설치가 NumPy 등 학습 의존성을 변경하는 일을 방지하기 위한 분리다.

`doctor`에 사용하는 프레임워크 정보에는 패키지 버전, 소스 커밋, 작업 트리 변경 여부와 변경분 해시가 들어간다. 기본 실행은 변경된 verl 소스를 허용하지 않는다. 벤치마크 계측 코드는 별도 모듈에서 실행되므로 upstream checkout을 수정할 필요가 없다.

공식 근거: [v0.9.1 의존성 선언](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/pyproject.toml), [v0.8.0 소스](https://github.com/verl-project/verl/tree/7aed6b230776f963fa09509c10d9c3a767d1102c), [NVIDIA CUDA 드라이버 호환성](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

## 실제 호출하는 학습 API

SFT는 `torch.distributed.run`에서 `benchkit.verl_entry sft`를 실행하고, 이 모듈이 `verl.trainer.sft_trainer`를 호출한다. 입력은 `messages` 열을 가진 Parquet이며, FSDP2와 BF16 전체 가중치 학습을 사용한다. 저장된 에이전트 대화의 assistant 토큰에 지도학습 손실을 적용한다. 시퀀스 길이는 상한이며, 처리량 분자는 실제 unpadded 입력 토큰 수다. 1,024토큰 이하 개발 smoke에서는 지도학습 토큰이 잘리지 않도록 실제 대화의 짧은 발췌를 사용하고 데이터 manifest에 표시한다.

초기 처리량 시험은 최종 체크포인트의 `extra` 정보만 저장했다. 최신 기본값은 아래 중간 산출물 보존 모드에 따라 모델·옵티마이저도 매 단계 저장한다. 기존 시험에서 가중치가 저장되었다고 소급하여 해석하지 않는다.

GRPO는 `benchkit.verl_entry grpo`에서 `verl.trainer.main_ppo`를 호출한다. vLLM이 실제 응답을 생성하고, 사용자 정의 보상 함수가 생성된 최종 정수 답변을 평가한다. verl이 그룹 보상으로 advantage를 계산하고, 이전 정책·참조 정책 log probability, 정책 손실·KL, 역전파, 옵티마이저 업데이트 및 vLLM 가중치 동기화를 수행한다. 임의 advantage를 생성하지 않는다.

RL 과제는 신뢰된 Python 코드 조각의 출력 예측이다. 생성된 코드는 실행하지 않는다. 정답 보상은 1, 파싱 가능한 오답은 0.1, 형식 오류는 0이다. 형식 보상은 명시적인 구성 요소이므로 정답률과 형식 통과율을 따로 보고한다. 이는 프레임워크 부하를 측정하는 단일 턴 대체 과제이며, 저장소 수정·도구 실행·SWE-bench 채점이나 제안서의 전체 다중 턴 하네스 검증을 대신하지 않는다.

Llama 기본 모델에는 명시적인 Llama-3 역할 템플릿을 제공한다. 모델 이름과 revision은 실행 전에 불변 로컬 스냅샷으로 해석한다. Qwen3.8-27B는 공식 모델 카드상 `qwen3_5` 구조를 사용하며, 고정 verl에는 해당 구조의 FSDP 지원 코드가 있다. 구조 지원 확인과 실제 27B 장비 실행 검증은 구분한다. [공식 Qwen3.8 모델](https://huggingface.co/Qwen/Qwen3.8-27B), [고정 버전 Qwen3.5 예제](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh).

기본 Llama의 템플릿은 run 디렉터리의 `model_assets`에도 적용한다. Native RL 데이터 로더는 actor 설정과 별도로 tokenizer를 로드하기 때문에 `actor_rollout_ref.model.custom_chat_template`만 설정하면 템플릿을 받지 못한다. 원본 가중치·설정 파일은 불변 symlink로 연결하고, `tokenizer_config.json`과 `chat_template.jinja`만 파생 생성한다. 원본 checkpoint는 수정하지 않는다. `model_assets_manifest.json`에 원본 모델 revision, 링크 대상, 원본·파생 tokenizer 설정 해시와 템플릿 해시를 기록한다.

## 시간과 메모리의 정의

Native file logger는 `trainer.logger=[console,file]`, `VERL_FILE_LOGGER_PATH`로 켠다. JSONL 한 줄은 `{"step": 1, "data": {"지표": 값}}` 구조다.

SFT 계측 버전 2는 `BaseEngine.train_batch` 호출 직전에 `torch.cuda.synchronize()`를 수행한 뒤 벽시계 측정을 시작한다. 순전파·역전파·옵티마이저 호출이 끝나면 다시 동기화한 뒤 측정을 종료한다. `timing_s/train_step`은 FSDP rank별 구간의 최댓값이다. 데이터 로딩·토큰화, 초기 모델 로딩, train-mode 진입/이탈에 따른 CPU offload, 최종 체크포인트 저장, 보고서 생성 시간은 이 구간에 포함하지 않는다. 기본 프로파일은 CPU offload를 끈다.

`bench/timing_instrumentation_version=2`가 현재 계측을 식별한다. 이전 버전은 native timer를 직접 노출했으며, 현재 값과 구분해야 한다. 원래 timer는 `timing_s/train_step_native`로 함께 남는다. Native 구현은 `optimizer.step()` 이후 `grad_norm.item()`으로 CUDA 스칼라를 CPU로 가져오지만, 버전 2는 이런 간접 동기화에 의존하지 않는다.

정확한 소스 경로는 [TrainingWorker timer](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/engine_workers.py#L364), [zero_grad → forward/backward → optimizer_step](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/engine/base.py#L124), [FSDP optimizer.step와 grad_norm.item](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/engine/fsdp/transformer_impl.py#L821)이다. 두 고정 버전의 SFT 콘솔 마지막 `Total time for train steps: 0.00s`는 누적되지 않는 upstream 변수에서 나오므로 측정값으로 사용하지 않는다.

GRPO의 `timing_s/step`은 두 고정 버전 모두 매 반복의 롤아웃·보상·정책 업데이트·**가중치 동기화**를 포함한다. v0.8은 `ray_trainer.py`의 step 블록 안에서 `update_weights`를 호출한다. v0.9.1의 동기식 trainer는 step 블록 안의 `on_step_end()`에서 이를 호출한다. 초기 가중치 동기화, 서버 시작, 검증과 전체 작업 종료 시간은 제외한다. v0.8의 rollout dump 호출은 step 안에 있고 v0.9.1은 밖에 있으므로 두 버전의 총 step 시간을 동일한 실행 환경으로 취급하지 않는다. [v0.8 step/sync 위치](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/trainer/ppo/ray_trainer.py#L1674), [v0.9.1 step 범위](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_base.py#L487), [v0.9.1 동기식 sync hook](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/v1/trainer_sync.py#L35).

SFT의 `perf/max_memory_allocated_gb`와 `perf/max_memory_reserved_gb`는 rank별 PyTorch peak를 최댓값으로 집계한다. 이는 GPU 전체 사용량과 다르며 프로세스 초기화 이후 누적 peak다. 필수 메모리 지표가 없으면 0을 대신 쓰지 않고 실패한다. RL은 native worker 메모리 지표를 보관한다. vLLM을 포함한 GPU 전체 peak와 비교하려면 별도 GPU telemetry를 함께 확인한다.

## 확인된 개발 이슈와 검증 한계

2026-10-16 초기 A6000 2-GPU 소형 모델 시험에서 실제 SFT와 GRPO 사이클, 비영(非零) gradient, 반복별 가중치 동기화가 확인되었다. 이 초기 시험은 SFT 계측 버전 1이었다. 후속 실행은 계측 버전 2의 key와 원시 로그로 구분한다.

초기 GRPO 시험은 prefix caching을 false로 요청했지만, v0.8의 CLI 변환이 false 플래그를 생략해 vLLM 기본값 true로 실행되었다. 이 시험을 설정이 일치한 최종 실측으로 재분류하지 않는다. 현재 프로파일은 true를 명시하고 개발 환경에서 false 요청을 거절한다. 초기 소형 모델은 정답률 0%였고 일부 형식 보상 차이로 GRPO 신호가 발생했다. 이를 문제 해결 학습의 성공으로 해석하지 않는다.

일부 Slurm 설정은 NVIDIA·AMD GPU visibility 환경변수를 동시에 제공한다. NVIDIA PyTorch 런타임을 확인한 경우에만 충돌하는 `ROCR_VISIBLE_DEVICES`와 `HIP_VISIBLE_DEVICES`를 제거하고, 할당된 `CUDA_VISIBLE_DEVICES`는 보존한다. 이 조정은 로그에 남는다.

현재 프로파일과 명령 생성은 두 고정 버전의 실제 Hydra 설정으로 검증했다. 원시 Hydra override는 기록된 실험 조건을 우회할 수 있어 허용하지 않는다. H100 전체 모델에서 OOM 또는 호환성 문제가 발생하면 모델·정밀도·배치·길이를 조용히 바꾸지 말고 새 프로파일과 실패 기록을 남긴다. H100 27B 장문맥 실행, 전체 다중 턴 도구 루프, GH200 환산 및 대규모 다중 노드 확장성은 별도 검증 대상이다.

## 중간 산출물 보존 모드 (capture version 1)

최신 기본값은 `artifact_capture=true`, `capture_every_step=1`, `save_checkpoint=true`, `checkpoint_every_steps=1`, `checkpoint_keep_all=true`다. 위의 초기 처리량 시험에서 사용한 `extra` 전용 저장과 달리, 새 실행은 매 단계 모델·옵티마이저·추가 상태를 native checkpoint로 보존한다. 과거 실행에 없던 체크포인트나 중간 텐서를 소급하여 존재한다고 표시하지 않는다. cadence는 명시적으로 늘릴 수 있지만 생성된 파일을 자동으로 지우거나 저장 실패를 무시하지 않는다.

`artifacts/<host>-pid<PID>-rank<RANK>/`에는 CPU로 복사한 실제 trainer payload `.pt`, 구조·dtype·shape·SHA256·크기·단계가 기록된 `.json`, 최종 해석된 native 설정 JSON이 생긴다. 프로세스별 `artifacts/manifest-*.jsonl`이 색인이다. `.pt`는 pickle 기반이므로 해당 실행에서 직접 만든 신뢰된 파일만 로드한다. 텐서 값을 JSON 통계로 대체하지 않는다.

SFT는 모든 rank의 `BaseEngine.train_batch` 입력(토큰·마스크·메타데이터)과 노출된 엔진 출력을 저장한다. RL v0.8은 실제 `DataProto.batch`, `non_tensor_batch`, `meta_info`를 보존한다. RL v0.9.1은 V1 trainer의 `KVBatchMeta` key·tag와 TransferQueue의 실제 전체 필드를 cleanup 이전에 읽어 보존한다. 보존 경계는 `rl_prompt_batch`, `rl_generated_reward_batch`, `rl_old_logprob_batch`, `rl_reference_batch`, `rl_reward_logprob_batch`, `rl_advantage_batch`, `rl_update_input`, `rl_update_output`이다. 생성과 보상은 native rollout에서 결합될 수 있으므로 생성 직후의 보상 전 상태를 별도로 보존한다고 주장하지 않는다. 내부의 모든 layer activation, gradient tensor 또는 CUDA 임시 버퍼를 수집하는 계측은 아니다. 옵티마이저·모델 상태는 native 단계별 체크포인트에 있다.

RL hook은 native TaskRunner를 감싸는 driver actor에만 설치한다. v0.9.1에서는 Ray ActorClass가 보존한 Python 구현을 사용하므로 해당 인터페이스가 없으면 명시적으로 실패한다. Native 학습·보상·advantage·업데이트 구현은 바꾸지 않는다. 요청한 핵심 capture 경계가 실행 후 누락되면 성공으로 처리하지 않는다. 두 고정 소스의 hook 위치와 Hydra 설정은 확인했지만 **신규 capture 모드의 H100 GPU 실행 검증은 아직 하지 않았다**. A6000 과거 성공은 capture 모드의 성공 증거가 아니다. H100 담당 에이전트는 작은 smoke에서 manifest, 텐서 재로드 및 checkpoint model/optimizer shard를 먼저 확인한다.

저장은 동기식이다. Manifest `elapsed_s`는 CPU 복사·payload 직렬화·SHA256 계산 시간이며, TransferQueue 조회와 마지막 metadata/manifest 기록 전체 시간을 포함하는 총 오버헤드가 아니다. SFT `timing_s/train_step`의 CUDA 구간에는 입력·출력 저장이 들어가지 않고 `bench/artifact_capture_s`에 payload 저장 시간의 합이 별도로 남는다. RL native `timing_s/step`에는 해당 블록 안에서 수행한 capture와 checkpoint 저장이 들어가며, prompt 준비처럼 블록 밖의 작업은 제외된다. 이 시간을 저장 기능이 없던 이전 실측과 같은 조건으로 비교하거나 저장 시간을 추정하여 빼지 않는다. 전체 프로세스 wall time과 파일 용량을 함께 사용한다. 매 단계 전체 checkpoint는 27B 모델에서 여러 TB를 차지할 수 있으므로 실행 전 여유 공간 검사와 명시적인 cadence가 필요하다.

v0.9.1의 `rl_update_output`은 업데이트 이후의 TransferQueue batch이며 optimizer 함수의 원시 반환값이나 gradient 전체가 아니다. 정책 업데이트 scalar는 native `metrics.jsonl`에도 보존된다. cadence가 1보다 커도 첫 학습 단계는 항상 보존한다.
