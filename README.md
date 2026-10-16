# H100 SFT·RL 실측 키트 (CoHA-Coder 제안서 증빙용)

## 측정 범위

이 저장소는 H100에서 롤아웃과 SFT·RL 정책 업데이트의 처리량·메모리·확장성을 측정하는 벤치마크다.
RL 업데이트는 verl 등의 RL 프레임워크 없이 PyTorch FSDP2로 직접 구현했다.
순전파·역전파·옵티마이저 업데이트는 실제 실행하지만, 기본 학습 가중치는 무작위 초기화이고 GRPO advantage도 임의 생성한다.
롤아웃은 저장된 대화를 재생하며 실제 코드 실행·보상 계산·공식 채점을 수행하지 않는다.
가중치 동기화 측정은 전체 state dict의 CPU 수집 시간이며 vLLM에 가중치를 적용하는 전체 과정은 포함하지 않는다.
따라서 단계별 연산 성능은 실측값이고, 이를 합친 전체 RL 비용과 GH200 노드시간은 환산·추정값이다.

기존 검증 PDF와 과거 학습 로그는 포함하지 않으며, 이 코드가 해당 PDF를 자동 생성하지는 않는다.
H100 실측 성공 여부는 실제 실행 결과로 확인해야 한다.

`data/sessions.jsonl`에는 SWE-bench 공개 과제의 프롬프트와 기록된 에이전트 대화 80건이 포함되어 있다.
과제 본문·인용 코드 등 제3자 자료의 권리는 각 원저작자에게 있다.

## 목적

클라우드 H100을 잠깐 빌려 **이번 과제 모델(Qwen3.8-27B 기반 CoHA-Coder-27B 초기 모델)로 SFT와 RL을 직접 돌리고**,
그 처리량·확장성을 제안서 5장 단가와 바로 비교하는 레포트를 만든다.

A6000 pilot(2026.10.15)과 같은 모델·같은 에이전트 궤적·같은 SW 버전(vLLM 0.19, transformers 5.5)을 쓰므로
"A6000 PoC → H100 확장 → 6호기 GH200" 순서로 그대로 이어진다.

## 무엇을 재나

| 단계 | 내용 | 산출 지표 | 5장 비교 대상 |
|---|---|---|---|
| RL 롤아웃 | pilot 실제 에이전트 궤적 80건 재현, GPU당 vLLM 1개(TP1), GPU 1·2·4·8장 | 시간당 스텝·과제, GPU당 유지율 | 과제당 0.0018 노드시간 |
| SFT | 27B, FSDP2, 8장, 시퀀스 16K·32K, 실제 궤적 토큰 | GPU당 토큰/초, MFU, 메모리 | 10억 토큰당 78 노드시간 |
| RL 정책 업데이트 | GRPO 손실(이전 정책 재계산 + 참조 모델 KL), 8장 + 전체 가중치 수집 시간 | GPU당 토큰/초 | 궤적당 0.020 노드시간(롤아웃+업데이트) |
| 학습 확장성 | 같은 구조 9B 모델 2·4·8장 (+ 2노드면 27B 8→16장) | GPU당 처리량 유지율 | 1분기 확장성 통과 기준 |

RL은 '롤아웃 → 정책 업데이트 → 가중치 동기화'를 단계별로 실측해 궤적 1건 비용으로 합친다.
샌드박스(Docker) 없이 GPU 부하만 재므로 어느 클라우드에서도 바로 돌아간다.

## 빌릴 자원

- **필수**: H100 80GB SXM × 8 (1노드), 디스크 300GB 이상, CPU RAM 1TB 권장(가중치 수집 측정)
- **선택**: 같은 사양 2노드 + InfiniBand/EFA → 27B SFT·RL 8장 → 16장 확장성
- 소요: 1노드 약 3시간(설치 30분 + 측정 2.5시간), 2노드 추가 측정 약 1시간

## 실행

```bash
# 1) 설치 + 모델 다운로드 (Qwen3.8-27B, Qwen3.5-9B)
bash setup.sh

# 2) 단일 노드 전체 측정 → results/report.md, fig_rollout.png, fig_train.png
bash run_all.sh
#    일부만: PHASES="sft grpo" bash run_all.sh  /  롤아웃 시간 단축: DUR=180 WARM=40 bash run_all.sh

# 3) (선택) 2노드: 두 노드에서 각각
NNODES=2 NODE_RANK=0 MASTER_ADDR=<노드0 IP> bash run_multinode.sh   # 노드0
NNODES=2 NODE_RANK=1 MASTER_ADDR=<노드0 IP> bash run_multinode.sh   # 노드1
python report.py      # 노드0에서 다시 실행하면 16장 결과까지 포함
```

`results/` 폴더(JSON, 로그, report.md, 그림)를 통째로 가져오면 된다.

## 파일

- `bench_rollout.py`: 롤아웃 처리량 (pilot 궤적 재현, 출력 길이 고정, prefix cache 재사용 패턴 동일)
- `bench_train.py`: SFT / GRPO 정책 업데이트 처리량 (PyTorch FSDP2, 활성화 체크포인팅, vocab 로짓 청크 계산)
- `serve_vllm.sh`: GPU마다 vLLM 서버 1개
- `report.py`: 결과 집계, 5장 단가 비교, 그림
- `data/sessions.jsonl`: pilot 에이전트 궤적 80건 (SWE-bench 공개 저장소 과제)
- `test_sjw.sbatch`: 연구실 A6000에서 소형 모델로 키트 동작을 검증한 스크립트 (H100에서는 불필요)

## 참고

- 학습 처리량 측정은 기본 무작위 초기화다(연산량은 가중치 값과 무관). 실제 가중치로 하려면 `--init pretrained`.
- H100 1장(80GB)에 27B BF16(54GB)을 올리면 KV 캐시가 약 18GB라, 6호기 GH200(96GB)보다 동시 실행 여유가 작다 → 롤아웃 단가는 6호기에서 더 좋아지는 쪽(보수적).
- 레포트에는 측정 일시·장비·버전이 `results/env.txt`와 각 JSON에 남는다.
