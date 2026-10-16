# Proposal methods and benchmark evidence

The benchmark collects measurements for a performance and scalability report.
Each reported value must identify its workload, hardware, software, and evidence type.
An H100 measurement does not establish GH200 performance or large-cluster scaling.

## Method coverage

| Proposal method | Benchmark workload | Measurements | Interpretation |
| --- | --- | --- | --- |
| SFT on successful agent trajectories | Train a pretrained model on trajectory-derived examples, with a recorded loss mask | Valid training tokens/s, tokens/s/GPU, step time, loss, peak GPU memory | Actual SFT workload for the selected dataset and sequence length |
| Verifiable-reward GRPO through verl | Generate grouped responses, evaluate a bounded code-output task, calculate advantages, and update the actor through verl | Rollout time, reward time, policy-update time, total step time, responses/s, reward distribution, actor gradient norm | Actual RL framework execution on a bounded proxy task |
| vLLM rollout and evaluation | Use the rollout backend selected in the verl configuration | Generated tokens, prompt tokens, response lengths, generation time, allocation size | Inference cost within the measured RL workload |
| Teacher-guided harness evolution | Compare teacher and student traces, revise skills, and evaluate revised harnesses | Requires separate harness experiments | Outside the bounded training benchmark |
| Teacher distillation / OPD | Obtain teacher token distributions and train the student | Requires teacher deployment and a distinct experiment | Not established by SFT or GRPO throughput |
| Challenger–Solver task generation | Generate, validate, and select adaptive tasks | Requires task-generation and selection measurements | The fixed proxy dataset does not reproduce this stage |
| Multi-turn repository repair | Execute tools, modify a repository, and run its tests in a sandbox | Tool time, sandbox time, task success, complete trajectory cost | Requires a separate agent-loop integration |
| Multi-node and MoE training | Run matching distributed workloads across node counts | Strong-scaling efficiency, communication time, all-to-all cost | A single-node dense-model result cannot establish these values |

The proxy exercises real generation, reward assignment, and optimizer updates.
It does not measure SWE-bench success or the full repository-repair workflow.
Report its total step time as the **bounded proxy RL step time**.
Do not call that value the proposal's complete agentic RL cost.

The original `bench_train.py`, `bench_rollout.py`, and `run_all.sh` remain legacy benchmarks.
Their random initialization, replay behavior, and synthetic advantages require separate evidence labels.
Their outputs must not establish successful execution of the new verl workflow.

## Model matrix and historical baseline

| Model | New measurement | Reason |
| --- | --- | --- |
| `meta-llama/Meta-Llama-3.1-8B` | SFT and verl GRPO | Directly measure the model family used in the historical training record |
| `Qwen/Qwen3.8-27B` | SFT and verl GRPO | Measure the proposal's 27B verification model |

Record the exact model revision and tokenizer revision for each run.
Confirm model access and architecture support before reserving a long GPU run.
Never replace an unavailable model silently.

The historical Llama record describes continued pretraining with DeepSpeed and fp16.
The corrected allocation was 24 GPUs, not the earlier 32-GPU claim.
Its stated settings were sequence length 8,192, microbatch 2/GPU, accumulation 8, and 8,742 optimizer steps.
The record reports about 27.50B tokens and 106 training hours.
The supplied validation report derives 3,004 tokens/s/GPU from that record.

The new Llama SFT and GRPO runs are separate experiments.
Differences in precision, loss masking, sequence length, trainer, and GPU allocation must remain visible.
The historical 3,004 value is a comparison baseline, not a target that the runner must reproduce.

The proposal also names EXAONE-4.5-33B and several larger domestic MoE candidates.
The two-model benchmark does not validate those models.

## Report tables and figures

| Output | Required columns or axes | Source format |
| --- | --- | --- |
| Environment table | GPU model/count, node count, GPU memory, topology, CPU RAM, driver, CUDA, package versions, model revision | Proposal §5(1), PDF pp. 13–15 |
| Workload table | Model, mode, data source, sequence lengths, batch, accumulation, rollout group, warmup, measured steps | Validation report §2.1 and §4.1 |
| SFT results | Tokens/s, tokens/s/GPU, step-time mean/median/p95, GPU peak memory, observed steps | Validation report §4.1 |
| RL results | Generation, reward, update, and total time, generated tokens, rewards, gradient norm, observed steps | Proposal §4.3 and §5 |
| Scaling results | GPU/node count, fixed workload definition, throughput, speedup, efficiency | Validation report §8 |
| Evidence table | Status, hardware, workload scope, source run, excluded warmup, missing metrics | Proposal §5(2), PDF p. 18 |
| Artifact inventory | Attempt, artifact category, relative path, file type, retained bytes, capture status, raw-file link | Run evidence and declared capture policy |
| Capture and checkpoint table | Capture policy, checkpoint cadence and contents, observed capture time, storage, missing artifacts | Runner and framework records |
| Throughput figure | GPU count or concurrency on x-axis, measured throughput on y-axis, separate workloads | Proposal Figure 5-1, PDF p. 16 |
| Training figure | Measured step on x-axis, loss and throughput on separate y-axes or panels | Validation report Figure 2 |
| Timing figure | Configuration on x-axis, measured stage time on y-axis | KIPS example Figure 3, PDF p. 3 |

The KIPS paper illustrates experimental reporting: hardware, configuration, measured comparisons, and a scaling figure.
Its workload measures Slurm scheduling with MPI jobs.
Its timings and system configuration are not model-training measurements or instructions for this repository.

Plot only observed configurations.
Show missing measurements as missing, not as zero.
Keep different models, sequence lengths, precisions, and reward workloads in separate comparison groups.

## Intermediate evidence and checkpoints

Retain every attempt, including successful, failed, interrupted, superseded, and review-excluded attempts.
Keep raw records alongside derived summaries.
Summary tables cannot replace per-step metrics or intermediate artifacts.

| Evidence | Retention and interpretation |
| --- | --- |
| Native step metrics | Preserve original keys, steps, values, and timing scope. Missing keys remain missing. |
| Framework and worker logs | Preserve available logs from each process and the scheduler. Record collection failures. |
| GRPO rollout records | Preserve prompts, responses, task identifiers, rewards, and available token or policy diagnostics. State omitted fields. |
| SFT intermediates | Preserve dataset provenance, tokenization and masking metadata, step loss, gradient diagnostics, and supported sample captures. |
| Checkpoints | Declare cadence and contents. Preserve every checkpoint produced under that policy. Identify whether it supports resuming. |
| Configuration and model assets | Preserve resolved framework configuration, immutable revisions, tokenizer metadata, templates, hashes, and local-reference manifests. |
| Capture manifest | Record the requested policy, observed artifacts, capture errors, retained bytes, and measured capture duration where available. |
| Review notes | Preserve original records. Add reasons for comparison exclusions in a separate review overlay. |

Artifact presence does not establish model convergence or a useful learning signal.
A checkpoint file does not prove that optimizer state exists or that resuming succeeds.
Report checkpoint contents and resume validation separately.
Do not infer absent tensor captures, checkpoints, or profiler traces from another metric.

Capture policy is part of the workload configuration.
Logging, profiling, rollout dumps, and checkpoint writes can change throughput and memory use.
Compare runs under matching capture policies unless the experiment explicitly measures capture overhead.
Keep native step timing, complete process timing, and post-run capture timing separate.
Do not subtract estimated capture overhead or sum overlapping timers.
Missing capture duration is unknown, not zero.
Native capture `elapsed_s` measures CPU copies, payload serialization, and hashing.
It excludes TransferQueue materialization and metadata or manifest writes.
It does not measure total capture overhead.
The v0.9 `rl_update_output` artifact contains the batch after the actor update.
It does not expose every gradient tensor or optimizer internal.

Artifact byte totals describe retained logical file sizes unless the collector states another definition.
They do not establish physical filesystem usage or total model-cache size.
Record symlinks as references without counting their targets as new retained data.
Check storage capacity before execution and record any limit that restricts capture.
Do not delete intermediate evidence automatically when storage is insufficient.

## Evidence and arithmetic rules

Use these descriptions in reports, even when the storage schema uses different field names:

- **H100 measurement:** a completed run whose recorded device is H100.
- **A6000 smoke measurement:** a development run on A6000, used to check execution and metrics.
- **Derived from measurement:** arithmetic based on an identified completed run.
- **Historical baseline:** a value from prior evidence, with its original conditions.
- **Projection:** a hardware or workload extrapolation with explicit assumptions.
- **Unsupported or missing:** a requested value with no qualifying observation.

For throughput, divide measured tokens by the corresponding measured wall time.
For GPU-hours, multiply allocated GPU count by measured seconds, then divide by 3,600.
For one billion tokens, divide one billion by measured tokens/s, then convert seconds to hours.
Multiply that duration by the actual allocated node count to obtain actual node-hours.

If needed, report `GPU-hours / 4` as **normalized four-GPU node-hours**.
This normalization does not convert an H100 server into a GH200 node.
Do not label normalized node-hours as observed GH200 node-hours.

For strong scaling, keep the total work and algorithm settings fixed.
Report efficiency as `throughput(N) / throughput(N0) × N0 / N`.
If the total batch grows with GPU count, label the experiment weak scaling instead.
Count generated responses consistently when reporting RL cost per response.
Do not call a single-turn proxy response a multi-turn agent trajectory.

The proposal's 78 four-GPU node-hours per billion tokens for 27B is a historical extrapolation.
Its 0.020 node-hours per coding RL trajectory is a literature-based estimate.
Neither value is an acceptance criterion for a successful benchmark run.

## Source map

These references identify supplied source documents without distributing private source material.
PDF page numbers count from the first page, including cover and template pages.

- **KISTI 거대연구 제안서 (코딩 에이전트)**: §4, pp. 10–13, defines the method and models.
  §5(1), pp. 13–15, maps vLLM, verl, FSDP2/Megatron-Core, and harnesses.
  §5(2), pp. 16–19, defines the report evidence and resource calculations.
- **성능 및 확장성 검증자료**, 2026-10-16 revision: §2, pp. 1–2, contains rollout conditions and outputs.
  §4.1, p. 3, contains the historical training baseline.
  §5, p. 4, contains cost assumptions.
  §8, p. 7, contains planned scaling tests.
- **모델 학습 로그 및 모델 config**: §§1–4 identifies Llama-3.1-8B, continued pretraining, and DeepSpeed.
  The subsequent correction to the allocation establishes 24 GPUs for the historical run.
- **KIPS_C2026A0193F**, “대형 클러스터 시스템의 작업 배치 스케줄러 확장성 검증 기법에 관한 연구”: pp. 1–3.
  Table 1 documents hardware, Table 2 documents configuration, and Figure 3 compares measured conditions.

Framework references:

- [verl documentation](https://verl.readthedocs.io/en/latest/index.html)
- [verl GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)

Consult the documentation for the installed version before changing framework options.
