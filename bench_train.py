"""H100 학습 처리량 벤치마크: SFT / GRPO(RL 정책 업데이트) — PyTorch FSDP2.

에이전트 궤적(data/sessions.jsonl)을 모델 토크나이저로 토큰화해 seq 길이로 이어 붙여 학습한다.
처리량 측정이 목적이므로 기본은 무작위 초기화(--init random). 실제 가중치는 --init pretrained.

  sft : 다음 토큰 CE (fwd+bwd)
  grpo: 이전 정책 logprob 재계산(fwd) + 참조 모델 logprob(fwd, --ref) + clipped PG + KL 손실(fwd+bwd)
        → verl 등 RL 프레임워크의 '정책 업데이트' 단계와 같은 연산 구성

usage (단일 노드 8장):
  torchrun --nproc_per_node 8 bench_train.py --model Qwen/Qwen3.8-27B --mode sft --seq 16384
usage (2노드):
  torchrun --nnodes 2 --node_rank $R --master_addr $ADDR --master_port 29500 --nproc_per_node 8 bench_train.py ...
"""
import argparse, json, os, platform, socket, time
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.8-27B")
p.add_argument("--mode", choices=["sft", "grpo"], default="sft")
p.add_argument("--seq", type=int, default=16384, help="GPU당 마이크로배치 시퀀스 길이")
p.add_argument("--micro-bs", type=int, default=1)
p.add_argument("--grad-accum", type=int, default=1)
p.add_argument("--steps", type=int, default=12)
p.add_argument("--warmup", type=int, default=3, help="처리량 계산에서 제외할 앞 step 수")
p.add_argument("--init", choices=["random", "pretrained"], default="random")
p.add_argument("--param-dtype", choices=["fp32", "bf16"], default="fp32", help="마스터 가중치 dtype")
p.add_argument("--ref", action="store_true", help="grpo: 참조 모델 KL 포함")
p.add_argument("--cpu-offload", action="store_true")
p.add_argument("--measure-sync", action="store_true", help="학습 후 전체 가중치 수집 시간(롤아웃 엔진 동기화 근사)")
p.add_argument("--loss-chunk", type=int, default=2048, help="vocab 로짓을 나눠 계산할 토큰 수")
p.add_argument("--data", default=os.path.join(HERE, "data", "sessions.jsonl"))
p.add_argument("--out", default=os.path.join(HERE, "results"))
p.add_argument("--tag", default="")
args = p.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
mesh = init_device_mesh("cuda", (world,))
log = (lambda *a: print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)) if rank == 0 else (lambda *a: None)

cfg = AutoConfig.from_pretrained(args.model)
tcfg = getattr(cfg, "text_config", cfg)        # Qwen3.5/3.8은 VLM 설정 → 텍스트 부분만 학습
pdtype = torch.float32 if args.param_dtype == "fp32" else torch.bfloat16


# ── 데이터: 실제 에이전트 궤적 토큰 스트림 ───────────────────────────────
def token_stream():
    tok = AutoTokenizer.from_pretrained(args.model)
    ids, n = [], 0
    for line in open(args.data):
        n += 1
        msgs = json.loads(line)["messages"]
        out = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            out = out["input_ids"]
        ids.extend(out)
    return torch.tensor(ids, dtype=torch.long), n


stream, n_sessions = token_stream()
n_traj_tokens = len(stream)
need = args.seq + 1
reps = (need * args.micro_bs * (rank + 2)) // len(stream) + 2
stream = stream.repeat(reps)
log(f"world={world} model={args.model} mode={args.mode} seq={args.seq} 궤적 토큰 {n_traj_tokens:,}")


def batch(step, micro):
    k = ((step * args.grad_accum + micro) * world + rank) * args.micro_bs
    rows = []
    for b in range(args.micro_bs):
        s = ((k + b) * args.seq) % (len(stream) - need)
        rows.append(stream[s:s + need])
    x = torch.stack(rows).to(dev, non_blocking=True)
    return x[:, :-1], x[:, 1:]


# ── 모델: meta 생성 → FSDP2 샤딩 → 실제 메모리 할당 ────────────────────────
def build(trainable):
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(tcfg, dtype=pdtype if trainable else torch.bfloat16,
                                             attn_implementation="sdpa")
    m.to(pdtype if trainable else torch.bfloat16)   # 일부 파라미터(A_log 등)가 다른 dtype으로 생성됨 → FSDP2는 단일 dtype 요구
    if trainable:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.config.use_cache = False
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kw = dict(mesh=mesh, mp_policy=mp)
    if args.cpu_offload and trainable:
        kw["offload_policy"] = CPUOffloadPolicy()
    for layer in m.model.layers:
        fully_shard(layer, **kw)
    fully_shard(m.model.embed_tokens, **kw)
    fully_shard(m, **kw)            # 루트 그룹: final norm + lm_head (손실 계산 중 전체 가중치 유지)
    m.to_empty(device="cpu" if (args.cpu_offload and trainable) else dev)
    # meta에서 만든 비영속 버퍼(RoPE) 재계산
    for name, mod in list(m.named_modules()):
        if hasattr(mod, "inv_freq") and hasattr(mod, "config"):
            new = type(mod)(config=mod.config, device=dev)
            parent = m.get_submodule(name.rsplit(".", 1)[0]) if "." in name else m
            setattr(parent, name.rsplit(".", 1)[-1], new)
    if args.init == "pretrained":
        full = None
        if rank == 0:
            full = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).state_dict()
        set_model_state_dict(m, full or {}, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        del full
    else:
        with torch.no_grad():
            for n, prm in m.named_parameters():
                if n.endswith("A_log"):
                    prm.uniform_(0.0, 2.7)
                elif prm.dim() == 1:
                    prm.zero_() if "norm" in n and "linear_attn" not in n else prm.fill_(1.0)
                else:
                    prm.normal_(0.0, 0.02)
    if not trainable:
        m.eval().requires_grad_(False)
    return m


t0 = time.time()
model = build(True)
ref = build(False) if (args.mode == "grpo" and args.ref) else None
n_param = sum(prm.numel() for prm in model.parameters())   # DTensor.numel()은 전체 크기
log(f"모델 준비 {time.time() - t0:.0f}s, 파라미터 {n_param / 1e9:.2f}B")
opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.95), weight_decay=0.0)


def token_logp(m, x, y):
    """[B,T] 토큰 logprob. vocab 로짓은 loss-chunk 단위로 만들고 버린다(역전파 시 재계산)."""
    h = m.model(input_ids=x).last_hidden_state
    w = m.lm_head.weight

    def piece(hc, yc):
        logits = (hc @ w.t()).float()
        return torch.gather(logits, -1, yc.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)

    outs = []
    for s in range(0, h.shape[1], args.loss_chunk):
        hc, yc = h[:, s:s + args.loss_chunk], y[:, s:s + args.loss_chunk]
        outs.append(checkpoint(piece, hc, yc, use_reentrant=False) if torch.is_grad_enabled() else piece(hc, yc))
    return torch.cat(outs, 1)


# forward를 청크 손실 계산으로 교체: m(x, y) 호출 시 FSDP2 루트 훅(final norm·lm_head all-gather)이 그대로 적용된다
for m_ in [model] + ([ref] if ref is not None else []):
    m_.forward = (lambda mm: (lambda x, y: token_logp(mm, x, y)))(m_)

times = []
peak = 0
for step in range(args.steps):
    torch.cuda.synchronize()
    dist.barrier()
    t = time.time()
    loss_v = 0.0
    for micro in range(args.grad_accum):
        x, y = batch(step, micro)
        if args.mode == "sft":
            loss = -model(x, y).mean()
        else:
            with torch.no_grad():
                old = model(x, y)                    # 이전 정책 logprob 재계산
                rl = ref(x, y) if ref is not None else None
            lp = model(x, y)
            adv = torch.randn(x.shape[0], 1, device=dev).sign()   # 그룹 정규화 이점(±1)
            ratio = torch.exp(lp - old)
            pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.28) * adv)
            if rl is not None:
                d = rl - lp
                pg = pg + 0.001 * (torch.exp(d) - d - 1)          # k3 KL
            loss = pg.mean()
        (loss / args.grad_accum).backward()
        loss_v += loss.item() / args.grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()
    dt = time.time() - t
    times.append(dt)
    peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    log(f"step {step + 1}/{args.steps} {dt:.2f}s loss {loss_v:.4f} peak {peak:.1f}GiB")

sync_s = None
if args.measure_sync:
    torch.cuda.synchronize()
    dist.barrier()
    t = time.time()
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    dist.barrier()
    sync_s = time.time() - t
    del sd
    log(f"전체 가중치 수집 {sync_s:.1f}s")

peak_all = torch.tensor([peak], device=dev)
dist.all_reduce(peak_all, op=dist.ReduceOp.MAX)
if rank == 0:
    meas = times[args.warmup:] or times
    step_s = sum(meas) / len(meas)
    tok_step = world * args.micro_bs * args.seq * args.grad_accum
    tok_gpu_s = tok_step / step_s / world
    passes = 3 if args.mode == "sft" else (3 + 1 + (1 if ref is not None else 0))   # fwd=1, bwd=2 단위
    flops_tok = 2 * n_param * passes
    gpu = torch.cuda.get_device_name()
    peak_tflops = 989e12 if "H100" in gpu or "H200" in gpu or "GH200" in gpu else None
    res = dict(
        kind="train", mode=args.mode, model=args.model, n_param=n_param, world=world,
        nodes=world // int(os.environ.get("LOCAL_WORLD_SIZE", world)),
        gpu=gpu, seq=args.seq, micro_bs=args.micro_bs, grad_accum=args.grad_accum, init=args.init,
        param_dtype=args.param_dtype, ref=ref is not None, cpu_offload=args.cpu_offload,
        steps=args.steps, warmup=args.warmup, step_times=times, step_s=step_s, tokens_per_step=tok_step,
        tok_per_gpu_s=tok_gpu_s, tflops_per_gpu=tok_gpu_s * flops_tok / 1e12,
        mfu=(tok_gpu_s * flops_tok / peak_tflops) if peak_tflops else None,
        peak_mem_gib=peak_all.item(), sync_s=sync_s, avg_traj_tokens=n_traj_tokens / n_sessions, host=socket.gethostname(),
        torch=torch.__version__, python=platform.python_version(), time=datetime.now().isoformat(),
    )
    os.makedirs(args.out, exist_ok=True)
    name = f"train_{args.mode}_{args.model.split('/')[-1]}_{world}gpu_s{args.seq}{args.tag}.json"
    json.dump(res, open(os.path.join(args.out, name), "w"), indent=1)
    log(f"결과: GPU당 {tok_gpu_s:,.0f} 토큰/초, step {step_s:.2f}s, "
        f"{res['tflops_per_gpu']:.0f} TFLOPS/GPU, peak {peak_all.item():.1f}GiB → {name}")
dist.destroy_process_group()
