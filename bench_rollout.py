"""H100 롤아웃(에이전트 추론) 처리량 벤치마크.

pilot에서 CoHA-Coder-27B 초기 모델이 실제로 만든 SWE-bench 에이전트 궤적 80건(data/sessions.jsonl)을
그대로 재현한다: 매 스텝마다 그때까지의 대화(시스템·과제·이전 행동·실행 결과)를 보내고,
원래 출력과 같은 길이(min_tokens = max_tokens, ignore_eos)를 생성시킨다. 다음 스텝 대화에는 원래
기록된 행동·실행 결과를 붙이므로 prefix cache 재사용 패턴까지 실제 에이전트 롤아웃과 같다.
Docker 샌드박스 없이 GPU 쪽 롤아웃 부하만 정확히 잰다(도구 실행 지연은 --tool-latency로 모사).

usage: python bench_rollout.py --ports 8100:8 --concurrency 256 --duration 300 --tag dp8
"""
import argparse, asyncio, json, os, random, re, time
from datetime import datetime

import aiohttp
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.8-27B", help="토크나이저 경로")
p.add_argument("--served-name", default="bench")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--ports", default="8100:1", help="시작포트:서버수 (서버 1개 = GPU 1장, TP1)")
p.add_argument("--gpus-per-server", type=int, default=1)
p.add_argument("--concurrency", type=int, default=32, help="동시 실행 에이전트 수(전체)")
p.add_argument("--duration", type=float, default=300, help="측정 시간(초)")
p.add_argument("--warmup", type=float, default=60, help="앞부분 제외(초)")
p.add_argument("--tool-latency", type=float, default=0.0, help="스텝 사이 도구 실행 지연(초)")
p.add_argument("--max-out", type=int, default=4096)
p.add_argument("--data", default=os.path.join(HERE, "data", "sessions.jsonl"))
p.add_argument("--out", default=os.path.join(HERE, "results"))
p.add_argument("--tag", default="")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

port0, nserv = map(int, args.ports.split(":"))
urls = [f"http://{args.host}:{port0 + i}" for i in range(nserv)]
tok = AutoTokenizer.from_pretrained(args.model)
sessions = []
for line in open(args.data):
    s = json.loads(line)
    msgs = s["messages"]
    turns = []   # (보낼 대화 끝 인덱스, 생성 토큰 수)
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            n = len(tok(m["content"], add_special_tokens=False)["input_ids"])
            turns.append((i, max(1, min(n, args.max_out))))
    sessions.append((s["id"], msgs, turns))
rng = random.Random(args.seed)
T0 = None
recs = []          # (t_end, prompt_tokens, completion_tokens, latency)
done_sessions = []  # (t_end, steps)
kv_max = [0.0] * nserv
errors = [0]


async def step(sess, url, msgs, n_out):
    body = {"model": args.served_name, "messages": msgs, "max_tokens": n_out, "min_tokens": n_out,
            "ignore_eos": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    async with sess.post(url + "/v1/chat/completions", json=body,
                         timeout=aiohttp.ClientTimeout(total=1800)) as r:
        j = await r.json()
    if "usage" not in j:
        raise RuntimeError(str(j)[:300])
    u = j["usage"]
    return time.time() - t, u["prompt_tokens"], u["completion_tokens"]


async def agent(k, sess):
    url = urls[k % nserv]      # 에이전트는 한 서버에 고정(prefix cache 재사용)
    order = list(range(len(sessions)))
    rng.shuffle(order)
    j = k
    while time.time() - T0 < args.duration:
        sid, msgs, turns = sessions[order[j % len(order)]]
        j += 1
        for idx, n_out in turns:
            if time.time() - T0 >= args.duration:
                return
            try:
                lat, pt, ct = await step(sess, url, msgs[:idx], n_out)
            except Exception as e:  # noqa: BLE001
                errors[0] += 1
                if errors[0] <= 5:
                    print("오류:", e, flush=True)
                await asyncio.sleep(1)
                continue
            recs.append((time.time() - T0, pt, ct, lat))
            if args.tool_latency:
                await asyncio.sleep(args.tool_latency)
        done_sessions.append((time.time() - T0, len(turns)))


async def kv_poll(sess):
    pat = re.compile(r'^vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)\{[^}]*\} ([0-9.e+-]+)', re.M)
    while time.time() - T0 < args.duration:
        for i, u in enumerate(urls):
            try:
                async with sess.get(u + "/metrics", timeout=aiohttp.ClientTimeout(total=10)) as r:
                    for v in pat.findall(await r.text()):
                        kv_max[i] = max(kv_max[i], float(v))
            except Exception:  # noqa: BLE001
                pass
        await asyncio.sleep(10)


async def main():
    global T0
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        for u in urls:
            for _ in range(120):
                try:
                    async with sess.get(u + "/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            break
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(5)
        T0 = time.time()
        print(f"[{datetime.now():%H:%M:%S}] 서버 {nserv}개, 동시 {args.concurrency}, {args.duration:.0f}s 측정 시작",
              flush=True)
        await asyncio.gather(kv_poll(sess), *[agent(k, sess) for k in range(args.concurrency)])


asyncio.run(main())
win = [r for r in recs if r[0] >= args.warmup]
span_h = (args.duration - args.warmup) / 3600
gpus = nserv * args.gpus_per_server
steps_h = len(win) / span_h
avg_steps = sum(len(t) for _, _, t in sessions) / len(sessions)
res = dict(
    kind="rollout", model=args.model, servers=nserv, gpus=gpus, concurrency=args.concurrency,
    duration=args.duration, warmup=args.warmup, tool_latency=args.tool_latency,
    steps=len(win), steps_per_h=steps_h, steps_per_h_per_gpu=steps_h / gpus,
    tasks_per_h_per_gpu=steps_h / gpus / avg_steps, avg_steps_per_task=avg_steps,
    prompt_tok_per_s=sum(r[1] for r in win) / (span_h * 3600),
    gen_tok_per_s=sum(r[2] for r in win) / (span_h * 3600),
    gen_tok_per_s_per_gpu=sum(r[2] for r in win) / (span_h * 3600) / gpus,
    latency_mean_s=sum(r[3] for r in win) / max(1, len(win)),
    latency_p95_s=sorted(r[3] for r in win)[int(0.95 * len(win))] if win else None,
    sessions_completed=len([s for s in done_sessions if s[0] >= args.warmup]),
    kv_cache_max=kv_max, errors=errors[0], time=datetime.now().isoformat(),
)
os.makedirs(args.out, exist_ok=True)
name = f"rollout_{args.model.split('/')[-1]}_{gpus}gpu_c{args.concurrency}{args.tag}.json"
json.dump(res, open(os.path.join(args.out, name), "w"), indent=1)
print(f"결과: 시간당 {steps_h:,.0f}스텝 (GPU당 {steps_h / gpus:,.0f}), GPU당 출력 {res['gen_tok_per_s_per_gpu']:,.0f} 토큰/초, "
      f"KV 최대 {max(kv_max) * 100:.0f}%, 오류 {errors[0]} → {name}", flush=True)
