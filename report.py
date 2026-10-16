"""results/*.json → results/report.md + 그림 2장. 제안서 산정 단가(5장)와 바로 비교한다.

usage: python report.py [results 디렉터리]
"""
import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
GPU_PER_NODE = 4          # 6호기 GH200 1노드 = 4장 (GPU당 성능은 H100과 같다고 보수 가정)
EST = dict(task27=0.0018, sft27=78.0, rl27=0.020)   # 5장 단가: 과제당 노드시간, SFT 10억 토큰당 노드시간, RL 궤적당 노드시간

rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "*.json")))]
roll = [r for r in rs if r.get("kind") == "rollout"]
train = [r for r in rs if r.get("kind") == "train"]
L = ["# H100 SFT·RL 실측 레포트", ""]
gpu_name = next((r["gpu"] for r in train), "H100")
L += [f"- 측정 환경: {gpu_name}", f"- 결과 파일 {len(rs)}개 (`{os.path.basename(R)}/`)", ""]

# ── 1. 롤아웃 ────────────────────────────────────────────────────────────
best_roll = {}
if roll:
    L += ["## 1. RL 롤아웃 (에이전트 추론) 처리량", "",
          "pilot 실제 에이전트 궤적 80건 재현, GPU당 vLLM 1개(TP1). 시간당 스텝·과제는 측정 구간(워밍업 제외) 기준.", "",
          "| GPU | 동시 실행 | 시간당 스텝 | GPU당 스텝/h | GPU당 과제/h | GPU당 출력 토큰/s | KV 최대 | p95 지연(s) | 오류 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(roll, key=lambda r: (r["gpus"], r["concurrency"])):
        L.append(f"| {r['gpus']} | {r['concurrency']} | {r['steps_per_h']:,.0f} | {r['steps_per_h_per_gpu']:,.0f} | "
                 f"{r['tasks_per_h_per_gpu']:.1f} | {r['gen_tok_per_s_per_gpu']:,.0f} | {max(r['kv_cache_max']) * 100:.0f}% | "
                 f"{r['latency_p95_s'] or 0:.1f} | {r['errors']} |")
        b = best_roll.get(r["gpus"])
        if b is None or r["steps_per_h"] > b["steps_per_h"]:
            best_roll[r["gpus"]] = r
    g0 = min(best_roll)
    base = best_roll[g0]["steps_per_h_per_gpu"]
    L += ["", "**확장성** (GPU 수별 최고 처리량, GPU당 처리량 유지율):", ""]
    for g, r in sorted(best_roll.items()):
        L.append(f"- {g}장: 시간당 {r['steps_per_h']:,.0f}스텝, GPU당 유지율 {r['steps_per_h_per_gpu'] / base * 100:.0f}% ({g0}장 대비)")
    top = best_roll[max(best_roll)]
    node_h_task = 1 / (top["tasks_per_h_per_gpu"] * GPU_PER_NODE)
    L += ["", f"**산정 단가 비교**: 과제당 {node_h_task:.4f} 노드시간(GH200 4장 = 1노드 환산) vs 5장 산정 {EST['task27']} "
              f"→ 산정값이 실측의 {EST['task27'] / node_h_task:.1f}배 (산정이 {'보수적' if EST['task27'] >= node_h_task else '낙관적'})", ""]

# ── 2. 학습 ──────────────────────────────────────────────────────────────
if train:
    L += ["## 2. SFT·RL 정책 업데이트 학습 처리량 (PyTorch FSDP2)", "",
          "| 단계 | 모델 | GPU(노드) | 시퀀스 | step(s) | GPU당 토큰/s | TFLOPS/GPU | MFU | 최대 메모리(GiB) | 가중치 수집(s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(train, key=lambda r: (r["mode"], r["model"], r["world"], r["seq"])):
        mode = "SFT" if r["mode"] == "sft" else "RL 업데이트" + ("(+참조)" if r["ref"] else "")
        L.append(f"| {mode} | {r['model'].split('/')[-1]} ({r['n_param'] / 1e9:.1f}B) | {r['world']} ({r['nodes']}) | "
                 f"{r['seq']:,} | {r['step_s']:.2f} | {r['tok_per_gpu_s']:,.0f} | {r['tflops_per_gpu']:.0f} | "
                 f"{(r['mfu'] or 0) * 100:.0f}% | {r['peak_mem_gib']:.0f} | {r['sync_s'] if r['sync_s'] is None else round(r['sync_s'], 1)} |")
    grp = defaultdict(list)
    for r in train:
        grp[(r["mode"], r["model"], r["seq"], r["ref"])].append(r)
    sc = [(k, sorted(v, key=lambda r: r["world"])) for k, v in grp.items() if len(v) > 1]
    if sc:
        L += ["", "**학습 확장성** (GPU당 처리량 유지율):", ""]
        for (mode, model, seq, _), v in sc:
            b = v[0]["tok_per_gpu_s"]
            L.append(f"- {mode.upper()} {model.split('/')[-1]} 시퀀스 {seq:,}: " + ", ".join(
                f"{r['world']}장 {r['tok_per_gpu_s']:,.0f}({r['tok_per_gpu_s'] / b * 100:.0f}%)" for r in v))
    main = [r for r in train if r["mode"] == "sft" and "27B" in r["model"]]
    if main:
        r = max(main, key=lambda r: r["tok_per_gpu_s"])
        sft_nh = 1e9 / (r["tok_per_gpu_s"] * GPU_PER_NODE) / 3600
        L += ["", f"**SFT 단가 비교**: 10억 토큰당 {sft_nh:.0f} 노드시간 vs 5장 산정 {EST['sft27']:.0f} "
                  f"→ 산정값이 실측의 {EST['sft27'] / sft_nh:.1f}배"]
    g = [r for r in train if r["mode"] == "grpo" and "27B" in r["model"]]
    if g and roll:
        r = max(g, key=lambda r: r["world"])
        upd_gpu_s = r["avg_traj_tokens"] / r["tok_per_gpu_s"]
        roll_gpu_s = 3600 / top["tasks_per_h_per_gpu"]
        nh = (roll_gpu_s + upd_gpu_s) / 3600 / GPU_PER_NODE
        L += ["", "## 3. RL 궤적 1건 비용 (롤아웃 + 정책 업데이트)", "",
              f"- 롤아웃: 궤적당 {roll_gpu_s:.0f} GPU초 (1장 실측, 평균 {top['avg_steps_per_task']:.1f}스텝)",
              f"- 정책 업데이트: 궤적 평균 {r['avg_traj_tokens']:,.0f} 토큰 ÷ GPU당 {r['tok_per_gpu_s']:,.0f} 토큰/s "
              f"= {upd_gpu_s:.1f} GPU초 ({r['world']}장 실측, 이전 정책·참조 모델 계산 포함)",
              f"- 합계 {nh:.4f} 노드시간/궤적 vs 5장 산정 {EST['rl27']} → 산정값이 실측의 {EST['rl27'] / nh:.1f}배. "
              "5장 단가는 thinking 모드 장기 과제(스텝·컨텍스트 2배 이상)와 가중치 동기화·대기 시간을 포함한 DeepSWE 공개치 기반이라 더 크다", ""]

open(os.path.join(R, "report.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ── 그림 ─────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if os.path.exists(f):
            from matplotlib import font_manager
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
    C = "#2a78d6"
    if best_roll:
        xs = sorted(best_roll)
        ys = [best_roll[x]["steps_per_h"] for x in xs]
        fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
        ax.plot(xs, [ys[0] / xs[0] * x for x in xs], color="#8a8984", lw=1, ls=(0, (3, 3)), label="선형 확장")
        ax.plot(xs, ys, color=C, lw=2, marker="o", ms=7, mec="white", mew=2, label="실측")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xlabel("H100 GPU 수 (GPU당 vLLM 1개)")
        ax.set_title("RL 롤아웃 처리량 (에이전트 스텝/시간)", loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e5e0")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_rollout.png"))
    if train:
        rows = sorted(train, key=lambda r: (r["mode"], r["model"], r["seq"], r["world"]))
        labels = [f"{'SFT' if r['mode'] == 'sft' else 'RL'} {r['model'].split('/')[-1].replace('Qwen', '')}\n"
                  f"{r['world']}장·{r['seq'] // 1024}K" for r in rows]
        fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(rows)), 3.4), dpi=200)
        bars = ax.bar(range(len(rows)), [r["tok_per_gpu_s"] for r in rows], color=C, width=0.6)
        for b_, r in zip(bars, rows):
            ax.annotate(f"{r['tok_per_gpu_s']:,.0f}", (b_.get_x() + b_.get_width() / 2, b_.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(range(len(rows)), labels, fontsize=7)
        ax.set_title("학습 처리량 (GPU당 토큰/초)", loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e5e0")
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_train.png"))
    print("그림 저장:", R)
except ImportError:
    pass
