"""Build native verl SFT/GRPO launches without importing GPU libraries."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


VERL_REVISIONS = {
    "h100": "1876b06d0a3e4e71e06230be10af14492ca8a75b",  # v0.9.1
    "a6000-dev": "7aed6b230776f963fa09509c10d9c3a767d1102c",  # v0.8.0
}

def _value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (str, list, dict)) else str(value)


def _overrides(values: dict[str, Any]) -> list[str]:
    return [f"{name}={_value(value)}" for name, value in values.items()]


def validate_framework() -> dict[str, Any]:
    """Report installed package/commit evidence; fail for missing native APIs."""
    packages = {}
    missing = []
    for name in ("verl", "torch", "vllm", "transformers", "ray", "pyarrow", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    spec = importlib.util.find_spec("verl")
    root = Path(spec.origin).parent.parent if spec and spec.origin else None
    commit = None
    source_status = None
    source_diff_sha256 = None
    if root:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
            status_result = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], text=True, capture_output=True, check=False)
            if status_result.returncode == 0:
                source_status = status_result.stdout
                if source_status:
                    diff_result = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"], capture_output=True, check=False)
                    # Dirty source is blocked, even if the difference is only an
                    # untracked file. The digest makes the rejected state clear.
                    source_diff_sha256 = hashlib.sha256(source_status.encode() + diff_result.stdout).hexdigest()
    supported = commit in VERL_REVISIONS.values()
    return {
        "available": not missing,
        "missing": missing,
        "packages": packages,
        "verl_source": str(root) if root else None,
        "verl_commit": commit,
        "pinned_revision_match": supported,
        "source_clean": source_status == "",
        "source_dirty": None if source_status is None else bool(source_status),
        "source_diff_sha256": source_diff_sha256,
        "lane": next((name for name, revision in VERL_REVISIONS.items() if revision == commit), "unverified"),
    }


def build_commands(config: dict[str, Any], run_dir: Path) -> list[list[str]]:
    """Build one native trainer invocation for a resolved, flat run config.

SFT is torchrun/FSDP2 and GRPO is verl's Ray trainer with vLLM. The latter
actually samples responses, computes rewards, updates the policy and loads its
new weights into the rollout engine. No hand-written loss or advantage is used.
    """
    run_dir = Path(run_dir).resolve()
    mode = config["mode"]
    if config.get("extra_overrides"):
        raise ValueError("extra_overrides are forbidden: add a validated config field so recorded settings remain authoritative")
    if mode not in {"sft", "grpo"}:
        raise ValueError("mode must be sft or grpo")
    gpus, nodes = int(config["gpus"]), int(config.get("nodes", 1))
    batch, micro = int(config["global_batch_size"]), int(config.get("micro_batch_size", 1))
    world = gpus * nodes
    if min(gpus, nodes, batch, micro) < 1 or batch % (world * micro):
        raise ValueError("global_batch_size must be positive and divisible by nodes*gpus*micro_batch_size")
    if config.get("strategy", "fsdp2") != "fsdp2":
        raise ValueError("This benchmark lane implements FSDP2; a Megatron run needs a separate validated profile")
    if config.get("dtype", "bfloat16") not in {"bfloat16", "bf16"}:
        raise ValueError("This validated lane uses BF16 model storage and compute; other dtypes require a separate profile")
    model = str(config["model"])
    if config.get("model_revision") not in (None, "", "main") and not Path(model).is_dir() and not config.get("_dry_run", False):
        raise ValueError("Resolve model_revision to a local HF snapshot before constructing trainer commands")
    steps = int(config.get("steps", 4))
    if steps < 1:
        raise ValueError("Need at least one training step")
    data = config.get("data_paths", {})
    paths = {key: str(data.get(key, run_dir / "data" / f"{key}.parquet")) for key in ("sft_train", "rl_train", "rl_val")}
    template = config.get("chat_template")
    if template and "{{" not in template and "{%" not in template:
        template_path = Path(template)
        if not template_path.is_absolute():
            template_path = Path(__file__).resolve().parents[1] / template_path
        template = template_path.read_text(encoding="utf-8")
    model_identity = str(config.get("requested_model", model)).lower()
    if not template and "llama" in model_identity and "instruct" not in model_identity:
        raise ValueError("Base Llama requires an explicit chat_template in the experiment config")
    offload = bool(config.get("allow_cpu_offload", False))
    capture_enabled = bool(config.get("artifact_capture", True))
    capture_every = int(config.get("capture_every_step", 1))
    save_checkpoint = bool(config.get("save_checkpoint", True))
    checkpoint_every = int(config.get("checkpoint_every_steps", 1))
    checkpoint_policy = config.get("checkpoint_policy", "periodic")
    if checkpoint_policy not in {"periodic", "final_only"}:
        raise ValueError("checkpoint_policy must be periodic or final_only")
    if checkpoint_policy == "final_only" and (not save_checkpoint or checkpoint_every != steps):
        raise ValueError("final_only requires save_checkpoint=true and checkpoint_every_steps=steps")
    if capture_every < 1 or checkpoint_every < 1:
        raise ValueError("Capture and checkpoint cadence must be positive")
    if not config.get("checkpoint_keep_all", True):
        raise ValueError("Checkpoint pruning is unsupported; preserve every produced checkpoint")
    seed = int(config.get("seed", 42))
    trainer = {
        "trainer.project_name": "kisti-h100-bench",
        "trainer.experiment_name": str(config.get("name", run_dir.name)),
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.logger": ["console", "file"],
        "trainer.total_training_steps": steps,
        "trainer.total_epochs": steps + 1,
        "trainer.save_freq": checkpoint_every if save_checkpoint else -1,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
        "trainer.nnodes": nodes,
        "trainer.n_gpus_per_node": gpus,
    }
    env = ["env", f"VERL_FILE_LOGGER_PATH={run_dir / 'metrics.jsonl'}", "TOKENIZERS_PARALLELISM=false", "WANDB_MODE=disabled"]
    if capture_enabled:
        env += [f"BENCH_ARTIFACT_ROOT={run_dir / 'artifacts'}", f"BENCH_CAPTURE_EVERY_STEP={capture_every}"]
    python = str(config.get("python", sys.executable))
    if mode == "sft":
        seq = int(config.get("seq_length", 16384))
        values = {
            "data.train_files": paths["sft_train"], "data.val_files": None,
            "data.messages_key": "messages", "data.train_batch_size": batch,
            "data.micro_batch_size_per_gpu": micro, "data.max_length": seq,
            "data.max_token_len_per_gpu": seq * micro,
            "data.use_dynamic_bsz": False, "data.pad_mode": "no_padding",
            "data.truncation": "right", "data.num_workers": 0,
            "data.ignore_input_ids_mismatch": True,
            "model.path": model, "model.use_remove_padding": True,
            "model.enable_gradient_checkpointing": True,
            "engine": "fsdp", "engine.strategy": "fsdp2",
            "engine.model_dtype": "bf16", "engine.dtype": "bfloat16",
            "engine.seed": seed, "engine.use_torch_compile": False,
            "engine.param_offload": offload, "engine.optimizer_offload": offload,
            "engine.offload_policy": offload,
            "optim.lr": float(config.get("learning_rate", 1e-6)),
            "checkpoint.save_contents": ["model", "optimizer", "extra"] if save_checkpoint else ["extra"],
            "trainer.max_ckpt_to_keep": None,
            "trainer.seed": seed,
            **trainer,
        }
        if template:
            values["model.custom_chat_template"] = template
        if nodes == 1:
            launch = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
        else:
            if "node_rank" not in config or not config.get("master_addr"):
                raise ValueError("Multi-node SFT requires node_rank and master_addr in config")
            launch = [python, "-m", "torch.distributed.run", f"--nnodes={nodes}", f"--nproc_per_node={gpus}",
                      f"--node_rank={config['node_rank']}", f"--master_addr={config['master_addr']}",
                      f"--master_port={config.get('master_port', 29500)}"]
        launch += ["-m", "benchkit.verl_entry", "sft"]
    else:
        if nodes > 1 and not config.get("ray_address"):
            raise ValueError("Multi-node GRPO requires an explicitly provisioned Ray cluster and ray_address")
        n = int(config.get("rollout_n", 4))
        tp = int(config.get("rollout_tp", 1))
        if n < 2:
            raise ValueError("GRPO requires rollout_n >= 2 for response-derived group advantages")
        if tp < 1 or world % tp:
            raise ValueError("rollout_tp must divide total GPU count")
        prompt = int(config.get("prompt_length", 8192))
        response = int(config.get("response_length", 8192))
        prefix_cache = bool(config.get("enable_prefix_caching", True))
        if config.get("framework_lane") in {"a6000", "a6000-dev"} and not prefix_cache:
            raise ValueError("verl v0.8.0 omits false vLLM bool CLI flags; enable_prefix_caching=false is unsupported in the development lane")
        values = {
            "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
            "data.train_files": paths["rl_train"], "data.val_files": paths["rl_val"],
            "data.train_batch_size": batch, "data.max_prompt_length": prompt,
            "data.max_response_length": response, "data.filter_overlong_prompts": True,
            "data.truncation": "error", "data.shuffle": False, "data.seed": seed,
            "data.dataloader_num_workers": 0,
            "actor_rollout_ref.model.path": model,
            "actor_rollout_ref.model.use_remove_padding": True,
            "actor_rollout_ref.model.enable_gradient_checkpointing": True,
            "actor_rollout_ref.actor.strategy": "fsdp2",
            "actor_rollout_ref.actor.optim.lr": float(config.get("learning_rate", 1e-6)),
            "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": micro,
            "actor_rollout_ref.actor.ppo_epochs": 1,
            "actor_rollout_ref.actor.use_dynamic_bsz": False,
            "actor_rollout_ref.actor.use_kl_loss": True,
            "actor_rollout_ref.actor.kl_loss_coef": 0.01,
            "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
            "actor_rollout_ref.actor.entropy_coeff": 0.0,
            "actor_rollout_ref.actor.use_torch_compile": False,
            "actor_rollout_ref.actor.entropy_from_logits_with_chunking": True,
            "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"] if save_checkpoint else ["extra"],
            "actor_rollout_ref.actor.checkpoint.async_save": False,
            "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
            "actor_rollout_ref.actor.fsdp_config.seed": seed,
            "actor_rollout_ref.actor.fsdp_config.param_offload": offload,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": offload,
            "actor_rollout_ref.actor.fsdp_config.offload_policy": offload,
            "actor_rollout_ref.ref.strategy": "fsdp2",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": micro,
            "actor_rollout_ref.ref.entropy_from_logits_with_chunking": True,
            "actor_rollout_ref.ref.use_torch_compile": False,
            "actor_rollout_ref.ref.fsdp_config.model_dtype": "bf16",
            "actor_rollout_ref.ref.fsdp_config.param_offload": offload,
            "actor_rollout_ref.ref.fsdp_config.offload_policy": offload,
            "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size": tp,
            "actor_rollout_ref.rollout.n": n,
            "actor_rollout_ref.rollout.temperature": float(config.get("temperature", 1.0)),
            "actor_rollout_ref.rollout.gpu_memory_utilization": float(config.get("rollout_memory_fraction", 0.5)),
            "actor_rollout_ref.rollout.max_model_len": prompt + response,
            "actor_rollout_ref.rollout.max_num_batched_tokens": max(2048, prompt + response),
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": micro,
            "actor_rollout_ref.rollout.free_cache_engine": True,
            "actor_rollout_ref.rollout.enforce_eager": bool(config.get("enforce_eager", False)),
            "actor_rollout_ref.rollout.enable_prefix_caching": prefix_cache,
            "reward.custom_reward_function.path": str(Path(__file__).with_name("reward.py")),
            "reward.custom_reward_function.name": "compute_score",
            "reward.num_workers": min(4, max(1, int(config.get("cpus", 8)) // 2)),
            "trainer.val_before_train": False,
            "trainer.rollout_data_dir": str(run_dir / "rollouts"),
            "trainer.validation_data_dir": str(run_dir / "validation"),
            "trainer.max_actor_ckpt_to_keep": None,
            "trainer.max_critic_ckpt_to_keep": None,
            "ray_kwargs.ray_init.num_cpus": int(config.get("cpus", os.environ.get("SLURM_CPUS_PER_TASK", 8))),
            **trainer,
        }
        if template:
            values["actor_rollout_ref.model.custom_chat_template"] = template
        if config.get("ray_address"):
            values["+ray_kwargs.ray_init.address"] = config["ray_address"]
        launch = [python, "-m", "benchkit.verl_entry", "grpo"]
    return [env + launch + _overrides(values)]
