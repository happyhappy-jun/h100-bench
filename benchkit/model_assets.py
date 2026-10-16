"""Derive run-local tokenizer metadata while preserving exact source weights.

verl loads separate tokenizers for the dataset, policy worker and rollout. A
custom template on HFModelConfig does not reach every one of those loaders.
The derived directory makes the declared template visible through ordinary
from_pretrained calls, without patching verl or editing the HF snapshot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare_model_assets(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Return a config pointing to a derived local bundle when a template exists.

Call after resolving the original model revision. Source metadata and weights
remain unchanged. The bundle belongs to this immutable attempt directory.
    """
    result = dict(config)
    declared = result.get("chat_template")
    if not declared:
        return result
    source = Path(result["model"]).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Resolve the model to a local snapshot before preparing tokenizer assets")
    if "{{" in declared or "{%" in declared:
        template = declared
        template_source = "inline_config"
    else:
        template_path = Path(declared).expanduser()
        if not template_path.is_absolute():
            template_path = Path(__file__).resolve().parents[1] / template_path
        template = template_path.read_text(encoding="utf-8")
        template_source = str(template_path.resolve())
    if not template.strip():
        raise ValueError("The declared chat template is empty")
    original_tokenizer_path = source / "tokenizer_config.json"
    original_bytes = original_tokenizer_path.read_bytes() if original_tokenizer_path.is_file() else b"{}"
    tokenizer_config = json.loads(original_bytes)
    if not isinstance(tokenizer_config, dict):
        raise ValueError("tokenizer_config.json must contain a JSON object")
    tokenizer_config["chat_template"] = template
    bundle = Path(run_dir).resolve() / "model_assets"
    if bundle == source or source in bundle.parents:
        raise ValueError("Derived assets must be outside the immutable source model directory")
    bundle.mkdir(parents=True, exist_ok=False)
    symlinks = []
    # Transformers may prefer chat_template.jinja or a named template directory
    # to tokenizer_config. Exclude all three original locations so one declared
    # template is used consistently by every loader.
    replaced = {"tokenizer_config.json", "chat_template.jinja", "chat_templates"}
    for entry in sorted(source.iterdir()):
        if entry.name.startswith(".") or entry.name in replaced:
            continue
        target = entry.resolve(strict=True)
        (bundle / entry.name).symlink_to(target, target_is_directory=target.is_dir())
        symlinks.append({"name": entry.name, "source": str(target)})
    updated_bytes = (json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (bundle / "tokenizer_config.json").write_bytes(updated_bytes)
    (bundle / "chat_template.jinja").write_text(template, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "operation": "derived tokenizer metadata; model weights are immutable symlinks",
        "source_model_path": str(source),
        "requested_model": result.get("requested_model", result["model"]),
        "resolved_model_revision": result.get("resolved_model_revision"),
        "template_source": template_source,
        "chat_template_sha256": _sha256(template.encode("utf-8")),
        "original_tokenizer_config_sha256": _sha256(original_bytes),
        "derived_tokenizer_config_sha256": _sha256(updated_bytes),
        "linked_assets": symlinks,
    }
    manifest_path = Path(run_dir).resolve() / "model_assets_manifest.json"
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["model_source_path"] = str(source)
    result["model"] = str(bundle)
    result["chat_template_sha256"] = metadata["chat_template_sha256"]
    result["derived_tokenizer_config_sha256"] = metadata["derived_tokenizer_config_sha256"]
    result["model_assets_manifest"] = str(manifest_path)
    return result
