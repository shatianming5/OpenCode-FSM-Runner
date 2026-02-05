from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env
from runner.dotenv import load_dotenv


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"not_json_object: {path}")
    return data


def _resolve_samples_path(repo_root: Path, rollout_json: dict[str, Any]) -> Path:
    paths = rollout_json.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("rollout_json_missing_paths")
    raw = paths.get("samples_jsonl")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("rollout_json_missing_paths.samples_jsonl")
    p = Path(raw.strip())
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _read_samples_jsonl(samples_path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and len(out) >= int(limit):
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            out.append(obj)
    return out


def _lazy_import_ml():
    """Lazy import optional ML deps (torch/transformers).

    This script supports a `--dry-run` mode which exercises env.setup/rollout/evaluation
    without requiring any ML dependencies.
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing ML dependencies. Install with:\n"
            "  pip install -r requirements-ml.txt\n"
            "(and ensure your torch build matches your platform/CUDA)"
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer


def _pick_device(torch: Any, device: str) -> str:
    d = str(device or "auto").strip().lower() or "auto"
    if d == "auto":
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):  # pragma: no cover
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):  # pragma: no cover
            return "mps"
        return "cpu"
    if d in ("cpu", "cuda", "mps"):
        return d
    raise ValueError(f"invalid --device: {device} (expected auto/cpu/cuda/mps)")


@dataclass(frozen=True)
class PpoLiteConfig:
    """A minimal PPO-style config (no benchmark-specific logic)."""

    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    epochs: int = 1
    minibatch_size: int = 4
    max_seq_len: int = 1024
    max_grad_norm: float = 1.0


def _build_batch(
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    max_seq_len: int,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    """Tokenize prompt+completion samples into a padded batch.

    Returns: input_ids, attention_mask, completion_mask, rewards
    - completion_mask marks token positions belonging to completion (same shape as input_ids).
    """
    torch = __import__("torch")

    input_ids_list: list[list[int]] = []
    completion_mask_list: list[list[int]] = []
    rewards_list: list[float] = []

    for s in samples:
        prompt = s.get("prompt")
        completion = s.get("completion")
        reward = s.get("reward")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        if not isinstance(reward, (int, float)):
            continue

        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        c_ids = tokenizer.encode(completion, add_special_tokens=False)
        if not p_ids:
            continue

        if len(p_ids) >= int(max_seq_len):
            # Too long prompt: skip (keeps boundary definition simple and stable).
            continue

        allowed_c = int(max_seq_len) - len(p_ids)
        if allowed_c <= 0:
            continue
        c_ids = c_ids[:allowed_c]
        if not c_ids:
            continue

        ids = list(p_ids) + list(c_ids)
        mask = [0] * len(p_ids) + [1] * len(c_ids)
        input_ids_list.append(ids)
        completion_mask_list.append(mask)
        rewards_list.append(float(reward))

    if not input_ids_list:
        raise RuntimeError("no_valid_samples (need prompt+completion+reward)")

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    max_len = max(len(x) for x in input_ids_list)
    batch_ids: list[list[int]] = []
    batch_attn: list[list[int]] = []
    batch_comp: list[list[int]] = []
    for ids, cm in zip(input_ids_list, completion_mask_list, strict=True):
        pad_n = max_len - len(ids)
        batch_ids.append(ids + [int(pad_id)] * pad_n)
        batch_attn.append([1] * len(ids) + [0] * pad_n)
        batch_comp.append(cm + [0] * pad_n)

    input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(batch_attn, dtype=torch.long, device=device)
    completion_mask = torch.tensor(batch_comp, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    return input_ids, attention_mask, completion_mask, rewards


def _logp_of_completions(model: Any, input_ids: Any, attention_mask: Any, completion_mask: Any) -> Any:
    torch = __import__("torch")

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    # Shift: logits[:, t] predicts token at t+1.
    logits2 = logits[:, :-1, :]
    target = input_ids[:, 1:]
    logp = torch.log_softmax(logits2, dim=-1)
    token_logp = logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].to(token_logp.dtype)
    return (token_logp * mask).sum(dim=-1)


def ppo_lite_update(
    *,
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    cfg: PpoLiteConfig,
    device: str,
) -> dict[str, Any]:
    """Minimal PPO-style update using (prompt, completion, reward) samples.

    Notes:
    - This intentionally avoids benchmark-specific assumptions.
    - KL is implemented as a bounded trust-region penalty using squared log-ratio vs `logp_old`.
    """
    torch = __import__("torch")

    model.train()
    if getattr(getattr(model, "config", None), "use_cache", None) is not None:
        try:
            model.config.use_cache = False  # type: ignore[attr-defined]
        except Exception:
            pass

    input_ids, attention_mask, completion_mask, rewards = _build_batch(
        tokenizer, samples, max_seq_len=int(cfg.max_seq_len), device=device
    )

    # Normalize rewards -> advantages.
    #
    # For very small smoke runs, we still want a meaningful gradient signal. With a single sample,
    # mean-centering would make advantage == 0 and produce a no-op update. Therefore we treat the
    # single-sample case as "no baseline" (adv = reward).
    if int(rewards.numel()) <= 1:
        adv = rewards.clone()
    else:
        adv = rewards - rewards.mean()
        adv = adv / (adv.std(unbiased=False) + 1e-6)

    with torch.no_grad():
        logp_old = _logp_of_completions(model, input_ids, attention_mask, completion_mask).detach()
        if not torch.isfinite(logp_old).all():
            raise RuntimeError("non_finite_logp_old")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))

    n = int(input_ids.shape[0])
    indices = list(range(n))
    total_loss = 0.0
    total_kl = 0.0
    steps = 0

    for _epoch in range(int(max(1, cfg.epochs))):
        random.shuffle(indices)
        for start in range(0, n, int(max(1, cfg.minibatch_size))):
            mb = indices[start : start + int(max(1, cfg.minibatch_size))]
            mb_ids = input_ids[mb]
            mb_attn = attention_mask[mb]
            mb_comp = completion_mask[mb]
            mb_adv = adv[mb]
            mb_logp_old = logp_old[mb]

            logp_new = _logp_of_completions(model, mb_ids, mb_attn, mb_comp)
            ratio = torch.exp(logp_new - mb_logp_old)
            clipped = torch.clamp(ratio, 1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps))

            pg1 = ratio * mb_adv
            pg2 = clipped * mb_adv
            # PPO clipping: min for positive adv, max for negative adv.
            pg = torch.where(mb_adv >= 0, torch.minimum(pg1, pg2), torch.maximum(pg1, pg2))
            loss_pg = -pg.mean()

            log_ratio = (logp_new - mb_logp_old)
            loss_kl = (log_ratio * log_ratio).mean()
            loss = loss_pg + float(cfg.kl_coef) * loss_kl

            if not torch.isfinite(loss).all():
                raise RuntimeError("non_finite_loss")

            loss.backward()
            if float(cfg.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().cpu().item())
            total_kl += float(loss_kl.detach().cpu().item())
            steps += 1

    with torch.no_grad():
        logp_final = _logp_of_completions(model, input_ids, attention_mask, completion_mask).detach()
        if not torch.isfinite(logp_final).all():
            raise RuntimeError("non_finite_logp_final")
        log_ratio_final = (logp_final - logp_old).to(torch.float32)
        approx_kl_final = float((log_ratio_final * log_ratio_final).mean().cpu().item())
        delta_logp_mean = float(log_ratio_final.mean().cpu().item())
        delta_logp_abs_mean = float(log_ratio_final.abs().mean().cpu().item())

    return {
        "ok": True,
        "steps": int(steps),
        "loss": (total_loss / max(1, steps)),
        "approx_kl": (total_kl / max(1, steps)),
        "approx_kl_final": approx_kl_final,
        "delta_logp_mean": delta_logp_mean,
        "delta_logp_abs_mean": delta_logp_abs_mean,
        "n_samples": int(n),
        "reward_mean": float(rewards.mean().detach().cpu().item()),
        "reward_std": float(rewards.std(unbiased=False).detach().cpu().item()) if int(rewards.numel()) > 1 else 0.0,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-file RL post-training runner: env.setup -> env.rollout -> PPO-style update -> env.evaluation."
    )
    p.add_argument("--bench", action="append", default=[], help="repo path or URL (repeatable)")
    p.add_argument("--out-root", required=True, help="output directory for checkpoints and summary")
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id or local dir (used when not --dry-run)",
    )
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps (used when not --dry-run)")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "fp32", "bf16", "fp16"),
        help="model dtype for training (auto prefers stability; fp16 may produce NaNs without AMP)",
    )
    p.add_argument("--seed", type=int, default=0)

    # OpenCode/scaffolding controls (benchmark-agnostic).
    p.add_argument("--opencode-model", default="", help="OpenCode model for scaffolding (provider/model)")
    p.add_argument("--opencode-url", default="", help="OpenCode server base URL (optional)")
    p.add_argument("--unattended", choices=("strict", "guided"), default="strict")
    p.add_argument(
        "--env-file",
        default="",
        help="dotenv file to load before running (empty disables). Example: /data/userdata/v-tiansha/.env",
    )
    p.add_argument("--env-override", action="store_true", help="override existing env vars with values from --env-file")

    # Bench stage controls (passed via env vars; contract scripts may choose to respect).
    p.add_argument("--eval-mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--eval-limit", type=int, default=20)

    # Training loop.
    p.add_argument("--segments", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="only run env.rollout+evaluation; skip ML training")

    # PPO-lite knobs.
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env_file = str(args.env_file or "").strip()
    if env_file:
        load_dotenv(env_file, override=bool(args.env_override))
    benches = [str(b).strip() for b in (args.bench or []) if str(b).strip()]
    if not benches:
        raise SystemExit("--bench is required (repeatable)")

    random.seed(int(args.seed or 0))
    out_root = Path(str(args.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env_overrides = {"AIDER_EVAL_MODE": str(args.eval_mode)}
    if int(args.eval_limit or 0) > 0:
        env_overrides["AIDER_EVAL_LIMIT"] = str(int(args.eval_limit))

    summary: dict[str, Any] = {
        "ts": _now_iso(),
        "dry_run": bool(args.dry_run),
        "base_model": str(args.base_model),
        "segments": int(args.segments),
        "benches": list(benches),
        "runs": [],
    }

    # Dry-run uses a dummy model dir (contract may start an echo server backend).
    dummy_model_dir = (out_root / "dummy_model_dir").resolve()
    dummy_model_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.dry_run):
        for bench in benches:
            sess = env.setup(
                bench,
                opencode_model=str(args.opencode_model),
                opencode_url=str(args.opencode_url),
                unattended=str(args.unattended),
            )
            rollout_res = env.rollout(
                dummy_model_dir,
                mode=str(args.eval_mode),
                require_samples=True,
                env_overrides=env_overrides,
            )
            eval_res = env.evaluation(mode=str(args.eval_mode), env_overrides=env_overrides)
            env.teardown()

            summary["runs"].append(
                {
                    "bench": str(bench),
                    "repo_root": str(sess.env.repo),
                    "rollout_ok": bool(rollout_res.ok),
                    "rollout_path": str(rollout_res.rollout_path) if rollout_res.rollout_path else "",
                    "eval_ok": bool(eval_res.ok),
                    "metrics": eval_res.metrics or {},
                }
            )
        (out_root / "rl_post_train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    # Training mode (requires ML deps).
    torch, AutoModelForCausalLM, AutoTokenizer = _lazy_import_ml()
    device = _pick_device(torch, str(args.device))
    torch.manual_seed(int(args.seed or 0))

    dtype_choice = str(args.dtype or "auto").strip().lower() or "auto"
    if dtype_choice == "auto":
        # Prefer stability for unattended smoke runs.
        dtype_choice = "fp32"
    if dtype_choice == "fp32":
        torch_dtype = torch.float32
    elif dtype_choice == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_choice == "fp16":
        torch_dtype = torch.float16
    else:
        raise SystemExit(f"invalid --dtype: {args.dtype}")

    model_kwargs = {"low_cpu_mem_usage": True}
    try:
        # Transformers v5+: prefer `dtype=...` (torch_dtype is deprecated).
        model = AutoModelForCausalLM.from_pretrained(str(args.base_model), dtype=torch_dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(args.base_model), torch_dtype=torch_dtype, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)

    cfg = PpoLiteConfig(
        lr=float(args.lr),
        clip_eps=float(args.clip_eps),
        kl_coef=float(args.kl_coef),
        epochs=int(args.epochs),
        minibatch_size=int(args.minibatch_size),
        max_seq_len=int(args.max_seq_len),
        max_grad_norm=float(args.max_grad_norm),
    )

    model_dir = (out_root / "model_dir").resolve()
    for seg_idx in range(int(max(1, args.segments))):
        seg_entry: dict[str, Any] = {"segment": seg_idx, "benches": [], "train": []}

        for bench in benches:
            # Export current policy for env.rollout/evaluation.
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)

            sess = env.setup(
                bench,
                opencode_model=str(args.opencode_model),
                opencode_url=str(args.opencode_url),
                unattended=str(args.unattended),
            )
            rollout_res = env.rollout(
                model_dir,
                mode=str(args.eval_mode),
                require_samples=True,
                env_overrides=env_overrides,
            )
            rollout_entry: dict[str, Any] = {
                "bench": str(bench),
                "repo_root": str(sess.env.repo),
                "rollout_ok": bool(rollout_res.ok),
                "rollout_path": str(rollout_res.rollout_path) if rollout_res.rollout_path else "",
            }

            train_metrics: dict[str, Any] = {"ok": False, "reason": "no_rollout"}
            if rollout_res.ok and rollout_res.rollout_path is not None:
                rollout_obj = _read_json(rollout_res.rollout_path)
                samples_path = _resolve_samples_path(sess.env.repo, rollout_obj)
                samples = _read_samples_jsonl(samples_path, limit=int(args.eval_limit or 0) or None)

                # Hybrid reward: if any sample is missing reward, fall back to eval.score later.
                need_eval_score = any(not isinstance(s.get("reward"), (int, float)) for s in samples)

                eval_res = env.evaluation(mode=str(args.eval_mode), env_overrides=env_overrides)
                rollout_entry["eval_ok"] = bool(eval_res.ok)
                rollout_entry["metrics"] = eval_res.metrics or {}
                score = float((eval_res.metrics or {}).get("score") or 0.0)

                if need_eval_score:
                    for s in samples:
                        if not isinstance(s.get("reward"), (int, float)):
                            s["reward"] = score

                # If we have no usable samples, do not silently proceed.
                if samples:
                    started = time.time()
                    train_metrics = ppo_lite_update(model=model, tokenizer=tokenizer, samples=samples, cfg=cfg, device=device)
                    train_metrics["wall_time_s"] = float(time.time() - started)
                else:
                    train_metrics = {"ok": False, "reason": "empty_samples_jsonl"}

            env.teardown()
            rollout_entry["train"] = train_metrics
            seg_entry["benches"].append(rollout_entry)
            seg_entry["train"].append(train_metrics)

        summary["runs"].append(seg_entry)
        (out_root / "rl_post_train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
