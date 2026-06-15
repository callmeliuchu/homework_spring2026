#!/usr/bin/env python3
"""A minimal, standalone DPO walkthrough for the final_project_llm_rl data.

This file intentionally does not import llm_rl_final_proj.*.  It is a compact
teaching version of the DPO path:

  JSONL rows -> preference examples -> tokenized chosen/rejected batches
  -> policy/reference log-probabilities -> DPO loss -> LoRA optimizer step
  -> preference evaluation -> generation check -> adapter save.

Run a tiny smoke experiment on the real local dataset:

  python minimal_dpo_walkthrough.py \
    --dataset_dir dataset/wildchat_min4_judged_5k_v1 \
    --train_limit 16 --eval_limit 8 --generation_limit 2 --max_steps 2

Run without any dataset folder, using two embedded toy examples:

  python minimal_dpo_walkthrough.py --toy_data --max_steps 1

For a real run, increase the limits/steps and use an A100-like GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# 1. Simple data containers
# -----------------------------


Message = dict[str, str]


@dataclass
class PreferenceExample:
    row_id: str
    prompt_messages: list[Message]
    prompt_text: str
    chosen_text: str
    rejected_text: str


@dataclass
class GenerationExample:
    row_id: str
    prompt_messages: list[Message]
    prompt_text: str
    reference_response: str | None


class PreferenceDataset(Dataset):
    def __init__(self, examples: list[PreferenceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]


@dataclass
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_response_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_response_mask: torch.Tensor
    row_ids: list[str]
    prompt_texts: list[str]
    chosen_texts: list[str]
    rejected_texts: list[str]

    def to(self, device: torch.device) -> "PreferenceBatch":
        return PreferenceBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            chosen_response_mask=self.chosen_response_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
            rejected_response_mask=self.rejected_response_mask.to(device),
            row_ids=self.row_ids,
            prompt_texts=self.prompt_texts,
            chosen_texts=self.chosen_texts,
            rejected_texts=self.rejected_texts,
        )


@dataclass
class SequenceScores:
    chosen_logp_sum: torch.Tensor
    rejected_logp_sum: torch.Tensor
    chosen_logp_mean: torch.Tensor
    rejected_logp_mean: torch.Tensor


# -----------------------------
# 2. Dataset loading
# -----------------------------


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Load local JSONL rows.  The project dataset stores one example per line."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def normalize_messages(obj: Any) -> list[Message]:
    """Accept the local prompt_messages format and a few common chat variants."""
    if obj is None:
        return []
    if isinstance(obj, str):
        return [{"role": "user", "content": obj}]
    if isinstance(obj, dict):
        if "messages" in obj:
            return normalize_messages(obj["messages"])
        return [{"role": str(obj.get("role", "user")), "content": str(obj.get("content", ""))}]
    if isinstance(obj, Iterable) and not isinstance(obj, (bytes, bytearray)):
        out: list[Message] = []
        for item in obj:
            if isinstance(item, dict):
                out.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
            else:
                out.append({"role": "user", "content": str(item)})
        return out
    return [{"role": "user", "content": str(obj)}]


def format_messages(messages: list[Message]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_preference_examples(dataset_dir: Path, split: str, limit: int) -> list[PreferenceExample]:
    """Turn train_prefs/test_prefs JSONL rows into chosen/rejected pairs."""
    rows = load_jsonl(dataset_dir / f"{split}.jsonl", limit=limit)
    examples: list[PreferenceExample] = []
    for i, row in enumerate(rows):
        prompt_messages = normalize_messages(row.get("prompt_messages", row.get("prompt")))
        examples.append(
            PreferenceExample(
                row_id=str(row.get("row_id", row.get("id", i))),
                prompt_messages=prompt_messages,
                prompt_text=str(row.get("prompt_text", format_messages(prompt_messages))),
                chosen_text=str(row["chosen_text"]),
                rejected_text=str(row["rejected_text"]),
            )
        )
    return examples


def build_generation_examples(dataset_dir: Path, split: str, limit: int) -> list[GenerationExample]:
    """Turn test_gen/train_gen rows into prompts for generation sanity checks."""
    rows = load_jsonl(dataset_dir / f"{split}.jsonl", limit=limit)
    examples: list[GenerationExample] = []
    for i, row in enumerate(rows):
        prompt_messages = normalize_messages(row.get("prompt_messages", row.get("prompt")))
        examples.append(
            GenerationExample(
                row_id=str(row.get("row_id", row.get("id", i))),
                prompt_messages=prompt_messages,
                prompt_text=str(row.get("prompt_text", format_messages(prompt_messages))),
                reference_response=row.get("chosen_text"),
            )
        )
    return examples


def build_toy_preference_examples() -> list[PreferenceExample]:
    """Tiny embedded preference data so the file can run without any dataset folder."""
    return [
        PreferenceExample(
            row_id="toy-1",
            prompt_messages=[{"role": "user", "content": "Give one safety tip for hiking."}],
            prompt_text="Give one safety tip for hiking.",
            chosen_text="Bring enough water and tell someone your route before you leave.",
            rejected_text="Just go wherever looks fun and do not worry about planning.",
        ),
        PreferenceExample(
            row_id="toy-2",
            prompt_messages=[{"role": "user", "content": "Explain photosynthesis in one sentence."}],
            prompt_text="Explain photosynthesis in one sentence.",
            chosen_text="Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar and oxygen.",
            rejected_text="Photosynthesis is when animals sleep during the day.",
        ),
    ]


def build_toy_generation_examples() -> list[GenerationExample]:
    return [
        GenerationExample(
            row_id="toy-gen-1",
            prompt_messages=[{"role": "user", "content": "Give one safety tip for hiking."}],
            prompt_text="Give one safety tip for hiking.",
            reference_response=None,
        )
    ]


# -----------------------------
# 3. Tokenization and batching
# -----------------------------


def tokenize_prompt_with_response(
    tokenizer,
    prompt_messages: list[Message],
    response_text: str,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> tuple[torch.Tensor, int]:
    """Build one full chat sequence and remember which final tokens are response."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0]
    full_ids = tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": response_text}],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )[0]

    # If the prompt is too long, drop earliest prompt tokens and keep the response.
    if prompt_ids.numel() > max_prompt_tokens:
        drop = int(prompt_ids.numel() - max_prompt_tokens)
        prompt_ids = prompt_ids[drop:]
        full_ids = full_ids[drop:]

    response_len = int(full_ids.numel() - prompt_ids.numel())
    if response_len <= 0:
        raise ValueError("Response produced zero tokens after chat-template tokenization.")

    # DPO scores only the completion, so cap the completion length.
    if response_len > max_response_tokens:
        full_ids = full_ids[: int(prompt_ids.numel() + max_response_tokens)]
        response_len = max_response_tokens
    return full_ids, response_len


def left_pad_with_response_mask(
    sequences: list[torch.Tensor],
    response_lengths: list[int],
    pad_token_id: int,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad input_ids and create a shifted mask over completion tokens.

    The model predicts token t from tokens < t, so logprobs have shape [B, L-1].
    response_mask therefore also has shape [B, L-1] and marks only answer tokens.
    """
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    response_mask = torch.zeros((len(sequences), max_len - 1), dtype=torch.float32)

    for i, (ids, response_len) in enumerate(zip(sequences, response_lengths)):
        n = int(ids.numel())
        input_ids[i, max_len - n :] = ids
        attention_mask[i, max_len - n :] = 1
        response_mask[i, (max_len - 1) - response_len :] = 1.0
    return input_ids, attention_mask, response_mask


class PreferenceCollator:
    """DataLoader collate_fn: examples -> one padded chosen/rejected tensor batch."""

    def __init__(self, tokenizer, max_prompt_tokens: int, max_response_tokens: int):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens

    def __call__(self, examples: list[PreferenceExample]) -> PreferenceBatch:
        chosen_ids, rejected_ids = [], []
        chosen_lens, rejected_lens = [], []

        for ex in examples:
            c_ids, c_len = tokenize_prompt_with_response(
                self.tokenizer, ex.prompt_messages, ex.chosen_text, self.max_prompt_tokens, self.max_response_tokens
            )
            r_ids, r_len = tokenize_prompt_with_response(
                self.tokenizer, ex.prompt_messages, ex.rejected_text, self.max_prompt_tokens, self.max_response_tokens
            )
            chosen_ids.append(c_ids)
            rejected_ids.append(r_ids)
            chosen_lens.append(c_len)
            rejected_lens.append(r_len)

        max_len = max(max(x.numel() for x in chosen_ids), max(x.numel() for x in rejected_ids))
        c_input, c_attn, c_mask = left_pad_with_response_mask(
            chosen_ids, chosen_lens, int(self.tokenizer.pad_token_id), int(max_len)
        )
        r_input, r_attn, r_mask = left_pad_with_response_mask(
            rejected_ids, rejected_lens, int(self.tokenizer.pad_token_id), int(max_len)
        )

        return PreferenceBatch(
            chosen_input_ids=c_input,
            chosen_attention_mask=c_attn,
            chosen_response_mask=c_mask,
            rejected_input_ids=r_input,
            rejected_attention_mask=r_attn,
            rejected_response_mask=r_mask,
            row_ids=[ex.row_id for ex in examples],
            prompt_texts=[ex.prompt_text for ex in examples],
            chosen_texts=[ex.chosen_text for ex in examples],
            rejected_texts=[ex.rejected_text for ex in examples],
        )


# -----------------------------
# 4. Model loading
# -----------------------------


def load_lora_policy(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    """Load base LM, wrap LoRA adapters, and freeze all non-LoRA weights."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=args.local_files_only)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if args.grad_checkpointing:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        base.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg).to(device)

    # Only LoRA matrices should train; the original model is the frozen reference.
    for name, p in model.named_parameters():
        train_this = "lora_" in name
        p.requires_grad_(train_this)
        if train_this:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable_params={trainable:,} total_params={total:,} fraction={trainable / total:.6f}")
    return model, tokenizer


def disable_adapter_context(model):
    """Temporarily disable LoRA to score the same batch with the frozen reference."""
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


# -----------------------------
# 5. Log-probs, DPO loss, eval
# -----------------------------


def per_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, enable_grad: bool) -> torch.Tensor:
    """Return log p(x_t | x_<t) for each observed next token, shape [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]       # predictions for next token
        targets = input_ids[:, 1:]           # actual next tokens
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def sequence_scores(model, batch: PreferenceBatch, enable_grad: bool) -> SequenceScores:
    """Score chosen and rejected responses under the current model."""
    input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0)
    attention_mask = torch.cat([batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0)
    response_mask = torch.cat([batch.chosen_response_mask, batch.rejected_response_mask], dim=0)

    token_logp = per_token_logprobs(model, input_ids, attention_mask, enable_grad=enable_grad)
    seq_sum = (token_logp * response_mask).sum(dim=1)
    seq_mean = masked_mean_per_row(token_logp, response_mask)
    chosen_sum, rejected_sum = seq_sum.chunk(2, dim=0)
    chosen_mean, rejected_mean = seq_mean.chunk(2, dim=0)
    return SequenceScores(chosen_sum, rejected_sum, chosen_mean, rejected_mean)


def policy_and_reference_scores(model, batch: PreferenceBatch, policy_grad: bool) -> tuple[SequenceScores, SequenceScores]:
    """Compute trainable policy scores and frozen-reference scores for DPO."""
    policy = sequence_scores(model, batch, enable_grad=policy_grad)
    with torch.no_grad():
        with disable_adapter_context(model):
            reference = sequence_scores(model, batch, enable_grad=False)
    return policy, reference


def dpo_loss(policy: SequenceScores, reference: SequenceScores, beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO: prefer chosen over rejected more than the frozen reference does."""
    policy_margin = policy.chosen_logp_sum - policy.rejected_logp_sum
    reference_margin = reference.chosen_logp_sum - reference.rejected_logp_sum
    logits = policy_margin - reference_margin
    loss = -F.logsigmoid(beta * logits).mean()
    metrics = {
        "loss": float(loss.detach().item()),
        "policy_accuracy_sum": float((policy_margin.detach() > 0).float().mean().item()),
        "reference_corrected_accuracy": float((logits.detach() > 0).float().mean().item()),
        "policy_margin_sum_mean": float(policy_margin.detach().mean().item()),
        "reference_margin_sum_mean": float(reference_margin.detach().mean().item()),
        "reference_corrected_margin_mean": float(logits.detach().mean().item()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_preferences(
    model,
    examples: list[PreferenceExample],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate whether chosen gets higher log-prob than rejected."""
    collator = PreferenceCollator(tokenizer, args.max_prompt_tokens, args.max_response_tokens)
    loader = DataLoader(examples, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
    policy_margins, corrected_margins = [], []
    model.eval()
    for batch in tqdm(loader, desc="eval[prefs]", dynamic_ncols=True):
        batch = batch.to(device)
        policy, reference = policy_and_reference_scores(model, batch, policy_grad=False)
        pm = policy.chosen_logp_sum - policy.rejected_logp_sum
        rm = reference.chosen_logp_sum - reference.rejected_logp_sum
        policy_margins.append(pm.cpu())
        corrected_margins.append((pm - rm).cpu())
    policy_all = torch.cat(policy_margins)
    corrected_all = torch.cat(corrected_margins)
    return {
        "pref_accuracy_sum_logp": float((policy_all > 0).float().mean().item()),
        "reference_corrected_pref_accuracy": float((corrected_all > 0).float().mean().item()),
        "pref_margin_sum_logp_mean": float(policy_all.mean().item()),
        "reference_corrected_margin_mean": float(corrected_all.mean().item()),
        "count_preference_pairs": float(policy_all.numel()),
    }


@torch.no_grad()
def generate_samples(model, tokenizer, examples: list[GenerationExample], args: argparse.Namespace, device: torch.device):
    """Generate a few completions so you can inspect behavior qualitatively."""
    if not examples:
        return []
    rows = []
    old_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.eval()
    for ex in examples:
        ids = tokenizer.apply_chat_template(
            ex.prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0][-args.max_prompt_tokens :].unsqueeze(0).to(device)
        attn = torch.ones_like(ids)
        gen_kwargs = {
            "input_ids": ids,
            "attention_mask": attn,
            "max_new_tokens": args.generation_max_new_tokens,
            "do_sample": args.generation_temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.generation_temperature > 0:
            gen_kwargs["temperature"] = args.generation_temperature
            gen_kwargs["top_p"] = args.generation_top_p
        out = model.generate(**gen_kwargs)
        text = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        rows.append({"row_id": ex.row_id, "prompt": ex.prompt_text, "model_response": text})
    model.config.use_cache = old_cache
    return rows


# -----------------------------
# 6. Training loop
# -----------------------------


def warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    scale = 1.0 if warmup_steps <= 0 else min(1.0, float(step + 1) / float(warmup_steps))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={device} dtype={dtype}")
    print(f"[setup] dataset_dir={args.dataset_dir} toy_data={args.toy_data}")

    # Load preference pairs for DPO and prompt-only rows for generation checks.
    # toy_data mode makes this file runnable without any project dataset folder.
    if args.toy_data:
        toy_prefs = build_toy_preference_examples()
        train_examples = toy_prefs[: args.train_limit or None]
        eval_examples = toy_prefs[: args.eval_limit or None]
        gen_examples = build_toy_generation_examples()[: args.generation_limit or None]
    else:
        train_examples = build_preference_examples(Path(args.dataset_dir), args.train_split, args.train_limit)
        eval_examples = build_preference_examples(Path(args.dataset_dir), args.eval_split, args.eval_limit)
        gen_examples = build_generation_examples(Path(args.dataset_dir), args.generation_split, args.generation_limit)
    print(f"[data] train={len(train_examples)} eval={len(eval_examples)} generation={len(gen_examples)}")
    print("[data] first row:", json.dumps(train_examples[0].__dict__, ensure_ascii=False)[:1000], "...")

    model, tokenizer = load_lora_policy(args, device, dtype)
    collator = PreferenceCollator(tokenizer, args.max_prompt_tokens, args.max_response_tokens)
    train_loader = DataLoader(
        PreferenceDataset(train_examples),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    print("[eval] baseline:")
    print(json.dumps(evaluate_preferences(model, eval_examples, tokenizer, args, device), indent=2))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    start_time = time.perf_counter()
    progress = tqdm(total=args.max_steps, desc="train[dpo]", dynamic_ncols=True)

    while optimizer_step < args.max_steps:
        for batch in train_loader:
            # One microbatch: score chosen/rejected under policy and frozen reference.
            batch = batch.to(device)
            policy, reference = policy_and_reference_scores(model, batch, policy_grad=True)

            # DPO objective: increase reference-corrected chosen-vs-rejected gap.
            loss, metrics = dpo_loss(policy, reference, beta=args.beta)
            (loss / args.grad_accum_steps).backward()
            micro_step += 1

            # Optimizer step can accumulate gradients over multiple microbatches.
            if micro_step % args.grad_accum_steps == 0:
                warmup_lr(optimizer, args.lr, optimizer_step, args.warmup_steps)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['policy_accuracy_sum']:.3f}")

                print(
                    json.dumps(
                        {
                            "step": optimizer_step,
                            "micro_step": micro_step,
                            "lr": optimizer.param_groups[0]["lr"],
                            "grad_norm": float(grad_norm.item()),
                            **metrics,
                        },
                        sort_keys=True,
                    )
                )

                if args.eval_interval > 0 and optimizer_step % args.eval_interval == 0:
                    print(f"[eval] step={optimizer_step}")
                    print(json.dumps(evaluate_preferences(model, eval_examples, tokenizer, args, device), indent=2))

                if optimizer_step >= args.max_steps:
                    break
        # Keep cycling through the small dataset until max_steps is reached.

    progress.close()

    print("[eval] final:")
    final_metrics = evaluate_preferences(model, eval_examples, tokenizer, args, device)
    print(json.dumps(final_metrics, indent=2))

    print("[generation] qualitative samples:")
    sample_rows = generate_samples(model, tokenizer, gen_examples, args, device)
    for row in sample_rows:
        print(json.dumps(row, ensure_ascii=False, indent=2)[:2000])

    # Save only the LoRA adapter, matching the original project convention.
    adapter_dir = output_dir / "adapter_final"
    model.save_pretrained(adapter_dir)
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(f"[done] saved_adapter={adapter_dir} elapsed_seconds={time.perf_counter() - start_time:.1f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone minimal DPO walkthrough.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset_dir", default="dataset/wildchat_min4_judged_5k_v1")
    p.add_argument("--train_split", default="train_prefs")
    p.add_argument("--eval_split", default="test_prefs")
    p.add_argument("--generation_split", default="test_gen")
    p.add_argument("--output_dir", default="runs/minimal_dpo_walkthrough")
    p.add_argument("--device", default="")
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--toy_data", action="store_true", help="Use embedded examples instead of reading dataset_dir.")

    # Tiny defaults are for learning the flow.  Increase for actual training.
    p.add_argument("--train_limit", type=int, default=16)
    p.add_argument("--eval_limit", type=int, default=8)
    p.add_argument("--generation_limit", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=2)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=1)

    p.add_argument("--beta", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max_prompt_tokens", type=int, default=700)
    p.add_argument("--max_response_tokens", type=int, default=512)
    p.add_argument("--generation_max_new_tokens", type=int, default=128)
    p.add_argument("--generation_temperature", type=float, default=0.0)
    p.add_argument("--generation_top_p", type=float, default=1.0)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--grad_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
