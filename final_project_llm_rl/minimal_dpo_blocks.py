# %%
"""Hard-coded, cell-by-cell DPO walkthrough.

This file is meant to be opened in VS Code / Cursor / PyCharm as Python cells.
Run one `# %%` block at a time and inspect the variables after each block.

It does not import `llm_rl_final_proj.*`; only normal Python packages are used.
The model is still Qwen by default.  Set USE_TOY_DATA=True if you want to avoid
reading the dataset folder while still exercising the full DPO pipeline.
"""

# %%
# 0. Imports and hard-coded configuration.

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Transformers reads some offline/cache flags at import time.  Set them before
# importing AutoTokenizer/AutoModel so notebook cell execution uses project cache.
_BOOT_PROJECT_DIR = Path("/mnt1/mnt1/nlp/lc/final_project_llm_rl")
if not _BOOT_PROJECT_DIR.exists():
    _BOOT_PROJECT_DIR = Path(__file__).resolve().parent
_BOOT_HF_HOME = _BOOT_PROJECT_DIR / ".cache/huggingface"
_BOOT_HF_HUB_CACHE = _BOOT_HF_HOME / "hub"
os.environ.setdefault("HF_HOME", str(_BOOT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_BOOT_HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_BOOT_HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path("/mnt1/mnt1/nlp/lc/final_project_llm_rl")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset/wildchat_min4_judged_5k_v1"
OUTPUT_DIR = PROJECT_DIR / "runs/minimal_dpo_blocks"
HF_HOME = PROJECT_DIR / ".cache/huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
LOCAL_QWEN_SNAPSHOT = (
    HF_HUB_CACHE
    / "models--Qwen--Qwen2.5-1.5B-Instruct"
    / "snapshots"
    / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
MODEL_NAME = str(LOCAL_QWEN_SNAPSHOT) if LOCAL_QWEN_SNAPSHOT.exists() else "Qwen/Qwen2.5-1.5B-Instruct"

USE_TOY_DATA = False
LOCAL_FILES_ONLY = True

TRAIN_SPLIT = "train_prefs"
EVAL_SPLIT = "test_prefs"
GENERATION_SPLIT = "test_gen"

TRAIN_LIMIT = 8
EVAL_LIMIT = 4
GENERATION_LIMIT = 1

MAX_PROMPT_TOKENS = 700
MAX_RESPONSE_TOKENS = 512
GENERATION_MAX_NEW_TOKENS = 96

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
MAX_STEPS = 1
GRAD_ACCUM_STEPS = 1

BETA = 0.005
LR = 5e-5
MAX_GRAD_NORM = 1.0

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
GRAD_CHECKPOINTING = True

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1" if LOCAL_FILES_ONLY else "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if LOCAL_FILES_ONLY else "0")
print({"project": str(PROJECT_DIR), "device": str(DEVICE), "dtype": str(DTYPE), "dataset": str(DATASET_DIR)})


# %%
# 1. Data structures.

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


# %%
# 2. Load raw JSONL rows or tiny embedded toy rows.

def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def toy_preference_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "toy-1",
            "prompt_messages": [{"role": "user", "content": "Give one safety tip for hiking."}],
            "prompt_text": "Give one safety tip for hiking.",
            "chosen_text": "Bring enough water and tell someone your route before you leave.",
            "rejected_text": "Just go wherever looks fun and do not worry about planning.",
        },
        {
            "row_id": "toy-2",
            "prompt_messages": [{"role": "user", "content": "Explain photosynthesis in one sentence."}],
            "prompt_text": "Explain photosynthesis in one sentence.",
            "chosen_text": "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar and oxygen.",
            "rejected_text": "Photosynthesis is when animals sleep during the day.",
        },
    ]


if USE_TOY_DATA:
    train_rows = toy_preference_rows()[:TRAIN_LIMIT]
    eval_rows = toy_preference_rows()[:EVAL_LIMIT]
    gen_rows = toy_preference_rows()[:GENERATION_LIMIT]
else:
    train_rows = load_jsonl(DATASET_DIR / f"{TRAIN_SPLIT}.jsonl", TRAIN_LIMIT)
    eval_rows = load_jsonl(DATASET_DIR / f"{EVAL_SPLIT}.jsonl", EVAL_LIMIT)
    gen_rows = load_jsonl(DATASET_DIR / f"{GENERATION_SPLIT}.jsonl", GENERATION_LIMIT)

print("raw train row keys:", sorted(train_rows[0].keys()))
print(json.dumps(train_rows[0], ensure_ascii=False, indent=2)[:1200])


# %%
# 3. Convert raw rows into simple examples used by the DPO pipeline.

def normalize_messages(obj: Any) -> list[Message]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [{"role": "user", "content": obj}]
    if isinstance(obj, dict):
        if "messages" in obj:
            return normalize_messages(obj["messages"])
        return [{"role": str(obj.get("role", "user")), "content": str(obj.get("content", ""))}]
    if isinstance(obj, Iterable) and not isinstance(obj, (bytes, bytearray)):
        out = []
        for item in obj:
            if isinstance(item, dict):
                out.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
            else:
                out.append({"role": "user", "content": str(item)})
        return out
    return [{"role": "user", "content": str(obj)}]


def format_messages(messages: list[Message]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def row_to_preference_example(row: dict[str, Any], i: int) -> PreferenceExample:
    messages = normalize_messages(row.get("prompt_messages", row.get("prompt")))
    return PreferenceExample(
        row_id=str(row.get("row_id", row.get("id", i))),
        prompt_messages=messages,
        prompt_text=str(row.get("prompt_text", format_messages(messages))),
        chosen_text=str(row["chosen_text"]),
        rejected_text=str(row["rejected_text"]),
    )


def row_to_generation_example(row: dict[str, Any], i: int) -> GenerationExample:
    messages = normalize_messages(row.get("prompt_messages", row.get("prompt")))
    return GenerationExample(
        row_id=str(row.get("row_id", row.get("id", i))),
        prompt_messages=messages,
        prompt_text=str(row.get("prompt_text", format_messages(messages))),
        reference_response=row.get("chosen_text"),
    )


train_examples = [row_to_preference_example(row, i) for i, row in enumerate(train_rows)]
eval_examples = [row_to_preference_example(row, i) for i, row in enumerate(eval_rows)]
generation_examples = [row_to_generation_example(row, i) for i, row in enumerate(gen_rows)]

print(train_examples[0])


# %%
# 4. Load Qwen tokenizer and model, then attach LoRA adapters.

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    local_files_only=LOCAL_FILES_ONLY,
    cache_dir=str(HF_HUB_CACHE),
)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    local_files_only=LOCAL_FILES_ONLY,
    cache_dir=str(HF_HUB_CACHE),
)

if GRAD_CHECKPOINTING:
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[x.strip() for x in LORA_TARGET_MODULES.split(",") if x.strip()],
    bias="none",
)

model = get_peft_model(base_model, lora_config).to(DEVICE)

# Freeze the base model; train only LoRA params.  Disabling LoRA later gives the reference model.
for name, p in model.named_parameters():
    train_this = "lora_" in name
    p.requires_grad_(train_this)
    if train_this:
        p.data = p.data.float()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print({"trainable_params": trainable_params, "total_params": total_params, "fraction": trainable_params / total_params})


# %%
# 5. Tokenize one prompt + chosen/rejected response pair.

def tokenize_prompt_with_response(
    prompt_messages: list[Message],
    response_text: str,
) -> tuple[torch.Tensor, int]:
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

    if prompt_ids.numel() > MAX_PROMPT_TOKENS:
        drop = int(prompt_ids.numel() - MAX_PROMPT_TOKENS)
        prompt_ids = prompt_ids[drop:]
        full_ids = full_ids[drop:]

    response_len = int(full_ids.numel() - prompt_ids.numel())
    if response_len > MAX_RESPONSE_TOKENS:
        full_ids = full_ids[: int(prompt_ids.numel() + MAX_RESPONSE_TOKENS)]
        response_len = MAX_RESPONSE_TOKENS
    if response_len <= 0:
        raise ValueError("response_len became zero")
    return full_ids, response_len


one = train_examples[0]
chosen_ids, chosen_response_len = tokenize_prompt_with_response(one.prompt_messages, one.chosen_text)
rejected_ids, rejected_response_len = tokenize_prompt_with_response(one.prompt_messages, one.rejected_text)

print("chosen tokens:", chosen_ids.shape, "chosen response tokens:", chosen_response_len)
print("rejected tokens:", rejected_ids.shape, "rejected response tokens:", rejected_response_len)
print("decoded chosen tail:", tokenizer.decode(chosen_ids[-80:], skip_special_tokens=False))


# %%
# 6. Build padded batch tensors and response masks.

def left_pad_with_response_mask(
    sequences: list[torch.Tensor],
    response_lengths: list[int],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.full((len(sequences), max_len), int(tokenizer.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    response_mask = torch.zeros((len(sequences), max_len - 1), dtype=torch.float32)

    for i, (ids, response_len) in enumerate(zip(sequences, response_lengths)):
        n = int(ids.numel())
        input_ids[i, max_len - n :] = ids
        attention_mask[i, max_len - n :] = 1
        response_mask[i, (max_len - 1) - response_len :] = 1.0
    return input_ids, attention_mask, response_mask


class PreferenceCollator:
    def __call__(self, examples: list[PreferenceExample]) -> PreferenceBatch:
        c_ids, r_ids, c_lens, r_lens = [], [], [], []
        for ex in examples:
            ids, n = tokenize_prompt_with_response(ex.prompt_messages, ex.chosen_text)
            c_ids.append(ids)
            c_lens.append(n)
            ids, n = tokenize_prompt_with_response(ex.prompt_messages, ex.rejected_text)
            r_ids.append(ids)
            r_lens.append(n)

        max_len = max(max(x.numel() for x in c_ids), max(x.numel() for x in r_ids))
        c_input, c_attn, c_mask = left_pad_with_response_mask(c_ids, c_lens, int(max_len))
        r_input, r_attn, r_mask = left_pad_with_response_mask(r_ids, r_lens, int(max_len))
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


collator = PreferenceCollator()
train_loader = DataLoader(PreferenceDataset(train_examples), batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collator)
eval_loader = DataLoader(PreferenceDataset(eval_examples), batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)

batch = next(iter(train_loader))
print("chosen_input_ids:", batch.chosen_input_ids.shape)
print("chosen_response_mask:", batch.chosen_response_mask.shape, "response token counts:", batch.chosen_response_mask.sum(dim=1))
print("row_ids:", batch.row_ids)


# %%
# 7. Compute per-token log-probs and sequence scores.

def per_token_logprobs(input_ids: torch.Tensor, attention_mask: torch.Tensor, enable_grad: bool) -> torch.Tensor:
    with torch.set_grad_enabled(enable_grad):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def sequence_scores(batch: PreferenceBatch, enable_grad: bool) -> SequenceScores:
    input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0)
    attention_mask = torch.cat([batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0)
    response_mask = torch.cat([batch.chosen_response_mask, batch.rejected_response_mask], dim=0)

    token_logp = per_token_logprobs(input_ids, attention_mask, enable_grad=enable_grad)
    seq_sum = (token_logp * response_mask).sum(dim=1)
    seq_mean = masked_mean_per_row(token_logp, response_mask)
    chosen_sum, rejected_sum = seq_sum.chunk(2, dim=0)
    chosen_mean, rejected_mean = seq_mean.chunk(2, dim=0)
    return SequenceScores(chosen_sum, rejected_sum, chosen_mean, rejected_mean)


batch = batch.to(DEVICE)
model.eval()
policy_scores_preview = sequence_scores(batch, enable_grad=False)
print(policy_scores_preview)
print("policy chosen-rejected margin:", policy_scores_preview.chosen_logp_sum - policy_scores_preview.rejected_logp_sum)


# %%
# 8. Compute frozen-reference scores by temporarily disabling LoRA.

def disable_adapter_context():
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


with torch.no_grad():
    with disable_adapter_context():
        reference_scores_preview = sequence_scores(batch, enable_grad=False)

print(reference_scores_preview)
print("reference chosen-rejected margin:", reference_scores_preview.chosen_logp_sum - reference_scores_preview.rejected_logp_sum)


# %%
# 9. Compute DPO loss for one batch.

def dpo_loss(policy: SequenceScores, reference: SequenceScores) -> tuple[torch.Tensor, dict[str, float]]:
    policy_margin = policy.chosen_logp_sum - policy.rejected_logp_sum
    reference_margin = reference.chosen_logp_sum - reference.rejected_logp_sum
    logits = policy_margin - reference_margin
    loss = -F.logsigmoid(BETA * logits).mean()
    metrics = {
        "loss": float(loss.detach().item()),
        "policy_accuracy_sum": float((policy_margin.detach() > 0).float().mean().item()),
        "reference_corrected_accuracy": float((logits.detach() > 0).float().mean().item()),
        "policy_margin_sum_mean": float(policy_margin.detach().mean().item()),
        "reference_margin_sum_mean": float(reference_margin.detach().mean().item()),
        "reference_corrected_margin_mean": float(logits.detach().mean().item()),
    }
    return loss, metrics


model.train()
policy_scores = sequence_scores(batch, enable_grad=True)
with torch.no_grad():
    with disable_adapter_context():
        reference_scores = sequence_scores(batch, enable_grad=False)
loss, metrics = dpo_loss(policy_scores, reference_scores)
print(metrics)


# %%
# 10. Run one optimizer step manually.

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.95))
optimizer.zero_grad(set_to_none=True)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print({"loss": float(loss.item()), "grad_norm": float(grad_norm.item())})


# %%
# 11. Small training loop for MAX_STEPS.

optimizer_step = 0
model.train()
progress = tqdm(total=MAX_STEPS, desc="train[dpo]", dynamic_ncols=True)

while optimizer_step < MAX_STEPS:
    for train_batch in train_loader:
        train_batch = train_batch.to(DEVICE)
        policy_scores = sequence_scores(train_batch, enable_grad=True)
        with torch.no_grad():
            with disable_adapter_context():
                reference_scores = sequence_scores(train_batch, enable_grad=False)

        loss, metrics = dpo_loss(policy_scores, reference_scores)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['policy_accuracy_sum']:.3f}")
        print({"step": optimizer_step, "grad_norm": float(grad_norm.item()), **metrics})
        if optimizer_step >= MAX_STEPS:
            break

progress.close()


# %%
# 12. Evaluate preference accuracy on eval examples.

@torch.no_grad()
def evaluate_preferences() -> dict[str, float]:
    model.eval()
    policy_margins, corrected_margins = [], []
    for eval_batch in tqdm(eval_loader, desc="eval[prefs]", dynamic_ncols=True):
        eval_batch = eval_batch.to(DEVICE)
        policy_scores = sequence_scores(eval_batch, enable_grad=False)
        with disable_adapter_context():
            reference_scores = sequence_scores(eval_batch, enable_grad=False)
        pm = policy_scores.chosen_logp_sum - policy_scores.rejected_logp_sum
        rm = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum
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


eval_metrics = evaluate_preferences()
print(json.dumps(eval_metrics, indent=2))


# %%
# 13. Generate from the trained policy for a qualitative check.

@torch.no_grad()
def generate_one(ex: GenerationExample) -> str:
    model.eval()
    old_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    ids = tokenizer.apply_chat_template(
        ex.prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0][-MAX_PROMPT_TOKENS:].unsqueeze(0).to(DEVICE)
    attention_mask = torch.ones_like(ids)
    out = model.generate(
        input_ids=ids,
        attention_mask=attention_mask,
        max_new_tokens=GENERATION_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.config.use_cache = old_cache
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


sample = generation_examples[0]
generated_text = generate_one(sample)
print("PROMPT:\n", sample.prompt_text)
print("MODEL RESPONSE:\n", generated_text)


# %%
# 14. Save the LoRA adapter and metrics.

adapter_dir = OUTPUT_DIR / "adapter_final"
model.save_pretrained(adapter_dir)
(OUTPUT_DIR / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
print("saved adapter:", adapter_dir)
