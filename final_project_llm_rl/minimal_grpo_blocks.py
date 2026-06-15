# %%
"""Hard-coded, cell-by-cell reward-model + GRPO walkthrough.

Open this file in VS Code / Cursor / PyCharm and run one `# %%` block at a time.

This is the minimal online RLHF pipeline in one file:

  train_prefs -> train Bradley-Terry reward model
  train_gen prompts -> Qwen policy rollout groups -> reward scores
  -> GRPO group advantages -> GRPO clipped loss -> optimizer step
  -> reward-model evaluation of policy generations -> save policy adapter.

It does not import `llm_rl_final_proj.*`.  It only uses normal Python packages.
By default it uses the real project dataset, not toy data, but very small limits.
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
from typing import Any, Iterable, Iterator

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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

PROJECT_DIR = Path("/mnt1/mnt1/nlp/lc/final_project_llm_rl")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset/wildchat_min4_judged_5k_v1"
OUTPUT_DIR = PROJECT_DIR / "runs/minimal_grpo_blocks"
REWARD_OUTPUT_DIR = OUTPUT_DIR / "reward_model"
HF_HOME = PROJECT_DIR / ".cache/huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
LOCAL_QWEN_SNAPSHOT = (
    HF_HUB_CACHE
    / "models--Qwen--Qwen2.5-1.5B-Instruct"
    / "snapshots"
    / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
MODEL_NAME = str(LOCAL_QWEN_SNAPSHOT) if LOCAL_QWEN_SNAPSHOT.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
REWARD_MODEL_NAME = MODEL_NAME

USE_TOY_DATA = False
LOCAL_FILES_ONLY = True

PREF_TRAIN_SPLIT = "train_prefs"
PREF_EVAL_SPLIT = "test_prefs"
GEN_TRAIN_SPLIT = "train_gen"
GEN_EVAL_SPLIT = "test_gen"

# Tiny defaults are for step-by-step learning.  Set limits to 0 or larger values for fuller runs.
RM_TRAIN_LIMIT = 8
RM_EVAL_LIMIT = 4
GEN_TRAIN_LIMIT = 8
GEN_EVAL_LIMIT = 4

MAX_PROMPT_TOKENS = 700
MAX_RESPONSE_TOKENS = 256

# Reward model training.
RM_BATCH_SIZE = 2
RM_MAX_STEPS = 1
RM_LR = 3e-5

# GRPO rollout and update.
PROMPT_BATCH_SIZE = 2
GROUP_SIZE = 4
MIN_NEW_TOKENS = 8
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_UPDATES = 1
PPO_EPOCHS = 1
MINIBATCH_SIZE = 4

POLICY_LR = 1e-5
CLIP_EPS = 0.2
KL_COEF = 0.01
ADV_CLIP = 5.0
MAX_GRAD_NORM = 0.5

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
REWARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
class RewardPairBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    row_ids: list[str]

    def to(self, device: torch.device) -> "RewardPairBatch":
        return RewardPairBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
            row_ids=self.row_ids,
        )


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    completion_texts: list[str]
    prompt_messages: list[list[Message]]
    prompt_texts: list[str]
    row_ids: list[str]


# %%
# 2. Load raw data: preference pairs for reward model, prompts for GRPO.

def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


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


def toy_preference_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "toy-1",
            "prompt_messages": [{"role": "user", "content": "Give one practical safety tip for hiking."}],
            "prompt_text": "Give one practical safety tip for hiking.",
            "chosen_text": "Tell someone your route and expected return time before you leave.",
            "rejected_text": "Go without telling anyone and ignore the weather.",
        },
        {
            "row_id": "toy-2",
            "prompt_messages": [{"role": "user", "content": "Explain photosynthesis in one sentence."}],
            "prompt_text": "Explain photosynthesis in one sentence.",
            "chosen_text": "Plants use sunlight, water, and carbon dioxide to make sugar and oxygen.",
            "rejected_text": "Photosynthesis is when animals sleep during the day.",
        },
    ]


if USE_TOY_DATA:
    rm_train_rows = toy_preference_rows()[:RM_TRAIN_LIMIT]
    rm_eval_rows = toy_preference_rows()[:RM_EVAL_LIMIT]
    gen_train_rows = toy_preference_rows()[:GEN_TRAIN_LIMIT]
    gen_eval_rows = toy_preference_rows()[:GEN_EVAL_LIMIT]
else:
    rm_train_rows = load_jsonl(DATASET_DIR / f"{PREF_TRAIN_SPLIT}.jsonl", RM_TRAIN_LIMIT)
    rm_eval_rows = load_jsonl(DATASET_DIR / f"{PREF_EVAL_SPLIT}.jsonl", RM_EVAL_LIMIT)
    gen_train_rows = load_jsonl(DATASET_DIR / f"{GEN_TRAIN_SPLIT}.jsonl", GEN_TRAIN_LIMIT)
    gen_eval_rows = load_jsonl(DATASET_DIR / f"{GEN_EVAL_SPLIT}.jsonl", GEN_EVAL_LIMIT)

rm_train_examples = [row_to_preference_example(row, i) for i, row in enumerate(rm_train_rows)]
rm_eval_examples = [row_to_preference_example(row, i) for i, row in enumerate(rm_eval_rows)]
train_examples = [row_to_generation_example(row, i) for i, row in enumerate(gen_train_rows)]
eval_examples = [row_to_generation_example(row, i) for i, row in enumerate(gen_eval_rows)]

print({"rm_train": len(rm_train_examples), "rm_eval": len(rm_eval_examples), "gen_train": len(train_examples), "gen_eval": len(eval_examples)})
print("first preference row:", rm_train_examples[0])
print("first GRPO prompt:", train_examples[0].prompt_text[:1000])


# %%
# 3. Load Qwen tokenizer and train a Bradley-Terry reward model with LoRA.

reward_tokenizer = AutoTokenizer.from_pretrained(
    REWARD_MODEL_NAME,
    use_fast=True,
    local_files_only=LOCAL_FILES_ONLY,
    cache_dir=str(HF_HUB_CACHE),
)
reward_tokenizer.padding_side = "left"
if reward_tokenizer.pad_token_id is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

base_reward = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_NAME,
    num_labels=1,
    dtype=DTYPE,
    local_files_only=LOCAL_FILES_ONLY,
    cache_dir=str(HF_HUB_CACHE),
)
if getattr(base_reward.config, "pad_token_id", None) is None:
    base_reward.config.pad_token_id = reward_tokenizer.pad_token_id
if GRAD_CHECKPOINTING:
    base_reward.gradient_checkpointing_enable()
    base_reward.config.use_cache = False
    base_reward.enable_input_require_grads()

reward_lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[x.strip() for x in LORA_TARGET_MODULES.split(",") if x.strip()],
    modules_to_save=["score"],
    bias="none",
)
reward_model = get_peft_model(base_reward, reward_lora_config).to(DEVICE)

for name, p in reward_model.named_parameters():
    train_this = ("lora_" in name) or ("score" in name)
    p.requires_grad_(train_this)
    if "lora_" in name:
        p.data = p.data.float()

rm_trainable = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
rm_total = sum(p.numel() for p in reward_model.parameters())
print({"reward_trainable_params": rm_trainable, "reward_total_params": rm_total})


# %%
# 4. Reward-model batching and scoring functions.

def tokenize_reward_prompt_response(prompt_messages: list[Message], response_text: str) -> torch.Tensor:
    prompt_ids = reward_tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0]
    full_ids = reward_tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": response_text.strip() or "[no response]"}],
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
    if response_len <= 0:
        raise ValueError("reward response has zero tokens")
    return full_ids


def left_pad_reward(ids_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(x.numel()) for x in ids_list)
    input_ids = torch.full((len(ids_list), max_len), int(reward_tokenizer.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, max_len - ids.numel():] = ids
        attention_mask[i, max_len - ids.numel():] = 1
    return input_ids, attention_mask


class RewardPairCollator:
    def __call__(self, examples: list[PreferenceExample]) -> RewardPairBatch:
        chosen_ids = [tokenize_reward_prompt_response(ex.prompt_messages, ex.chosen_text) for ex in examples]
        rejected_ids = [tokenize_reward_prompt_response(ex.prompt_messages, ex.rejected_text) for ex in examples]
        max_len = max(max(x.numel() for x in chosen_ids), max(x.numel() for x in rejected_ids))
        c_input, c_attn = left_pad_reward(chosen_ids)
        r_input, r_attn = left_pad_reward(rejected_ids)
        if c_input.shape[1] < max_len:
            c_input, c_attn = left_pad_reward([torch.cat([torch.full((max_len - x.numel(),), int(reward_tokenizer.pad_token_id)), x]) for x in chosen_ids])
        if r_input.shape[1] < max_len:
            r_input, r_attn = left_pad_reward([torch.cat([torch.full((max_len - x.numel(),), int(reward_tokenizer.pad_token_id)), x]) for x in rejected_ids])
        return RewardPairBatch(c_input, c_attn, r_input, r_attn, [ex.row_id for ex in examples])


def reward_model_scores(input_ids: torch.Tensor, attention_mask: torch.Tensor, enable_grad: bool) -> torch.Tensor:
    with torch.set_grad_enabled(enable_grad):
        logits = reward_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        return logits[:, 0] if logits.ndim == 2 else logits


def reward_pair_loss(batch: RewardPairBatch) -> tuple[torch.Tensor, dict[str, float]]:
    chosen = reward_model_scores(batch.chosen_input_ids, batch.chosen_attention_mask, enable_grad=True)
    rejected = reward_model_scores(batch.rejected_input_ids, batch.rejected_attention_mask, enable_grad=True)
    margin = chosen - rejected
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "pair_accuracy": float((margin.detach() > 0).float().mean().item()),
        "margin_mean": float(margin.detach().mean().item()),
        "chosen_score_mean": float(chosen.detach().mean().item()),
        "rejected_score_mean": float(rejected.detach().mean().item()),
    }


rm_train_loader = DataLoader(PreferenceDataset(rm_train_examples), batch_size=RM_BATCH_SIZE, shuffle=True, collate_fn=RewardPairCollator())
rm_eval_loader = DataLoader(PreferenceDataset(rm_eval_examples), batch_size=RM_BATCH_SIZE, shuffle=False, collate_fn=RewardPairCollator())
rm_batch = next(iter(rm_train_loader)).to(DEVICE)
rm_loss_preview, rm_metrics_preview = reward_pair_loss(rm_batch)
print(rm_metrics_preview)


# %%
# 5. Train the reward model for a few tiny steps.

rm_optimizer = torch.optim.AdamW([p for p in reward_model.parameters() if p.requires_grad], lr=RM_LR, betas=(0.9, 0.95))
reward_model.train()
rm_step = 0
rm_progress = tqdm(total=RM_MAX_STEPS, desc="train[reward_model]", dynamic_ncols=True)

while rm_step < RM_MAX_STEPS:
    for batch in rm_train_loader:
        batch = batch.to(DEVICE)
        loss, metrics = reward_pair_loss(batch)
        rm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        rm_optimizer.step()
        rm_step += 1
        rm_progress.update(1)
        print({"rm_step": rm_step, "grad_norm": float(grad_norm.item()), **metrics})
        if rm_step >= RM_MAX_STEPS:
            break

rm_progress.close()
reward_adapter_dir = REWARD_OUTPUT_DIR / "adapter_final"
reward_model.save_pretrained(reward_adapter_dir)
print("saved reward adapter:", reward_adapter_dir)


# %%
# 6. Evaluate reward model pair accuracy on held-out preference pairs.

@torch.no_grad()
def evaluate_reward_model() -> dict[str, float]:
    reward_model.eval()
    margins, chosen_scores, rejected_scores = [], [], []
    for batch in tqdm(rm_eval_loader, desc="eval[reward_model]", dynamic_ncols=True):
        batch = batch.to(DEVICE)
        chosen = reward_model_scores(batch.chosen_input_ids, batch.chosen_attention_mask, enable_grad=False)
        rejected = reward_model_scores(batch.rejected_input_ids, batch.rejected_attention_mask, enable_grad=False)
        margins.append((chosen - rejected).detach().cpu())
        chosen_scores.append(chosen.detach().cpu())
        rejected_scores.append(rejected.detach().cpu())
    margin = torch.cat(margins)
    chosen = torch.cat(chosen_scores)
    rejected = torch.cat(rejected_scores)
    return {
        "rm_pair_accuracy": float((margin > 0).float().mean().item()),
        "rm_margin_mean": float(margin.mean().item()),
        "rm_chosen_score_mean": float(chosen.mean().item()),
        "rm_rejected_score_mean": float(rejected.mean().item()),
        "count": float(margin.numel()),
    }


rm_eval_metrics = evaluate_reward_model()
print(json.dumps(rm_eval_metrics, indent=2))


# %%
# 7. Load Qwen policy model with LoRA adapter to train by GRPO.

policy_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    local_files_only=LOCAL_FILES_ONLY,
    cache_dir=str(HF_HUB_CACHE),
)
policy_tokenizer.padding_side = "left"
if policy_tokenizer.pad_token_id is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

base_policy = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    local_files_only=LOCAL_FILES_ONLY,
    cache_dir=str(HF_HUB_CACHE),
)
if GRAD_CHECKPOINTING:
    base_policy.gradient_checkpointing_enable()
    base_policy.config.use_cache = False
    base_policy.enable_input_require_grads()

policy_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[x.strip() for x in LORA_TARGET_MODULES.split(",") if x.strip()],
    bias="none",
)
policy_model = get_peft_model(base_policy, policy_lora_config).to(DEVICE)

for name, p in policy_model.named_parameters():
    train_this = "lora_" in name
    p.requires_grad_(train_this)
    if train_this:
        p.data = p.data.float()

policy_trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
policy_total = sum(p.numel() for p in policy_model.parameters())
print({"policy_trainable_params": policy_trainable, "policy_total_params": policy_total})


# %%
# 8. Tokenize prompts for policy generation.

def tokenize_chat_prompts(messages_list: list[list[Message]]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = []
    for messages in messages_list:
        ids = policy_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
        if ids.numel() > MAX_PROMPT_TOKENS:
            ids = ids[-MAX_PROMPT_TOKENS:]
        encoded.append(ids)
    max_len = max(x.numel() for x in encoded)
    input_ids = torch.full((len(encoded), max_len), int(policy_tokenizer.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, max_len - ids.numel():] = ids
        attention_mask[i, max_len - ids.numel():] = 1
    return input_ids.to(DEVICE), attention_mask.to(DEVICE)


prompt_batch = train_examples[:PROMPT_BATCH_SIZE]
prompt_input_ids, prompt_attention_mask = tokenize_chat_prompts([ex.prompt_messages for ex in prompt_batch])
print("prompt_input_ids:", prompt_input_ids.shape)
print(policy_tokenizer.decode(prompt_input_ids[0], skip_special_tokens=False)[-1000:])


# %%
# 9. Rollout: sample GROUP_SIZE completions per prompt.

@torch.no_grad()
def generate_grouped_completions(prompt_examples: list[GenerationExample]) -> dict[str, Any]:
    input_ids, attention_mask = tokenize_chat_prompts([ex.prompt_messages for ex in prompt_examples])
    prompt_len = int(input_ids.shape[1])
    was_training = bool(policy_model.training)
    had_gc = bool(getattr(policy_model, "is_gradient_checkpointing", False))
    if had_gc and hasattr(policy_model, "gradient_checkpointing_disable"):
        policy_model.gradient_checkpointing_disable()
        policy_model.config.use_cache = True
    policy_model.eval()
    try:
        sequences = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=MIN_NEW_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=GROUP_SIZE,
            num_beams=1,
            pad_token_id=policy_tokenizer.pad_token_id,
            eos_token_id=policy_tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        if had_gc and hasattr(policy_model, "gradient_checkpointing_enable"):
            policy_model.gradient_checkpointing_enable()
            policy_model.enable_input_require_grads()
            policy_model.config.use_cache = False
        if was_training:
            policy_model.train()

    attention = (sequences != int(policy_tokenizer.pad_token_id)).long()
    completion_ids = sequences[:, prompt_len:]
    completion_texts = []
    for row in completion_ids:
        if (row == int(policy_tokenizer.pad_token_id)).any():
            row = row[: int((row != int(policy_tokenizer.pad_token_id)).sum().item())]
        completion_texts.append(policy_tokenizer.decode(row, skip_special_tokens=True))
    repeated_examples = []
    for ex in prompt_examples:
        repeated_examples.extend([ex] * GROUP_SIZE)
    return {"sequences": sequences, "attention_mask": attention, "prompt_len": prompt_len, "completion_texts": completion_texts, "examples": repeated_examples}


rollout_raw = generate_grouped_completions(prompt_batch)
print("sequences:", rollout_raw["sequences"].shape)
for i, text in enumerate(rollout_raw["completion_texts"][: min(4, len(rollout_raw["completion_texts"]))]):
    print(f"--- completion {i} ---\n{text[:1000]}")


# %%
# 10. Compute old policy logprobs, frozen reference logprobs, and completion mask.

def per_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, enable_grad: bool) -> torch.Tensor:
    with torch.set_grad_enabled(enable_grad):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def completion_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_len: int) -> torch.Tensor:
    mask = torch.ones((input_ids.shape[0], input_ids.shape[1] - 1), device=input_ids.device)
    mask[:, : prompt_len - 1] = 0.0
    return mask * attention_mask[:, 1:].float()


def disable_policy_adapter_context():
    if hasattr(policy_model, "disable_adapter"):
        return policy_model.disable_adapter()
    return nullcontext()


with torch.no_grad():
    old_logprobs = per_token_logprobs(policy_model, rollout_raw["sequences"], rollout_raw["attention_mask"], False)
    with disable_policy_adapter_context():
        ref_logprobs = per_token_logprobs(policy_model, rollout_raw["sequences"], rollout_raw["attention_mask"], False)
    comp_mask = completion_mask(rollout_raw["sequences"], rollout_raw["attention_mask"], rollout_raw["prompt_len"])

print("old_logprobs:", old_logprobs.shape)
print("ref_logprobs:", ref_logprobs.shape)
print("completion token counts:", comp_mask.sum(dim=1))


# %%
# 11. Score rollout completions with the reward model trained above.

@torch.no_grad()
def reward_scores_for_rollout(rollout: dict[str, Any]) -> torch.Tensor:
    ids_list = [tokenize_reward_prompt_response(ex.prompt_messages, response) for ex, response in zip(rollout["examples"], rollout["completion_texts"])]
    input_ids, attention_mask = left_pad_reward(ids_list)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    scores = reward_model_scores(input_ids, attention_mask, enable_grad=False)
    return scores.detach().float()


rewards = reward_scores_for_rollout(rollout_raw)
print("rewards:", rewards)
print("grouped rewards:", rewards.reshape(-1, GROUP_SIZE))


# %%
# 12. Compute GRPO group advantages.

def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    grouped = rewards.reshape(-1, GROUP_SIZE)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (centered / std).reshape(-1)


advantages = group_advantages(rewards).to(DEVICE)
print("advantages:", advantages)
print("grouped advantages:", advantages.reshape(-1, GROUP_SIZE))


# %%
# 13. Assemble rollout batch.

rollout_batch = RolloutBatch(
    input_ids=rollout_raw["sequences"],
    attention_mask=rollout_raw["attention_mask"],
    completion_mask=comp_mask,
    old_logprobs=old_logprobs,
    ref_logprobs=ref_logprobs,
    rewards=rewards.to(DEVICE),
    advantages=advantages,
    completion_texts=rollout_raw["completion_texts"],
    prompt_messages=[ex.prompt_messages for ex in rollout_raw["examples"]],
    prompt_texts=[ex.prompt_text for ex in rollout_raw["examples"]],
    row_ids=[ex.row_id for ex in rollout_raw["examples"]],
)
print(rollout_batch)


# %%
# 14. GRPO loss on one minibatch.

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl(new_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    delta = torch.clamp(ref_logp - new_logp, min=-20.0, max=20.0)
    return masked_mean(torch.exp(delta) - delta - 1.0, mask)


def grpo_minibatch_loss(mb: RolloutBatch) -> tuple[torch.Tensor, dict[str, float]]:
    adv = mb.advantages.clamp(-ADV_CLIP, ADV_CLIP).detach()
    mask = mb.completion_mask
    new_logp = per_token_logprobs(policy_model, mb.input_ids, mb.attention_mask, enable_grad=True)
    log_ratio = torch.clamp(new_logp - mb.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv.unsqueeze(1)
    clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
    clipped = clipped_ratio * adv.unsqueeze(1)
    per_token_obj = torch.minimum(unclipped, clipped) * mask
    seq_obj = masked_mean_per_row(per_token_obj, mask)
    policy_loss = -seq_obj.mean()
    kl = approx_kl(new_logp, mb.ref_logprobs, mask)
    entropy = -masked_mean(new_logp, mask)
    clipfrac = masked_mean((ratio != clipped_ratio).float(), mask)
    loss = policy_loss + KL_COEF * kl
    return loss, {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl": float(kl.detach().item()),
        "entropy": float(entropy.detach().item()),
        "clipfrac": float(clipfrac.detach().item()),
    }


policy_model.train()
grpo_loss_preview, grpo_metrics_preview = grpo_minibatch_loss(rollout_batch)
print(grpo_metrics_preview)


# %%
# 15. One optimizer update over rollout minibatches.

def iter_minibatches(batch: RolloutBatch, minibatch_size: int, shuffle: bool = True) -> Iterator[RolloutBatch]:
    n = batch.input_ids.shape[0]
    indices = torch.randperm(n, device=batch.input_ids.device) if shuffle else torch.arange(n, device=batch.input_ids.device)
    for start in range(0, n, minibatch_size):
        idx = indices[start : start + minibatch_size]
        yield RolloutBatch(
            input_ids=batch.input_ids[idx],
            attention_mask=batch.attention_mask[idx],
            completion_mask=batch.completion_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx],
            advantages=batch.advantages[idx],
            completion_texts=[batch.completion_texts[i] for i in idx.tolist()],
            prompt_messages=[batch.prompt_messages[i] for i in idx.tolist()],
            prompt_texts=[batch.prompt_texts[i] for i in idx.tolist()],
            row_ids=[batch.row_ids[i] for i in idx.tolist()],
        )


policy_optimizer = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=POLICY_LR, betas=(0.9, 0.95))
policy_model.train()
policy_optimizer.zero_grad(set_to_none=True)
update_metrics = []
for _epoch in range(PPO_EPOCHS):
    for mb in iter_minibatches(rollout_batch, MINIBATCH_SIZE, shuffle=True):
        loss, metrics = grpo_minibatch_loss(mb)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), MAX_GRAD_NORM)
        policy_optimizer.step()
        policy_optimizer.zero_grad(set_to_none=True)
        update_metrics.append({**metrics, "grad_norm": float(grad_norm.item())})
print(update_metrics)


# %%
# 16. Full tiny GRPO training loop: rollout -> reward -> advantage -> update.

def sample_prompt_batch() -> list[GenerationExample]:
    return [random.choice(train_examples) for _ in range(PROMPT_BATCH_SIZE)]


for update_idx in range(1, MAX_UPDATES + 1):
    prompt_batch = sample_prompt_batch()
    rollout_raw = generate_grouped_completions(prompt_batch)
    with torch.no_grad():
        old_logprobs = per_token_logprobs(policy_model, rollout_raw["sequences"], rollout_raw["attention_mask"], False)
        with disable_policy_adapter_context():
            ref_logprobs = per_token_logprobs(policy_model, rollout_raw["sequences"], rollout_raw["attention_mask"], False)
        comp_mask = completion_mask(rollout_raw["sequences"], rollout_raw["attention_mask"], rollout_raw["prompt_len"])
    rewards = reward_scores_for_rollout(rollout_raw).to(DEVICE)
    advantages = group_advantages(rewards).to(DEVICE)
    rollout_batch = RolloutBatch(
        input_ids=rollout_raw["sequences"],
        attention_mask=rollout_raw["attention_mask"],
        completion_mask=comp_mask,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=rewards,
        advantages=advantages,
        completion_texts=rollout_raw["completion_texts"],
        prompt_messages=[ex.prompt_messages for ex in rollout_raw["examples"]],
        prompt_texts=[ex.prompt_text for ex in rollout_raw["examples"]],
        row_ids=[ex.row_id for ex in rollout_raw["examples"]],
    )
    policy_model.train()
    metrics_accum = []
    for _epoch in range(PPO_EPOCHS):
        for mb in iter_minibatches(rollout_batch, MINIBATCH_SIZE, shuffle=True):
            loss, metrics = grpo_minibatch_loss(mb)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), MAX_GRAD_NORM)
            policy_optimizer.step()
            policy_optimizer.zero_grad(set_to_none=True)
            metrics_accum.append({**metrics, "grad_norm": float(grad_norm.item())})
    print({
        "update": update_idx,
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std(unbiased=False).item()),
        "last_update_metrics": metrics_accum[-1],
    })


# %%
# 17. Evaluate policy by generating one response per eval prompt and reward-scoring it.

@torch.no_grad()
def generate_one_per_prompt(prompt_examples: list[GenerationExample]) -> dict[str, Any]:
    old_group_size = globals()["GROUP_SIZE"]
    globals()["GROUP_SIZE"] = 1
    try:
        out = generate_grouped_completions(prompt_examples)
    finally:
        globals()["GROUP_SIZE"] = old_group_size
    return out


eval_rollout = generate_one_per_prompt(eval_examples[:GEN_EVAL_LIMIT])
eval_rewards = reward_scores_for_rollout(eval_rollout)
print({"eval_reward_mean": float(eval_rewards.mean().item()), "eval_reward_std": float(eval_rewards.std(unbiased=False).item())})
for ex, text, score in zip(eval_rollout["examples"], eval_rollout["completion_texts"], eval_rewards.tolist()):
    print("\nPROMPT:", ex.prompt_text[:500])
    print("SCORE:", score)
    print("RESPONSE:", text[:1000])


# %%
# 18. Save reward adapter, policy adapter, and eval scores.

policy_adapter_dir = OUTPUT_DIR / "policy_adapter_final"
policy_model.save_pretrained(policy_adapter_dir)
(OUTPUT_DIR / "eval_reward_scores.json").write_text(
    json.dumps({"scores": [float(x) for x in eval_rewards.tolist()], "reward_model_eval": rm_eval_metrics}, indent=2),
    encoding="utf-8",
)
print("saved reward adapter:", reward_adapter_dir)
print("saved policy adapter:", policy_adapter_dir)
