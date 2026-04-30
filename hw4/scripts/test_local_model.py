from __future__ import annotations

import argparse

import torch

from hw4.models.load import (
    load_inference_model_and_tokenizer,
    resolve_adapter_path,
    tokenize_chat_prompts,
)


DEFAULT_PROMPT = "Solve briefly: what is 12 + 30?"


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    input_ids, attention_mask = tokenize_chat_prompts(
        tokenizer,
        [build_messages(prompt)],
        add_generation_prompt=True,
        max_prompt_tokens=None,
        device=device,
    )

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    prompt_len = int(input_ids.shape[1])
    completion_ids = out[0, prompt_len:]
    if tokenizer.pad_token_id is not None and (completion_ids == tokenizer.pad_token_id).any():
        completion_ids = completion_ids[completion_ids != tokenizer.pad_token_id]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a local or Hugging Face causal LM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Local model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional LoRA adapter directory to load on top of the base model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Single user prompt to send to the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, cuda:0, mps, ...",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be >= 1, got {args.max_new_tokens}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(device)
    adapter_path = None if args.adapter_path is None else resolve_adapter_path(args.adapter_path)

    print(f"Loading model from: {args.model_name}")
    print(f"Using device={device} dtype={dtype}")
    if adapter_path is not None:
        print(f"Loading adapter from: {adapter_path}")

    loaded = load_inference_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    completion = generate_completion(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Completion ===")
    print(completion)


if __name__ == "__main__":
    main()
