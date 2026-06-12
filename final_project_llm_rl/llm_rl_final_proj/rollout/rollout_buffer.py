from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    N = batch.input_ids.shape[0]
    if shuffle:
        indexs = torch.randperm(N,generator=generator,device=batch.input_ids.device)
    else:
        indexs = torch.arange(N,device=batch.input_ids.device)
    i = 0
    while i < N:
        indices = indexs[i:i+minibatch_size]
        mb = RolloutBatch(
                input_ids=batch.input_ids[indices]    ,  # [N, L]
                attention_mask= batch.attention_mask[indices] ,   # [N, L]
                completion_mask=batch.completion_mask[indices]  ,  # [N, L-1] float
                old_logprobs=batch.old_logprobs[indices]  ,    # [N, L-1]
                ref_logprobs=batch.ref_logprobs[indices]  ,     # [N, L-1]
                rewards=batch.rewards[indices]      ,      # [N]
                advantages=batch.advantages[indices],      # [N]
                task_names=[batch.task_names[j] for j in indices.tolist()] if batch.task_names is not None else None,
                completion_texts=[batch.completion_texts[j] for j in indices.tolist()] if batch.completion_texts is not None else None,
        )
        if device is not None:
            mb = mb.to(device)
        yield mb
        i += minibatch_size
