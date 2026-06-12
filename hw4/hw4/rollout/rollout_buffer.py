from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

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

    # Optional debug
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
    # TODO(student): yield RolloutBatch minibatches of size minibatch_size.
    # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    # raise NotImplementedError("student TODO: iter_minibatches")
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

