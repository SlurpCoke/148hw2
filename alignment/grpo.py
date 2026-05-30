from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels.

    Returns a dict with keys:
        input_ids      (batch, max_len-1)  – prompt+output tokens, last token dropped
        labels         (batch, max_len-1)  – shifted right (first token dropped)
        response_mask  (batch, max_len-1)  – 1 for response positions in labels
    """
    full_sequences = [
        tokenizer.encode(p, add_special_tokens=False) + tokenizer.encode(o, add_special_tokens=False)
        for p, o in zip(prompt_strs, output_strs)
    ]
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompt_strs]
    response_lens = [len(tokenizer.encode(o, add_special_tokens=False)) for o in output_strs]

    max_len = max(len(seq) - 1 for seq in full_sequences)
    pad_id = tokenizer.pad_token_id

    input_ids_list, labels_list, mask_list = [], [], []
    for seq, p_len, r_len in zip(full_sequences, prompt_lens, response_lens):
        seq_len = len(seq) - 1
        pad = [pad_id] * (max_len - seq_len)

        input_ids_list.append(seq[:-1] + pad)
        labels_list.append(seq[1:] + pad)
        # False for (prompt_len-1) prompt-prediction positions,
        # True for response_len positions,
        # False for padding
        mask = [False] * (p_len - 1) + [True] * r_len + [False] * (max_len - seq_len)
        mask_list.append(mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(mask_list, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Entropy and log-probability utilities
# ---------------------------------------------------------------------------

def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropy over the vocabulary dimension (numerically stable)."""
    log_probs = F.log_softmax(logits, dim=-1)          # (batch, seq, vocab)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)             # (batch, seq)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples.

    Args:
        model:                HuggingFace causal LM (or compatible).
        input_ids:            (batch, seq)
        labels:               (batch, seq) – token ids that are the targets.
        return_token_entropy: if True, also return per-token entropies.

    Returns dict with:
        log_probs       (batch, seq) – log p_θ(x_t | x_{<t})
        token_entropy   (batch, seq) – optional
    """
    logits = model(input_ids).logits                   # (batch, seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    result: dict[str, Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


# ---------------------------------------------------------------------------
# Masked normalisation
# ---------------------------------------------------------------------------

def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum masked elements along *dim* and divide by *normalize_constant*."""
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


# ---------------------------------------------------------------------------
# Generation logging
# ---------------------------------------------------------------------------

def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serialisable per-generation logs for debugging training runs."""
    logs: list[dict[str, Any]] = []
    for i, (prompt, response, gt, info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos)
    ):
        entry: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "format_reward": info.get("format_reward", 0.0),
            "answer_reward": info.get("answer_reward", 0.0),
            "total_reward": info.get("reward", 0.0),
            "response_length": len(response.split()),
        }
        if token_entropies is not None:
            ent = token_entropies[i]
            entry["avg_token_entropy"] = float(ent) if hasattr(ent, "__float__") else ent
        logs.append(entry)
    return logs


# ---------------------------------------------------------------------------
# GRPO advantage estimation
# ---------------------------------------------------------------------------

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalised advantages.

    Returns:
        advantages        (rollout_batch_size,)
        raw_rewards       (rollout_batch_size,)
        metadata          dict of scalar stats
    """
    raw = [
        reward_fn(resp, gt)["reward"]
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw, dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)          # (n_prompts, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = (centered / (std + advantage_eps)).reshape(-1)
    else:
        advantages = centered.reshape(-1)

    metadata: dict[str, float] = {
        "mean_reward": float(raw_rewards.mean()),
        "std_reward": float(raw_rewards.std()),
        "max_reward": float(raw_rewards.max()),
        "min_reward": float(raw_rewards.min()),
    }
    return advantages, raw_rewards, metadata


# ---------------------------------------------------------------------------
# GRPO-Clip loss
# ---------------------------------------------------------------------------

def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-token GRPO-Clip loss (to be *minimised*, i.e. negative of the objective).

    Args:
        advantages:       (batch, 1)   per-example advantage A^(i)
        policy_log_probs: (batch, seq) log π_θ(o_t | q, o_{<t})
        old_log_probs:    (batch, seq) log π_θ_old(o_t | q, o_{<t})
        cliprange:        ε

    Returns:
        loss     (batch, seq) per-token clipped loss
        metadata dict
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)         # (batch, seq)
    clipped = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    adv = advantages.expand_as(policy_log_probs)                 # broadcast

    loss = -torch.minimum(ratios * adv, clipped * adv)           # (batch, seq)

    is_clipped = (ratios * adv < clipped * adv).detach()
    metadata: dict[str, Tensor] = {
        "clip_fraction": is_clipped.float().mean(),
        "ratio_mean": ratios.detach().mean(),
    }
    return loss, metadata


# ---------------------------------------------------------------------------
# GRPO microbatch train step
# ---------------------------------------------------------------------------

def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Forward-and-backward on one microbatch; accumulates gradients.

    Args:
        policy_log_probs:          (batch, seq) – requires_grad
        response_mask:             (batch, seq) – bool or int, 1 for response tokens
        gradient_accumulation_steps: divides loss for correct gradient scaling
        advantages:                (batch, 1)
        old_log_probs:             (batch, seq) – detached
        cliprange:                 ε

    Returns:
        loss      scalar tensor (detached value)
        metadata  dict
    """
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    mask_f = response_mask.to(per_token_loss.dtype)
    masked_sum = (per_token_loss * mask_f).sum(dim=1)             # (batch,)
    per_example_loss = masked_sum / mask_f.sum(dim=1)             # (batch,)
    loss = per_example_loss.mean() / gradient_accumulation_steps

    loss.backward()

    return loss.detach(), metadata


# ---------------------------------------------------------------------------
# Full GRPO training loop
# ---------------------------------------------------------------------------

def _generate_responses(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    """Batch-generate completions with the HuggingFace model."""
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    encoded = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=pad_id,
        )

    responses = []
    for ids in output_ids:
        new_ids = ids[prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        # Truncate at </answer> if present
        if "</answer>" in text:
            text = text[: text.index("</answer>") + len("</answer>")]
        responses.append(text)

    tokenizer.padding_side = orig_padding_side
    return responses


def _eval_validation(
    model: torch.nn.Module,
    tokenizer,
    val_examples: list[dict[str, Any]],
    reward_fn: Callable,
    n_examples: int,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    """Run a quick validation pass and return mean rewards."""
    subset = val_examples[:n_examples]
    prompts = [prompt_template.format(question=ex["question"]) for ex in subset]
    gts = [ex["answer"] for ex in subset]

    model.eval()
    responses = _generate_responses(model, tokenizer, prompts, max_new_tokens, 1, temperature, device)
    model.train()

    rewards = [reward_fn(r, gt)["reward"] for r, gt in zip(responses, gts)]
    format_rewards = [reward_fn(r, gt)["format_reward"] for r, gt in zip(responses, gts)]
    answer_rewards = [reward_fn(r, gt)["answer_reward"] for r, gt in zip(responses, gts)]

    return {
        "val_reward": float(sum(rewards) / len(rewards)),
        "val_format_reward": float(sum(format_rewards) / len(format_rewards)),
        "val_answer_reward": float(sum(answer_rewards) / len(answer_rewards)),
    }


def train_grpo(
    *,
    model: torch.nn.Module,
    tokenizer,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    n_grpo_steps: int = 50,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    device: torch.device | str | None = None,
    output_dir: str | None = None,
    log_every: int = 5,
    n_val_examples: int = 256,
    prompt_template: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5.

    Returns a history dict with per-step training metrics and periodic
    validation rewards.
    """
    from alignment.prompts import COT_PROMPT_TEMPLATE
    from alignment.drgrpo_grader import r1_zero_reward_fn

    if prompt_template is None:
        prompt_template = str(COT_PROMPT_TEMPLATE)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.train()

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    history: dict[str, Any] = {"steps": [], "val": []}

    for step in range(n_grpo_steps):
        # ------------------------------------------------------------------ #
        # 1.  Sample questions and build repeated prompts
        # ------------------------------------------------------------------ #
        indices = random.sample(range(len(train_examples)), n_prompts_per_rollout_batch)
        questions = [train_examples[i]["question"] for i in indices]
        gts = [train_examples[i]["answer"] for i in indices]

        # Repeat each question group_size times
        repeated_prompts = [prompt_template.format(question=q) for q in questions for _ in range(group_size)]
        repeated_gts = [gt for gt in gts for _ in range(group_size)]

        # ------------------------------------------------------------------ #
        # 2.  Generate rollouts with the current (old) policy
        # ------------------------------------------------------------------ #
        model.eval()
        rollout_responses = _generate_responses(
            model, tokenizer, repeated_prompts,
            max_new_tokens=sampling_max_tokens,
            min_new_tokens=sampling_min_tokens,
            temperature=sampling_temperature,
            device=device,
        )
        model.train()

        # ------------------------------------------------------------------ #
        # 3.  Compute rewards and advantages
        # ------------------------------------------------------------------ #
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )
        advantages = advantages.to(device)

        # ------------------------------------------------------------------ #
        # 4.  Tokenise rollouts
        # ------------------------------------------------------------------ #
        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        # ------------------------------------------------------------------ #
        # 5.  Cache old log-probs (no gradient)
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            old_result = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            old_log_probs = old_result["log_probs"]  # (rollout_batch_size, seq)

        # ------------------------------------------------------------------ #
        # 6.  Train epochs (typically 1 on-policy epoch)
        # ------------------------------------------------------------------ #
        step_losses: list[float] = []
        for _epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            mb_count = rollout_batch_size // micro_train_batch_size
            for mb_idx in range(mb_count):
                s = mb_idx * micro_train_batch_size
                e = s + micro_train_batch_size

                mb_input_ids = input_ids[s:e]
                mb_labels = labels[s:e]
                mb_mask = response_mask[s:e]
                mb_adv = advantages[s:e].unsqueeze(1)      # (mb, 1)
                mb_old_lp = old_log_probs[s:e].detach()

                policy_result = get_response_log_probs(model, mb_input_ids, mb_labels)
                policy_lp = policy_result["log_probs"]

                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_lp,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=cliprange,
                )
                step_losses.append(float(loss))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        step_info: dict[str, Any] = {
            "step": step,
            "loss": float(sum(step_losses) / len(step_losses)),
            "mean_reward": reward_meta["mean_reward"],
            "mean_advantage": float(advantages.mean()),
            "grad_norm": float(
                sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            ),
        }
        history["steps"].append(step_info)

        print(
            f"Step {step+1}/{n_grpo_steps} | loss={step_info['loss']:.4f} "
            f"| reward={step_info['mean_reward']:.3f}"
        )

        # ------------------------------------------------------------------ #
        # 7.  Periodic validation
        # ------------------------------------------------------------------ #
        if (step + 1) % log_every == 0:
            val_metrics = _eval_validation(
                model, tokenizer, val_examples, reward_fn,
                n_val_examples, prompt_template,
                sampling_max_tokens, sampling_temperature, device,
            )
            val_metrics["step"] = step
            history["val"].append(val_metrics)
            print(
                f"  [VAL] answer_reward={val_metrics['val_answer_reward']:.3f} "
                f"format_reward={val_metrics['val_format_reward']:.3f}"
            )

            # Log some sample generations
            sample_prompts = repeated_prompts[:4]
            sample_responses = rollout_responses[:4]
            sample_gts = repeated_gts[:4]
            sample_infos = [reward_fn(r, gt) for r, gt in zip(sample_responses, sample_gts)]
            gen_logs = log_generations(sample_prompts, sample_responses, sample_gts, sample_infos)
            for idx, entry in enumerate(gen_logs):
                print(
                    f"  [GEN {idx}] reward={entry['total_reward']:.1f} "
                    f"len={entry['response_length']} | "
                    f"{entry['response'][:120]!r}"
                )

    if output_dir is not None:
        import json, os
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return history
