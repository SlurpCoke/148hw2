from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main")
    return list(dataset[split])


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=ex["question"]) for ex in examples]


# ---------------------------------------------------------------------------
# vLLM evaluation
# ---------------------------------------------------------------------------

def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serialisable artifacts."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)

    responses = [out.outputs[0].text for out in outputs]
    return {"prompts": list(prompts), "responses": responses}


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialise generations and scores to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_path}")


# ---------------------------------------------------------------------------
# Direct-prediction baseline (§3.1)
# ---------------------------------------------------------------------------

def run_direct_baseline(output_path: Path) -> dict[str, Any]:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn

    output_path = Path(output_path)
    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = [ex["answer"] for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    artifacts = evaluate_vllm(llm, answer_tag_reward_fn, prompts, sampling_params)
    responses = artifacts["responses"]

    scores = [answer_tag_reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]
    artifacts["ground_truths"] = ground_truths
    artifacts["scores"] = scores

    n = len(scores)
    correct = sum(1 for s in scores if s["answer_reward"] == 1.0)
    format_ok = sum(1 for s in scores if s["format_reward"] == 1.0)
    artifacts["accuracy"] = correct / n
    artifacts["format_rate"] = format_ok / n

    print(f"Direct Prediction  accuracy={correct/n:.4f}  format_rate={format_ok/n:.4f}")
    write_evaluation_results(artifacts, output_path)
    return artifacts


# ---------------------------------------------------------------------------
# Chain-of-Thought baseline (§3.2)
# ---------------------------------------------------------------------------

def run_cot_baseline(output_path: Path) -> dict[str, Any]:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    from vllm import LLM, SamplingParams
    from .drgrpo_grader import r1_zero_reward_fn

    output_path = Path(output_path)
    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [ex["answer"] for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    artifacts = evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params)
    responses = artifacts["responses"]

    scores = [r1_zero_reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]
    artifacts["ground_truths"] = ground_truths
    artifacts["scores"] = scores

    n = len(scores)
    correct = sum(1 for s in scores if s["answer_reward"] == 1.0)
    format_ok = sum(1 for s in scores if s["format_reward"] == 1.0)
    artifacts["accuracy"] = correct / n
    artifacts["format_rate"] = format_ok / n

    print(f"CoT  accuracy={correct/n:.4f}  format_rate={format_ok/n:.4f}")
    write_evaluation_results(artifacts, output_path)
    return artifacts


# ---------------------------------------------------------------------------
# Self-consistency baseline (§3.2)
# ---------------------------------------------------------------------------

def run_self_consistency_baseline(output_path: Path, k: int = 5) -> dict[str, Any]:
    """Evaluate the self-consistency baseline from Section 3.2.

    For each question we sample K responses and take a majority vote.
    """
    from vllm import LLM, SamplingParams
    from .drgrpo_grader import r1_zero_reward_fn
    from .rewards import extract_answer_from_tags

    output_path = Path(output_path)
    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [ex["answer"] for ex in examples]
    n = len(examples)

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=k,  # generate k responses per prompt
    )

    outputs = llm.generate(prompts, sampling_params)

    all_responses: list[list[str]] = []
    for out in outputs:
        all_responses.append([o.text for o in out.outputs])

    # Majority vote over extracted answers
    voted_answers: list[str | None] = []
    for responses_for_q in all_responses:
        extracted = [extract_answer_from_tags(r) for r in responses_for_q]
        valid = [a for a in extracted if a is not None]
        if valid:
            voted_answers.append(Counter(valid).most_common(1)[0][0])
        else:
            voted_answers.append(None)

    # Score voted answers against ground truths
    from .drgrpo_grader import grade
    correct = 0
    scores = []
    for voted, gt in zip(voted_answers, ground_truths):
        if voted is not None and grade(voted, gt):
            correct += 1
            scores.append({"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0})
        else:
            scores.append({"answer_reward": 0.0, "format_reward": float(voted is not None), "reward": 0.0})

    artifacts = {
        "prompts": prompts,
        "all_responses": all_responses,
        "voted_answers": voted_answers,
        "ground_truths": ground_truths,
        "scores": scores,
        "accuracy": correct / n,
        "k": k,
    }

    print(f"Self-Consistency (k={k})  accuracy={correct/n:.4f}")
    write_evaluation_results(artifacts, output_path)
    return artifacts


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_prompt_template(use_cot: bool) -> str:
    return str(COT_PROMPT_TEMPLATE) if use_cot else DIRECT_PROMPT_TEMPLATE
