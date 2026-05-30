from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small":  ModelSpec(d_model=512,  d_ff=2048,  num_layers=8,  num_heads=8),
    "medium": ModelSpec(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "large":  ModelSpec(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    from basics import BasicsTransformerLM

    spec = MODEL_SPECS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10000.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if config.compile_model:
        model = torch.compile(model)
    return model


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# Single step execution
# ---------------------------------------------------------------------------

def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    with autocast_context:
        logits = model(batch)
        if mode in ("forward-backward", "train-step"):
            # Use .float() to avoid issues with BF16 reduction
            loss = logits.float().mean()
            loss.backward()

    if mode == "train-step" and optimizer is not None:
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    batch = make_random_batch(config, device)
    autocast_ctx = make_autocast_context(config.use_bf16)

    optimizer: torch.optim.Optimizer | None = None
    if config.mode == "train-step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # Warm-up (annotate as NVTX range so it can be filtered out in nsys)
    # ------------------------------------------------------------------
    with nvtx.range("warmup"):
        for _ in range(config.warmup_steps):
            run_single_step(model, batch, config.mode, autocast_ctx, optimizer)
            if optimizer is not None:
                optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Start memory profiling (if requested) after warm-up
    # ------------------------------------------------------------------
    maybe_start_memory_history(config.use_memory_profiler)

    # ------------------------------------------------------------------
    # Timed measurement steps
    # ------------------------------------------------------------------
    times: list[float] = []
    with nvtx.range("measure"):
        for _ in range(config.measure_steps):
            if optimizer is not None:
                optimizer.zero_grad()
            t0 = timeit.default_timer()
            run_single_step(model, batch, config.mode, autocast_ctx, optimizer)
            t1 = timeit.default_timer()
            times.append(t1 - t0)

    # ------------------------------------------------------------------
    # Save memory snapshot and stop profiling
    # ------------------------------------------------------------------
    config.output_dir.mkdir(parents=True, exist_ok=True)
    maybe_dump_memory_snapshot(
        config.use_memory_profiler,
        config.output_dir / f"memory_{config.model_size}_{config.mode}.pickle",
    )

    mean_ms = statistics.mean(times) * 1000.0
    std_ms = (statistics.stdev(times) if len(times) > 1 else 0.0) * 1000.0

    print(
        f"[{config.model_size:6s}] ctx={config.context_length:4d} "
        f"mode={config.mode:16s} bf16={config.use_bf16} "
        f"mean={mean_ms:8.2f}ms  std={std_ms:6.2f}ms"
    )

    return {
        "model_size": config.model_size,
        "context_length": config.context_length,
        "mode": config.mode,
        "use_bf16": config.use_bf16,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "times_ms": [t * 1000.0 for t in times],
    }


# ---------------------------------------------------------------------------
# NVTX-annotated attention (for nsys profiling)
# ---------------------------------------------------------------------------

def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention wrapped with NVTX ranges for nsys profiling."""
    from basics.nn_utils import softmax

    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))

    with nvtx.range("computing softmax"):
        weights = softmax(scores, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(weights, V, "... query key, ... key d_v -> ... query d_v")

    return output


# ---------------------------------------------------------------------------
# Memory profiling helpers
# ---------------------------------------------------------------------------

def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to {output_path}")


# ---------------------------------------------------------------------------
# Mixed-precision context helper
# ---------------------------------------------------------------------------

def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
