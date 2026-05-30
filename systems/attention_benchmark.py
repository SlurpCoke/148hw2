from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


# ---------------------------------------------------------------------------
# QKV creation
# ---------------------------------------------------------------------------

def make_qkv(
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    shape = (batch_size, sequence_length, head_dim)
    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    return q, k, v


# ---------------------------------------------------------------------------
# Single-configuration benchmark
# ---------------------------------------------------------------------------

def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: AttentionBenchmarkConfig | None = None,
) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    from basics.model import scaled_dot_product_attention

    n_fwd = config.forward_passes if config is not None else 100
    n_bwd = config.backward_passes if config is not None else 100
    n_warmup = 5

    attn_fn = scaled_dot_product_attention
    if config is not None and config.compile_attention:
        attn_fn = torch.compile(scaled_dot_product_attention)

    device = q.device

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    for _ in range(n_warmup):
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = attn_fn(q_, k_, v_)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Time forward passes
    # ------------------------------------------------------------------
    fwd_times: list[float] = []
    for _ in range(n_fwd):
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        t0 = timeit.default_timer()
        _ = attn_fn(q_, k_, v_)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        fwd_times.append(t1 - t0)

    # ------------------------------------------------------------------
    # Measure peak memory AFTER one forward pass (graph still live)
    # ------------------------------------------------------------------
    q_ = q.detach().requires_grad_(True)
    k_ = k.detach().requires_grad_(True)
    v_ = v.detach().requires_grad_(True)
    out_mem = attn_fn(q_, k_, v_)
    torch.cuda.synchronize()
    mem_before_bwd_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    del out_mem, q_, k_, v_
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Time backward passes (forward + backward; time only the backward)
    # ------------------------------------------------------------------
    bwd_times: list[float] = []
    for _ in range(n_bwd):
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = attn_fn(q_, k_, v_)
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        out.sum().backward()
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        bwd_times.append(t1 - t0)

    return {
        "fwd_mean_ms": statistics.mean(fwd_times) * 1000.0,
        "fwd_std_ms": (statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0.0) * 1000.0,
        "bwd_mean_ms": statistics.mean(bwd_times) * 1000.0,
        "bwd_std_ms": (statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0.0) * 1000.0,
        "mem_before_bwd_mb": mem_before_bwd_mb,
    }


# ---------------------------------------------------------------------------
# Full grid benchmark
# ---------------------------------------------------------------------------

def benchmark_attention_grid(
    config: AttentionBenchmarkConfig,
) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, float | int | str]] = []

    print(
        f"{'head_dim':>8}  {'seq_len':>7}  "
        f"{'fwd_ms':>8}  {'bwd_ms':>8}  {'mem_mb':>8}"
    )
    print("-" * 55)

    for head_dim, seq_len in iter_benchmark_shapes(config):
        row: dict[str, float | int | str] = {
            "head_dim": head_dim,
            "seq_len": seq_len,
        }
        try:
            q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device)
            metrics = benchmark_attention_once(q, k, v, config)
            row.update(metrics)
            print(
                f"{head_dim:>8}  {seq_len:>7}  "
                f"{metrics['fwd_mean_ms']:>8.3f}  "
                f"{metrics['bwd_mean_ms']:>8.3f}  "
                f"{metrics['mem_before_bwd_mb']:>8.2f}"
            )
        except torch.cuda.OutOfMemoryError:
            row.update({
                "fwd_mean_ms": float("nan"),
                "bwd_mean_ms": float("nan"),
                "mem_before_bwd_mb": float("nan"),
                "oom": True,
            })
            print(f"{head_dim:>8}  {seq_len:>7}  {'OOM':>8}  {'OOM':>8}  {'OOM':>8}")
            torch.cuda.empty_cache()

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
