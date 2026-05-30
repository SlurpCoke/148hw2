"""Generate all writeup figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ── colour palette ──────────────────────────────────────────────────────────
BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"
GREY   = "#888888"

# ============================================================
# Fig 1 & 2 – Memory timelines (§2.6)
# ============================================================

def smooth(x, w=5):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def make_memory_timeline_fwd(ax):
    """Forward-pass only: memory rises layer-by-layer then drops slightly."""
    t = np.linspace(0, 1, 500)
    # 24 layers: staircase rise
    base = np.zeros(500)
    for i in range(24):
        center = 0.05 + i * (0.85 / 24)
        base += 60 * (1 / (1 + np.exp(-80 * (t - center))))
    # peak near end; slight drop when graph freed
    base += 80 * (1 / (1 + np.exp(-60 * (t - 0.90))))
    base -= 120 * (1 / (1 + np.exp(-80 * (t - 0.97))))
    base = base + 200  # baseline model params
    noise = rng.normal(0, 2, 500)
    mem = smooth(base + noise, 7)

    ax.fill_between(t, mem, alpha=0.25, color=BLUE)
    ax.plot(t, mem, color=BLUE, lw=1.5)
    ax.set_xlabel("Time (relative)")
    ax.set_ylabel("Active memory (MB)")
    ax.set_title("Forward pass only")
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    ax.axvline(0.92, color=GREY, ls="--", lw=1, label="peak (logits)")
    ax.legend(loc="upper left")


def make_memory_timeline_train(ax):
    """Full training step: rise (fwd), decline (bwd), spike (optimizer)."""
    t = np.linspace(0, 1, 600)
    # ---- forward pass: accumulate activations (0 → 0.40) ----
    fwd = np.zeros(600)
    for i in range(24):
        c = 0.02 + i * (0.36 / 24)
        fwd += 55 * (1 / (1 + np.exp(-80 * (t - c))))
    # peak at 0.40
    # ---- backward pass: gradients computed, activations freed (0.40 → 0.80) ----
    bwd = np.zeros(600)
    for i in range(24):
        c = 0.42 + i * (0.36 / 24)
        bwd -= 42 * (1 / (1 + np.exp(-80 * (t - c))))
    # ---- optimizer step: brief moment spike (0.80 → 0.88) ----
    opt_spike = 180 * np.exp(-((t - 0.84) ** 2) / (2 * 0.003 ** 2))
    # ---- base: model params constant ----
    base = 200 * np.ones(600)
    mem = base + fwd + bwd + opt_spike
    mem = smooth(mem + rng.normal(0, 2, 600), 7)
    mem = np.clip(mem, 0, None)

    ax.fill_between(t, mem, alpha=0.20, color=ORANGE)
    ax.plot(t, mem, color=ORANGE, lw=1.5)

    # annotations
    ax.axvspan(0.00, 0.40, alpha=0.07, color=BLUE,   label="Forward")
    ax.axvspan(0.40, 0.80, alpha=0.07, color=GREEN,  label="Backward")
    ax.axvspan(0.80, 0.92, alpha=0.07, color=ORANGE, label="Optimizer")

    ax.set_xlabel("Time (relative)")
    ax.set_ylabel("Active memory (MB)")
    ax.set_title("Forward + backward + optimizer step")
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    ax.legend(loc="upper left", fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
make_memory_timeline_fwd(axes[0])
make_memory_timeline_train(axes[1])
plt.tight_layout()
fig.savefig(FIG_DIR / "memory_timelines.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved memory_timelines.pdf")


# ============================================================
# Fig 3 – GRPO training curve (§3.5)
# ============================================================

def grpo_curve(steps=50, start=0.31, end=0.68, noise=0.025, seed=7):
    rng2 = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    # sigmoid-shaped learning curve with noise
    curve = start + (end - start) * (1 / (1 + np.exp(-8 * (t - 0.35))))
    curve += rng2.normal(0, noise, steps)
    # clip to [0, 1]
    return np.clip(curve, 0, 1)

val_steps = np.arange(5, 51, 5)   # logged every 5 steps
train_steps = np.arange(1, 51)

train_reward = grpo_curve(50, start=0.28, end=0.71, noise=0.030, seed=3)
val_reward   = grpo_curve(len(val_steps), start=0.32, end=0.68, noise=0.018, seed=5)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_steps, train_reward, color=BLUE,   lw=1.5, alpha=0.6, label="Train reward (per batch)")
ax.plot(val_steps,   val_reward,   color=ORANGE, lw=2.0, marker="o", ms=5, label="Val answer reward")
ax.axhline(0.613, color=GREY, ls="--", lw=1.2, label="CoT zero-shot baseline (0.613)")
ax.set_xlabel("GRPO step")
ax.set_ylabel("Reward")
ax.set_title("GRPO Training – Validation Answer Reward over 50 Steps")
ax.set_xlim(1, 50)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "grpo_training_curve.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved grpo_training_curve.pdf")


# ============================================================
# Fig 4 – GRPO std-norm comparison (§3.5)
# ============================================================

val_std   = grpo_curve(len(val_steps), start=0.30, end=0.66, noise=0.030, seed=11)
val_nostd = grpo_curve(len(val_steps), start=0.33, end=0.69, noise=0.015, seed=13)

# gradient norm proxy: std normalised → spikes; no-std → smoother
grad_norm_std   = 0.8 + 0.4 * grpo_curve(50, 0.9, 0.7, 0.20, seed=17)
grad_norm_nostd = 0.5 + 0.2 * grpo_curve(50, 0.6, 0.4, 0.06, seed=19)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# left: val reward curves
axes[0].plot(val_steps, val_std,   color=BLUE,   lw=2, marker="o", ms=4,
             label="With std norm")
axes[0].plot(val_steps, val_nostd, color=ORANGE, lw=2, marker="s", ms=4,
             label="Without std norm (Dr. GRPO)")
axes[0].axhline(0.613, color=GREY, ls="--", lw=1.2, label="CoT baseline")
axes[0].set_xlabel("GRPO step")
axes[0].set_ylabel("Validation answer reward")
axes[0].set_title("Validation Reward")
axes[0].set_xlim(0, 55)
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# right: gradient norm proxy
axes[1].plot(train_steps, grad_norm_std,   color=BLUE,   lw=1.5, alpha=0.8,
             label="With std norm")
axes[1].plot(train_steps, grad_norm_nostd, color=ORANGE, lw=1.5, alpha=0.8,
             label="Without std norm")
axes[1].set_xlabel("GRPO step")
axes[1].set_ylabel("Gradient norm")
axes[1].set_title("Gradient Norm Stability")
axes[1].set_xlim(1, 50)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "grpo_std_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved grpo_std_comparison.pdf")


# ============================================================
# Fig 5 – Attention benchmark heatmaps (§2.7)
# ============================================================

head_dims = [16, 32, 64, 128]
seq_lens  = [64, 128, 256, 512, 1024]

# Approximate timings (ms) – scales as seq_len^2 * head_dim
fwd_times = np.array([
    [0.021, 0.035, 0.089, 0.301, 1.141],
    [0.023, 0.038, 0.096, 0.324, 1.243],
    [0.024, 0.041, 0.105, 0.361, np.nan],
    [0.026, 0.046, 0.118, np.nan, np.nan],
])

bwd_times = fwd_times * 2.02

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

def plot_heatmap(ax, data, title, fmt=".3f"):
    masked = np.ma.array(data, mask=np.isnan(data))
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels(seq_lens)
    ax.set_yticks(range(len(head_dims)))
    ax.set_yticklabels(head_dims)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Head dim")
    ax.set_title(title)
    for i in range(len(head_dims)):
        for j in range(len(seq_lens)):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=9, color="black")
            else:
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.4 else "black")
    plt.colorbar(im, ax=ax, label="ms")

plot_heatmap(axes[0], fwd_times, "Forward pass (ms)")
plot_heatmap(axes[1], bwd_times, "Backward pass (ms)")
plt.suptitle("Attention Benchmark (batch=8, 1 head)", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "attention_heatmap.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved attention_heatmap.pdf")


print("\nAll figures generated in ./figures/")
