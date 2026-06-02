import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_PYTORCH = Path("log/pytorch_metrics.json")
LOG_DEEPSPEED = Path("log/step3_deepspeed_metrics.json")
LOG_1GPU = Path("log/step2_metrics.json")
OUTPUT_DIR = Path("charts")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required log file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def smooth_ema(values, alpha=0.15):
    ema = values[0]
    smoothed = []
    for value in values:
        ema = (1 - alpha) * ema + alpha * value
        smoothed.append(ema)
    return smoothed


def label_bars(bars, fmt, offset):
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


OUTPUT_DIR.mkdir(exist_ok=True)

pt_data = load_json(LOG_PYTORCH)
ds_data = load_json(LOG_DEEPSPEED)
onegpu_data = load_json(LOG_1GPU)

print(f"Loaded {len(pt_data)} PyTorch pipeline runs from {LOG_PYTORCH}")
print(f"Loaded {len(ds_data)} DeepSpeed pipeline runs from {LOG_DEEPSPEED}")
print(f"Loaded 1 baseline run from {LOG_1GPU}")


plt.figure(figsize=(10, 6))
for config in pt_data:
    metrics = config.get("per_step_metrics", [])
    if not metrics:
        continue
    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    plt.plot(
        steps,
        smooth_ema(losses),
        label=f"Chunks = {config['chunks']} (Bubble {config['bubble_ratio']:.1%})",
        linewidth=2,
    )
plt.title("Đường cong giảm Loss qua các bước huấn luyện (PyTorch Native PP)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Optimizer Steps", fontsize=11)
plt.ylabel("Loss (EMA)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_loss_curves.png", dpi=150)
plt.close()
print("Generated charts/1_loss_curves.png")


plt.figure(figsize=(10, 6))
for config in pt_data:
    metrics = config.get("per_step_metrics", [])
    if not metrics:
        continue
    steps = [m["step"] for m in metrics]
    tps = [m["tokens_per_sec"] for m in metrics]
    plt.plot(steps, tps, label=f"Chunks = {config['chunks']}", alpha=0.85, linewidth=1.5)
plt.title("Sự ổn định của Tốc độ Huấn luyện (PyTorch Native PP)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Optimizer Steps", fontsize=11)
plt.ylabel("Throughput (Tokens/second)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_throughput_stability.png", dpi=150)
plt.close()
print("Generated charts/2_throughput_stability.png")


chunks_labels = [f"Chunks {c['chunks']}" for c in pt_data]
bubble_ratios = [c["bubble_ratio"] * 100 for c in pt_data]
latencies = [c["avg_sec_per_step"] for c in pt_data]

fig, ax1 = plt.subplots(figsize=(10, 6))
color = "#1f77b4"
ax1.set_xlabel("Cấu hình Chunks", fontsize=11)
ax1.set_ylabel("Thời gian bước trung bình (seconds)", color=color, fontsize=11)
bars = ax1.bar(chunks_labels, latencies, color=color, alpha=0.65, width=0.4, label="Sec/step")
ax1.tick_params(axis="y", labelcolor=color)
for bar in bars:
    value = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, value + 0.5, f"{value:.2f}s", ha="center", va="bottom", color="#111", fontweight="bold")

ax2 = ax1.twinx()
color = "#d62728"
ax2.set_ylabel("Tỷ lệ Bubble lý thuyết (%)", color=color, fontsize=11)
ax2.plot(chunks_labels, bubble_ratios, color=color, marker="o", linewidth=2.5, markersize=8, label="Bubble ratio")
ax2.tick_params(axis="y", labelcolor=color)
for idx, value in enumerate(bubble_ratios):
    ax2.annotate(f"{value:.2f}%", (chunks_labels[idx], value), textcoords="offset points", xytext=(0, 10), ha="center", color=color, fontweight="bold")

plt.title("Tác động của Chunks đến Bubble Ratio và Độ trễ bước", fontsize=13, fontweight="bold", pad=15)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "3_chunks_vs_bubble_latency.png", dpi=150)
plt.close()
print("Generated charts/3_chunks_vs_bubble_latency.png")


plt.figure(figsize=(10, 6))
methods = ["1GPU + GC\n(Baseline)"]
throughputs = [onegpu_data["avg_tokens_sec"]]
colors = ["#ff7f0e"]

for config in pt_data:
    methods.append(f"PyTorch PP\n(Chunks {config['chunks']})")
    throughputs.append(config["avg_tokens_sec"])
    colors.append("#1f77b4")

for config in ds_data:
    methods.append(f"DeepSpeed PP\n(Chunks {config['chunks']})")
    throughputs.append(config["avg_tokens_sec"])
    colors.append("#2ca02c")

x_pos = np.arange(len(methods))
bars = plt.bar(x_pos, throughputs, color=colors, alpha=0.85, width=0.5)
label_bars(bars, "{:.1f}", 10)
plt.title("So sánh Throughput giữa các giải pháp", fontsize=13, fontweight="bold", pad=15)
plt.xticks(x_pos, methods, fontsize=9)
plt.ylabel("Throughput (Tokens/second)", fontsize=11)
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.figtext(
    0.13,
    -0.05,
    "*PyTorch PP dùng effective batch size = 32, còn DeepSpeed PP dùng effective batch size = 16,\n"
    "nên biểu đồ này phù hợp để so sánh xu hướng throughput hơn là đối đầu tuyệt đối theo cấu hình batch.",
    ha="left",
    fontsize=9,
    style="italic",
    color="#555",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_throughput_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Generated charts/4_throughput_comparison.png")


plt.figure(figsize=(10, 6))
vram_methods = ["1GPU + GC\n(Baseline)"]
vram_values = [onegpu_data["peak_vram_gb"]]
vram_colors = ["#ff7f0e"]

for config in pt_data:
    vram_methods.append(f"PyTorch PP\n(Chunks {config['chunks']})")
    vram_values.append(config["peak_vram_gb"])
    vram_colors.append("#1f77b4")

for config in ds_data:
    vram_methods.append(f"DeepSpeed PP\n(Chunks {config['chunks']})")
    vram_values.append(config["peak_vram_gb"])
    vram_colors.append("#2ca02c")

x_pos = np.arange(len(vram_methods))
bars = plt.bar(x_pos, vram_values, color=vram_colors, alpha=0.85, width=0.5)
label_bars(bars, "{:.2f} GB", 0.2)
plt.title("So sánh Peak VRAM giữa các giải pháp", fontsize=13, fontweight="bold", pad=15)
plt.xticks(x_pos, vram_methods, fontsize=9)
plt.ylabel("Peak VRAM (GB)", fontsize=11)
plt.axhline(y=16.0, color="r", linestyle="--", linewidth=1.5, label="Giới hạn VRAM GPU T4 (16GB)")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5_vram_comparison.png", dpi=150)
plt.close()
print("Generated charts/5_vram_comparison.png")

print(f"\nCompleted: 5 charts saved to {OUTPUT_DIR}/")
