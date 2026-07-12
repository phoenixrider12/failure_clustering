"""Reproduce the paper's result figures.

Each function renders one figure from the paper using the reported numbers
(kept here as clearly-labelled constants). Run the whole module to regenerate
every figure, or import a single function.

    python plot_results.py --output-dir .

Figures:
    monitoring_f1        Runtime-monitoring F1 (In-D vs OOD) for car & robot.
    vlm_comparison       VLM comparison on manipulation descriptions (CS/ROUGE-L/LLM-J).
    finetuned_comparison Fine-tuned failure models (AHA, RoboFAC) vs Gemini.
    frame_ablation       Frame-sampling ablation (fixed fps vs semantic downsampling).
    clustering_comparison Taxonomy-discovery comparison (BERTopic vs LLM ensemble).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Consistent styling across all figures.
sns.set(style="white", font_scale=2.0)
plt.rcParams["font.family"] = "Times New Roman"
COLORS = ["#4C72B0", "#55A868", "#C44E52"]  # metric palette
IND_OOD = ["#4C72B0", "#DD8452"]


def _annotate(ax, bars, fmt="{:.2f}", fontsize=16):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=fontsize)


def _grouped_metric_panel(ax, groups, cs, rouge, llmj, title=None, annotate="{:.3f}"):
    x = np.arange(len(groups))
    width = 0.22
    b1 = ax.bar(x - width, cs, width, label="CS ↑", color=COLORS[0])
    b2 = ax.bar(x, rouge, width, label="ROUGE-L ↑", color=COLORS[1])
    b3 = ax.bar(x + width, llmj, width, label="LLM-J ↑", color=COLORS[2])
    for bars in (b1, b2, b3):  # highlight the last group (our method)
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2.5)
        _annotate(ax, bars, fmt=annotate)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15)
    ax.set_ylabel("Score")
    if title:
        ax.set_title(title, fontweight="bold")


# --------------------------------------------------------------------------- #
# Figure: runtime-monitoring F1
# --------------------------------------------------------------------------- #
def monitoring_f1(output_dir: str) -> None:
    car_methods = ["VideoMAE-BC", "LLM-AD", "NoContext", "Ours"]
    car_ind, car_ood = [65.3, 12.3, 54.1, 71.4], [25.2, 49.7, 69.6, 77.9]
    car_lead = ["506.6 ms", "166.6 ms", "473.3 ms", "610 ms"]

    robot_methods = ["ENet-BC", "LLM-AD", "NoContext", "Ours"]
    robot_ind, robot_ood = [78.8, 40.0, 67.4, 77.2], [22.4, 27.2, 40.5, 50.0]
    robot_lead = ["1.01 s", "1.38 s", "0.76 s", "1.21 s"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    def panel(ax, methods, ind, ood, lead, title):
        x = np.arange(len(methods))
        width = 0.35
        b1 = ax.bar(x - width / 2, ind, width, label="In-Distribution", color=IND_OOD[0])
        b2 = ax.bar(x + width / 2, ood, width, label="Out-of-Distribution", color=IND_OOD[1])
        for i, m in enumerate(methods):
            if m == "Ours":
                for b in (b1[i], b2[i]):
                    b.set_edgecolor("black")
                    b.set_linewidth(2.5)
        for i, txt in enumerate(lead):
            ax.text(x[i], max(ind[i], ood[i]) + 6, txt, ha="center", va="bottom",
                    fontsize=18, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("F1 Score")

    panel(axes[0], car_methods, car_ind, car_ood, car_lead, "Real-World Car Crash Videos")
    panel(axes[1], robot_methods, robot_ind, robot_ood, robot_lead, "Vision-Based Indoor Robot Navigation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, output_dir, "monitoring_f1.png")


# --------------------------------------------------------------------------- #
# Figure: VLM comparison (manipulation descriptions)
# --------------------------------------------------------------------------- #
def vlm_comparison(output_dir: str) -> None:
    models = ["LLaVA-NeXT", "Qwen2.5-VL-7B", "OpenAI o4-mini", "Cosmos-Reason1-7B", "Gemini 2.5 Pro"]
    sim = ([0.4846, 0.5200, 0.5557, 0.5324, 0.6003],
           [0.2017, 0.2273, 0.2422, 0.1541, 0.2589],
           [0.10, 0.26, 0.46, 0.20, 0.76])
    real = ([0.6305, 0.6890, 0.6770, 0.6930, 0.6280],
            [0.2657, 0.3570, 0.3640, 0.2290, 0.3420],
            [0.103, 0.333, 0.370, 0.133, 0.567])
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    _grouped_metric_panel(axes[0], models, *sim, title="Simulation", annotate="{:.2f}")
    _grouped_metric_panel(axes[1], models, *real, title="Real-World", annotate="{:.2f}")
    axes[0].set_ylim(0, 0.85)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, output_dir, "vlm_comparison.png")


# --------------------------------------------------------------------------- #
# Figure: fine-tuned failure models vs Gemini
# --------------------------------------------------------------------------- #
def finetuned_comparison(output_dir: str) -> None:
    models = ["AHA-13B", "RoboFAC-7B", "Gemini 2.5 Pro"]
    fig, ax = plt.subplots(figsize=(12, 7))
    _grouped_metric_panel(ax, models,
                          cs=[0.471, 0.452, 0.628],
                          rouge=[0.280, 0.137, 0.342],
                          llmj=[0.465, 0.133, 0.550])
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    _save(fig, output_dir, "finetuned_comparison.png")


# --------------------------------------------------------------------------- #
# Figure: frame-sampling ablation
# --------------------------------------------------------------------------- #
def frame_ablation(output_dir: str) -> None:
    methods = ["1 fps", "0.5 fps", "0.25 fps", "Ours"]
    fig, ax = plt.subplots(figsize=(10, 8))
    _grouped_metric_panel(ax, methods,
                          cs=[0.5927, 0.5885, 0.5917, 0.6003],
                          rouge=[0.2545, 0.2582, 0.2512, 0.2589],
                          llmj=[0.66, 0.68, 0.69, 0.76])
    ax.set_xticklabels(methods, rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    plt.tight_layout()
    _save(fig, output_dir, "frame_ablation.png")


# --------------------------------------------------------------------------- #
# Figure: taxonomy-discovery (clustering) comparison
# --------------------------------------------------------------------------- #
def clustering_comparison(output_dir: str) -> None:
    methods = ["BERTopic", "BERTopic-LLM", "Ours (Single Run)", "Ours (Aggregation)"]
    metrics = ["CP ↑", "TC ↑", "SAS ↑"]
    values = np.array([
        [0.785, 0.625, 0.696],
        [0.875, 0.875, 0.875],
        [0.818, 0.900, 0.849],
        [0.920, 1.000, 0.958],
    ])
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(methods))
    width = 0.22
    for k in range(3):
        bars = ax.bar(x + (k - 1) * width, values[:, k], width, label=metrics[k])
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2.5)
        _annotate(ax, bars, fmt="{:.3f}", fontsize=18)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    _save(fig, output_dir, "clustering_comparison.png")


def _save(fig, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


FIGURES = {
    "monitoring_f1": monitoring_f1,
    "vlm_comparison": vlm_comparison,
    "finetuned_comparison": finetuned_comparison,
    "frame_ablation": frame_ablation,
    "clustering_comparison": clustering_comparison,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=".", help="Directory to write figures into.")
    p.add_argument("--figure", choices=list(FIGURES), default=None,
                   help="Render only this figure (default: all).")
    args = p.parse_args()

    to_render = [args.figure] if args.figure else list(FIGURES)
    for name in to_render:
        FIGURES[name](args.output_dir)


if __name__ == "__main__":
    main()
