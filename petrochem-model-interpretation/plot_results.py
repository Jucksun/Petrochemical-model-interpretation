#!/usr/bin/env python3
"""Generate publication-ready benchmark & interpretability charts for GitHub / Drive."""
import argparse
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def plot_behavioral_summary(summary_csv: str, out_dir: str):
    with open(summary_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    models = [short_name(r["model"]) for r in rows]
    cats = [c for c in rows[0].keys() if c not in ("model", "overall")]

    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    colors = ["#2b5c8f", "#d97724", "#2e8b57", "#7b4ea3"]

    for i, r in enumerate(rows):
        vals = [float(r[c]) * 100 for c in cats]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=models[i], color=colors[i % len(colors)])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Petrochemical OSINT Safety & Factuality Benchmark (Local Open-Weight Models)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cats])
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig1_behavioral_benchmark.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_interpretability(probe_json: str, out_dir: str):
    with open(probe_json, encoding="utf-8") as f:
        data = json.load(f)

    colors = ["#2b5c8f", "#d97724", "#2e8b57", "#7b4ea3"]

    # Figure 2: Layer-by-Layer Linear Probe Accuracy (Factual vs. Abstain)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    for i, (model_id, mdata) in enumerate(data.items()):
        accs = [a * 100 for a in mdata["layer_probe_accuracy_fact_vs_abstain"]]
        # Normalize layer depth to 0-100% so models with different layer counts align cleanly
        depth_pct = np.linspace(0, 100, len(accs))
        ax.plot(
            depth_pct,
            accs,
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"{short_name(model_id)} ({len(accs)}L)",
            color=colors[i % len(colors)],
        )

    ax.axhline(50, color="gray", linestyle=":", label="Random Chance (50%)")
    ax.set_xlabel("Relative Model Depth (% of Transformer Layers)")
    ax.set_ylabel("LOO Linear Probe Accuracy (%)")
    ax.set_title("Internal Entity Familiarity: Linear Probe Separating Real vs. Fabricated Facilities")
    ax.set_ylim(20, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    plt.tight_layout()

    out_path2 = os.path.join(out_dir, "fig2_linear_probe_layers.png")
    plt.savefig(out_path2)
    plt.close()
    print(f"Saved {out_path2}")

    # Figure 3: Refusal Direction Cosine Similarity & Logit Lens Trajectory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    # Left panel: Refusal direction projections
    categories = ["refuse_items", "comply_items", "factual_items", "abstain_items"]
    cat_labels = ["Harmful\n(safe-*)", "Worker PPE\n(help-*)", "Known Facts\n(fact-*)", "Fake Entities\n(abst-*)"]
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(data))

    for i, (model_id, mdata) in enumerate(data.items()):
        proj = mdata["refusal_projections"]
        means = [np.mean(list(proj[c].values())) for c in categories]
        offset = (i - (len(data) - 1) / 2) * width
        ax1.bar(x + offset, means, width, label=short_name(model_id), color=colors[i % len(colors)])

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels)
    ax1.set_ylabel("Mean Cosine Similarity w/ Refusal Vector")
    ax1.set_title("Refusal Direction Sensitivity (75% Depth)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    # Right panel: Logit lens probability of "NOT" on ground-02
    for i, (model_id, mdata) in enumerate(data.items()):
        probs = mdata["logit_lens_ground02_NOT_prob"]
        depth_pct = np.linspace(0, 100, len(probs))
        ax2.plot(
            depth_pct,
            probs,
            marker="s",
            markersize=3.5,
            linewidth=2,
            label=short_name(model_id),
            color=colors[i % len(colors)],
        )

    ax2.set_xlabel("Relative Model Depth (%)")
    ax2.set_ylabel("P(next token = 'NOT')")
    ax2.set_title("Logit Lens: Grounded Abstention ('NOT IN RECORD')")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path3 = os.path.join(out_dir, "fig3_refusal_and_logit_lens.png")
    plt.savefig(out_path3)
    plt.close()
    print(f"Saved {out_path3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    plot_behavioral_summary(os.path.join(args.out_dir, "summary.csv"), args.out_dir)
    plot_interpretability(os.path.join(args.out_dir, "probe_metrics.json"), args.out_dir)
