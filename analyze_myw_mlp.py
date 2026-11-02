import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

from compare_myw_mlp import fit_myw, response_db
from iirnet.mlp import MLPModel


COLORS = {"MLP": "tab:red", "MYW": "tab:orange", "Target": "tab:blue"}


def load_rows(filename):
    with filename.open(newline="") as file:
        return list(csv.DictReader(file))


def paired_errors(rows, family):
    selected = [r for r in rows if r["family"] == family and r["myw_status"] == "ok"]
    myw = np.array([float(r["myw_rmse_db"]) for r in selected])
    mlp = np.array([float(r["mlp_rmse_db"]) for r in selected])
    return selected, myw, mlp


def plot_distributions(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, family in zip(axes, ("C", "D")):
        _, myw, mlp = paired_errors(rows, family)
        for label, values in (("MYW", myw), ("MLP", mlp)):
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=label, color=COLORS[label])
        ax.set(xlabel="Per-filter RMSE (dB)", title=f"Family {family}")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Fraction of filters at or below error")
    fig.suptitle("Error distributions: typical accuracy vs. upper-tail risk")
    fig.tight_layout()
    fig.savefig(output / "error_distributions.png", dpi=180)
    plt.close(fig)


def plot_paired_scatter(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, family in zip(axes, ("C", "D")):
        _, myw, mlp = paired_errors(rows, family)
        ax.scatter(myw, mlp, s=9, alpha=0.35, color="tab:purple")
        low = max(1e-4, min(myw.min(), mlp.min()) * 0.8)
        high = max(myw.max(), mlp.max()) * 1.2
        ax.plot([low, high], [low, high], "k--", linewidth=1, label="Equal RMSE")
        ax.set(xlabel="MYW RMSE (dB)", ylabel="MLP RMSE (dB)",
               title=f"Family {family}", xlim=(low, high), ylim=(low, high))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Below diagonal: MLP wins; above diagonal: MYW wins")
    fig.tight_layout()
    fig.savefig(output / "paired_scatter.png", dpi=180)
    plt.close(fig)


def plot_error_vs_radius(rows, test_sets, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, family in zip(axes, ("C", "D")):
        paired, myw, mlp = paired_errors(rows, family)
        radii = []
        for row in paired:
            target_sos = test_sets[family][int(row["example"])]
            _, poles, _ = signal.sos2zpk(target_sos)
            radii.append(float(np.max(np.abs(poles))))
        ax.scatter(radii, myw, s=10, alpha=0.3, label="MYW", color=COLORS["MYW"])
        ax.scatter(radii, mlp, s=10, alpha=0.3, label="MLP", color=COLORS["MLP"])
        ax.set(xlabel="Maximum target pole radius", title=f"Family {family}")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Per-filter RMSE (dB)")
    fig.suptitle("Does reconstruction error increase for poles near the unit circle?")
    fig.tight_layout()
    fig.savefig(output / "error_vs_target_radius.png", dpi=180)
    plt.close(fig)


def plot_example(family, label, row, target_sos, model, num_points, order, output):
    frequencies, target_h = signal.sosfreqz(target_sos, worN=num_points)
    target_db = 20 * np.log10(np.abs(target_h) + 1e-8)
    model_input = torch.tensor(np.clip(target_db, -128, 128) / 128,
                               dtype=torch.float32)[None]
    with torch.inference_mode():
        mlp_sos, _ = model(model_input)
    mlp_sos = mlp_sos.squeeze(0).cpu().numpy()
    myw_sos, _, _ = fit_myw(target_sos, np.abs(target_h), num_points, order)

    mlp_db = response_db(mlp_sos, frequencies)
    myw_db = response_db(myw_sos, frequencies)
    freq_hz = frequencies / (2 * np.pi) * 44100
    valid = freq_hz > 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name, curve in (("Target", target_db), ("MLP", mlp_db), ("MYW", myw_db)):
        axes[0].plot(freq_hz[valid], curve[valid], label=name, color=COLORS[name])
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)",
                title="Frequency-response reconstruction")
    axes[0].set_xscale("log")
    axes[0].legend()

    for name, curve in (("MLP", mlp_db), ("MYW", myw_db)):
        axes[1].plot(freq_hz[valid], (curve - target_db)[valid],
                     label=name, color=COLORS[name])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Prediction − target (dB)",
                title="Error by frequency")
    axes[1].set_xscale("log")
    axes[1].legend()

    theta = np.linspace(0, 2 * np.pi, 400)
    axes[2].plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)
    for name, sos in (("Target", target_sos), ("MLP", mlp_sos), ("MYW", myw_sos)):
        _, poles, _ = signal.sos2zpk(sos)
        axes[2].scatter(poles.real, poles.imag, marker="x", s=24,
                        color=COLORS[name], label=f"{name} poles")
    axes[2].set(xlabel="Real", ylabel="Imaginary", title="Pole locations",
                xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Family {family}, example {row['example']}: {label} | "
        f"MLP {float(row['mlp_rmse_db']):.2f} dB, "
        f"MYW {float(row['myw_rmse_db']):.2f} dB"
    )
    fig.tight_layout()
    fig.savefig(output / f"{family}_{label}.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path,
                        default=Path("results/myw_mlp_paired/per_filter.csv"))
    parser.add_argument("--filters", type=Path,
                        default=Path("results/cd_eval/test_filters.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/myw_mlp_analysis"))
    args = parser.parse_args()

    rows = load_rows(args.results)
    with np.load(args.filters) as saved:
        test_sets = {family: saved[family] for family in ("C", "D")}
    args.output.mkdir(parents=True, exist_ok=True)

    plot_distributions(rows, args.output)
    plot_paired_scatter(rows, args.output)
    plot_error_vs_radius(rows, test_sets, args.output)

    print("Loading trained MLP for representative response plots...", flush=True)
    model = MLPModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.eval()
    num_points = int(model.hparams.num_points)
    order = int(getattr(model.hparams, "max_train_order", model.hparams.model_order))

    summary = {}
    for family in ("C", "D"):
        paired, myw, mlp = paired_errors(rows, family)
        win_count = int(np.sum(myw < mlp))
        summary[family] = {
            "paired_filters": len(paired),
            "myw_lower_rmse": win_count,
            "mlp_lower_rmse": int(np.sum(mlp < myw)),
            "myw_over_5db": int(np.sum(myw > 5)),
            "mlp_over_5db": int(np.sum(mlp > 5)),
            "myw_over_10db": int(np.sum(myw > 10)),
            "mlp_over_10db": int(np.sum(mlp > 10)),
        }
        print(f"Family {family}: {json.dumps(summary[family])}", flush=True)

        selections = {
            "myw_worst": max(paired, key=lambda r: float(r["myw_rmse_db"])),
            "myw_advantage": max(
                paired, key=lambda r: float(r["mlp_rmse_db"]) - float(r["myw_rmse_db"])
            ),
        }
        for label, row in selections.items():
            target_sos = test_sets[family][int(row["example"])]
            plot_example(family, label, row, target_sos,
                         model, num_points, order, args.output)

    with (args.output / "analysis_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved 7 plots and the analysis summary to: {args.output}")


if __name__ == "__main__":
    main()
