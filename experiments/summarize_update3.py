import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/update3")
ARCHITECTURES = ("1x32", "3x32", "5x32", "3x16", "3x64")


def summarize_sweep():
    print("\nArchitecture sweep — validation pooled RMSE (dB)")
    print("model      C        D     combined")
    ranked = []

    for model in ARCHITECTURES:
        path = ROOT / "validation" / model / "summary.csv"
        if not path.exists():
            print(f"{model:6s}  pending")
            continue

        with path.open(newline="") as file:
            rows = {row["family"]: row for row in csv.DictReader(file)}
        if not {"C", "D"} <= rows.keys():
            print(f"{model:6s}  awaiting complete summary")
            continue

        errors = [float(rows[f]["pooled_rmse_db"]) for f in ("C", "D")]
        counts = [int(rows[f]["num_filters"]) for f in ("C", "D")]
        combined = np.sqrt(np.average(np.square(errors), weights=counts))
        ranked.append((combined, model))
        print(f"{model:6s}  {errors[0]:6.3f}   {errors[1]:6.3f}   {combined:7.3f}")

    if len(ranked) == len(ARCHITECTURES):
        score, winner = min(ranked)
        checkpoint = (
            "checkpoints/cd_mlp_3x32.ckpt" if winner == "3x32"
            else f"checkpoints/update3_cd_{winner}.ckpt"
        )
        print(f"\nLowest combined validation RMSE: {winner} ({score:.3f} dB)")
        print(f"Checkpoint: {checkpoint}")


def plot_refinement(csv_path):
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, family in zip(axes, ("C", "D")):
        successful = [r for r in rows if r["family"] == family and r["status"] == "ok"]
        myw = {r["example"]: float(r["rmse_db"]) for r in successful
               if r["method"] == "MYW"}
        if not myw:
            raise ValueError(f"No successful MYW fits for Family {family}")

        for method, label in (("MLP", "MLP start"), ("random", "Random start")):
            by_step = {}
            for r in successful:
                if r["method"] == method and r["example"] in myw:
                    by_step.setdefault(int(r["step"]), {})[r["example"]] = float(r["rmse_db"])

            steps = sorted(by_step)
            if not steps or any(by_step[s].keys() != myw.keys() for s in steps):
                raise ValueError(f"Incomplete paired results for {family}, {method}")
            medians = [np.median(list(by_step[s].values())) for s in steps]
            ax.plot(steps, medians, marker="o", label=label)

        ax.axhline(np.median(list(myw.values())), color="black",
                   linestyle="--", label="Modified Yule–Walker")
        ax.set(title=f"Family {family} (n={len(myw)})", xlabel="Gradient steps",
               yscale="log")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Median per-filter RMSE (dB, log scale)")
    axes[1].legend()
    fig.tight_layout()
    output = csv_path.with_name(csv_path.stem + "_convergence.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "hybrid_3x32_val25.csv")
    args = parser.parse_args()
    summarize_sweep()
    plot_refinement(args.csv)
