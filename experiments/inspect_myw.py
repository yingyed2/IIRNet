import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from experiments.paper_filters import generate_paper_filter
from iirnet.baselines.yw import yulewalk


def select_examples(rows):
    successful = [row for row in rows if row["status"] == "ok"]
    ranked = sorted(successful, key=lambda row: float(row["rmse_db"]))

    if not ranked:
        raise ValueError("No successful examples found")

    return {
        "median": ranked[int(round(0.50 * (len(ranked) - 1)))],
        "p95": ranked[int(round(0.95 * (len(ranked) - 1)))],
        "worst": ranked[-1],
    }


def plot_example(
    family, label, row, mag, target_sos, points, order, output
):
    # Match the benchmark's MYW inputs exactly.
    frequencies = np.arange(points) / points
    _, endpoint_h = signal.sosfreqz(target_sos, worN=[np.pi])

    design_frequencies = np.append(frequencies, 1.0)
    design_magnitude = np.append(mag, np.abs(endpoint_h[0]))

    b, a = yulewalk(
        order,
        design_frequencies,
        design_magnitude,
        npt=points,
    )
    fitted_sos = signal.tf2sos(b.reshape(-1), a.reshape(-1))

    if not np.isfinite(fitted_sos).all():
        raise ValueError("Nonfinite fitted coefficients")

    # Match the benchmark's response evaluation exactly.
    w, target_h = signal.sosfreqz(target_sos, worN=points)
    _, fitted_h = signal.sosfreqz(fitted_sos, worN=w)

    target_db = 20 * np.log10(np.abs(target_h) + 1e-8)
    fitted_db = 20 * np.log10(np.abs(fitted_h) + 1e-8)

    if not np.isfinite(fitted_db).all():
        raise ValueError("Nonfinite fitted response")

    error_db = fitted_db - target_db
    rmse = float(np.sqrt(np.mean(error_db ** 2)))
    saved_rmse = float(row["rmse_db"])

    if not np.isclose(rmse, saved_rmse, rtol=1e-5, atol=1e-6):
        print(
            f"WARNING: {family} example {row['example']} | "
            f"saved RMSE={saved_rmse:.6f} dB | "
            f"regenerated RMSE={rmse:.6f} dB | "
            f"difference={rmse - saved_rmse:+.6f} dB"
        )

    zeros, poles, _ = signal.sos2zpk(target_sos)
    _, fitted_poles, _ = signal.sos2zpk(fitted_sos)

    max_target_radius = float(np.max(np.abs(poles)))
    max_fitted_radius = float(np.max(np.abs(fitted_poles)))
    stable = bool(max_fitted_radius < 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(w / np.pi, target_db, label="Target")
    axes[0].plot(w / np.pi, fitted_db, label="MYW")
    axes[0].set(
        xlabel="Normalized frequency (1 = Nyquist)",
        ylabel="Magnitude (dB)",
        title=f"RMSE: {rmse:.2f} dB",
    )
    axes[0].legend()

    axes[1].plot(w / np.pi, error_db)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Normalized frequency (1 = Nyquist)",
        ylabel="MYW − target (dB)",
        title="Matching error",
    )

    theta = np.linspace(0, 2 * np.pi, 400)
    axes[2].plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)
    axes[2].scatter(
        zeros.real,
        zeros.imag,
        marker="o",
        facecolors="none",
        edgecolors="C0",
        label="Target zeros",
    )
    axes[2].scatter(
        poles.real,
        poles.imag,
        marker="x",
        color="C0",
        label="Target poles",
    )
    axes[2].scatter(
        fitted_poles.real,
        fitted_poles.imag,
        marker="+",
        color="C1",
        label="MYW poles",
    )
    axes[2].set(
        xlabel="Real",
        ylabel="Imaginary",
        title=f"Max target pole radius: {max_target_radius:.4f}",
    )
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)

    example = int(row["example"])
    fig.suptitle(f"Family {family} — {label} — example {example}")
    fig.tight_layout()
    fig.savefig(output / f"{family}_{label}.png", dpi=160)
    plt.close(fig)

    np.savez(
        output / f"{family}_{label}.npz",
        target_sos=target_sos,
        fitted_sos=fitted_sos,
        normalized_frequency=w / np.pi,
        target_db=target_db,
        fitted_db=fitted_db,
    )

    print(
        f"{family} {label}: example={example} | "
        f"RMSE={rmse:.3f} dB | "
        f"max target pole radius={max_target_radius:.6f} | "
        f"MYW stable={stable}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/myw_1000_stabilized"),
    )
    args = parser.parse_args()

    with (args.results / "summary.json").open() as handle:
        settings = json.load(handle)["settings"]

    with (args.results / "per_filter.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    output = args.results / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)

    for family in ("C", "D"):
        family_rows = [row for row in rows if row["family"] == family]
        selections = select_examples(family_rows)

        selected_indices = {}
        for label, row in selections.items():
            index = int(row["example"])
            selected_indices.setdefault(index, []).append((label, row))

        np.random.seed(settings["family_seeds"][family])

        # Replay generation in order to reproduce each selected target.
        for index in range(max(selected_indices) + 1):
            mag, _, _, _, target_sos = generate_paper_filter(
                family=family,
                num_points=settings["frequency_points"],
                order=settings["order"],
            )

            for label, row in selected_indices.get(index, []):
                plot_example(
                    family=family,
                    label=label,
                    row=row,
                    mag=mag,
                    target_sos=target_sos,
                    points=settings["frequency_points"],
                    order=settings["order"],
                    output=output,
                )

    print(f"\nSaved six diagnostic plots to {output}")


if __name__ == "__main__":
    main()