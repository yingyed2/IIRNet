from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from iirnet.baselines.yw import yulewalk
from experiments.paper_filters import generate_paper_filter


def main():
    output_dir = Path("results/myw_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    order = 16
    num_points = 512

    for family in ("C", "D"):
        mag, _, _, _, target_sos = generate_paper_filter(
            family=family,
            num_points=num_points,
            order=order,
        )

        # The generator samples [0, pi), excluding Nyquist.
        frequencies = np.arange(num_points) / num_points

        # Append the actual Nyquist magnitude for MYW's specification.
        _, endpoint_h = signal.sosfreqz(target_sos, worN=[np.pi])
        design_frequencies = np.append(frequencies, 1.0)
        design_magnitude = np.append(mag, np.abs(endpoint_h[0]))

        start = time.perf_counter()
        b, a = yulewalk(
            order,
            design_frequencies,
            design_magnitude,
            npt=num_points,
        )
        fitted_sos = signal.tf2sos(b.reshape(-1), a.reshape(-1))
        elapsed_ms = (time.perf_counter() - start) * 1000

        actual_order = len(a.reshape(-1)) - 1
        assert actual_order == order, "Unexpected fitted filter order"
        assert np.isfinite(fitted_sos).all(), "Nonfinite coefficients"

        # Compare both responses on exactly the same frequency grid.
        w, target_h = signal.sosfreqz(target_sos, worN=num_points)
        _, fitted_h = signal.sosfreqz(fitted_sos, worN=w)

        target_db = 20 * np.log10(np.abs(target_h) + 1e-8)
        fitted_db = 20 * np.log10(np.abs(fitted_h) + 1e-8)

        assert np.isfinite(fitted_db).all(), "Nonfinite fitted response"

        mse = np.mean((fitted_db - target_db) ** 2)
        rmse = np.sqrt(mse)

        _, target_poles, _ = signal.sos2zpk(target_sos)
        _, fitted_poles, _ = signal.sos2zpk(fitted_sos)

        assert np.all(np.abs(target_poles) < 1), "Unstable target filter"

        max_radius = np.max(np.abs(fitted_poles))
        stable = bool(max_radius < 1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(w / np.pi, target_db, label="Target")
        axes[0].plot(w / np.pi, fitted_db, label="MYW")
        axes[0].set(
            xlabel="Normalized frequency (1 = Nyquist)",
            ylabel="Magnitude (dB)",
            title=f"Family {family} | RMSE: {rmse:.2f} dB",
        )
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        theta = np.linspace(0, 2 * np.pi, 400)
        axes[1].plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)

        axes[1].scatter(
            target_poles.real,
            target_poles.imag,
            marker="o",
            facecolors="none",
            edgecolors="C0",
            label="Target poles",
        )
        axes[1].scatter(
            fitted_poles.real,
            fitted_poles.imag,
            marker="x",
            color="C1",
            label="MYW poles",
        )
        axes[1].set(
            xlabel="Real",
            ylabel="Imaginary",
            title=f"MYW stable: {stable}",
        )
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"family_{family}.png", dpi=160)
        plt.close(fig)

        print(
            f"Family {family}: order={actual_order} | "
            f"MSE={mse:.4f} dB² | RMSE={rmse:.4f} dB | "
            f"time={elapsed_ms:.2f} ms | "
            f"max pole radius={max_radius:.6f} | stable={stable}"
        )


if __name__ == "__main__":
    main()