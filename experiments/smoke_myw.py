from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch

from iirnet.baselines.yw import YuleWalkerFilterDesign
from iirnet.filter import (
    generate_uniform_disk_filter,
    generate_uniform_mag_disk_filter,
)


def main():
    output_dir = Path("results/myw_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    order = 16
    num_points = 512
    designer = YuleWalkerFilterDesign(N=order)

    # Use upstream names for now; their paper C/D mapping needs verification.
    generators = {
        "uniform_disk": generate_uniform_disk_filter,
        "uniform_mag_disk": generate_uniform_mag_disk_filter,
    }

    for name, generator in generators.items():
        mag, _, _, _, target_sos = generator(num_points, order)
        target_db = 20 * np.log10(mag + 1e-8)
        target_tensor = torch.tensor(target_db, dtype=torch.float32)

        start = time.perf_counter()
        fitted_sos = designer(target_tensor).squeeze(0).numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Evaluate both filters on exactly the same frequency grid.
        w, target_h = signal.sosfreqz(target_sos, worN=num_points)
        _, fitted_h = signal.sosfreqz(fitted_sos, worN=w)

        target_db = 20 * np.log10(np.abs(target_h) + 1e-8)
        fitted_db = 20 * np.log10(np.abs(fitted_h) + 1e-8)

        assert np.isfinite(fitted_sos).all(), "Nonfinite coefficients"
        assert np.isfinite(fitted_db).all(), "Nonfinite response"

        mse = np.mean((fitted_db - target_db) ** 2)

        _, target_poles, _ = signal.sos2zpk(target_sos)
        _, fitted_poles, _ = signal.sos2zpk(fitted_sos)
        max_radius = np.max(np.abs(fitted_poles))
        stable = bool(max_radius < 1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(w / np.pi, target_db, label="Target")
        axes[0].plot(w / np.pi, fitted_db, label="MYW")
        axes[0].set(
            xlabel="Normalized frequency (1 = Nyquist)",
            ylabel="Magnitude (dB)",
            title=name,
        )
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        theta = np.linspace(0, 2 * np.pi, 400)
        axes[1].plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)
        axes[1].scatter(
            target_poles.real, target_poles.imag,
            marker="o", facecolors="none", edgecolors="C0",
            label="Target poles",
        )
        axes[1].scatter(
            fitted_poles.real, fitted_poles.imag,
            marker="x", color="C1", label="MYW poles",
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
        fig.savefig(output_dir / f"{name}.png", dpi=160)
        plt.close(fig)

        print(
            f"{name}: MSE={mse:.4f} dB² | "
            f"time={elapsed_ms:.2f} ms | "
            f"max pole radius={max_radius:.6f} | stable={stable}"
        )


if __name__ == "__main__":
    main()