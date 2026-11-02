"""Evaluate MYW and a trained MLP on identical saved C/D IIR filters.

Run from the IIRNet repository root. Uses SciPy frequency responses for BOTH
models so neither gets a different numerical definition of dB error.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from iirnet.baselines.yw import yulewalk
from iirnet.mlp import MLPModel


EPS = 1e-8


def response_db(sos, frequencies):
    _, h = signal.sosfreqz(sos, worN=frequencies)
    response = 20 * np.log10(np.abs(h) + EPS)
    if not np.isfinite(response).all():
        raise ValueError("Nonfinite frequency response")
    return response


def fit_myw(target_sos, target_magnitude, num_points, order):
    # Match experiments/benchmark_myw.py: include the exact Nyquist endpoint.
    frequencies = np.arange(num_points) / num_points
    _, nyquist_h = signal.sosfreqz(target_sos, worN=[np.pi])
    design_frequencies = np.append(frequencies, 1.0)
    design_magnitude = np.append(target_magnitude, np.abs(nyquist_h[0]))

    start = time.perf_counter()
    b, a = yulewalk(order, design_frequencies, design_magnitude, npt=num_points)
    fitted_sos = signal.tf2sos(b.reshape(-1), a.reshape(-1))
    elapsed_ms = 1000 * (time.perf_counter() - start)

    if len(a.reshape(-1)) - 1 != order:
        raise ValueError("MYW returned the wrong filter order")
    if not np.isfinite(fitted_sos).all():
        raise ValueError("MYW returned nonfinite coefficients")

    _, poles, _ = signal.sos2zpk(fitted_sos)
    max_radius = float(np.max(np.abs(poles)))
    return fitted_sos, elapsed_ms, max_radius


def mse_statistics(rows, field):
    mse = np.array([float(row[field]) for row in rows if row[field] != ""])
    if len(mse) == 0:
        return None
    individual_rmse = np.sqrt(mse)
    return {
        "n": int(len(mse)),
        "pooled_rmse_db": float(np.sqrt(np.mean(mse))),
        "median_rmse_db": float(np.median(individual_rmse)),
        "p95_rmse_db": float(np.percentile(individual_rmse, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--filters", type=Path, default=Path("results/cd_eval/test_filters.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/myw_mlp_paired")
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N filters per family (smoke test).")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    print("Loading saved test filters and trained MLP...", flush=True)
    model = MLPModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.eval()
    num_points = int(model.hparams.num_points)
    order = int(getattr(model.hparams, "max_train_order", model.hparams.model_order))

    rows = []
    family_summaries = {}
    with np.load(args.filters) as saved:
        for family in ("C", "D"):
            filters = np.asarray(saved[family], dtype=np.float64)
            if args.limit is not None:
                filters = filters[:args.limit]
            if filters.ndim != 3 or filters.shape[1:] != ((order + 1) // 2, 6):
                raise ValueError(f"Unexpected {family} SOS shape: {filters.shape}")

            print(f"Evaluating Family {family}: {len(filters)} saved filters", flush=True)
            family_rows = []
            for index, target_sos in enumerate(filters):
                frequencies, target_h = signal.sosfreqz(target_sos, worN=num_points)
                target_magnitude = np.abs(target_h)
                target_db = 20 * np.log10(target_magnitude + EPS)

                # Exactly the input normalization from IIRFilterDataset.__getitem__.
                normalized_input = np.clip(target_db, -128, 128) / 128
                model_input = torch.tensor(normalized_input, dtype=torch.float32)[None]
                with torch.inference_mode():
                    predicted_sos, _ = model(model_input)
                predicted_sos = predicted_sos.squeeze(0).cpu().numpy()

                mlp_db = response_db(predicted_sos, frequencies)
                mlp_mse = float(np.mean((mlp_db - target_db) ** 2))

                row = {
                    "family": family,
                    "example": index,
                    "mlp_mse_db2": mlp_mse,
                    "mlp_rmse_db": float(np.sqrt(mlp_mse)),
                    "myw_status": "ok",
                    "myw_mse_db2": "",
                    "myw_rmse_db": "",
                    "myw_stable": "",
                    "myw_max_pole_radius": "",
                    "myw_elapsed_ms": "",
                    "myw_error": "",
                }

                try:
                    _, target_poles, _ = signal.sos2zpk(target_sos)
                    if not np.all(np.abs(target_poles) < 1):
                        raise ValueError("Target filter is unstable")
                    myw_sos, elapsed_ms, max_radius = fit_myw(
                        target_sos, target_magnitude, num_points, order
                    )
                    myw_db = response_db(myw_sos, frequencies)
                    myw_mse = float(np.mean((myw_db - target_db) ** 2))
                    row.update({
                        "myw_mse_db2": myw_mse,
                        "myw_rmse_db": float(np.sqrt(myw_mse)),
                        "myw_stable": bool(max_radius < 1),
                        "myw_max_pole_radius": max_radius,
                        "myw_elapsed_ms": elapsed_ms,
                    })
                except Exception as error:
                    row["myw_status"] = "failed"
                    row["myw_error"] = f"{type(error).__name__}: {error}"

                family_rows.append(row)
                rows.append(row)
                if (index + 1) % 100 == 0:
                    print(f"  {family}: {index + 1}/{len(filters)}", flush=True)

            myw_rows = [row for row in family_rows if row["myw_status"] == "ok"]
            stable_rows = [row for row in myw_rows if row["myw_stable"] is True]
            summary = {
                "attempted": len(family_rows),
                "myw_successful": len(myw_rows),
                "myw_stable": len(stable_rows),
                "mlp": mse_statistics(family_rows, "mlp_mse_db2"),
                "myw_all_successful": mse_statistics(myw_rows, "myw_mse_db2"),
                "myw_stable_only": mse_statistics(stable_rows, "myw_mse_db2"),
            }
            family_summaries[family] = summary
            print(f"Family {family} summary:\n{json.dumps(summary, indent=2)}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_filter.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "test_filters": str(args.filters),
        "frequency_points": num_points,
        "filter_order": order,
        "families": family_summaries,
    }
    with (args.output / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved paired comparison to: {args.output}")


if __name__ == "__main__":
    main()
