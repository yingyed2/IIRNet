import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy import signal

from experiments.paper_filters import generate_paper_filter
from iirnet.baselines.yw import yulewalk


def evaluate_one(family, order, num_points):
    mag, _, _, _, target_sos = generate_paper_filter(
        family=family,
        num_points=num_points,
        order=order,
    )

    _, target_poles, _ = signal.sos2zpk(target_sos)
    if not np.all(np.abs(target_poles) < 1):
        raise ValueError("Target filter is unstable")

    frequencies = np.arange(num_points) / num_points
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
    if actual_order != order:
        raise ValueError(f"Expected order {order}, got {actual_order}")
    if not np.isfinite(fitted_sos).all():
        raise ValueError("Nonfinite fitted coefficients")

    w, target_h = signal.sosfreqz(target_sos, worN=num_points)
    _, fitted_h = signal.sosfreqz(fitted_sos, worN=w)

    target_db = 20 * np.log10(np.abs(target_h) + 1e-8)
    fitted_db = 20 * np.log10(np.abs(fitted_h) + 1e-8)

    if not np.isfinite(fitted_db).all():
        raise ValueError("Nonfinite fitted response")

    mse = float(np.mean((fitted_db - target_db) ** 2))
    _, fitted_poles, _ = signal.sos2zpk(fitted_sos)
    max_radius = float(np.max(np.abs(fitted_poles)))

    return {
        "mse_db2": mse,
        "rmse_db": float(np.sqrt(mse)),
        "elapsed_ms": elapsed_ms,
        "max_pole_radius": max_radius,
        "stable": bool(max_radius < 1),
    }


def summarize(rows):
    successful = [row for row in rows if row["status"] == "ok"]

    summary = {
        "attempted": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
    }

    if not successful:
        return summary

    mse = np.array([row["mse_db2"] for row in successful])
    rmse = np.array([row["rmse_db"] for row in successful])
    timings = np.array([row["elapsed_ms"] for row in successful])
    stable_count = sum(row["stable"] for row in successful)

    summary.update({
        "stable": stable_count,
        "unstable": len(successful) - stable_count,
        "mean_mse_db2": float(np.mean(mse)),
        "pooled_rmse_db": float(np.sqrt(np.mean(mse))),
        "median_rmse_db": float(np.median(rmse)),
        "p95_rmse_db": float(np.percentile(rmse, 95)),
        "mean_elapsed_ms": float(np.mean(timings)),
        "median_elapsed_ms": float(np.median(timings)),
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/myw_batch"))
    args = parser.parse_args()

    if args.examples < 1:
        parser.error("--examples must be positive")
    if args.order < 2 or args.order % 2:
        parser.error("--order must be a positive even integer")
    if args.points < 4 * args.order:
        parser.error("--points must be at least 4 times --order")
    if args.points & (args.points - 1):
        parser.error("--points must be a power of two")
    if not 0 <= args.seed < 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 2")

    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    summaries = {}
    fields = [
        "family", "example", "seed", "status",
        "mse_db2", "rmse_db", "elapsed_ms",
        "max_pole_radius", "stable", "error",
    ]

    with (args.output / "per_filter.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for family_index, family in enumerate(("C", "D")):
            family_seed = args.seed + family_index
            np.random.seed(family_seed)

            # Warm up the design code without advancing the experiment RNG.
            rng_state = np.random.get_state()
            try:
                evaluate_one(family, args.order, args.points)
            except Exception as error:
                print(f"Family {family} warmup failed: {error}")
            finally:
                np.random.set_state(rng_state)

            family_rows = []

            for example in range(args.examples):
                row = {
                    "family": family,
                    "example": example,
                    "seed": family_seed,
                    "status": "ok",
                    "error": "",
                }

                try:
                    row.update(evaluate_one(family, args.order, args.points))
                except Exception as error:
                    row["status"] = "failed"
                    row["error"] = f"{type(error).__name__}: {error}"

                writer.writerow(row)
                handle.flush()
                family_rows.append(row)
                rows.append(row)

                if (example + 1) % 100 == 0:
                    print(f"Family {family}: {example + 1}/{args.examples}")

            summary = summarize(family_rows)
            summaries[family] = summary

            print(f"\nFamily {family}")
            print(json.dumps(summary, indent=2))

    report = {
        "settings": {
            "examples_per_family": args.examples,
            "order": args.order,
            "frequency_points": args.points,
            "family_seeds": {"C": args.seed, "D": args.seed + 1},
            "magnitude_epsilon": 1e-8,
            "timing_scope": "MYW design plus transfer-function-to-SOS conversion",
        },
        "families": summaries,
    }

    with (args.output / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()