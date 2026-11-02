import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from iirnet.data import IIRFilterDataset
from iirnet.mlp import MLPModel
from iirnet import signal


def evaluate_family(model, method, num_examples, batch_size, num_points, order):
    dataset = IIRFilterDataset(
        method=method,
        num_points=num_points,
        max_order=order,
        num_examples=num_examples,
        precompute=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    mse_batches = []
    target_sos_batches = []

    with torch.inference_mode():
        for batch in loader:
            magnitude_db_normalized = batch[1]
            target_sos = batch[-1]

            predicted_sos, _ = model(magnitude_db_normalized)
            _, predicted_response = signal.sosfreqz(predicted_sos, worN=num_points)
            _, target_response = signal.sosfreqz(target_sos, worN=num_points)

            predicted_db = 20 * torch.log10(signal.mag(predicted_response) + 1e-8)
            target_db = 20 * torch.log10(signal.mag(target_response) + 1e-8)

            mse_per_filter = ((predicted_db - target_db) ** 2).mean(dim=-1)
            mse_batches.append(mse_per_filter.cpu().numpy())
            target_sos_batches.append(target_sos.cpu().numpy())

    mse = np.concatenate(mse_batches)
    rmse_per_filter = np.sqrt(mse)
    metrics = {
        "num_filters": len(mse),
        "pooled_rmse_db": float(np.sqrt(np.mean(mse))),
        "mean_rmse_db": float(np.mean(rmse_per_filter)),
        "median_rmse_db": float(np.median(rmse_per_filter)),
        "p95_rmse_db": float(np.percentile(rmse_per_filter, 95)),
    }
    return metrics, np.concatenate(target_sos_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cd_eval"))
    args = parser.parse_args()

    model = MLPModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.eval()
    num_points = int(model.hparams.num_points)
    order = int(getattr(model.hparams, "max_train_order", model.hparams.model_order))

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation: {num_points} frequency bins, order {order}")

    results = []
    saved_filters = {}

    for offset, (label, method) in enumerate(
        (("C", "uniform_disk"), ("D", "uniform_mag_disk"))
    ):
        torch.manual_seed(args.seed + offset)
        np.random.seed(args.seed + offset)

        metrics, target_sos = evaluate_family(
            model, method, args.num_examples, args.batch_size, num_points, order
        )
        results.append({"family": label, **metrics})
        saved_filters[label] = target_sos

        print(
            f"Family {label}: n={metrics['num_filters']}, "
            f"pooled RMSE={metrics['pooled_rmse_db']:.3f} dB, "
            f"median={metrics['median_rmse_db']:.3f} dB, "
            f"p95={metrics['p95_rmse_db']:.3f} dB"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    filters_path = args.output_dir / "test_filters.npz"
    np.savez_compressed(filters_path, **saved_filters)
    print(f"Saved summary: {summary_path}")
    print(f"Saved fixed test filters for later MYW comparison: {filters_path}")


if __name__ == "__main__":
    main()
