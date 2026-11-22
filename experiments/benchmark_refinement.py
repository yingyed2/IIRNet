import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from compare_myw_mlp import fit_myw, response_db
from experiments.refine_cd import refine
from iirnet.mlp import MLPModel


def benchmark_target(model, target_sos, seed):
    points = int(model.hparams.num_points)
    order = int(model.hparams.model_order)
    frequencies, target_h = signal.sosfreqz(target_sos, worN=points)
    target_db = response_db(target_sos, frequencies)
    model_input = torch.tensor(
        np.clip(target_db, -128, 128) / 128, dtype=torch.float32
    )[None, :]

    with torch.no_grad():
        mlp_raw = model.predict_raw(model_input)

    generator = torch.Generator().manual_seed(seed)
    random_raw = torch.randn(mlp_raw.shape, generator=generator)
    random_raw[0, 0] = -np.log(99.0)  # Unit initial gain, as in the smoke test.

    rows = []
    for method, raw in (("MLP", mlp_raw), ("random", random_raw)):
        for step, rmse, radius in refine(
            model, raw, target_db, frequencies, method, verbose=False
        ):
            rows.append((method, step, rmse, radius, "ok"))

    try:
        fitted, _, radius = fit_myw(target_sos, np.abs(target_h), points, order)
        fitted_db = response_db(fitted, frequencies)
        rmse = float(np.sqrt(np.mean((fitted_db - target_db) ** 2)))
        rows.append(("MYW", "", rmse, radius, "ok"))
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as err:
        rows.append(("MYW", "", "", "", f"failed: {err}"))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--checkpoint", default="checkpoints/cd_mlp_3x32.ckpt")
    parser.add_argument("--filters", default="results/update3/validation/3x32/test_filters.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model = MLPModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    model.requires_grad_(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.filters) as saved, args.output.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["family", "example", "method", "step", "rmse_db",
                         "max_pole_radius", "status"])

        for family_index, family in enumerate(("C", "D")):
            filters = saved[family]
            if not 1 <= args.limit <= len(filters):
                parser.error(f"--limit must be between 1 and {len(filters)} for {family}")

            paired = []
            for index in range(args.limit):
                seed = 14 + 1_000_000 * family_index + index
                rows = benchmark_target(model, filters[index].astype(np.float64), seed)
                for row in rows:
                    writer.writerow((family, index, *row))

                # Compare every method on the same targets with a successful MYW fit.
                scores = {(method, step): rmse for method, step, rmse, _, status in rows
                          if status == "ok"}
                if ("MYW", "") in scores:
                    paired.append(scores)

            out.flush()
            print(f"\nFamily {family}: {len(paired)}/{args.limit} paired MYW fits")
            if not paired:
                continue
            print("method        pooled dB   median dB    p95 dB")
            for key in (("MLP", 0), ("MLP", 10), ("MLP", 50),
                        ("random", 50), ("MYW", "")):
                errors = np.array([scores[key] for scores in paired])
                pooled = np.sqrt(np.mean(errors ** 2))
                name = f"{key[0]} {key[1]}".strip()
                print(f"{name:12s}  {pooled:8.3f}  {np.median(errors):10.3f}"
                      f"  {np.percentile(errors, 95):8.3f}")

            beats_mlp = sum(s["MLP", 50] < s["MLP", 0] for s in paired)
            beats_myw = sum(s["MLP", 50] < s["MYW", ""] for s in paired)
            print(f"Hybrid at step 50 beat plain MLP on {beats_mlp}/{len(paired)}")
            print(f"Hybrid at step 50 beat MYW on       {beats_myw}/{len(paired)}")

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
