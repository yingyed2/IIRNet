import numpy as np
import torch
from scipy import signal

from compare_myw_mlp import response_db
from iirnet.loss import LogMagTargetFrequencyLoss
from iirnet.mlp import MLPModel


def refine(model, initial_raw, target_db, frequencies, label, *, verbose=True):
    measurements = []
    raw = initial_raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=0.01)
    loss_fn = LogMagTargetFrequencyLoss()
    target = torch.tensor(target_db, dtype=raw.dtype)[None, :]

    for step in range(51):
        sos, (_, poles, _) = model.decode_filter(raw)
        loss = loss_fn(sos, target)

        if not torch.isfinite(loss):
            raise ValueError(f"{label}: nonfinite loss at step {step}")

        if step in (0, 1, 5, 10, 25, 50):
            predicted_db = response_db(sos[0].detach().numpy(), frequencies)
            rmse = np.sqrt(np.mean((predicted_db - target_db) ** 2))
            radius = poles.detach().abs().max().item()
            if radius >= 1:
                raise ValueError(f"{label}: unstable filter at step {step}")
            measurements.append((step, float(rmse), radius))
            if verbose:
                print(f"{label:6s}  {step:4d}  {rmse:9.4f}  {radius:8.4f}")

        if step < 50:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return measurements


def main():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    model = MLPModel.load_from_checkpoint(
        "checkpoints/cd_mlp_3x32.ckpt", map_location="cpu"
    )
    model.eval()
    model.requires_grad_(False)

    filters_path = "results/update3/validation/3x32/test_filters.npz"
    with np.load(filters_path) as saved:
        for family in ("C", "D"):
            target_sos = saved[family][0].astype(np.float64)
            frequencies, _ = signal.sosfreqz(
                target_sos, worN=int(model.hparams.num_points)
            )
            target_db = response_db(target_sos, frequencies)
            model_input = torch.tensor(
                np.clip(target_db, -128, 128) / 128, dtype=torch.float32
            )[None, :]

            with torch.no_grad():
                plain_sos, _ = model(model_input)
                mlp_raw = model.predict_raw(model_input)
                initial_sos, _ = model.decode_filter(mlp_raw)
                torch.testing.assert_close(initial_sos, plain_sos)

            # Random poles and zeros; initialize overall gain to 1.
            # The decoder maps the first raw gain through 100 * sigmoid(x).
            random_raw = torch.randn_like(mlp_raw)
            random_raw[0, 0] = -np.log(99.0)

            print(f"\nFamily {family}, validation target 0")
            print("start   step    RMSE_dB    max|p|")
            refine(model, mlp_raw, target_db, frequencies, "MLP")
            refine(model, random_raw, target_db, frequencies, "random")


if __name__ == "__main__":
    main()
