import unittest
import numpy as np
import torch

from iirnet.data import IIRFilterDataset
from iirnet.mlp import MLPModel


class TestCDPipeline(unittest.TestCase):

    def test_generated_roots_are_bounded(self):
        for method in ("uniform_disk", "uniform_mag_disk"):
            for order in (9, 10):
                dataset = IIRFilterDataset(
                    method=method,
                    num_points=64,
                    max_order=order,
                    num_examples=16,
                    precompute=False,
                )

                for _ in range(16):
                    sample = dataset[0]
                    sos = sample[-1].numpy()

                    self.assertEqual(
                        sos.shape,
                        ((order + 1) // 2, 6)
                    )
                    self.assertTrue(np.isfinite(sos).all())

                    for section in sos:
                        zeros = np.roots(section[:3])
                        poles = np.roots(section[3:])

                        self.assertLessEqual(
                            max(abs(zeros)), 0.900001
                        )
                        self.assertLessEqual(
                            max(abs(poles)), 0.900001
                        )

                # Repeated indexing should generate new samples.
                self.assertFalse(
                    torch.equal(dataset[0][-1], dataset[0][-1])
                )

    def test_small_mlp_forward(self):
        model = MLPModel(
            num_points=64,
            num_layers=3,
            hidden_dim=32,
            model_order=10,
            max_root_radius=0.9,
            max_epochs=1,
        )

        x = torch.randn(4, 64)
        sos, (zeros, poles, gains) = model(x)

        self.assertEqual(sos.shape, (4, 5, 6))
        self.assertTrue(torch.isfinite(sos).all())
        self.assertTrue((zeros.abs() <= 0.900001).all())
        self.assertTrue((poles.abs() <= 0.900001).all())

        sos.sum().backward()
        self.assertIsNotNone(model.layers[0][0].weight.grad)


if __name__ == "__main__":
    unittest.main()