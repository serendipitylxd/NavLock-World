import unittest

from navlock_world.datasets import NavLockSceneDataset
from navlock_world.models import NavLockTemporalBaseline
from navlock_world.training.losses import masked_sequence_loss
from navlock_world.training import NavLockTensorizer, navlock_tensor_collate


class TestTemporalBaseline(unittest.TestCase):
    def test_forward_and_loss(self):
        dataset = NavLockSceneDataset(data_root="data", split="train", mode="all")
        tensorizer = NavLockTensorizer()
        samples = [tensorizer.tensorize_sample(dataset[0]), tensorizer.tensorize_sample(dataset[1])]
        batch = navlock_tensor_collate(samples)

        model = NavLockTemporalBaseline(input_dim=tensorizer.feature_dim, hidden_dim=16, num_layers=1)
        outputs = model(batch["features"])
        self.assertEqual(outputs["upper_gate_logits"].shape[:2], batch["features"].shape[:2])
        self.assertEqual(outputs["ship_intention_logits"].shape[-1], 3)

        loss, metrics = masked_sequence_loss(outputs, batch)
        self.assertGreater(float(loss), 0.0)
        self.assertIn("water_loss", metrics)


if __name__ == "__main__":
    unittest.main()

