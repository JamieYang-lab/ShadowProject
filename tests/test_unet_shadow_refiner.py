import unittest

import torch

from losses.mask_losses import bce_dice_loss, dice_loss
from models.unet_shadow_refiner import UNetShadowRefiner


class UNetShadowRefinerTestCase(unittest.TestCase):
    def test_forward_pass_with_fake_input(self):
        model = UNetShadowRefiner(input_channels=18, output_channels=1, base_channels=8)
        x = torch.randn(2, 18, 128, 128)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 1, 128, 128))

    def test_loss_returns_finite_value(self):
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        total, bce, dice = bce_dice_loss(logits, targets)

        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(bce))
        self.assertTrue(torch.isfinite(dice))
        self.assertTrue(torch.isfinite(dice_loss(logits, targets)))

    def test_one_tiny_cpu_training_step_works(self):
        model = UNetShadowRefiner(input_channels=18, output_channels=1, base_channels=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(1, 18, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, _, _ = bce_dice_loss(logits, target)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))

    def test_model_supports_input_channels_18(self):
        model = UNetShadowRefiner(input_channels=18)

        self.assertEqual(model.enc1.block[0].in_channels, 18)


if __name__ == "__main__":
    unittest.main()
