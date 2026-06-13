import keras
import numpy as np

from litert_tunner import distillation


def test__kl_loss():
    """Verifies that kl_loss correctly computes mean KL divergence."""
    teacher = keras.ops.convert_to_tensor([[0.8, 0.2], [0.5, 0.5]], dtype="float32")
    student = keras.ops.convert_to_tensor([[0.8, 0.2], [0.1, 0.9]], dtype="float32")

    # Batch 1: KL = 0.8 * log(0.8/0.8) + 0.2 * log(0.2/0.2) = 0.0
    # Batch 2: KL = 0.5 * log(0.5/0.1) + 0.5 * log(0.5/0.9)
    kl_1 = 0.0
    kl_2 = 0.5 * np.log(0.5 / 0.1) + 0.5 * np.log(0.5 / 0.9)
    expected_loss = (kl_1 + kl_2) / 2.0

    loss = distillation.kl_loss(student, teacher)  # pyright: ignore[reportArgumentType]
    np.testing.assert_allclose(loss, expected_loss, atol=1e-5)


def test__mse_loss():
    """Verifies that mse_loss correctly computes mean squared error."""
    y_true = keras.ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    y_pred = keras.ops.convert_to_tensor([[1.0, 1.0], [0.0, 4.0]], dtype="float32")

    # Batch 1: mse = ((1-1)^2 + (1-2)^2) / 2 = 0.5
    # Batch 2: mse = ((0-3)^2 + (4-4)^2) / 2 = 4.5
    # Wait, ops.mean across all dimensions.
    # Total mean: (0 + 1 + 9 + 0) / 4 = 10 / 4 = 2.5
    expected_loss = 2.5

    loss = distillation.mse_loss(y_pred, y_true)  # pyright: ignore[reportArgumentType]
    np.testing.assert_allclose(loss, expected_loss, atol=1e-5)


def test__cosine_loss():
    """Verifies that cosine_loss correctly computes mean cosine distance."""
    y_true = keras.ops.convert_to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype="float32")
    y_pred = keras.ops.convert_to_tensor([[1.0, 0.0, 0.0], [0.0, -1.0, -1.0]], dtype="float32")

    # Batch 1: sim = 1.0 -> loss = 0.0
    # Batch 2: sim = -1.0 -> loss = 2.0
    # Mean loss: 1.0
    loss = distillation.cosine_loss(y_pred, y_true)  # pyright: ignore[reportArgumentType]
    np.testing.assert_allclose(loss, 1.0, atol=1e-5)
