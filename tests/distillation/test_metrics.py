import keras
import numpy as np

from litert_tunner import distillation


def test__cosine_similarity_metric():
    """Verifies that cosine_similarity_metric correctly computes mean cosine sim."""
    y_true = keras.ops.convert_to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype="float32")
    y_pred = keras.ops.convert_to_tensor([[1.0, 0.0, 0.0], [0.0, -1.0, -1.0]], dtype="float32")

    # Batch 1: [1, 0, 0] & [1, 0, 0] -> cosine sim = 1.0
    # Batch 2: [0, 1, 1] & [0, -1, -1] -> cosine sim = -1.0
    # Mean across batch: 0.0
    sim = distillation.cosine_similarity_metric(y_pred, y_true)  # pyright: ignore[reportArgumentType]
    np.testing.assert_allclose(sim, 0.0, atol=1e-5)


def test__cosine_similarity_metric_identical():
    """Verifies that cosine_similarity_metric returns 1.0 for identical vectors."""
    y_true = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    y_pred = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

    sim = distillation.cosine_similarity_metric(y_pred, y_true)  # pyright: ignore[reportArgumentType]
    np.testing.assert_allclose(sim, 1.0, atol=1e-5)
