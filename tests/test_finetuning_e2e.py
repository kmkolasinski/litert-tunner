"""Fine-tuning end-to-end smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np

import litert_tunner


def test__finetuning_smoke_test(
    make_dense_tflite: Callable,
    run_interpreter: Callable,
    tmp_path: Path,
):
    """Verify that we can load a model, fine-tune it (update biases/scales), and reduce the prediction gap."""
    # 1. Create a quantized dense model with float I/O to make validation simpler
    model_path = make_dense_tflite(
        num_features=8,
        num_units=1,
        use_bias=True,
        activation=None,
        float_io=True,
    )
    saved_path = tmp_path / "finetuned.tflite"

    # 2. Load model
    tunner_model = litert_tunner.load_model(str(model_path))

    # Generate some training inputs and target outputs
    np.random.seed(42)
    x_train = np.random.uniform(-1.0, 1.0, (32, 8)).astype(np.float32)
    initial_outputs = tunner_model.predict(x_train)
    y_targets = initial_outputs + 0.5

    # 3. Fine-tune the tunner model
    tunner_model.compile(optimizer="adam", loss="mse")

    # Check initial loss
    initial_loss = tunner_model.evaluate(x_train, y_targets, verbose=0)

    # Train for 15 epochs to ensure convergence/loss reduction
    tunner_model.fit(x_train, y_targets, epochs=15, batch_size=8, verbose=0)

    # Check that loss has decreased after training
    final_loss = tunner_model.evaluate(x_train, y_targets, verbose=0)
    assert final_loss < initial_loss

    # 4. Save and reload model
    litert_tunner.save_model(tunner_model, str(saved_path))

    # 5. Verify the saved model has the updated predictions in the Interpreter
    ref_inputs = np.random.uniform(-1.0, 1.0, (5, 8)).astype(np.float32)

    original_saved_outputs = run_interpreter(model_path, ref_inputs)
    finetuned_saved_outputs = run_interpreter(saved_path, ref_inputs)

    # The predictions should be different because the model parameters were modified/fine-tuned
    assert not np.allclose(original_saved_outputs, finetuned_saved_outputs, atol=1e-5)
