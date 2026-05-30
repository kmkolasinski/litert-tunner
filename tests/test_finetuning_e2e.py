"""Fine-tuning end-to-end smoke test."""

from pathlib import Path
from typing import Callable

import keras
import numpy as np

import litert_tunner


def test__finetuning_smoke_test(
    make_dense_tflite: Callable,
    run_interpreter: Callable,
    tmp_path: Path,
):
    """Verify that we can load a model, fine-tune it (update biases/scales),
    and reduce the prediction gap.
    """
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

    # Freeze scales/zero_points to make fine-tuning focus on the bias
    for v in tunner_model.trainable_variables:
        if "bias" not in v.name:
            v.trainable = False

    # Generate some training inputs and target outputs
    np.random.seed(42)
    x_train = np.random.uniform(-1.0, 1.0, (32, 8)).astype(np.float32)
    initial_outputs = tunner_model.predict(x_train)
    y_targets = initial_outputs + 0.5

    # 3. Fine-tune the tunner model
    # Use SGD with a learning rate of 0.2 so the bias can quickly adapt to the +0.5 shift
    tunner_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2), loss="mse")

    # Check initial loss
    initial_loss = float(tunner_model.evaluate(x_train, y_targets, verbose=0))  # type: ignore
    assert initial_loss > 0.2, f"Initial loss should be worse (higher), but got: {initial_loss}"

    # Train for 15 epochs to ensure convergence/loss reduction
    tunner_model.fit(x_train, y_targets, epochs=15, batch_size=8, verbose=0)  # type: ignore

    # Check that loss has decreased after training
    final_loss = float(tunner_model.evaluate(x_train, y_targets, verbose=0))  # type: ignore
    assert final_loss < 0.05, f"Final loss should be improved, but got: {final_loss}"
    assert final_loss < initial_loss, (
        f"Final loss ({final_loss}) is not better than initial loss ({initial_loss})"
    )

    # 4. Save and reload model
    litert_tunner.save_model(tunner_model, str(saved_path))

    # 5. Verify the saved model has the updated predictions in the Interpreter
    original_saved_outputs = run_interpreter(model_path, x_train)
    finetuned_saved_outputs = run_interpreter(saved_path, x_train)

    original_mse = float(np.mean((original_saved_outputs - y_targets) ** 2))
    finetuned_mse = float(np.mean((finetuned_saved_outputs - y_targets) ** 2))

    # Confirm that original Interpreter predictions are worse (higher MSE)
    assert original_mse > 0.2, f"Original Interpreter MSE should be worse, but got: {original_mse}"

    # Confirm that finetuned Interpreter predictions are better (lower MSE)
    assert finetuned_mse < 0.05, (
        f"Finetuned Interpreter MSE should be improved, but got: {finetuned_mse}"
    )
    assert finetuned_mse < original_mse, (
        f"Finetuned Interpreter MSE ({finetuned_mse}) is not better than "
        f"original MSE ({original_mse})"
    )
