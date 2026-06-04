"""Fine-tuning end-to-end smoke test."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import keras
import numpy as np

import litert_tunner
from litert_tunner.testing_utils import assert_cosine_similarity
from tests.conftest import export_quantized_tflite_model


def test__finetuning_smoke_test(
    run_interpreter: Callable,
    tmp_path: Path,
):
    """Verify that we can load a model, fine-tune it (update biases/scales),
    and reduce the prediction gap using a teacher model.
    """
    # 1. Create a simple single Dense layer keras model
    keras.utils.set_random_seed(42)
    inputs = keras.Input(shape=(8,))
    outputs = keras.layers.Dense(
        units=1,
        use_bias=True,
        activation=None,
        kernel_initializer=cast("Any", keras.initializers.RandomUniform(-0.5, 0.5)),
        bias_initializer=cast("Any", keras.initializers.RandomUniform(-0.1, 0.1)),
    )(inputs)
    teacher_model = keras.Model(inputs=inputs, outputs=outputs)

    # 2. Export it to int8 litert format
    model_path = tmp_path / "original.tflite"
    export_quantized_tflite_model(
        input_shape=(8,),
        model=teacher_model,
        float_io=True,
        output_path=model_path,
    )

    # 3. Load via litert tunner
    tunner_model = litert_tunner.load_model(str(model_path))

    # Simulate that the teacher model was fine-tuned on new data and its bias shifted
    teacher_model.layers[-1].bias.assign_add(np.array([0.5], dtype=np.float32))

    # Generate some training dataset
    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (32, 8)).astype(np.float32)

    # Get teacher outputs
    y_targets = teacher_model.predict(x_train, verbose=0)  # type: ignore[reportArgumentType]

    # Check initial loss between tunner model (quantized) and teacher model
    tunner_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2), loss="mse")
    initial_loss = float(tunner_model.evaluate(x_train, y_targets, verbose=0))  # type: ignore[reportGeneralTypeIssues]

    assert initial_loss > 0.2, f"Initial loss should be worse (higher), but got: {initial_loss}"

    # Train to align tunner model with teacher model
    tunner_model.fit(x_train, y_targets, epochs=15, batch_size=8, verbose=0)  # type: ignore[reportGeneralTypeIssues]

    final_loss = float(tunner_model.evaluate(x_train, y_targets, verbose=0))  # type: ignore[reportGeneralTypeIssues]

    # The loss should decrease significantly
    assert final_loss < 0.05, f"Final loss should be improved, but got: {final_loss}"
    assert final_loss < initial_loss, (
        f"Final loss ({final_loss}) is not better than initial loss ({initial_loss})"
    )

    # 4. Export finetuned model to tflite and load it to see if model is now
    # more aligned with original keras model
    saved_path = tmp_path / "finetuned.tflite"
    litert_tunner.save_model(tunner_model, str(saved_path))

    # Evaluate using the Interpreter
    original_saved_outputs = run_interpreter(model_path, x_train)
    finetuned_saved_outputs = run_interpreter(saved_path, x_train)

    original_mse = float(np.mean((original_saved_outputs - y_targets) ** 2))
    finetuned_mse = float(np.mean((finetuned_saved_outputs - y_targets) ** 2))

    # Confirm that original Interpreter predictions are worse (higher MSE)
    assert original_mse > 0.2, f"Original Interpreter MSE should be worse, but got: {original_mse}"

    # Confirm that finetuned Interpreter predictions are better aligned with the teacher
    assert finetuned_mse < 0.05, (
        f"Finetuned Interpreter MSE should be improved, but got: {finetuned_mse}"
    )
    assert finetuned_mse < original_mse, (
        f"Finetuned Interpreter MSE ({finetuned_mse}) is not better than "
        f"original MSE ({original_mse})"
    )

    # Check cosine similarity between the tunner model and the saved interpreter model
    tunner_outputs = tunner_model.predict(x_train, verbose=0)  # type: ignore[reportArgumentType]
    assert_cosine_similarity(finetuned_saved_outputs, tunner_outputs)
