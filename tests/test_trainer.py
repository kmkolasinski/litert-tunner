import logging
from typing import Any

import keras
import numpy as np

import litert_tunner
from tests import conftest


def test__trainer_finetuning_e2e(temp_model_dir, caplog):
    """Verifies that the generic Trainer works for fine-tuning a quantized model."""
    # 1. Create a simple base Keras model
    input_shape = (8,)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Dense(8, activation="relu")(inputs)
    outputs = keras.layers.Dense(2, activation=None)(x)
    base_model = keras.Model(inputs=inputs, outputs=outputs)

    # 2. Export it to INT8 TFLite
    output_path = temp_model_dir / "test_trainer_model.tflite"
    conftest.export_quantized_tflite_model(
        input_shape, base_model, float_io=True, output_path=output_path
    )

    # 3. Load the model via litert_tunner
    litert_model = litert_tunner.load_model(str(output_path))

    # 4. Prepare for fine-tuning (freeze weights, unfreeze biases and scales)
    with caplog.at_level(logging.INFO):
        litert_tunner.prepare_for_finetuning(litert_model)

    # Verify that the variables taken for training were logged
    assert len(caplog.records) > 0
    training_log_found = False
    stats_log_found = False
    for record in caplog.records:
        if "Variable taken for training: path=" in record.message:
            training_log_found = True
            # Check for path, dtype, shape in the log message
            assert "path=" in record.message
            assert "dtype=" in record.message
            assert "shape=" in record.message
        elif "Finetuning statistics:" in record.message:
            stats_log_found = True
            assert "Trainable variables:" in record.message
            assert "Trainable parameters:" in record.message
    assert training_log_found, "Logging for trainable variables not found."
    assert stats_log_found, "Finetuning statistics log not found."

    # Verify trainability (only biases and scales should be trainable)
    trainable_count = 0
    for v in litert_model.trainable_weights:
        assert v.path.endswith("/bias") or v.path.endswith("/weight_scale")
        trainable_count += 1
    assert trainable_count > 0, "No trainable variables found after preparation."

    non_trainable_count = 0
    for v in litert_model.non_trainable_weights:
        assert not v.path.endswith("/bias")
        assert not v.path.endswith("/weight_scale")
        non_trainable_count += 1
    assert non_trainable_count > 0, "No frozen weights found."

    # 5. Initialize the Trainer
    def cosine_similarity(y_pred: Any, y_true: Any) -> keras.KerasTensor:
        # Flatten and normalize
        y_p = keras.ops.reshape(y_pred, (keras.ops.shape(y_pred)[0], -1))
        y_t = keras.ops.reshape(y_true, (keras.ops.shape(y_true)[0], -1))
        y_p = keras.ops.normalize(y_p, axis=1)
        y_t = keras.ops.normalize(y_t, axis=1)
        return keras.ops.mean(keras.ops.sum(y_p * y_t, axis=1))  # pyright: ignore[reportReturnType]

    trainer = litert_tunner.Trainer(
        litert_model=litert_model,
        base_model=base_model,
        l2_weight_decay=0.01,
        extra_metrics={"similarity": cosine_similarity},
    )

    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))

    # 6. Generate dummy training data
    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (100, 8)).astype(np.float32)

    # 7. Fit the model
    history = trainer.fit(x_train, epochs=3, batch_size=10, verbose="auto")

    # 8. Verification
    metrics = history.history
    assert "distill_loss" in metrics, "Distillation loss not tracked."
    assert "l2_loss" in metrics, "L2 loss not tracked."
    assert "similarity" in metrics, "Custom extra metric not tracked."

    # Loss should generally decrease or at least be computed
    final_loss = metrics["loss"][-1]
    assert final_loss >= 0.0, "Total loss should be non-negative."

    # Make sure we can predict with the trainer (which just calls student)
    predictions = trainer.predict(x_train[:5], verbose="auto")
    assert predictions.shape == (5, 2)
