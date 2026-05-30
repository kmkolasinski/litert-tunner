"""Tests for flatbuffer load/save round-trip correctness."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

import litert_tunner


def test__load_save_identity(
    make_dense_tflite: Callable,
    run_interpreter: Callable,
    tmp_path: Path,
):
    """Verify that loading and saving without training produces a bit-exact identical model."""
    model_path = make_dense_tflite(
        num_features=8,
        num_units=4,
        use_bias=True,
        activation="relu",
        float_io=False,
    )
    saved_path = tmp_path / "roundtrip.tflite"

    # 1. Load model in litert-tunner
    tunner_model = litert_tunner.load_model(str(model_path))

    # 2. Save it back immediately without changes
    litert_tunner.save_model(tunner_model, str(saved_path))

    # 3. Verify outputs of original and saved models are bit-exact identical
    rng = np.random.default_rng(42)
    inputs = rng.uniform(-1.0, 1.0, (5, 8)).astype(np.float32)

    original_out = run_interpreter(model_path, inputs)
    saved_out = run_interpreter(saved_path, inputs)

    np.testing.assert_array_equal(original_out, saved_out)


def test__saved_model_loadable_by_interpreter(make_dense_tflite: Callable, tmp_path: Path):
    """Verify that saving a modified model can be loaded by Interpreter and works."""
    model_path = make_dense_tflite(
        num_features=8,
        num_units=2,
        use_bias=True,
        activation=None,
        float_io=False,
    )
    saved_path = tmp_path / "modified.tflite"

    # Load model
    tunner_model = litert_tunner.load_model(str(model_path))

    # Modify a parameter (e.g., bias) to a new value
    dense_layer = None
    for layer in tunner_model.layers:
        if "quantized_dense" in layer.name:
            dense_layer = layer
            break

    assert dense_layer is not None

    # Update the bias weights in the Keras model
    original_bias = dense_layer.bias.numpy()
    new_bias = original_bias + 10.0
    dense_layer.bias.assign(new_bias)

    # Save modified model
    litert_tunner.save_model(tunner_model, str(saved_path))

    # Try to load and run with interpreter to check if it parses and runs successfully
    interpreter = Interpreter(model_path=str(saved_path))
    interpreter.allocate_tensors()
