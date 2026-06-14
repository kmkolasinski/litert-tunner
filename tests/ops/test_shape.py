"""Tests for the SHAPE op."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


class TestShape:
    """Tests for the SHAPE op."""

    def test__shape_is_registered(self):
        """SHAPE must be present in the op registry."""
        assert "SHAPE" in registry.registered_ops()

    def test__shape_returns_correct_shape(self):
        """SHAPE must return the shape of the input tensor."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 4, 4, 3), dtype=types.DTYPE_INT8
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=1, shape=(4,), dtype=types.DTYPE_INT32
        )

        op = op_test_utils.make_operator(op_type="SHAPE", input_indices=(0,), output_indices=(1,))
        tensors = (input_tensor, output_tensor)

        input_data = np.zeros((2, 4, 4, 3), dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([2, 4, 4, 3], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)

    def test__shape_not_writable(self):
        """SHAPE layer must not implement the Writable protocol."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 8), dtype=types.DTYPE_INT8
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=1, shape=(2,), dtype=types.DTYPE_INT32
        )

        op = op_test_utils.make_operator(op_type="SHAPE", input_indices=(0,), output_indices=(1,))
        tensors = (input_tensor, output_tensor)

        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__shape_integration(temp_model_dir, run_interpreter, quantization: str, dtype_policy: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(None, 4))
    x = keras.layers.Lambda(tf.shape)(inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    input_shape = (1, 1, 4)

    output_path = temp_model_dir / f"{quantization}_{dtype_policy}_shape_integration.tflite"
    conftest.export_tflite_model(
        input_shape=input_shape[1:],
        model=model,
        quantization=quantization,
        float_io=True,
        output_path=output_path,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    original_policy = keras.config.dtype_policy()
    try:
        keras.config.set_dtype_policy(dtype_policy)
        atol = op_test_utils.get_default_atol(dtype_policy)
        op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter, atol=atol)
    finally:
        keras.config.set_dtype_policy(original_policy)
    op_test_utils.verify_model_contains_operator(output_path, "SHAPE")
