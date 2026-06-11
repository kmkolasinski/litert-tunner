"""Tests for the RESHAPE op."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry, reshape
from tests import conftest
from tests.ops import op_test_utils


class TestReshape:
    """Tests for the RESHAPE op."""

    def test__reshape_is_registered(self):
        """RESHAPE must be present in the op registry."""
        assert "RESHAPE" in registry.registered_ops()

    def test__static_reshape_output_shape(self):
        """Static reshape must produce correct output shape."""
        input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[0])
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 8), dtype=types.DTYPE_INT8, quantization=input_quant
        )
        shape_data = np.array([1, 2, 4], dtype=np.int32)
        shape_tensor = op_test_utils.make_tensor(
            name="shape", index=1, shape=(3,), dtype=types.DTYPE_INT32, data=shape_data
        )
        output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[0])
        output_tensor = op_test_utils.make_tensor(
            name="output",
            index=2,
            shape=(1, 2, 4),
            dtype=types.DTYPE_INT8,
            quantization=output_quant,
        )

        op = op_test_utils.make_operator(
            op_type="RESHAPE", input_indices=(0, 1), output_indices=(2,)
        )
        tensors = (input_tensor, shape_tensor, output_tensor)

        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 8)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 2, 4))

    def test__reshape_preserves_values(self):
        """Reshape must preserve tensor values."""
        input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[0])
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 6), dtype=types.DTYPE_INT8, quantization=input_quant
        )
        shape_data = np.array([1, 2, 3], dtype=np.int32)
        shape_tensor = op_test_utils.make_tensor(
            name="shape", index=1, shape=(3,), dtype=types.DTYPE_INT32, data=shape_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 2, 3), dtype=types.DTYPE_INT8
        )

        op = op_test_utils.make_operator(
            op_type="RESHAPE", input_indices=(0, 1), output_indices=(2,)
        )
        tensors = (input_tensor, shape_tensor, output_tensor)

        input_data = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)

    def test__reshape_not_writable(self):
        """RESHAPE layer must not implement the Writable protocol."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 8), dtype=types.DTYPE_INT8
        )
        shape_data = np.array([1, 2, 4], dtype=np.int32)
        shape_tensor = op_test_utils.make_tensor(
            name="shape", index=1, shape=(3,), dtype=types.DTYPE_INT32, data=shape_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 2, 4), dtype=types.DTYPE_INT8
        )

        op = op_test_utils.make_operator(
            op_type="RESHAPE", input_indices=(0, 1), output_indices=(2,)
        )
        tensors = (input_tensor, shape_tensor, output_tensor)

        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_layer_not_writable(layer)

    def test__reshape_dynamic_shape_bug_reproduction(self):
        """Verify dynamic shape inputs work without crashing."""
        layer = reshape.Reshape(target_shape=(1, 1, 32), name="reshape_test")

        data_input = keras.Input(shape=(32,), dtype="float32")
        shape_input = keras.Input(shape=(4,), dtype="float32")

        model = keras.Model(
            inputs=[data_input, shape_input],
            outputs=layer([data_input, shape_input]),
        )

        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 32)).astype(np.float32)
        shape_vec = np.array([2, 1, 1, 32], dtype=np.float32)
        shape_batch = np.tile(shape_vec, (2, 1))
        out = model.predict([data, shape_batch])
        assert out.shape == (2, 1, 1, 32)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__reshape_expand_dims_tile_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4, 4))
    x = keras.layers.Reshape((16,))(inputs)
    x = keras.ops.expand_dims(x, axis=-1)
    x = keras.ops.tile(x, [1, 2])
    model = keras.Model(inputs=inputs, outputs=x)
    input_shape = (1, 4, 4)

    output_path = temp_model_dir / f"{quantization}_reshape_integration.tflite"
    conftest.export_tflite_model(
        input_shape=input_shape[1:],
        model=model,
        quantization=quantization,
        float_io=True,
        output_path=output_path,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "RESHAPE")
    op_test_utils.verify_model_contains_operator(output_path, "EXPAND_DIMS")
    op_test_utils.verify_model_contains_operator(output_path, "TILE")
