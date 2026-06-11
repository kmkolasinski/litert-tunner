"""Tests for the PAD op."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


class TestPad:
    """Tests for the PAD op."""

    def test__pad_is_registered(self):
        """PAD must be present in the op registry."""
        assert "PAD" in registry.registered_ops()

    def test__pad_output_shape(self):
        """PAD must produce correctly padded output shape."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 4, 4, 3), dtype=types.DTYPE_FLOAT32
        )
        paddings_data = np.array([[0, 0], [1, 1], [2, 2], [0, 0]], dtype=np.int32)
        paddings_tensor = op_test_utils.make_tensor(
            name="paddings", index=1, shape=(4, 2), dtype=types.DTYPE_INT32, data=paddings_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 6, 8, 3), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(op_type="PAD", input_indices=(0, 1), output_indices=(2,))
        tensors = (input_tensor, paddings_tensor, output_tensor)

        input_data = np.ones((1, 4, 4, 3), dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 6, 8, 3))

    def test__pad_preserves_values_and_pads_with_zeros(self):
        """PAD must pad with zeros and preserve original values."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 2), dtype=types.DTYPE_FLOAT32
        )
        paddings_data = np.array([[0, 0], [1, 2]], dtype=np.int32)
        paddings_tensor = op_test_utils.make_tensor(
            name="paddings", index=1, shape=(2, 2), dtype=types.DTYPE_INT32, data=paddings_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 5), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(op_type="PAD", input_indices=(0, 1), output_indices=(2,))
        tensors = (input_tensor, paddings_tensor, output_tensor)

        input_data = np.array([[3.0, 7.0]], dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([[0.0, 3.0, 7.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)

    def test__pad_raises_without_constant_paddings(self):
        """PAD builder must raise if paddings tensor has no data."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32
        )
        paddings_tensor = op_test_utils.make_tensor(
            name="paddings", index=1, shape=(2, 2), dtype=types.DTYPE_INT32
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 6), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(op_type="PAD", input_indices=(0, 1), output_indices=(2,))
        tensors = (input_tensor, paddings_tensor, output_tensor)

        with pytest.raises(ValueError, match="requires a constant paddings tensor"):
            op_test_utils.build_layer_from_registry(op, tensors)

    def test__pad_layer_name_contains_output_index(self):
        """Layer name must end with output tensor index."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32
        )
        paddings_data = np.array([[0, 0], [1, 1]], dtype=np.int32)
        paddings_tensor = op_test_utils.make_tensor(
            name="paddings", index=1, shape=(2, 2), dtype=types.DTYPE_INT32, data=paddings_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=5, shape=(1, 6), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(op_type="PAD", input_indices=(0, 1), output_indices=(5,))
        tensors = (input_tensor, paddings_tensor, output_tensor)

        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert layer.name.endswith("_5")

    def test__pad_not_writable(self):
        """PAD layer must not implement the Writable protocol."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32
        )
        paddings_data = np.array([[0, 0], [1, 1]], dtype=np.int32)
        paddings_tensor = op_test_utils.make_tensor(
            name="paddings", index=1, shape=(2, 2), dtype=types.DTYPE_INT32, data=paddings_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 6), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(op_type="PAD", input_indices=(0, 1), output_indices=(2,))
        tensors = (input_tensor, paddings_tensor, output_tensor)

        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__pad_integration(temp_model_dir, run_interpreter, quantization: str):
    """PAD: load → predict → save → reload → compare."""
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(2, 3))
    x = keras.layers.ZeroPadding1D(padding=(1, 2))(inputs)  # pyright: ignore[reportArgumentType]
    model = keras.Model(inputs=inputs, outputs=x)
    input_shape = (1, 2, 3)

    output_path = temp_model_dir / f"{quantization}_pad_integration.tflite"
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
    op_test_utils.verify_model_contains_operator(output_path, "PAD")
