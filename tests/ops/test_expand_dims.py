"""Tests for the EXPAND_DIMS op."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


class TestExpandDims:
    """Tests for the EXPAND_DIMS op."""

    def test__expand_dims_is_registered(self):
        """EXPAND_DIMS must be present in the op registry."""
        assert "EXPAND_DIMS" in registry.registered_ops()

    def test__expand_dims_output_shape(self):
        """EXPAND_DIMS must correctly expand the shape."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 4, 4, 3), dtype=types.DTYPE_FLOAT32
        )
        axis_data = np.array(2, dtype=np.int32)
        axis_tensor = op_test_utils.make_tensor(
            name="axis", index=1, shape=(), dtype=types.DTYPE_INT32, data=axis_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 4, 1, 4, 3), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(
            op_type="EXPAND_DIMS", input_indices=(0, 1), output_indices=(2,)
        )
        tensors = (input_tensor, axis_tensor, output_tensor)

        input_data = np.ones((1, 4, 4, 3), dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4, 1, 4, 3))
        np.testing.assert_array_equal(output, np.ones((1, 4, 1, 4, 3), dtype=np.float32))

    def test__expand_dims_not_writable(self):
        """EXPAND_DIMS layer must not implement the Writable protocol."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(1, 8), dtype=types.DTYPE_FLOAT32
        )
        axis_data = np.array(1, dtype=np.int32)
        axis_tensor = op_test_utils.make_tensor(
            name="axis", index=1, shape=(), dtype=types.DTYPE_INT32, data=axis_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(1, 1, 8), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(
            op_type="EXPAND_DIMS", input_indices=(0, 1), output_indices=(2,)
        )
        tensors = (input_tensor, axis_tensor, output_tensor)

        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__expand_dims_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4, 4))
    x = keras.ops.expand_dims(inputs, axis=-1)
    model = keras.Model(inputs=inputs, outputs=x)
    input_shape = (1, 4, 4)

    output_path = temp_model_dir / f"{quantization}_expand_dims_integration.tflite"
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
    op_test_utils.verify_model_contains_operator(output_path, "EXPAND_DIMS")
