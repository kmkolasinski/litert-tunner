"""Tests for the PACK op."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


class TestPack:
    """Tests for the PACK op."""

    def test__pack_is_registered(self):
        """PACK must be present in the op registry."""
        assert "PACK" in registry.registered_ops()

    def test__pack_all_dynamic(self):
        """PACK with all dynamic inputs should stack tensors."""
        input1_tensor = op_test_utils.make_tensor(
            name="input1", index=0, shape=(), dtype=types.DTYPE_FLOAT32
        )
        input2_tensor = op_test_utils.make_tensor(
            name="input2", index=1, shape=(), dtype=types.DTYPE_FLOAT32
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=2, shape=(2,), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(
            op_type="PACK",
            input_indices=(0, 1),
            output_indices=(2,),
            options={"Axis": 0, "ValuesCount": 2},
        )
        tensors = (input1_tensor, input2_tensor, output_tensor)

        input_data = [
            np.array(5.0, dtype=np.float32),
            np.array(10.0, dtype=np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([5.0, 10.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)

    def test__pack_mixed_constant_dynamic(self):
        """PACK with mixed constant and dynamic inputs should merge correctly."""
        # Dynamic input at position 0, constants at positions 1 and 2
        input_tensor = op_test_utils.make_tensor(
            name="dynamic", index=0, shape=(), dtype=types.DTYPE_INT32
        )
        const1_tensor = op_test_utils.make_tensor(
            name="const1",
            index=1,
            shape=(),
            dtype=types.DTYPE_INT32,
            data=np.array(7, dtype=np.int32),
        )
        const2_tensor = op_test_utils.make_tensor(
            name="const2",
            index=2,
            shape=(),
            dtype=types.DTYPE_INT32,
            data=np.array(3, dtype=np.int32),
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=3, shape=(3,), dtype=types.DTYPE_INT32
        )

        op = op_test_utils.make_operator(
            op_type="PACK",
            input_indices=(0, 1, 2),
            output_indices=(3,),
            options={"Axis": 0, "ValuesCount": 3},
        )
        tensors = (input_tensor, const1_tensor, const2_tensor, output_tensor)

        input_data = np.array(42.0, dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([42.0, 7.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)

    def test__pack_not_writable(self):
        """PACK layer must not implement the Writable protocol."""
        input_tensor = op_test_utils.make_tensor(
            name="input1", index=0, shape=(), dtype=types.DTYPE_FLOAT32
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=1, shape=(1,), dtype=types.DTYPE_FLOAT32
        )

        op = op_test_utils.make_operator(
            op_type="PACK",
            input_indices=(0,),
            output_indices=(1,),
            options={"Axis": 0, "ValuesCount": 1},
        )
        tensors = (input_tensor, output_tensor)

        layer, _ = op_test_utils.build_and_call(op, tensors, np.array(1.0, dtype=np.float32))
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__pack_integration(temp_model_dir, run_interpreter, quantization: str, dtype_policy: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(4)(inputs)
    y = keras.layers.Dense(4)(inputs)
    outputs = keras.ops.stack([x, y], axis=1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_{dtype_policy}_pack_integration.tflite"
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
    op_test_utils.verify_model_contains_operator(output_path, "PACK")
