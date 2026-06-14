"""Tests for the STRIDED_SLICE op."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


class TestStridedSlice:
    """Tests for the STRIDED_SLICE op."""

    def test__strided_slice_is_registered(self):
        """STRIDED_SLICE must be present in the op registry."""
        assert "STRIDED_SLICE" in registry.registered_ops()

    def test__scalar_extraction_with_shrink_mask(self):
        """STRIDED_SLICE with ShrinkAxisMask should extract a scalar."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(4,), dtype=types.DTYPE_INT32
        )
        begin_data = np.array([0], dtype=np.int32)
        begin_tensor = op_test_utils.make_tensor(
            name="begin", index=1, shape=(1,), dtype=types.DTYPE_INT32, data=begin_data
        )
        end_data = np.array([1], dtype=np.int32)
        end_tensor = op_test_utils.make_tensor(
            name="end", index=2, shape=(1,), dtype=types.DTYPE_INT32, data=end_data
        )
        strides_data = np.array([1], dtype=np.int32)
        strides_tensor = op_test_utils.make_tensor(
            name="strides", index=3, shape=(1,), dtype=types.DTYPE_INT32, data=strides_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=4, shape=(), dtype=types.DTYPE_INT32
        )

        op = op_test_utils.make_operator(
            op_type="STRIDED_SLICE",
            input_indices=(0, 1, 2, 3),
            output_indices=(4,),
            options={"ShrinkAxisMask": 1},
        )
        tensors = (input_tensor, begin_tensor, end_tensor, strides_tensor, output_tensor)

        input_data = np.array([10, 20, 30, 40], dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        assert output.shape == ()
        assert float(output) == 10.0

    def test__range_slice(self):
        """STRIDED_SLICE should extract a range of values."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(6,), dtype=types.DTYPE_INT32
        )
        begin_data = np.array([1], dtype=np.int32)
        begin_tensor = op_test_utils.make_tensor(
            name="begin", index=1, shape=(1,), dtype=types.DTYPE_INT32, data=begin_data
        )
        end_data = np.array([4], dtype=np.int32)
        end_tensor = op_test_utils.make_tensor(
            name="end", index=2, shape=(1,), dtype=types.DTYPE_INT32, data=end_data
        )
        strides_data = np.array([1], dtype=np.int32)
        strides_tensor = op_test_utils.make_tensor(
            name="strides", index=3, shape=(1,), dtype=types.DTYPE_INT32, data=strides_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=4, shape=(3,), dtype=types.DTYPE_INT32
        )

        op = op_test_utils.make_operator(
            op_type="STRIDED_SLICE",
            input_indices=(0, 1, 2, 3),
            output_indices=(4,),
        )
        tensors = (input_tensor, begin_tensor, end_tensor, strides_tensor, output_tensor)

        input_data = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([20, 30, 40], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)

    def test__strided_slice_not_writable(self):
        """STRIDED_SLICE layer must not implement the Writable protocol."""
        input_tensor = op_test_utils.make_tensor(
            name="input", index=0, shape=(4,), dtype=types.DTYPE_INT32
        )
        begin_data = np.array([0], dtype=np.int32)
        begin_tensor = op_test_utils.make_tensor(
            name="begin", index=1, shape=(1,), dtype=types.DTYPE_INT32, data=begin_data
        )
        end_data = np.array([1], dtype=np.int32)
        end_tensor = op_test_utils.make_tensor(
            name="end", index=2, shape=(1,), dtype=types.DTYPE_INT32, data=end_data
        )
        strides_data = np.array([1], dtype=np.int32)
        strides_tensor = op_test_utils.make_tensor(
            name="strides", index=3, shape=(1,), dtype=types.DTYPE_INT32, data=strides_data
        )
        output_tensor = op_test_utils.make_tensor(
            name="output", index=4, shape=(), dtype=types.DTYPE_INT32
        )

        op = op_test_utils.make_operator(
            op_type="STRIDED_SLICE",
            input_indices=(0, 1, 2, 3),
            output_indices=(4,),
            options={"ShrinkAxisMask": 1},
        )
        tensors = (input_tensor, begin_tensor, end_tensor, strides_tensor, output_tensor)

        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.array([1, 2, 3, 4], dtype=np.float32)
        )
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__strided_slice_integration(
    temp_model_dir, run_interpreter, quantization: str, dtype_policy: str
):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4, 4))
    x = inputs[:, 1:3, :]  # pyright: ignore[reportCallIssue,reportArgumentType]
    model = keras.Model(inputs=inputs, outputs=x)
    input_shape = (1, 4, 4)

    output_path = temp_model_dir / f"{quantization}_{dtype_policy}_strided_slice_integration.tflite"
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
    op_test_utils.verify_model_contains_operator(output_path, "STRIDED_SLICE")
