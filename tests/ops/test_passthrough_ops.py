"""Tests for passthrough ops: RESHAPE, SHAPE, STRIDED_SLICE, PACK."""

from __future__ import annotations

import keras
import numpy as np

from litert_tunner.graph import types
from litert_tunner.ops import registry, reshape
from tests.conftest import export_quantized_tflite_model
from tests.ops import op_test_utils

# ===================================================================
# RESHAPE tests
# ===================================================================


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


# ===================================================================
# SHAPE tests
# ===================================================================


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


# ===================================================================
# STRIDED_SLICE tests
# ===================================================================


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


# ===================================================================
# PACK tests
# ===================================================================


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


def test__passthrough_ops_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4, 4))
    x = keras.layers.Reshape((16,))(inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    input_shape = (1, 4, 4)

    output_path = temp_model_dir / "passthrough_ops_integration.tflite"
    export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "RESHAPE")
