"""Tests for ADD operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.ops import op_test_utils


@pytest.fixture
def add_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal ADD op with INT8 I/O."""
    input1_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input1_tensor = op_test_utils.make_tensor(
        name="input1_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input1_quant
    )

    input2_quant = op_test_utils.make_quant_params(scales=[0.2], zero_points=[0])
    input2_tensor = op_test_utils.make_tensor(
        name="input2_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input2_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=2, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input1_tensor, input2_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="ADD",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestAddBuild:
    def test__add_is_registered(self):
        assert "ADD" in registry.registered_ops()

    def test__build_returns_keras_layer(self, add_setup):
        op, tensors = add_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, add_setup):
        op, tensors = add_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestAddCall:
    def test__output_shape_matches_expected(self, add_setup):
        op, tensors = add_setup
        input_data = [
            np.random.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            np.random.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        ]
        layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__add_formula_matches_expected(self, add_setup):
        """Verify add computation produces correct simulated INT8 output."""
        op, tensors = add_setup
        input1_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)
        input2_data = np.array([[10, -10, 0, 20]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        expected = np.array([[15.0, 7.0, 11.0, 20.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestAddTrainableWeights:
    def test__trainable_weights(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(
            layer, {"output_scale", "output_zero_point"}
        )

    def test__non_trainable_weights(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input1_scale",
                "input1_zero_point",
                "input2_scale",
                "input2_zero_point",
            },
        )


class TestAddWriteOps:
    def test__is_writable(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            tensors,
            expected_buffer_writes=0,
            expected_quant_writes=3,
        )

    def test__write_ops_quant_indices(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op, tensors)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.input_indices[1], op.output_indices[0]}
        )
