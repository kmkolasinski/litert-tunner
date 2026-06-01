"""Tests for RELU operator."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.conftest import export_quantized_tflite_model
from tests.ops import op_test_utils


@pytest.fixture
def relu_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal RELU op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.25], zero_points=[-4])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.25], zero_points=[-4])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="RELU",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestReluBuild:
    def test__relu_is_registered(self) -> None:
        assert "RELU" in registry.registered_ops()

    def test__build_returns_keras_layer(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestReluCall:
    def test__output_shape_matches_expected(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-2.0, 2.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__relu_formula_matches_expected(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify relu computation produces correct simulated INT8 output."""
        op, tensors = relu_setup
        # With zero point -4 and scale 0.25:
        # [-6.0, -4.0, -2.0, 0.0] dequantizes to [-0.5, 0.0, 0.5, 1.0]
        # ReLU of [-0.5, 0.0, 0.5, 1.0] is [0.0, 0.0, 0.5, 1.0]
        # [0.0, 0.0, 0.5, 1.0] quantizes back to [-4.0, -4.0, -2.0, 0.0]
        input_data = np.array([[-6.0, -4.0, -2.0, 0.0]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        expected = np.array([[-4.0, -4.0, -2.0, 0.0]], dtype=np.float32)

        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestReluTrainableWeights:
    def test__trainable_weights(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        inputs = np.ones((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(
            layer,
            {
                "output_scale",
                "output_zero_point",
            },
        )

    def test__non_trainable_weights(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        inputs = np.ones((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input_scale",
                "input_zero_point",
            },
        )


class TestReluWriteOps:
    def test__is_writable(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        inputs = np.ones((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        inputs = np.ones((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(
        self, relu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = relu_setup
        inputs = np.ones((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )


def test__relu_integration(temp_model_dir, run_interpreter) -> None:
    """Verify model with RELU operator maps and runs correctly."""
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Activation("relu")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / "relu_integration.tflite"
    export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "RELU")
