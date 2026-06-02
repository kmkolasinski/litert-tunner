"""Tests for RSQRT operator."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.conftest import export_quantized_tflite_model
from tests.ops import op_test_utils


@pytest.fixture
def rsqrt_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal RSQRT op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.25], zero_points=[0])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 2), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=1, shape=(1, 2), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="RSQRT",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestRsqrtBuild:
    def test__rsqrt_is_registered(self) -> None:
        assert "RSQRT" in registry.registered_ops()

    def test__build_returns_keras_layer(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestRsqrtCall:
    def test__output_shape_matches_expected(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        rng = np.random.default_rng(42)
        # Avoid negative values since we are computing square root
        input_data = rng.uniform(0.1, 1.0, (1, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 2))

    def test__rsqrt_formula_matches_expected(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify rsqrt computation produces correct simulated INT8 output."""
        op, tensors = rsqrt_setup
        input_data = np.array([[4.0, 16.0]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Compute expected outputs
        expected = np.array([[5.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestRsqrtTrainableWeights:
    def test__trainable_weights(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        inputs = np.ones((1, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(
            layer,
            set(),
        )

    def test__non_trainable_weights(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        inputs = np.ones((1, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer, {"input_scale", "input_zero_point", "output_scale", "output_zero_point"}
        )


class TestRsqrtWriteOps:
    def test__is_writable(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        inputs = np.ones((1, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        inputs = np.ones((1, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(
        self, rsqrt_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = rsqrt_setup
        inputs = np.ones((1, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )


def test__rsqrt_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    # Ensure positive inputs for rsqrt
    x = keras.layers.Activation("relu")(inputs) + 0.1
    outputs = keras.ops.rsqrt(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / "rsqrt_integration.tflite"
    export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "RSQRT")
