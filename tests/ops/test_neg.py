"""Tests for NEG operator."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.conftest import export_quantized_tflite_model
from tests.ops import op_test_utils


@pytest.fixture
def neg_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal NEG op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[5])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="NEG",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


@pytest.fixture
def neg_setup_no_quant() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal NEG op with float32 I/O (no quantization params)."""
    input_tensor = op_test_utils.make_tensor(
        name="input_float32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )
    output_tensor = op_test_utils.make_tensor(
        name="output_float32", index=1, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="NEG",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestNegBuild:
    def test__neg_is_registered(self) -> None:
        assert "NEG" in registry.registered_ops()

    def test__build_returns_keras_layer(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestNegCall:
    def test__output_shape_matches_expected(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__neg_formula_matches_expected(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify negation computation produces correct simulated INT8 output."""
        op, tensors = neg_setup
        input_data = np.array([[-5, 5, -15, 15]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Compute expected negation outputs
        expected = np.array([[5, -5, 15, -15]], dtype=np.float32)

        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestNegTrainableWeights:
    def test__trainable_weights(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(
            layer,
            {
                "output_scale",
                "output_zero_point",
            },
        )

    def test__non_trainable_weights(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input_scale",
                "input_zero_point",
            },
        )


class TestNegWriteOps:
    def test__is_writable(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(
        self, neg_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = neg_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )


class TestNegNoQuant:
    def test__build_layer_no_quant_vars(
        self, neg_setup_no_quant: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify layer handles None quantization params during build."""
        op, tensors = neg_setup_no_quant
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert getattr(layer, "input_quant", None) is None
        assert getattr(layer, "output_quant", None) is None

    def test__call_no_quant(
        self, neg_setup_no_quant: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify forward pass correctly operates on float tensors when quant params are None."""
        op, tensors = neg_setup_no_quant
        input_data = np.array([[-5.0, 5.0, -15.0, 15.0]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.array([[5.0, -5.0, 15.0, -15.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test__write_ops_empty(
        self, neg_setup_no_quant: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify layer generates no quant write ops when quant params are None."""
        op, tensors = neg_setup_no_quant
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=0,
        )


def test__neg_integration(temp_model_dir, run_interpreter):
    tf.random.set_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = -inputs
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / "neg_integration.tflite"
    export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)
