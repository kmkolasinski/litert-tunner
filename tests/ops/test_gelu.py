"""Tests for GELU operator."""

from __future__ import annotations

import typing

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def float_gelu_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal GELU op with float32 I/O (no quantization)."""
    input_tensor = op_test_utils.make_tensor(
        name="input_f32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=1, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="GELU",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


@pytest.fixture
def gelu_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal GELU op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[0])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="GELU",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestGeluBuild:
    def test__gelu_is_registered(self) -> None:
        assert "GELU" in registry.registered_ops()

    def test__build_returns_keras_layer(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestGeluCall:
    def test__output_shape_matches_expected(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__gelu_formula_matches_expected(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        """Verify gelu computation produces correct simulated INT8 output."""
        op, tensors = gelu_setup
        input_data = np.array([[0, 2, -2, 4]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Compute expected outputs
        expected = np.array([[-5.0, 3.0, -7.0, 15.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestGeluTrainableWeights:
    def test__trainable_weights(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(
            layer,
            set(),
        )

    def test__non_trainable_weights(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer, {"input_scale", "input_zero_point", "output_scale", "output_zero_point"}
        )


class TestGeluWriteOps:
    def test__is_writable(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(
        self, gelu_setup: tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]
    ) -> None:
        op, tensors = gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )


class TestFloatGeluBuild:
    def test__float_gelu_build_returns_keras_layer(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_gelu_layer_name_contains_output_index(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestFloatGeluCall:
    def test__float_gelu_output_shape(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__float_gelu_formula_matches_numpy(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        input_data = np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Calculate expected using ops.gelu since numpy doesn't have gelu
        expected = typing.cast(
            "typing.Any", keras.ops.convert_to_numpy(keras.ops.gelu(input_data, approximate=False))
        )
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatGeluTrainableWeights:
    def test__float_gelu_trainable_weights(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__float_gelu_non_trainable_weights(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatGeluWriteOps:
    def test__float_gelu_not_writable(self, float_gelu_setup):
        op, tensors = float_gelu_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_not_writable(layer)


def test__gelu_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Activation("gelu")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / "gelu_integration.tflite"
    conftest.export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "GELU")


def test__float32_gelu_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Activation("gelu")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / "float32_gelu_integration.tflite"
    conftest.export_float32_tflite_model(input_shape[1:], model, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "GELU")
