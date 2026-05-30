"""Tests for fully connected (Dense) operator."""

from typing import Callable

import keras
import numpy as np
import pytest

import litert_tunner
from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.ops import op_test_utils


def test__dense_no_activation(make_dense_tflite: Callable, run_interpreter: Callable):
    """Verify that a single dense layer model matches Interpreter output."""
    model_path = make_dense_tflite(
        num_features=8, num_units=1, use_bias=True, activation=None, float_io=False
    )

    # Load model in litert-tunner
    tunner_model = litert_tunner.load_model(str(model_path))

    # Generate random test inputs
    np.random.seed(123)
    inputs = np.random.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)

    # 1. Run Interpreter
    interpreter_output = run_interpreter(model_path, inputs)

    # 2. Run Tunner Model
    graph_def = tunner_model._graph_def
    input_tensor = graph_def.tensors[graph_def.input_indices[0]]
    input_scale = input_tensor.quantization.scales[0]
    input_zp = input_tensor.quantization.zero_points[0]

    # Quantize inputs to simulated int8 space (represented as float32 in Keras)
    simulated_int8_inputs = np.round(inputs / input_scale) + input_zp
    simulated_int8_inputs = np.clip(simulated_int8_inputs, -128, 127).astype(np.float32)

    tunner_output_sim = tunner_model.predict(simulated_int8_inputs)

    # Dequantized float32 output to INT8 to match Interpreter output
    output_tensor = graph_def.tensors[graph_def.output_indices[0]]
    output_scale = output_tensor.quantization.scales[0]
    output_zp = output_tensor.quantization.zero_points[0]

    tunner_output_int8 = np.round(tunner_output_sim / output_scale) + output_zp
    tunner_output_int8 = np.clip(tunner_output_int8, -128, 127).astype(np.int8)

    # They must match within atol=1
    np.testing.assert_allclose(tunner_output_int8, interpreter_output, atol=1, rtol=0)


def test__dense_multiple_units(make_dense_tflite: Callable, run_interpreter: Callable):
    """Verify multiple output units match Interpreter output."""
    model_path = make_dense_tflite(
        num_features=8, num_units=4, use_bias=True, activation="relu", float_io=False
    )
    tunner_model = litert_tunner.load_model(str(model_path))

    inputs = np.random.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)
    interpreter_output = run_interpreter(model_path, inputs)

    graph_def = tunner_model._graph_def
    input_tensor = graph_def.tensors[graph_def.input_indices[0]]
    input_scale = input_tensor.quantization.scales[0]
    input_zp = input_tensor.quantization.zero_points[0]

    simulated_int8_inputs = np.round(inputs / input_scale) + input_zp
    simulated_int8_inputs = np.clip(simulated_int8_inputs, -128, 127).astype(np.float32)

    tunner_output_sim = tunner_model.predict(simulated_int8_inputs)

    output_tensor = graph_def.tensors[graph_def.output_indices[0]]
    output_scale = output_tensor.quantization.scales[0]
    output_zp = output_tensor.quantization.zero_points[0]

    tunner_output_int8 = np.round(tunner_output_sim / output_scale) + output_zp
    tunner_output_int8 = np.clip(tunner_output_int8, -128, 127).astype(np.int8)

    np.testing.assert_allclose(tunner_output_int8, interpreter_output, atol=1, rtol=0)


def test__dense_float32_io(make_dense_tflite: Callable, run_interpreter: Callable):
    """Verify float32 I/O models match Interpreter output with small tolerance."""
    model_path = make_dense_tflite(
        num_features=8, num_units=2, use_bias=True, activation="relu6", float_io=True
    )
    tunner_model = litert_tunner.load_model(str(model_path))

    inputs = np.random.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)
    interpreter_output = run_interpreter(model_path, inputs)

    # For float32 I/O, QUANTIZE/DEQUANTIZE handle the scale/zp internally
    tunner_output = tunner_model.predict(inputs)

    np.testing.assert_allclose(tunner_output, interpreter_output, atol=0.01, rtol=1e-3)


# ---------------------------------------------------------------------------
# Fixtures — FULLY_CONNECTED
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal FULLY_CONNECTED op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    weight_quant = op_test_utils.make_quant_params(scales=[0.2], zero_points=[0])
    weight_data = np.array([[10, 20, 30, 40], [-10, -20, -30, -40]], dtype=np.int8)
    weight_tensor = op_test_utils.make_tensor(
        name="weight_int8",
        index=1,
        shape=(2, 4),
        dtype=types.DTYPE_INT8,
        quantization=weight_quant,
        data=weight_data,
    )

    bias_data = np.array([1, -1], dtype=np.int32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_int32", index=2, shape=(2,), dtype=types.DTYPE_INT32, data=bias_data
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=3, shape=(1, 2), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, weight_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="FULLY_CONNECTED",
        input_indices=(0, 1, 2),
        output_indices=(3,),
    )
    return op, tensors


# ===================================================================
# FULLY_CONNECTED op tests
# ===================================================================


class TestDenseBuild:
    """Tests for the FULLY_CONNECTED op builder."""

    def test__dense_is_registered(self):
        """FULLY_CONNECTED must be present in the op registry."""
        assert "FULLY_CONNECTED" in registry.registered_ops()

    def test__build_returns_keras_layer(self, dense_setup):
        """The builder must return a Keras layer."""
        op, tensors = dense_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, dense_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = dense_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_weights(self, dense_setup):
        """Builder must raise if the weight tensor has no data."""
        op, tensors = dense_setup
        # Modify weight tensor to have None data
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_int8",
            index=1,
            shape=(2, 4),
            dtype=types.DTYPE_INT8,
            quantization=tensors[1].quantization,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


class TestDenseCall:
    """Tests for calling the FULLY_CONNECTED layer."""

    def test__output_shape_matches_expected(self, dense_setup):
        """Output shape must match expected matmul shape."""
        op, tensors = dense_setup
        input_data = np.random.uniform(-1.0, 1.0, (2, 4)).astype(np.float32)
        layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 2))

    def test__dense_formula_matches_expected(self, dense_setup):
        """Verify dense computation with fake quantization."""
        op, tensors = dense_setup
        input_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Expected based on formulas:
        expected = np.array([[12.5, -12.5]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestDenseTrainableWeights:
    """Tests for FULLY_CONNECTED layer trainable parameters."""

    def test__trainable_weights(self, dense_setup):
        """FULLY_CONNECTED layer must have trainable bias, output_scale, output_zero_point."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_trainable_weight_names(
            layer, {"bias", "output_scale", "output_zero_point"}
        )

    def test__non_trainable_weights(self, dense_setup):
        """FULLY_CONNECTED layer must have frozen weights and I/O scales/zps."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "weight_int8",
                "input_scale",
                "input_zero_point",
                "weight_scale",
                "weight_zero_point",
            },
        )


class TestDenseWriteOps:
    """Tests for FULLY_CONNECTED layer collect_write_ops."""

    def test__is_writable(self, dense_setup):
        """FULLY_CONNECTED layer must implement the Writable protocol."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, dense_setup):
        """FULLY_CONNECTED must emit 2 buffer writes (weight, bias) and 3 quant writes."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            tensors,
            expected_buffer_writes=2,
            expected_quant_writes=3,
        )

    def test__write_ops_buffer_indices(self, dense_setup):
        """Buffer writes must target weight and bias tensor indices."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        buffer_writes, _ = layer.collect_write_ops(op, tensors)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )

    def test__write_ops_quant_indices(self, dense_setup):
        """Quant writes must target input, weight, and output tensor indices."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        _, quant_writes = layer.collect_write_ops(op, tensors)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.input_indices[1], op.output_indices[0]}
        )
