"""Tests for fully connected (Dense) operator."""

from collections.abc import Callable

import keras
import numpy as np
import pytest

import litert_tunner
from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


def test__dense_no_activation(make_dense_tflite: Callable, run_interpreter: Callable):
    """Verify that a single dense layer model matches Interpreter output."""
    model_path = make_dense_tflite(
        num_features=8, num_units=1, use_bias=True, activation=None, float_io=False
    )

    # Load model in litert-tunner
    tunner_model = litert_tunner.load_model(str(model_path))

    # Generate random test inputs
    rng = np.random.default_rng(123)
    inputs = rng.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)

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

    # QuantizedDense outputs simulated INT8 values (float32 tensor with integer values)
    tunner_output_sim = tunner_model.predict(simulated_int8_inputs)

    # Cast directly to int8 — the output is already in INT8 range
    tunner_output_int8 = np.clip(np.round(tunner_output_sim), -128, 127).astype(np.int8)

    # They must match within atol=1
    np.testing.assert_allclose(tunner_output_int8, interpreter_output, atol=1, rtol=0)


def test__dense_multiple_units(make_dense_tflite: Callable, run_interpreter: Callable):
    """Verify multiple output units match Interpreter output."""
    model_path = make_dense_tflite(
        num_features=8, num_units=4, use_bias=True, activation="relu", float_io=False
    )
    tunner_model = litert_tunner.load_model(str(model_path))

    rng = np.random.default_rng(42)
    inputs = rng.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)
    interpreter_output = run_interpreter(model_path, inputs)

    graph_def = tunner_model._graph_def
    input_tensor = graph_def.tensors[graph_def.input_indices[0]]
    input_scale = input_tensor.quantization.scales[0]
    input_zp = input_tensor.quantization.zero_points[0]

    simulated_int8_inputs = np.round(inputs / input_scale) + input_zp
    simulated_int8_inputs = np.clip(simulated_int8_inputs, -128, 127).astype(np.float32)

    # QuantizedDense outputs simulated INT8 values (float32 tensor with integer values)
    tunner_output_sim = tunner_model.predict(simulated_int8_inputs)

    # Cast directly to int8 — the output is already in INT8 range
    tunner_output_int8 = np.clip(np.round(tunner_output_sim), -128, 127).astype(np.int8)

    np.testing.assert_allclose(tunner_output_int8, interpreter_output, atol=1, rtol=0)


def test__dense_float32_io(make_dense_tflite: Callable, run_interpreter: Callable):
    """Verify float32 I/O models match Interpreter output with small tolerance."""
    model_path = make_dense_tflite(
        num_features=8, num_units=2, use_bias=True, activation="relu6", float_io=True
    )
    tunner_model = litert_tunner.load_model(str(model_path))

    rng = np.random.default_rng(42)
    inputs = rng.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)
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
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 2))

    def test__dense_formula_matches_expected(self, dense_setup):
        """Verify dense computation produces correct simulated INT8 output.

        With input=[-2, 1, 0, 3], input_scale=0.1, input_zp=-5:
          dequantized = 0.1 * ([-2,1,0,3] - (-5)) = [0.3, 0.6, 0.5, 0.8]
        With weight_scale=0.2, weight_zp=0:
          dequantized_w = 0.2 * [[10,20,30,40],[-10,-20,-30,-40]]
        Matmul + bias = [12.42, -12.42]
        With output_scale=0.5, output_zp=10:
          quantized = round(12.42/0.5) + 10 = 35
          quantized = round(-12.42/0.5) + 10 = -15
        """
        op, tensors = dense_setup
        input_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Expected: simulated INT8 values (quantized, not dequantized)
        expected = np.array([[35.0, -15.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestDenseTrainableWeights:
    """Tests for FULLY_CONNECTED layer trainable parameters."""

    def test__trainable_weights(self, dense_setup):
        """FULLY_CONNECTED layer must have trainable bias, output_scale, output_zero_point."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_trainable_weight_names(layer, {"bias", "weight_int8", "weight_scale"})

    def test__non_trainable_weights(self, dense_setup):
        """FULLY_CONNECTED layer must have frozen weights and I/O scales/zps."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input_scale",
                "input_zero_point",
                "output_scale",
                "output_zero_point",
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
        """FULLY_CONNECTED must emit 2 buffer writes (weight, bias) and 4 quant writes."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=4,
        )

    def test__write_ops_buffer_indices(self, dense_setup):
        """Buffer writes must target weight and bias tensor indices."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        buffer_writes, _ = layer.collect_write_ops(op)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )

    def test__write_ops_quant_indices(self, dense_setup):
        """Quant writes must target input, weight, bias, and output tensor indices."""
        op, tensors = dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes,
            {op.input_indices[0], op.input_indices[1], op.input_indices[2], op.output_indices[0]},
        )


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__dense_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(8)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_dense_integration.tflite"
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

    op_test_utils.verify_model_contains_operator(output_path, "FULLY_CONNECTED")


# ===================================================================
# Float32 Dense — Fixtures
# ===================================================================


@pytest.fixture
def float_dense_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal FULLY_CONNECTED op with float32 I/O (no quantization)."""
    rng = np.random.default_rng(42)

    input_tensor = op_test_utils.make_tensor(
        name="input_f32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    kernel_data = rng.uniform(-0.5, 0.5, (2, 4)).astype(np.float32)
    weight_tensor = op_test_utils.make_tensor(
        name="weight_f32",
        index=1,
        shape=(2, 4),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
        data=kernel_data,
    )

    bias_data = np.array([0.1, -0.2], dtype=np.float32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_f32",
        index=2,
        shape=(2,),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
        data=bias_data,
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=3, shape=(1, 2), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input_tensor, weight_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="FULLY_CONNECTED",
        input_indices=(0, 1, 2),
        output_indices=(3,),
    )
    return op, tensors


# ===================================================================
# Float32 Dense tests
# ===================================================================


class TestFloatDenseBuild:
    """Tests for the FULLY_CONNECTED op builder with float32 tensors."""

    def test__float_dense_build_returns_keras_layer(self, float_dense_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_dense_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_dense_layer_name_contains_output_index(self, float_dense_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = float_dense_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__float_dense_build_raises_without_weights(self, float_dense_setup):
        """Builder must raise if the weight tensor has no data."""
        op, tensors = float_dense_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_f32",
            index=1,
            shape=(2, 4),
            dtype=types.DTYPE_FLOAT32,
            quantization=None,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


class TestFloatDenseCall:
    """Tests for calling the float32 FULLY_CONNECTED layer."""

    def test__float_dense_output_shape(self, float_dense_setup):
        """Output shape must match expected matmul shape."""
        op, tensors = float_dense_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 2))

    def test__float_dense_formula_matches_numpy(self, float_dense_setup):
        """Float32 dense must match simple numpy matmul + bias."""
        op, tensors = float_dense_setup
        rng = np.random.default_rng(123)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)

        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Manual computation: output = input @ weight^T + bias
        kernel = tensors[1].data
        bias = tensors[2].data
        expected = input_data @ kernel.T + bias
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatDenseTrainableWeights:
    """Tests for float32 FULLY_CONNECTED layer trainable parameters."""

    def test__float_dense_trainable_weights(self, float_dense_setup):
        """Float32 FullyConnected must have trainable kernel and bias."""
        op, tensors = float_dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_trainable_weight_names(layer, {"kernel", "bias"})

    def test__float_dense_no_non_trainable_weights(self, float_dense_setup):
        """Float32 FullyConnected must have no non-trainable weights."""
        op, tensors = float_dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatDenseWriteOps:
    """Tests for float32 FULLY_CONNECTED layer collect_write_ops."""

    def test__float_dense_is_writable(self, float_dense_setup):
        """Float32 FullyConnected must implement the Writable protocol."""
        op, tensors = float_dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_layer_is_writable(layer)

    def test__float_dense_write_ops_counts(self, float_dense_setup):
        """Float32 FullyConnected must emit 2 buffer writes (kernel, bias) and 0 quant writes."""
        op, tensors = float_dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=0,
        )

    def test__float_dense_write_ops_buffer_indices(self, float_dense_setup):
        """Buffer writes must target kernel and bias tensor indices."""
        op, tensors = float_dense_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        buffer_writes, _ = layer.collect_write_ops(op)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )


# ===================================================================
# Float32 Dense integration tests
# ===================================================================


def test__float32_dense_forward_matches_interpreter(
    make_float32_dense_tflite: Callable, run_interpreter: Callable
):
    """Float32 dense model Keras output must match LiteRT Interpreter output."""
    model_path = make_float32_dense_tflite(
        num_features=8, num_units=4, use_bias=True, activation=None
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (2, 8)).astype(np.float32)

    op_test_utils.verify_model_outputs(model_path, x_train, run_interpreter)


def test__float32_dense_with_relu_matches_interpreter(
    make_float32_dense_tflite: Callable, run_interpreter: Callable
):
    """Float32 dense with fused relu must match Interpreter output."""
    model_path = make_float32_dense_tflite(
        num_features=8, num_units=4, use_bias=True, activation="relu"
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (2, 8)).astype(np.float32)

    op_test_utils.verify_model_outputs(model_path, x_train, run_interpreter)


def test__dense_weight_int8_trainable_save_roundtrip(
    make_dense_tflite: Callable, run_interpreter: Callable
):
    """Verify that perturbing trainable weight_int8 saves correctly to tflite.

    Flow: load → make weight_int8 trainable → perturb weights → save →
    reload → compare Keras output vs Interpreter output.
    """
    model_path = make_dense_tflite(
        num_features=8, num_units=4, use_bias=True, activation=None, float_io=True
    )

    keras_model = litert_tunner.load_model(str(model_path))

    # Perturb weight_int8 values slightly (add small float offsets)
    for v in keras_model.variables:
        if v.path.endswith("weight_int8"):
            current = v.numpy()
            rng = np.random.default_rng(123)
            perturbation = rng.uniform(-2.0, 2.0, current.shape).astype(np.float32)
            v.assign(current + perturbation)

    # Generate test inputs
    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (2, 8)).astype(np.float32)

    # Get Keras output before save
    keras_output_before = keras_model.predict(x_train)

    # Save and reload
    litert_tunner.save_model(keras_model, str(model_path))
    saved_outputs = run_interpreter(model_path, x_train)

    # Outputs must match: Keras forward (with quantize_to_int8_ste snap) ≈ Interpreter
    np.testing.assert_allclose(keras_output_before, saved_outputs, atol=1e-3)
