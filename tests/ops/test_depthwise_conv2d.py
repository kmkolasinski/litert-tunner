"""Tests for DepthwiseConv2D operator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import keras
import numpy as np
import pytest

import litert_tunner
from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils

# ---------------------------------------------------------------------------
# Fixtures — DEPTHWISE_CONV_2D
# ---------------------------------------------------------------------------


@pytest.fixture
def depthwise_conv2d_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal DEPTHWISE_CONV_2D op with INT8 I/O and per-channel quantization."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    # Per-channel weight quantization: 2 output channels (depth_multiplier=1, in_ch=2)
    weight_quant = op_test_utils.make_quant_params(
        scales=[0.2, 0.3],
        zero_points=[0, 0],
        quantized_dimension=3,
    )
    # Weight shape: (1, kH=3, kW=3, C_in * depth_multiplier = 2)
    rng = np.random.default_rng(42)
    weight_data = rng.integers(-20, 20, (1, 3, 3, 2)).astype(np.int8)
    weight_tensor = op_test_utils.make_tensor(
        name="weight_int8",
        index=1,
        shape=(1, 3, 3, 2),
        dtype=types.DTYPE_INT8,
        quantization=weight_quant,
        data=weight_data,
    )

    # Bias: per-channel, INT32
    bias_data = np.array([1, -1], dtype=np.int32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_int32", index=2, shape=(2,), dtype=types.DTYPE_INT32, data=bias_data
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=3,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, weight_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="DEPTHWISE_CONV_2D",
        input_indices=(0, 1, 2),
        output_indices=(3,),
        options={"Padding": 0, "StrideH": 1, "StrideW": 1, "DepthMultiplier": 1},
    )
    return op, tensors


@pytest.fixture
def float_depthwise_conv2d_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal DEPTHWISE_CONV_2D op with float32 I/O (no quantization)."""
    input_tensor = op_test_utils.make_tensor(
        name="input_f32",
        index=0,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
    )

    rng = np.random.default_rng(42)
    weight_data = rng.uniform(-1.0, 1.0, (1, 3, 3, 2)).astype(np.float32)
    weight_tensor = op_test_utils.make_tensor(
        name="weight_f32",
        index=1,
        shape=(1, 3, 3, 2),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
        data=weight_data,
    )

    bias_data = np.array([0.5, -0.5], dtype=np.float32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_f32", index=2, shape=(2,), dtype=types.DTYPE_FLOAT32, data=bias_data
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32",
        index=3,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
    )

    tensors = (input_tensor, weight_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="DEPTHWISE_CONV_2D",
        input_indices=(0, 1, 2),
        output_indices=(3,),
        options={"Padding": 0, "StrideH": 1, "StrideW": 1, "DepthMultiplier": 1},
    )
    return op, tensors


# ===================================================================
# DEPTHWISE_CONV_2D build tests
# ===================================================================


class TestDepthwiseConv2DBuild:
    """Tests for the DEPTHWISE_CONV_2D op builder."""

    def test__depthwise_conv2d_is_registered(self):
        """DEPTHWISE_CONV_2D must be present in the op registry."""
        assert "DEPTHWISE_CONV_2D" in registry.registered_ops()

    def test__build_returns_keras_layer(self, depthwise_conv2d_setup):
        """The builder must return a Keras layer."""
        op, tensors = depthwise_conv2d_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, depthwise_conv2d_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = depthwise_conv2d_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_weights(self, depthwise_conv2d_setup):
        """Builder must raise if the weight tensor has no data."""
        op, tensors = depthwise_conv2d_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_int8",
            index=1,
            shape=(1, 3, 3, 2),
            dtype=types.DTYPE_INT8,
            quantization=tensors[1].quantization,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


# ===================================================================
# DEPTHWISE_CONV_2D call tests
# ===================================================================


class TestDepthwiseConv2DCall:
    """Tests for calling the DEPTHWISE_CONV_2D layer."""

    def test__output_shape_matches_expected(self, depthwise_conv2d_setup):
        """Output shape must match expected depthwise conv2d output shape (same padding)."""
        op, tensors = depthwise_conv2d_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 4, 4, 2))

    def test__output_values_in_int8_range(self, depthwise_conv2d_setup):
        """Output values must be in the INT8 range [-128, 127]."""
        op, tensors = depthwise_conv2d_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-10.0, 10.0, (1, 4, 4, 2)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        assert output.min() >= -128.0
        assert output.max() <= 127.0


# ===================================================================
# DEPTHWISE_CONV_2D trainable weight tests
# ===================================================================


class TestDepthwiseConv2DTrainableWeights:
    """Tests for DEPTHWISE_CONV_2D layer trainable parameters."""

    def test__trainable_weights(self, depthwise_conv2d_setup):
        """DEPTHWISE_CONV_2D layer must have trainable bias, output_scale, output_zero_point."""
        op, tensors = depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_trainable_weight_names(layer, {"bias", "weight_int8", "weight_scale"})

    def test__non_trainable_weights(self, depthwise_conv2d_setup):
        """DEPTHWISE_CONV_2D layer must have frozen weights and I/O scales/zps."""
        op, tensors = depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
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


# ===================================================================
# DEPTHWISE_CONV_2D write ops tests
# ===================================================================


class TestDepthwiseConv2DWriteOps:
    """Tests for DEPTHWISE_CONV_2D layer collect_write_ops."""

    def test__is_writable(self, depthwise_conv2d_setup):
        """DEPTHWISE_CONV_2D layer must implement the Writable protocol."""
        op, tensors = depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, depthwise_conv2d_setup):
        """DEPTHWISE_CONV_2D must emit 2 buffer writes (weight, bias) and 4 quant writes."""
        op, tensors = depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=4,
        )

    def test__write_ops_buffer_indices(self, depthwise_conv2d_setup):
        """Buffer writes must target weight and bias tensor indices."""
        op, tensors = depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        buffer_writes, _ = layer.collect_write_ops(op)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )

    def test__write_ops_quant_indices(self, depthwise_conv2d_setup):
        """Quant writes must target input, weight, bias, and output tensor indices."""
        op, tensors = depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes,
            {op.input_indices[0], op.input_indices[1], op.input_indices[2], op.output_indices[0]},
        )


# ===================================================================
# FLOAT32 DEPTHWISE_CONV_2D tests
# ===================================================================


class TestFloatDepthwiseConv2DBuild:
    def test__float_depthwise_conv2d_build_returns_keras_layer(self, float_depthwise_conv2d_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_depthwise_conv2d_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_depthwise_conv2d_layer_name_contains_output_index(
        self, float_depthwise_conv2d_setup
    ):
        """Layer name must end with output tensor index."""
        op, tensors = float_depthwise_conv2d_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")

    def test__float_depthwise_conv2d_build_raises_without_weights(
        self, float_depthwise_conv2d_setup
    ):
        """Builder must raise if weight tensor has no data."""
        op, tensors = float_depthwise_conv2d_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_f32",
            index=1,
            shape=(1, 3, 3, 2),
            dtype=types.DTYPE_FLOAT32,
            quantization=None,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


class TestFloatDepthwiseConv2DCall:
    def test__float_depthwise_conv2d_output_shape(self, float_depthwise_conv2d_setup):
        """Output shape must match expected shape."""
        op, tensors = float_depthwise_conv2d_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 4, 4, 2))

    def test__float_depthwise_conv2d_formula_matches_numpy(self, float_depthwise_conv2d_setup):
        """Float32 op output must match keras operations closely."""
        op, tensors = float_depthwise_conv2d_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)

        kernel = tensors[op.input_indices[1]].data
        bias = tensors[op.input_indices[2]].data

        kernel_reshaped = np.reshape(kernel[0], (3, 3, 2, 1))

        expected = keras.ops.depthwise_conv(
            input_data,
            kernel_reshaped,
            strides=1,
            padding="same",
            dilation_rate=1,
        )
        expected = expected + bias
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatDepthwiseConv2DTrainableWeights:
    def test__float_depthwise_conv2d_trainable_weights(self, float_depthwise_conv2d_setup):
        op, tensors = float_depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_trainable_weight_names(layer, {"kernel", "bias"})

    def test__float_depthwise_conv2d_non_trainable_weights(self, float_depthwise_conv2d_setup):
        op, tensors = float_depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatDepthwiseConv2DWriteOps:
    def test__float_depthwise_conv2d_is_writable(self, float_depthwise_conv2d_setup):
        op, tensors = float_depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_layer_is_writable(layer)

    def test__float_depthwise_conv2d_write_ops_counts(self, float_depthwise_conv2d_setup):
        op, tensors = float_depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=0,
        )

    def test__float_depthwise_conv2d_write_ops_buffer_indices(self, float_depthwise_conv2d_setup):
        op, tensors = float_depthwise_conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        buffer_writes, _ = layer.collect_write_ops(op)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )


# ===================================================================
# Integration tests — DepthwiseConv2D through fixture
# ===================================================================


@pytest.fixture
def make_depthwise_conv_tflite(tmp_path) -> Callable:
    """Fixture returning a function to create fully quantized INT8 DepthwiseConv2D models."""

    def _make(
        input_shape: tuple[int, int, int] = (8, 8, 3),
        depth_multiplier: int = 1,
        kernel_size: int = 3,
        use_bias: bool = True,
        activation: str | None = None,
        float_io: bool = True,
        seed: int = 42,
    ):
        keras.utils.set_random_seed(seed)

        inputs = keras.Input(shape=input_shape)
        x = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            padding="same",
            use_bias=use_bias,
            activation=activation,
            depthwise_initializer=cast("Any", keras.initializers.RandomUniform(-0.5, 0.5)),
            bias_initializer=cast("Any", keras.initializers.RandomUniform(-0.1, 0.1)),
        )(inputs)
        # Add a dense layer to finalize
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(
            units=2,
            kernel_initializer=cast("Any", keras.initializers.RandomUniform(-0.5, 0.5)),
            bias_initializer=cast("Any", keras.initializers.RandomUniform(-0.1, 0.1)),
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        output_path = tmp_path / f"dw_conv_{depth_multiplier}_{activation}_{float_io}.tflite"
        conftest.export_quantized_tflite_model(input_shape, model, float_io, output_path)
        return output_path

    return _make


def test__depthwise_conv2d_float32_io(
    make_depthwise_conv_tflite: Callable, run_interpreter: Callable
):
    """Verify float32 I/O DepthwiseConv2D model matches Interpreter output."""
    model_path = make_depthwise_conv_tflite(
        input_shape=(8, 8, 3),
        depth_multiplier=1,
        kernel_size=3,
        use_bias=True,
        activation=None,
        float_io=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (4, 8, 8, 3)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)


def test__depthwise_conv2d_with_relu(
    make_depthwise_conv_tflite: Callable, run_interpreter: Callable
):
    """Verify DepthwiseConv2D with fused ReLU matches Interpreter output."""
    model_path = make_depthwise_conv_tflite(
        input_shape=(8, 8, 3),
        depth_multiplier=1,
        kernel_size=3,
        use_bias=True,
        activation="relu",
        float_io=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (4, 8, 8, 3)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)


def test__depthwise_conv2d_save_roundtrip(
    make_depthwise_conv_tflite: Callable, run_interpreter: Callable
):
    """Verify save roundtrip preserves DepthwiseConv2D model outputs."""
    model_path = make_depthwise_conv_tflite(
        input_shape=(8, 8, 3),
        depth_multiplier=1,
        kernel_size=3,
        use_bias=True,
        activation=None,
        float_io=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (4, 8, 8, 3)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    keras_model = litert_tunner.load_model(str(model_path))
    litert_tunner.save_model(keras_model, str(model_path))

    saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(litert_outputs, saved_outputs, atol=1e-3)


def test__depthwise_conv2d_weight_int8_trainable_save_roundtrip(
    make_depthwise_conv_tflite: Callable, run_interpreter: Callable
):
    """Verify that perturbing trainable weight_int8 saves correctly to tflite.

    Flow: load → make weight_int8 trainable → perturb weights → save →
    reload → compare Keras output vs Interpreter output.
    """
    model_path = make_depthwise_conv_tflite(
        input_shape=(8, 8, 3),
        depth_multiplier=1,
        kernel_size=3,
        use_bias=True,
        activation=None,
        float_io=True,
    )

    keras_model = litert_tunner.load_model(str(model_path))

    # Perturb weight_int8 values slightly
    for v in keras_model.variables:
        if v.path.endswith("weight_int8"):
            current = v.numpy()
            rng = np.random.default_rng(123)
            perturbation = rng.uniform(-2.0, 2.0, current.shape).astype(np.float32)
            v.assign(current + perturbation)

    # Generate test inputs
    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (2, 8, 8, 3)).astype(np.float32)

    # Get Keras output before save
    keras_output_before = keras_model.predict(x_train)

    # Save and reload
    litert_tunner.save_model(keras_model, str(model_path))
    saved_outputs = run_interpreter(model_path, x_train)

    # Outputs must match: Keras forward (with quantize_to_int8_ste snap) ≈ Interpreter
    np.testing.assert_allclose(keras_output_before, saved_outputs, atol=1e-3)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__depthwise_conv2d_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(8, 8, 3))
    outputs = keras.layers.DepthwiseConv2D(3)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 8, 8, 3)

    output_path = temp_model_dir / f"{quantization}_depthwise_conv2d_integration.tflite"
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

    op_test_utils.verify_model_contains_operator(output_path, "DEPTHWISE_CONV_2D")
