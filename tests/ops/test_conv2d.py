"""Tests for Conv2D operator."""

from collections.abc import Callable

import keras
import numpy as np
import pytest

import litert_tunner
from litert_tunner import testing_utils
from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils

# ---------------------------------------------------------------------------
# Fixtures — CONV_2D
# ---------------------------------------------------------------------------


@pytest.fixture
def conv2d_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal CONV_2D op with INT8 I/O and per-channel quantization."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    # Per-channel weight quantization: 3 output channels
    weight_quant = op_test_utils.make_quant_params(
        scales=[0.2, 0.3, 0.1],
        zero_points=[0, 0, 0],
        quantized_dimension=0,
    )
    # Weight shape: (out_ch=3, kH=1, kW=1, in_ch=2) — 1x1 conv for simplicity
    weight_data = np.array(
        [
            [[[10, 20]]],
            [[[-10, -20]]],
            [[[5, 5]]],
        ],
        dtype=np.int8,
    )
    weight_tensor = op_test_utils.make_tensor(
        name="weight_int8",
        index=1,
        shape=(3, 1, 1, 2),
        dtype=types.DTYPE_INT8,
        quantization=weight_quant,
        data=weight_data,
    )

    # Bias: per-channel, INT32
    bias_data = np.array([1, -1, 0], dtype=np.int32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_int32", index=2, shape=(3,), dtype=types.DTYPE_INT32, data=bias_data
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=3,
        shape=(1, 4, 4, 3),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, weight_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="CONV_2D",
        input_indices=(0, 1, 2),
        output_indices=(3,),
        options={"Padding": 0, "StrideH": 1, "StrideW": 1},
    )
    return op, tensors


# ===================================================================
# CONV_2D build tests
# ===================================================================


class TestConv2DBuild:
    """Tests for the CONV_2D op builder."""

    def test__conv2d_is_registered(self):
        """CONV_2D must be present in the op registry."""
        assert "CONV_2D" in registry.registered_ops()

    def test__build_returns_keras_layer(self, conv2d_setup):
        """The builder must return a Keras layer."""
        op, tensors = conv2d_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, conv2d_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = conv2d_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_weights(self, conv2d_setup):
        """Builder must raise if the weight tensor has no data."""
        op, tensors = conv2d_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_int8",
            index=1,
            shape=(3, 1, 1, 2),
            dtype=types.DTYPE_INT8,
            quantization=tensors[1].quantization,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


# ===================================================================
# CONV_2D call tests
# ===================================================================


class TestConv2DCall:
    """Tests for calling the CONV_2D layer."""

    def test__output_shape_matches_expected(self, conv2d_setup):
        """Output shape must match expected conv2d output shape (same padding)."""
        op, tensors = conv2d_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 4, 4, 3))

    def test__output_values_in_int8_range(self, conv2d_setup):
        """Output values must be in the INT8 range [-128, 127]."""
        op, tensors = conv2d_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-10.0, 10.0, (1, 4, 4, 2)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        assert output.min() >= -128.0
        assert output.max() <= 127.0


# ===================================================================
# CONV_2D trainable weight tests
# ===================================================================


class TestConv2DTrainableWeights:
    """Tests for CONV_2D layer trainable parameters."""

    def test__trainable_weights(self, conv2d_setup):
        """CONV_2D layer must have trainable bias, output_scale, output_zero_point."""
        op, tensors = conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_trainable_weight_names(layer, {"bias", "weight_scale", "weight_int8"})

    def test__non_trainable_weights(self, conv2d_setup):
        """CONV_2D layer must have frozen weights and I/O scales/zps."""
        op, tensors = conv2d_setup
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
# CONV_2D write ops tests
# ===================================================================


class TestConv2DWriteOps:
    """Tests for CONV_2D layer collect_write_ops."""

    def test__is_writable(self, conv2d_setup):
        """CONV_2D layer must implement the Writable protocol."""
        op, tensors = conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, conv2d_setup):
        """CONV_2D must emit 2 buffer writes (weight, bias) and 4 quant writes."""
        op, tensors = conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=4,
        )

    def test__write_ops_buffer_indices(self, conv2d_setup):
        """Buffer writes must target weight and bias tensor indices."""
        op, tensors = conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        buffer_writes, _ = layer.collect_write_ops(op)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )

    def test__write_ops_quant_indices(self, conv2d_setup):
        """Quant writes must target input, weight, bias, and output tensor indices."""
        op, tensors = conv2d_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes,
            {op.input_indices[0], op.input_indices[1], op.input_indices[2], op.output_indices[0]},
        )


# ===================================================================
# Integration tests — Conv2D through make_resnet_tflite
# ===================================================================


def test__conv2d_float32_io(make_resnet_tflite: Callable, run_interpreter: Callable):
    """Verify float32 I/O ResNet-like CNN model matches Interpreter output."""
    model_path = make_resnet_tflite(
        input_shape=(8, 8, 3),
        filters=[8],
        kernel_size=3,
        use_bias=True,
        activation="relu",
        float_io=True,
        add_skip_connections=False,
        add_batchnorm=False,
        pooling_type=None,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (4, 8, 8, 3)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=testing_utils.QUANT_STEP)


def test__conv2d_save_roundtrip(make_resnet_tflite: Callable, run_interpreter: Callable):
    """Verify save roundtrip preserves Conv2D model outputs."""
    model_path = make_resnet_tflite(
        input_shape=(8, 8, 3),
        filters=[8],
        kernel_size=3,
        use_bias=True,
        activation=None,
        float_io=True,
        add_skip_connections=False,
        add_batchnorm=False,
        pooling_type=None,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (4, 8, 8, 3)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    keras_model = litert_tunner.load_model(str(model_path))
    litert_tunner.save_model(keras_model, str(model_path))

    saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(litert_outputs, saved_outputs, atol=testing_utils.QUANT_STEP)


def test__conv2d_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(8, 8, 3))
    outputs = keras.layers.Conv2D(8, 3)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 8, 8, 3)

    output_path = temp_model_dir / "conv2d_integration.tflite"
    conftest.export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "CONV_2D")


def test__conv2d_weight_int8_trainable_save_roundtrip(
    make_resnet_tflite: Callable, run_interpreter: Callable
):
    """Verify that perturbing trainable weight_int8 saves correctly to tflite.

    Flow: load → make weight_int8 trainable → perturb weights → save →
    reload → compare Keras output vs Interpreter output.
    """
    model_path = make_resnet_tflite(
        input_shape=(8, 8, 3),
        filters=[8],
        kernel_size=3,
        use_bias=True,
        activation=None,
        float_io=True,
        add_skip_connections=False,
        add_batchnorm=False,
        pooling_type=None,
    )

    # Load and make weight_int8 trainable
    keras_model = litert_tunner.load_model(str(model_path))
    litert_tunner.prepare_for_finetuning(
        keras_model, trainable_pattern=r".*(bias|weight_scale|weight_int8)$"
    )

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
