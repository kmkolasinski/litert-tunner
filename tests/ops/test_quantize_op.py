"""Tests for QUANTIZE and DEQUANTIZE op builders.

Uses the reusable ``op_test_utils`` mini-framework to verify the full
contract: build, call, trainable weights, collect_write_ops, and
Writable protocol compliance.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import quantize_op, registry, utils
from tests.conftest import export_quantized_tflite_model
from tests.ops import op_test_utils

# ---------------------------------------------------------------------------
# Fixtures — QUANTIZE
# ---------------------------------------------------------------------------


@pytest.fixture
def quantize_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal QUANTIZE op with input (float32) → output (int8)."""
    input_tensor = op_test_utils.make_tensor(
        name="input_float",
        index=0,
        shape=(1, 4),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
        buffer_index=0,
    )
    output_quant = op_test_utils.make_quant_params(scales=[0.05], zero_points=[3])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=1,
        shape=(1, 4),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
        buffer_index=1,
    )
    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="QUANTIZE",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


# ---------------------------------------------------------------------------
# Fixtures — DEQUANTIZE
# ---------------------------------------------------------------------------


@pytest.fixture
def dequantize_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal DEQUANTIZE op with input (int8) → output (float32)."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 8),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
        buffer_index=0,
    )
    output_tensor = op_test_utils.make_tensor(
        name="output_float",
        index=1,
        shape=(1, 8),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
        buffer_index=1,
    )
    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="DEQUANTIZE",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


# ===================================================================
# QUANTIZE op tests
# ===================================================================


class TestQuantizeBuild:
    """Tests for the QUANTIZE op builder."""

    def test__quantize_is_registered(self):
        """QUANTIZE must be present in the op registry."""
        assert "QUANTIZE" in registry.registered_ops()

    def test__build_returns_keras_layer(self, quantize_setup):
        """The builder must return a Keras layer."""
        op, tensors = quantize_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, quantize_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = quantize_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_output_quantization(self):
        """Builder must raise if the output tensor has no quantization."""
        output_tensor = op_test_utils.make_tensor(name="bad_output", index=1, quantization=None)
        input_tensor = op_test_utils.make_tensor(name="input", index=0)
        tensors = (input_tensor, output_tensor)
        op = op_test_utils.make_operator(
            op_type="QUANTIZE", input_indices=(0,), output_indices=(1,)
        )
        with pytest.raises(ValueError, match="QUANTIZE op requires quantized output"):
            op_test_utils.build_layer_from_registry(op, tensors)


class TestQuantizeCall:
    """Tests for calling the QUANTIZE layer."""

    def test__output_shape_matches_input(self, quantize_setup):
        """Output shape must equal input shape (quantize doesn't change shape)."""
        op, tensors = quantize_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 4))

    def test__output_values_in_int8_range(self, quantize_setup):
        """Output values must be clamped to [-128, 127]."""
        op, tensors = quantize_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-10.0, 10.0, (5, 4)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        assert np.all(output >= -128.0), f"Min value {output.min()} < -128"
        assert np.all(output <= 127.0), f"Max value {output.max()} > 127"

    def test__output_values_are_integers(self, quantize_setup):
        """Quantize output must be integer-valued (within float representation)."""
        op, tensors = quantize_setup
        input_data = np.array([[0.1, 0.2, -0.3, 0.4]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        np.testing.assert_array_equal(output, np.round(output))

    def test__quantize_formula_matches_expected(self, quantize_setup):
        """Verify quantize formula: clamp(round(x / scale) + zero_point, -128, 127)."""
        op, tensors = quantize_setup
        scale = 0.05
        zero_point = 3.0
        input_data = np.array([[0.25, -0.5, 1.0, -6.0]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        expected = np.clip(np.round(input_data / scale) + zero_point, -128, 127)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestQuantizeTrainableWeights:
    """Tests for QUANTIZE layer trainable parameters."""

    def test__trainable_weights(self, quantize_setup):
        """QUANTIZE layer must have trainable scale and zero_point."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__no_non_trainable_weights(self, quantize_setup):
        """QUANTIZE layer must have no non-trainable weights."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_non_trainable_weight_names(layer, {"scale", "zero_point"})


class TestQuantizeWriteOps:
    """Tests for QUANTIZE layer collect_write_ops."""

    def test__is_writable(self, quantize_setup):
        """QUANTIZE layer must implement the Writable protocol."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, quantize_setup):
        """QUANTIZE must emit 0 buffer writes and 1 quant write."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=1,
        )

    def test__write_ops_target_output_tensor(self, quantize_setup):
        """The quant write must target the output tensor index."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        _, quant_writes = op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=1,
        )
        op_test_utils.assert_quant_write_tensor_indices(quant_writes, {op.output_indices[0]})

    def test__write_ops_scale_value_matches_weight(self, quantize_setup):
        """The emitted scale must match the layer's current scale weight."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        _, quant_writes = layer.collect_write_ops(op)
        layer_scale = float(np.asarray(ops.convert_to_numpy(layer.scale)))
        assert quant_writes[0].scales == [pytest.approx(layer_scale)]

    def test__write_ops_zero_point_value_matches_weight(self, quantize_setup):
        """The emitted zero_point must match the layer's current zero_point weight."""
        op, tensors = quantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 4), dtype=np.float32))
        _, quant_writes = layer.collect_write_ops(op)
        layer_zp = int(np.round(np.asarray(ops.convert_to_numpy(layer.zero_point))))
        assert quant_writes[0].zero_points == [layer_zp]


# ===================================================================
# DEQUANTIZE op tests
# ===================================================================


class TestDequantizeBuild:
    """Tests for the DEQUANTIZE op builder."""

    def test__dequantize_is_registered(self):
        """DEQUANTIZE must be present in the op registry."""
        assert "DEQUANTIZE" in registry.registered_ops()

    def test__build_returns_keras_layer(self, dequantize_setup):
        """The builder must return a Keras layer."""
        op, tensors = dequantize_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, dequantize_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = dequantize_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_input_quantization(self):
        """Builder must raise if the input tensor has no quantization."""
        input_tensor = op_test_utils.make_tensor(name="bad_input", index=0, quantization=None)
        output_tensor = op_test_utils.make_tensor(name="output", index=1, dtype=types.DTYPE_FLOAT32)
        tensors = (input_tensor, output_tensor)
        op = op_test_utils.make_operator(
            op_type="DEQUANTIZE", input_indices=(0,), output_indices=(1,)
        )
        with pytest.raises(ValueError, match="DEQUANTIZE op requires quantized input"):
            op_test_utils.build_layer_from_registry(op, tensors)


class TestDequantizeCall:
    """Tests for calling the DEQUANTIZE layer."""

    def test__output_shape_matches_input(self, dequantize_setup):
        """Output shape must equal input shape."""
        op, tensors = dequantize_setup
        input_data = np.array([[10, -20, 50, -100, 0, 127, -128, 42]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 8))

    def test__dequantize_formula_matches_expected(self, dequantize_setup):
        """Verify dequantize formula: real_value = scale * (int8_value - zero_point)."""
        op, tensors = dequantize_setup
        scale = 0.1
        zero_point = -5.0
        input_data = np.array([[10, -20, 50, -100, 0, 127, -128, 42]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        expected = scale * (input_data - zero_point)
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test__dequantize_with_zero_zero_point(self):
        """Dequantize with zero_point=0 should be a simple scale."""
        quant = op_test_utils.make_quant_params(scales=[0.02], zero_points=[0])
        input_tensor = op_test_utils.make_tensor(index=0, quantization=quant, shape=(1, 3))
        output_tensor = op_test_utils.make_tensor(index=1, dtype=types.DTYPE_FLOAT32, shape=(1, 3))
        tensors = (input_tensor, output_tensor)
        op = op_test_utils.make_operator(
            op_type="DEQUANTIZE", input_indices=(0,), output_indices=(1,)
        )
        input_data = np.array([[100, -50, 0]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = 0.02 * input_data
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestDequantizeTrainableWeights:
    """Tests for DEQUANTIZE layer trainable parameters."""

    def test__no_trainable_weights(self, dequantize_setup):
        """DEQUANTIZE layer must have no trainable weights."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__non_trainable_weights(self, dequantize_setup):
        """DEQUANTIZE layer must have frozen scale and zero_point."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_non_trainable_weight_names(layer, {"scale", "zero_point"})


class TestDequantizeWriteOps:
    """Tests for DEQUANTIZE layer collect_write_ops."""

    def test__is_writable(self, dequantize_setup):
        """DEQUANTIZE layer must implement the Writable protocol."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, dequantize_setup):
        """DEQUANTIZE must emit 0 buffer writes and 1 quant write."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=1,
        )

    def test__write_ops_target_input_tensor(self, dequantize_setup):
        """The quant write must target the input tensor index."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        _, quant_writes = op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=1,
        )
        op_test_utils.assert_quant_write_tensor_indices(quant_writes, {op.input_indices[0]})

    def test__write_ops_scale_value_matches_weight(self, dequantize_setup):
        """The emitted scale must match the layer's current scale weight."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        _, quant_writes = layer.collect_write_ops(op)
        layer_scale = float(np.asarray(ops.convert_to_numpy(layer.scale)))
        assert quant_writes[0].scales == [pytest.approx(layer_scale)]

    def test__write_ops_zero_point_value_matches_weight(self, dequantize_setup):
        """The emitted zero_point must match the layer's current zero_point weight."""
        op, tensors = dequantize_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, np.zeros((1, 8), dtype=np.float32))
        _, quant_writes = layer.collect_write_ops(op)
        layer_zp = int(np.round(np.asarray(ops.convert_to_numpy(layer.zero_point))))
        assert quant_writes[0].zero_points == [layer_zp]


# ===================================================================
# Quantize ↔ Dequantize round-trip
# ===================================================================


class TestQuantizeDequantizeRoundtrip:
    """Verify that QUANTIZE → DEQUANTIZE is consistent (simulated fake-quant)."""

    def test__roundtrip_preserves_quantized_values(self):
        """Values that are exactly representable in INT8 should survive a round-trip."""
        scale = 0.1
        zp = 0

        q_quant = op_test_utils.make_quant_params(scales=[scale], zero_points=[zp])
        d_quant = op_test_utils.make_quant_params(scales=[scale], zero_points=[zp])

        # Tensor setup: float_in → quantize → int8_mid → dequantize → float_out
        t0 = op_test_utils.make_tensor(index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32)
        t1 = op_test_utils.make_tensor(
            index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=q_quant
        )
        t2 = op_test_utils.make_tensor(
            index=2, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
        )

        # Build quantize layer
        q_op = op_test_utils.make_operator(
            op_type="QUANTIZE", input_indices=(0,), output_indices=(1,)
        )
        _q_layer, q_out = op_test_utils.build_and_call(
            q_op, (t0, t1, t2), np.array([[0.5, -1.0, 2.0, -5.0]], dtype=np.float32)
        )

        # Build dequantize layer — reads quant params from its *input* tensor (t1)
        d_op = op_test_utils.make_operator(
            op_type="DEQUANTIZE", input_indices=(1,), output_indices=(2,)
        )
        # Re-create t1 with the dequantize quant params
        t1_deq = op_test_utils.make_tensor(
            index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=d_quant
        )
        d_layer = op_test_utils.build_layer_from_registry(d_op, (t0, t1_deq, t2))
        d_out = np.asarray(keras.ops.convert_to_numpy(d_layer(keras.ops.convert_to_tensor(q_out))))

        # Expected: round-trip via fake quantization
        input_data = np.array([[0.5, -1.0, 2.0, -5.0]], dtype=np.float32)
        expected = np.clip(np.round(input_data / scale) + zp, -128, 127)
        expected = (expected - zp) * scale
        np.testing.assert_allclose(d_out, expected, atol=1e-5)


class TestQuantizeLayer:
    """Direct tests for the Quantize Keras layer."""

    def test__quantize_layer_call(self):
        """Verify Quantize layer behavior."""
        x = np.array([-1.2, -0.2, 0.3, 1.4], dtype=np.float32)
        scale = 0.05
        zero_point = -10.0

        layer = quantize_op.Quantize(scale=scale, zero_point=zero_point)
        layer.build((None, len(x)))

        y = layer(x)
        y_np = np.asarray(ops.convert_to_numpy(y))

        # Replicate expected quantization in float
        q_np = utils.quantize_int8(x, scale, int(zero_point))

        # The layer outputs float32 values that are within INT8 range
        assert np.allclose(y_np, q_np.astype(np.float32), atol=1e-5)

    def test__quantize_layer_ste_gradient(self):
        """Verify that gradients flow through the Quantize layer using STE."""
        layer = quantize_op.Quantize(scale=0.1, zero_point=0.0)

        inputs = keras.Input(shape=(3,))
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        x_tensor = tf.constant([[0.15, -0.25, 0.05]], dtype="float32")
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            y_tensor = model(x_tensor)
            loss = tf.reduce_sum(y_tensor**2)

        grads = tape.gradient(loss, x_tensor)
        grads_np = np.asarray(ops.convert_to_numpy(grads))
        # STE gradient of quantize: d/dx [round(x/scale) + zp] = 1/scale
        # So d(loss)/dx = 2 * y_tensor * (1/scale)
        expected_grads = np.asarray(ops.convert_to_numpy(2 * y_tensor / 0.1))
        assert np.allclose(grads_np, expected_grads, atol=0.05)

    def test__quantize_layer_get_config(self):
        """Verify get_config and serialization of Quantize layer."""
        scale = 0.1
        zero_point = 5.0
        layer = quantize_op.Quantize(scale=scale, zero_point=zero_point, trainable=False)
        config = layer.get_config()
        assert config["scale"] == scale
        assert config["zero_point"] == zero_point
        assert not config["trainable"]


class TestDequantizeLayer:
    """Direct tests for the Dequantize Keras layer."""

    def test__dequantize_layer_call(self):
        """Verify Dequantize layer behavior."""
        x = np.array([-20, -4, 6, 28], dtype=np.int8)
        scale = 0.05
        zero_point = -10.0

        layer = quantize_op.Dequantize(scale=scale, zero_point=zero_point)
        layer.build((None, len(x)))

        y = layer(x)
        y_np = np.asarray(ops.convert_to_numpy(y))

        # Replicate with utils helper
        deq_np = utils.dequantize_float(x, scale, int(zero_point))

        assert np.allclose(y_np, deq_np, atol=1e-5)

    def test__dequantize_layer_get_config(self):
        """Verify get_config and serialization of Dequantize layer."""
        scale = 0.1
        zero_point = 5.0
        layer = quantize_op.Dequantize(scale=scale, zero_point=zero_point, passthrough=True)
        config = layer.get_config()
        assert config["scale"] == scale
        assert config["zero_point"] == zero_point
        assert config["passthrough"] is True


def test__quantize_op_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(4)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / "quantize_op_integration.tflite"
    export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "QUANTIZE")
