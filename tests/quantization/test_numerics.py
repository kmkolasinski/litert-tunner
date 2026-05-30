"""Tests for quantization numerical helpers and Keras layers."""

import numpy as np
import tensorflow as tf
from keras import ops

from litert_tunner import quantization


def test__compute_requantize_multiplier():
    """Verify compute_requantize_multiplier logic."""
    input_scale = 0.5
    weight_scale = np.array([0.1, 0.2, 0.3])
    output_scale = 0.05

    multiplier = quantization.compute_requantize_multiplier(input_scale, weight_scale, output_scale)
    expected = (input_scale * weight_scale) / output_scale
    assert np.allclose(multiplier, expected)


def test__quantize_layer():
    """Verify Quantize layer behavior and STE gradients."""
    x = np.array([-1.2, -0.2, 0.3, 1.4], dtype=np.float32)
    scale = 0.05
    zero_point = -10.0

    layer = quantization.Quantize(scale=scale, zero_point=zero_point)
    layer.build((None, len(x)))

    y = layer(x)
    y_np = np.asarray(ops.convert_to_numpy(y))

    # Replicate with numpy
    q_np = quantization.quantize_int8(x, scale, int(zero_point))

    # The layer outputs float32 values that are within INT8 range
    assert np.allclose(y_np, q_np.astype(np.float32), atol=1e-5)

    # Check gradients flow through Quantize layer
    x_tensor = tf.constant([[0.15, -0.25, 0.05]], dtype="float32")
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        y_tensor = layer(x_tensor)
        loss = tf.reduce_sum(y_tensor**2)

    grads = tape.gradient(loss, x_tensor)
    grads_np = np.asarray(ops.convert_to_numpy(grads))
    # STE gradient of quantize: d/dx [round(x/scale) + zp] = 1/scale
    # So d(loss)/dx = 2 * y_tensor * (1/scale)
    expected_grads = np.asarray(ops.convert_to_numpy(2 * y_tensor / scale))
    assert np.allclose(grads_np, expected_grads, atol=0.05)


def test__dequantize_layer():
    """Verify Dequantize layer behavior."""
    x = np.array([-20, -4, 6, 28], dtype=np.int8)
    scale = 0.05
    zero_point = -10.0

    layer = quantization.Dequantize(scale=scale, zero_point=zero_point)
    layer.build((None, len(x)))

    y = layer(x)
    y_np = np.asarray(ops.convert_to_numpy(y))

    # Replicate with numpy
    deq_np = quantization.dequantize_float(x, scale, int(zero_point))

    assert np.allclose(y_np, deq_np, atol=1e-5)
