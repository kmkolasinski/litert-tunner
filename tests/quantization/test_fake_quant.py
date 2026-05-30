"""Tests for quantization numerical helpers and Keras layers."""

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from litert_tunner import quantization


def test__quantize_dequantize_roundtrip():
    """Verify that quantize -> dequantize roundtrip is within quantization noise."""
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 5.0], dtype=np.float32)
    scale = 0.1
    zero_point = 5

    # Numpy helpers
    q = quantization.quantize_int8(x, scale, zero_point)
    assert q.dtype == np.int8
    deq = quantization.dequantize_float(q, scale, zero_point)

    # Error must be <= scale / 2 (under nearest-rounding) for unclipped values
    unclipped_indices = (q > -128) & (q < 127)
    assert np.all(np.abs(deq[unclipped_indices] - x[unclipped_indices]) <= (scale / 2.0 + 1e-6))


def test__fake_quantize_layer():
    """Verify the FakeQuantize layer matches numpy quantization roundtrip."""
    x = np.array([-1.2, -0.2, 0.3, 1.4], dtype=np.float32)
    scale = 0.05
    zero_point = -10.0

    layer = quantization.FakeQuantize(scale=scale, zero_point=zero_point)
    layer.build((None, len(x)))

    y = layer(x)
    y_np = np.asarray(ops.convert_to_numpy(y))

    # Replicate with numpy
    q_np = quantization.quantize_int8(x, scale, int(zero_point))
    deq_np = quantization.dequantize_float(q_np, scale, int(zero_point))

    assert np.allclose(y_np, deq_np, atol=1e-5)


def test__fake_quantize_ste_gradient():
    """Verify that gradients flow through the FakeQuantize layer using STE."""
    inputs = keras.Input(shape=(3,))
    layer = quantization.FakeQuantize(scale=0.1, zero_point=0.0)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    x = tf.constant([[0.15, -0.25, 0.05]], dtype="float32")

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
        loss = tf.reduce_sum(y**2)

    grads = tape.gradient(loss, x)
    grads_np = np.asarray(ops.convert_to_numpy(grads))
    expected_grads = np.asarray(ops.convert_to_numpy(2 * y))
    assert np.allclose(grads_np, expected_grads, atol=0.05)


def test__layer_get_config():
    """Verify get_config for all quantization layers."""
    scale = 0.1
    zero_point = 5.0

    fq_layer = quantization.FakeQuantize(scale=scale, zero_point=zero_point, trainable=False)
    fq_config = fq_layer.get_config()
    assert fq_config["scale"] == scale
    assert fq_config["zero_point"] == zero_point
    assert not fq_config["trainable"]

    q_layer = quantization.Quantize(scale=scale, zero_point=zero_point, trainable=False)
    q_config = q_layer.get_config()
    assert q_config["scale"] == scale
    assert q_config["zero_point"] == zero_point
    assert not q_config["trainable"]

    dq_layer = quantization.Dequantize(scale=scale, zero_point=zero_point)
    dq_config = dq_layer.get_config()
    assert dq_config["scale"] == scale
    assert dq_config["zero_point"] == zero_point
