"""Unit tests for utility functions in litert_tunner.ops.utils."""

from __future__ import annotations

import typing

import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import utils
from tests.ops import op_test_utils


def _to_numpy(x: typing.Any) -> np.ndarray:
    """Helper to convert to numpy and cast to np.ndarray for Pyright."""
    return typing.cast("np.ndarray", ops.convert_to_numpy(x))


def test__expand_dims_if_not_scalar():
    """Verify expand_dims_if_not_scalar behavior on scalars and multi-dimensional tensors."""
    # Scalar case
    scalar = ops.convert_to_tensor(5.0)
    res_scalar = utils.expand_dims_if_not_scalar(scalar, axis=0)
    np.testing.assert_allclose(_to_numpy(res_scalar), 5.0)

    # Array case
    arr = ops.convert_to_tensor([1.0, 2.0])
    res_arr = utils.expand_dims_if_not_scalar(arr, axis=0)
    assert res_arr.shape == (1, 2)
    np.testing.assert_allclose(_to_numpy(res_arr), [[1.0, 2.0]])


def test__quantize_to_int8():
    """Verify quantize_to_int8 correctly rounds and converts to int8."""
    tensor = ops.convert_to_tensor([1.2, 2.8, -0.6, -1.2])
    res = utils.quantize_to_int8(tensor)
    assert res.dtype == np.int8
    np.testing.assert_array_equal(res, [1, 3, -1, -1])


def test__quantize_bias_to_int32():
    """Verify quantize_bias_to_int32 computes bias_scale and rounds to int32."""
    bias = ops.convert_to_tensor([0.15, -0.3])
    input_scale = ops.convert_to_tensor(0.5)
    weight_scale = ops.convert_to_tensor([0.1, 0.2])

    res = utils.quantize_bias_to_int32(bias, input_scale, weight_scale)
    assert res.dtype == np.int32

    # bias_scale = 0.5 * [0.1, 0.2] = [0.05, 0.1]
    # bias / bias_scale = [0.15 / 0.05, -0.3 / 0.1] = [3.0, -3.0]
    np.testing.assert_array_equal(res, [3, -3])


def test__get_quant_param_value():
    """Verify get_quant_param_value converts arrays to scalars if len 1, else returns array."""
    # Length 1 case
    arr1 = np.array([2.5], dtype=np.float32)
    res1 = utils.get_quant_param_value(arr1)
    assert isinstance(res1, float)
    assert res1 == 2.5

    # Length > 1 case
    arr2 = np.array([1.0, 2.0], dtype=np.float32)
    res2 = utils.get_quant_param_value(arr2)
    assert isinstance(res2, np.ndarray)
    assert res2.dtype == np.float32
    np.testing.assert_array_equal(res2, [1.0, 2.0])


def test__get_bias_float32():
    """Verify get_bias_float32 reads from tensors list or defaults to zeros."""
    # Setup dummy operator and tensors
    # Case 1: Bias is present
    op = op_test_utils.make_operator(
        op_type="FULLY_CONNECTED",
        input_indices=(0, 1, 2),
        output_indices=(3,),
    )
    bias_data = np.array([10, -20], dtype=np.int32)
    tensors = (
        op_test_utils.make_tensor(name="in", index=0, shape=(1, 2), dtype=types.DTYPE_INT8),
        op_test_utils.make_tensor(name="w", index=1, shape=(2, 2), dtype=types.DTYPE_INT8),
        op_test_utils.make_tensor(
            name="bias", index=2, shape=(2,), dtype=types.DTYPE_INT32, data=bias_data
        ),
        op_test_utils.make_tensor(name="out", index=3, shape=(1, 2), dtype=types.DTYPE_INT8),
    )

    input_scale = 0.5
    weight_scales = np.array([0.1, 0.2])

    bias_f32 = utils.get_bias_float32(op, tensors, input_scale, weight_scales, output_units=2)
    # Expected: bias_data * input_scale * weight_scales = [10 * 0.05, -20 * 0.1] = [0.5, -2.0]
    np.testing.assert_allclose(bias_f32, [0.5, -2.0], atol=1e-5)

    # Case 2: Bias index is -1 (absent)
    op_nobias = op_test_utils.make_operator(
        op_type="FULLY_CONNECTED",
        input_indices=(0, 1, -1),
        output_indices=(3,),
    )
    bias_nobias = utils.get_bias_float32(
        op_nobias, tensors, input_scale, weight_scales, output_units=2
    )
    np.testing.assert_allclose(bias_nobias, [0.0, 0.0])


def test__apply_fused_activation():
    """Verify apply_fused_activation behaves correctly for NONE, RELU, RELU6, RELU_N1_TO_1."""
    x = ops.convert_to_tensor([-2.0, 0.5, 7.0])

    # NONE
    np.testing.assert_allclose(
        _to_numpy(utils.apply_fused_activation(x, utils.FUSED_ACTIVATION_NONE)), [-2.0, 0.5, 7.0]
    )

    # RELU
    np.testing.assert_allclose(
        _to_numpy(utils.apply_fused_activation(x, utils.FUSED_ACTIVATION_RELU)), [0.0, 0.5, 7.0]
    )

    # RELU6
    np.testing.assert_allclose(
        _to_numpy(utils.apply_fused_activation(x, utils.FUSED_ACTIVATION_RELU6)), [0.0, 0.5, 6.0]
    )

    # RELU_N1_TO_1
    np.testing.assert_allclose(
        _to_numpy(utils.apply_fused_activation(x, utils.FUSED_ACTIVATION_RELU_N1_TO_1)),
        [-1.0, 0.5, 1.0],
    )


def test__quantize_int8():
    """Verify numpy-based quantize_int8 function."""
    x = np.array([-1.5, 0.0, 1.5, 20.0], dtype=np.float32)
    scale = 0.1
    zero_point = 5
    res = utils.quantize_int8(x, scale, zero_point)
    assert res.dtype == np.int8
    expected = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
    np.testing.assert_array_equal(res, expected)


def test__dequantize_float():
    """Verify numpy-based dequantize_float function."""
    x = np.array([-15, 0, 15, 127], dtype=np.int8)
    scale = 0.1
    zero_point = 5
    res = utils.dequantize_float(x, scale, zero_point)
    assert res.dtype == np.float32
    expected = scale * (x.astype(np.float32) - np.float32(zero_point))
    np.testing.assert_allclose(res, expected)


def test__compute_requantize_multiplier():
    """Verify compute_requantize_multiplier logic."""
    input_scale = 0.5
    weight_scale = np.array([0.1, 0.2, 0.3])
    output_scale = 0.05
    res = utils.compute_requantize_multiplier(input_scale, weight_scale, output_scale)
    expected = (input_scale * weight_scale) / output_scale
    np.testing.assert_allclose(res, expected)


def test__round_ste():
    """Verify _round_ste returns rounded values forward and passes gradients backward."""
    x = tf.constant([1.2, 2.7, -0.6], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = utils._round_ste(x)

    np.testing.assert_allclose(_to_numpy(y), [1.0, 3.0, -1.0])
    grads = tape.gradient(y, x)
    # STE: gradient is identity (all ones)
    np.testing.assert_allclose(_to_numpy(grads), [1.0, 1.0, 1.0])


def test__clip_ste():
    """Verify _clip_ste clips forward and passes gradients backward."""
    x = tf.constant([-130.0, 0.0, 150.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = utils._clip_ste(x, -128.0, 127.0)

    np.testing.assert_allclose(_to_numpy(y), [-128.0, 0.0, 127.0])
    grads = tape.gradient(y, x)
    # STE: gradient is identity (all ones)
    np.testing.assert_allclose(_to_numpy(grads), [1.0, 1.0, 1.0])


def test__quantize_ste():
    """Verify quantize_ste simulates quantization and propagates gradients."""
    x = tf.constant([-1.2, -0.2, 0.3, 1.4], dtype=tf.float32)
    scale = 0.05
    zero_point = -10.0

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = utils.quantize_ste(x, scale, zero_point)

    # Replicate forward
    expected = np.clip(np.round(_to_numpy(x) / scale) + zero_point, -128.0, 127.0)
    np.testing.assert_allclose(_to_numpy(y), expected, atol=1e-5)

    grads = tape.gradient(y, x)
    # STE gradient is scale inverse
    np.testing.assert_allclose(_to_numpy(grads), np.ones(4) / scale, atol=1e-5)


def test__dequantize_ste():
    """Verify dequantize_ste simulates dequantization."""
    x = ops.convert_to_tensor([-20.0, -4.0, 6.0, 28.0], dtype="float32")
    scale = 0.05
    zero_point = -10.0

    y = utils.dequantize_ste(x, scale, zero_point)
    expected = scale * (_to_numpy(x) - zero_point)
    np.testing.assert_allclose(_to_numpy(y), expected, atol=1e-5)


def test__fake_quantize():
    """Verify _fake_quantize helper performs quantize and then dequantize."""
    x = tf.constant([-1.2, -0.2, 0.3, 1.4], dtype=tf.float32)
    scale = 0.05
    zero_point = -10.0

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = utils.fake_quantize(x, scale, zero_point)

    # Replicate forward
    q = np.clip(np.round(_to_numpy(x) / scale) + zero_point, -128.0, 127.0)
    expected = scale * (q - zero_point)
    np.testing.assert_allclose(_to_numpy(y), expected, atol=1e-5)

    grads = tape.gradient(y, x)
    # STE gradient is constant one
    np.testing.assert_allclose(_to_numpy(grads), [1.0, 1.0, 1.0, 1.0])


def test__to_float_list():
    """Verify to_float_list behaves correctly on tensor and scalar."""
    tensor = ops.convert_to_tensor([1.2, 2.8])
    res = utils.to_float_list(tensor)
    assert isinstance(res, list)
    assert all(isinstance(val, float) for val in res)
    assert res == pytest.approx([1.2, 2.8])

    scalar = ops.convert_to_tensor(5.5)
    res_s = utils.to_float_list(scalar)
    assert res_s == [5.5]


def test__to_int_list():
    """Verify to_int_list behaves correctly on tensor and scalar."""
    tensor = ops.convert_to_tensor([1.2, 2.8])
    res = utils.to_int_list(tensor)
    assert isinstance(res, list)
    assert all(isinstance(val, int) for val in res)
    assert res == [1, 3]

    scalar = ops.convert_to_tensor(5.5)
    res_s = utils.to_int_list(scalar)
    assert res_s == [6]


def test__make_quant_write_op():
    """Verify make_quant_write_op formats QuantizationWriteOp correctly."""
    scale = ops.convert_to_tensor([0.1, 0.2])
    zp = ops.convert_to_tensor([3, 4])
    op = utils.make_quant_write_op(tensor_index=7, scale_tensor=scale, zp_tensor=zp)

    assert isinstance(op, types.QuantizationWriteOp)
    assert op.tensor_index == 7
    assert op.scales == pytest.approx([0.1, 0.2])
    assert op.zero_points == [3, 4]
