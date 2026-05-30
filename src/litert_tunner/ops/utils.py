"""Utility helper functions for operators in litert_tunner.

These functions provide reusable logic for tensor reshaping, quantization parameters,
and converting tensors to types suitable for flatbuffer writing.
"""

from __future__ import annotations

import typing

import numpy as np
from keras import ops

from litert_tunner.graph import types

# Fused activation function codes from TFLite schema
FUSED_ACTIVATION_NONE = 0
FUSED_ACTIVATION_RELU = 1
FUSED_ACTIVATION_RELU_N1_TO_1 = 2
FUSED_ACTIVATION_RELU6 = 3


def expand_dims_if_not_scalar(tensor: typing.Any, axis: int) -> typing.Any:
    """Expands the dimensions of a tensor if it is not a scalar (ndim > 0).

    Args:
        tensor: The input tensor or variable.
        axis: The axis along which to expand.

    Returns:
        The tensor with expanded dimensions if not a scalar, otherwise the original tensor.
    """
    if len(tensor.shape) > 0:
        return ops.expand_dims(tensor, axis)
    return tensor


def quantize_to_int8(tensor: typing.Any) -> np.ndarray:
    """Converts a tensor to a rounded numpy INT8 array.

    Args:
        tensor: The input tensor to round and convert.

    Returns:
        NumPy array with dtype int8.
    """
    val = typing.cast(np.ndarray, ops.convert_to_numpy(tensor))
    return np.round(val).astype(np.int8)


def quantize_bias_to_int32(
    bias_tensor: typing.Any,
    input_scale_tensor: typing.Any,
    weight_scale_tensor: typing.Any,
) -> np.ndarray:
    """Quantizes a float32 bias tensor to INT32 using input and weight scales.

    Bias scale = input_scale * weight_scale.

    Args:
        bias_tensor: Float32 bias tensor.
        input_scale_tensor: Input scale tensor (scalar).
        weight_scale_tensor: Weight scale tensor (scalar or array).

    Returns:
        NumPy array with dtype int32.
    """
    bias_val = typing.cast(np.ndarray, ops.convert_to_numpy(bias_tensor))
    input_scale_val = float(typing.cast(typing.Any, ops.convert_to_numpy(input_scale_tensor)))
    weight_scale_val = np.asarray(
        typing.cast(np.ndarray, ops.convert_to_numpy(weight_scale_tensor))
    )
    bias_scale = input_scale_val * weight_scale_val
    return np.round(bias_val / bias_scale).astype(np.int32)


def get_quant_param_value(param_array: np.ndarray) -> float | np.ndarray:
    """Extracts a scalar float if the array has length 1, else returns float32 array.

    Args:
        param_array: The quantization parameter array (scales or zero points).

    Returns:
        A float scalar if length is 1, otherwise a NumPy float32 array.
    """
    if len(param_array) == 1:
        return float(param_array[0])
    return param_array.astype(np.float32)


def get_bias_float32(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    input_scale: float,
    weight_scales: np.ndarray,
    output_units: int,
) -> np.ndarray:
    """Extracts the bias tensor as float32, or initializes it to zeros if not present.

    Args:
        op: The parsed operator info.
        tensors: All tensors in the graph.
        input_scale: The input quantization scale.
        weight_scales: NumPy array of weight quantization scales.
        output_units: The number of output channels/units to fallback to if bias is missing.

    Returns:
        NumPy array with dtype float32.
    """
    if len(op.input_indices) > 2 and op.input_indices[2] >= 0:
        bias_tensor = tensors[op.input_indices[2]]
        if bias_tensor.data is not None:
            # TFLite stores bias as INT32; convert to float32
            bias_scale = input_scale * weight_scales.astype(np.float64)
            return bias_tensor.data.astype(np.float32) * bias_scale.astype(np.float32)
    return np.zeros(output_units, dtype=np.float32)


def apply_fused_activation(x: typing.Any, fused_activation: int) -> typing.Any:
    """Applies a TFLite fused activation function to a tensor.

    Args:
        x: Input Keras tensor.
        fused_activation: Code of the fused activation function.

    Returns:
        The tensor after applying the activation.
    """
    if fused_activation == FUSED_ACTIVATION_NONE:
        return x
    elif fused_activation == FUSED_ACTIVATION_RELU:
        return ops.relu(x)
    elif fused_activation == FUSED_ACTIVATION_RELU6:
        return ops.minimum(ops.relu(x), 6.0)
    elif fused_activation == FUSED_ACTIVATION_RELU_N1_TO_1:
        return ops.clip(x, -1.0, 1.0)
    else:
        msg = f"Unsupported fused activation: {fused_activation}"
        raise ValueError(msg)
