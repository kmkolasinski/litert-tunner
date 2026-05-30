"""Quantization numeric helpers for litert_tunner.

Pure numpy functions implementing TFLite's affine quantization scheme.
These are used by both the parser (for verification) and the fake-quant
Keras layers.
"""

from __future__ import annotations

import numpy as np

# INT8 quantization range constants
INT8_MIN = -128
INT8_MAX = 127


def quantize_int8(
    x: np.ndarray,
    scale: np.ndarray | float,
    zero_point: np.ndarray | int,
) -> np.ndarray:
    """Quantize float32 values to INT8 using TFLite's affine scheme.

    Formula: int8_value = clamp(round(x / scale) + zero_point, -128, 127)

    Args:
        x: Float32 values to quantize.
        scale: Quantization scale (per-tensor or per-channel).
        zero_point: Quantization zero point (per-tensor or per-channel).

    Returns:
        INT8 quantized values as int8 numpy array.
    """
    scaled = np.round(x / scale) + zero_point
    clamped = np.clip(scaled, INT8_MIN, INT8_MAX)
    return clamped.astype(np.int8)


def dequantize_float(
    x: np.ndarray,
    scale: np.ndarray | float,
    zero_point: np.ndarray | int,
) -> np.ndarray:
    """Dequantize INT8 values to float32 using TFLite's affine scheme.

    Formula: real_value = scale * (int8_value - zero_point)

    Args:
        x: INT8 quantized values.
        scale: Quantization scale (per-tensor or per-channel).
        zero_point: Quantization zero point (per-tensor or per-channel).

    Returns:
        Float32 dequantized values.
    """
    return scale * (x.astype(np.float32) - np.float32(zero_point))


def compute_requantize_multiplier(
    input_scale: float,
    weight_scale: np.ndarray | float,
    output_scale: float,
) -> np.ndarray | float:
    """Compute the requantization multiplier for fused ops.

    In TFLite, the accumulator (INT32) is rescaled to the output
    quantization domain using:
        multiplier = (input_scale * weight_scale) / output_scale

    Args:
        input_scale: Scale of the input activation tensor.
        weight_scale: Scale of the weight tensor (scalar or per-channel array).
        output_scale: Scale of the output activation tensor.

    Returns:
        Requantization multiplier (scalar or per-channel array).
    """
    return (input_scale * weight_scale) / output_scale
