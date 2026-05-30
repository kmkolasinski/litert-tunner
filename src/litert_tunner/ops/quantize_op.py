"""Quantize and Dequantize op builders for litert_tunner."""

from __future__ import annotations

import keras

from litert_tunner.graph import types
from litert_tunner.ops import registry
from litert_tunner.quantization import fake_quant


@registry.register_op("QUANTIZE")
def build_quantize(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    graph_def: types.GraphDef | None = None,
) -> keras.Layer:
    """Build a Quantize layer from parsed TFLite operator info.

    TFLite QUANTIZE inputs:
        [0] input tensor (FLOAT32 or INT8)
    TFLite QUANTIZE outputs:
        [0] output tensor (INT8)
    """
    output_tensor = tensors[op.output_indices[0]]
    output_quant = output_tensor.quantization

    if output_quant is None:
        msg = "QUANTIZE op requires quantized output tensor"
        raise ValueError(msg)

    scale = float(output_quant.scales[0])
    zero_point = float(output_quant.zero_points[0])

    return fake_quant.Quantize(
        scale=scale,
        zero_point=zero_point,
        trainable=True,
        name=f"quantize_{op.output_indices[0]}",
    )


@registry.register_op("DEQUANTIZE")
def build_dequantize(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    graph_def: types.GraphDef | None = None,
) -> keras.Layer:
    """Build a Dequantize layer from parsed TFLite operator info.

    TFLite DEQUANTIZE inputs:
        [0] input tensor (INT8)
    TFLite DEQUANTIZE outputs:
        [0] output tensor (FLOAT32)
    """
    input_tensor = tensors[op.input_indices[0]]
    input_quant = input_tensor.quantization

    if input_quant is None:
        msg = "DEQUANTIZE op requires quantized input tensor"
        raise ValueError(msg)

    scale = float(input_quant.scales[0])
    zero_point = float(input_quant.zero_points[0])

    # Check if the input tensor is produced by another operator in the graph
    passthrough = False
    if graph_def is not None:
        for other_op in graph_def.operators:
            if op.input_indices[0] in other_op.output_indices:
                passthrough = True
                break

    return fake_quant.Dequantize(
        scale=scale,
        zero_point=zero_point,
        passthrough=passthrough,
        name=f"dequantize_{op.output_indices[0]}",
    )
