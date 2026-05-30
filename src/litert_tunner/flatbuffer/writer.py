"""Flatbuffer writer for litert_tunner.

Updates buffer data and quantization parameters in a TFLite flatbuffer using
values from a trained Keras model.
"""

from pathlib import Path

import flatbuffers
import flatbuffers.number_types
import keras
import numpy as np
import tflite
from keras import ops

from litert_tunner.graph import types


def save_tflite(model: keras.Model, path: str | Path):
    """Write the current parameter values of the tunner model back into a .tflite file.

    This performs binary surgery on the original flatbuffer bytes since the Object API
    is not available in the pip `tflite` package.
    """
    if not hasattr(model, "_graph_def"):
        raise ValueError("Model was not created by litert_tunner.load_model")

    graph_def: types.GraphDef = model._graph_def

    buf = bytearray(graph_def.raw_model_bytes)
    model_obj = tflite.Model.GetRootAs(buf, 0)
    subgraph_t = model_obj.Subgraphs(0)

    def overwrite_buffer(buffer_idx: int, new_data: bytes):
        """Overwrite the buffer data in-place at the specified index."""
        buffer_t = model_obj.Buffers(buffer_idx)
        o = flatbuffers.number_types.UOffsetTFlags.py_type(buffer_t._tab.Offset(4))
        if o != 0:
            offset = buffer_t._tab.Vector(o)
            length = buffer_t.DataLength()
            if len(new_data) != length:
                raise ValueError(
                    f"Size mismatch in buffer {buffer_idx}: expected {length}, got {len(new_data)}"
                )
            buf[offset : offset + length] = new_data

    def overwrite_quantization(tensor_idx: int, scales: list[float], zero_points: list[int]):
        """Overwrite scales and zero points in-place for a specific tensor."""
        tensor_t = subgraph_t.Tensors(tensor_idx)
        quant_t = tensor_t.Quantization()
        if quant_t is None:
            return

        o_scale = flatbuffers.number_types.UOffsetTFlags.py_type(quant_t._tab.Offset(4))
        if o_scale != 0:
            offset = quant_t._tab.Vector(o_scale)
            length = quant_t.ScaleLength()
            if len(scales) != length:
                raise ValueError(f"Scale length mismatch in tensor {tensor_idx}")
            new_data = np.array(scales, dtype=np.float32).tobytes()
            buf[offset : offset + len(new_data)] = new_data

        o_zp = flatbuffers.number_types.UOffsetTFlags.py_type(quant_t._tab.Offset(6))
        if o_zp != 0:
            offset = quant_t._tab.Vector(o_zp)
            length = quant_t.ZeroPointLength()
            if len(zero_points) != length:
                raise ValueError(f"Zero point length mismatch in tensor {tensor_idx}")
            new_data = np.array(zero_points, dtype=np.int64).tobytes()
            buf[offset : offset + len(new_data)] = new_data

    # Iterate over operators and match with Keras layers to extract trained parameters
    for op in graph_def.operators:
        if op.op_type == "FULLY_CONNECTED":
            layer_name = f"quantized_dense_{op.output_indices[0]}"
            layer = None
            for lyr in model.layers:
                if lyr.name == layer_name:
                    layer = lyr
                    break
            if layer is None:
                continue

            # Update weight_int8
            weight_val = ops.convert_to_numpy(layer.weight_int8)
            weight_int8 = np.round(weight_val).astype(np.int8)
            weight_tensor_idx = op.input_indices[1]
            weight_tensor_t = subgraph_t.Tensors(weight_tensor_idx)
            overwrite_buffer(weight_tensor_t.Buffer(), bytes(weight_int8.tobytes()))

            # Update bias (if present)
            if len(op.input_indices) > 2 and op.input_indices[2] >= 0:
                bias_val = ops.convert_to_numpy(layer.bias)
                input_scale_val = float(ops.convert_to_numpy(layer.input_scale))
                weight_scale_val = float(ops.convert_to_numpy(layer.weight_scale))
                bias_scale = input_scale_val * weight_scale_val
                bias_int32 = np.round(bias_val / bias_scale).astype(np.int32)

                bias_tensor_idx = op.input_indices[2]
                bias_tensor_t = subgraph_t.Tensors(bias_tensor_idx)
                overwrite_buffer(bias_tensor_t.Buffer(), bytes(bias_int32.tobytes()))

            # Update quantization params
            input_tensor_idx = op.input_indices[0]
            in_scale = float(ops.convert_to_numpy(layer.input_scale))
            in_zp = int(np.round(ops.convert_to_numpy(layer.input_zero_point)))
            overwrite_quantization(input_tensor_idx, [in_scale], [in_zp])

            w_scale = float(ops.convert_to_numpy(layer.weight_scale))
            w_zp = int(np.round(ops.convert_to_numpy(layer.weight_zero_point)))
            overwrite_quantization(weight_tensor_idx, [w_scale], [w_zp])

            output_tensor_idx = op.output_indices[0]
            out_scale = float(ops.convert_to_numpy(layer.output_scale))
            out_zp = int(np.round(ops.convert_to_numpy(layer.output_zero_point)))
            overwrite_quantization(output_tensor_idx, [out_scale], [out_zp])

        elif op.op_type == "QUANTIZE":
            layer_name = f"quantize_{op.output_indices[0]}"
            layer = None
            for lyr in model.layers:
                if lyr.name == layer_name:
                    layer = lyr
                    break
            if layer is None:
                continue

            output_tensor_idx = op.output_indices[0]
            scale_val = float(ops.convert_to_numpy(layer.scale))
            zp_val = int(np.round(ops.convert_to_numpy(layer.zero_point)))
            overwrite_quantization(output_tensor_idx, [scale_val], [zp_val])

        elif op.op_type == "DEQUANTIZE":
            layer_name = f"dequantize_{op.output_indices[0]}"
            layer = None
            for lyr in model.layers:
                if lyr.name == layer_name:
                    layer = lyr
                    break
            if layer is None:
                continue

            input_tensor_idx = op.input_indices[0]
            scale_val = float(ops.convert_to_numpy(layer.scale))
            zp_val = int(np.round(ops.convert_to_numpy(layer.zero_point)))
            overwrite_quantization(input_tensor_idx, [scale_val], [zp_val])

    with open(path, "wb") as f:
        f.write(bytes(buf))
