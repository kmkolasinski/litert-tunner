"""Flatbuffer writer for litert_tunner.

Updates buffer data and quantization parameters in a TFLite flatbuffer using
values from a trained Keras model. The writer is fully generic — it delegates
parameter extraction to each layer's ``collect_write_ops`` method, so adding
support for a new op requires zero changes here.
"""

from pathlib import Path

import flatbuffers
import flatbuffers.number_types
import keras
import numpy as np
import tflite

from litert_tunner.graph import types


def save_tflite(model: keras.Model, path: str | Path) -> None:
    """Write the current parameter values of the tunner model back into a .tflite file.

    This performs binary surgery on the original flatbuffer bytes. The writer is
    op-agnostic — each Keras layer that implements the ``Writable`` protocol
    provides its own write instructions via ``collect_write_ops``.

    Args:
        model: A Keras model created by ``litert_tunner.load_model``.
        path: Output path for the updated .tflite file.

    Raises:
        ValueError: If the model was not created by ``litert_tunner.load_model``.
    """
    if not hasattr(model, "_graph_def"):
        raise ValueError("Model was not created by litert_tunner.load_model")

    graph_def: types.GraphDef = model._graph_def  # noqa: SLF001

    buf = bytearray(graph_def.raw_model_bytes)
    model_obj = tflite.Model.GetRootAs(buf, 0)
    if model_obj is None:
        raise ValueError("Failed to parse TFLite model from bytes")
    subgraph_t = model_obj.Subgraphs(0)
    if subgraph_t is None:
        raise ValueError("Model has no subgraph at index 0")

    # Build a mapping from layer name → Keras layer for fast lookup
    layer_map: dict[str, keras.Layer] = {lyr.name: lyr for lyr in model.layers}

    # Collect all write instructions from writable layers
    buffer_writes: list[types.BufferWriteOp] = []
    quant_writes: list[types.QuantizationWriteOp] = []

    for op in graph_def.operators:
        layer = _find_layer_for_op(op, layer_map)
        if layer is None or not isinstance(layer, types.Writable):
            continue

        op_buf_writes, op_quant_writes = layer.collect_write_ops(op)
        buffer_writes.extend(op_buf_writes)
        quant_writes.extend(op_quant_writes)

    # Apply all buffer writes
    for write_op in buffer_writes:
        tensor_t = subgraph_t.Tensors(write_op.tensor_index)
        if tensor_t is None:
            msg = f"Tensor at index {write_op.tensor_index} is None"
            raise ValueError(msg)
        _overwrite_buffer(buf, model_obj, tensor_t.Buffer(), write_op.data)

    # Apply all quantization writes
    for write_op in quant_writes:
        _overwrite_quantization(buf, subgraph_t, write_op)

    with Path(path).open("wb") as f:
        f.write(bytes(buf))


def _find_layer_for_op(
    op: types.OperatorInfo,
    layer_map: dict[str, keras.Layer],
) -> keras.Layer | None:
    """Find the Keras layer corresponding to a flatbuffer operator.

    Layers are matched by the naming convention established in the op builders:
    each builder names its layer using the first output tensor index as a suffix
    (e.g., ``quantized_dense_3``, ``quantize_1``, ``dequantize_5``).

    Args:
        op: The operator to find a layer for.
        layer_map: Mapping from layer name to Keras layer.

    Returns:
        The matching layer, or None if no layer matches.
    """
    output_idx = op.output_indices[0]
    # Try common naming patterns used by registered op builders.
    # Each builder names its layer as ``{prefix}_{output_index}``.
    for layer_name, layer in layer_map.items():
        if layer_name.endswith(f"_{output_idx}"):
            return layer
    return None


def _overwrite_buffer(
    buf: bytearray,
    model_obj: tflite.Model,
    buffer_idx: int,
    new_data: bytes,
) -> None:
    """Overwrite the buffer data in-place at the specified index.

    Args:
        buf: Mutable bytearray of the full flatbuffer.
        model_obj: Parsed Model object (for buffer offset lookup).
        buffer_idx: Index of the buffer to overwrite.
        new_data: New raw bytes (must match existing buffer length).

    Raises:
        ValueError: If the new data size does not match the existing buffer.
    """
    buffer_t = model_obj.Buffers(buffer_idx)
    if buffer_t is None:
        msg = f"Buffer at index {buffer_idx} is None"
        raise ValueError(msg)
    o = flatbuffers.number_types.UOffsetTFlags.py_type(buffer_t._tab.Offset(4))  # noqa: SLF001
    if o != 0:
        offset = buffer_t._tab.Vector(o)  # noqa: SLF001
        length = buffer_t.DataLength()
        if len(new_data) != length:
            msg = f"Size mismatch in buffer {buffer_idx}: expected {length}, got {len(new_data)}"
            raise ValueError(msg)
        buf[offset : offset + length] = new_data


def _overwrite_quantization(
    buf: bytearray,
    subgraph_t: tflite.SubGraph,
    write_op: types.QuantizationWriteOp,
) -> None:
    """Overwrite scales and zero points in-place for a specific tensor.

    Args:
        buf: Mutable bytearray of the full flatbuffer.
        subgraph_t: Parsed SubGraph object (for tensor lookup).
        write_op: Write instruction with tensor index, scales, and zero points.

    Raises:
        ValueError: If the scales or zero-points length does not match the existing data.
    """
    tensor_t = subgraph_t.Tensors(write_op.tensor_index)
    if tensor_t is None:
        msg = f"Tensor at index {write_op.tensor_index} is None"
        raise ValueError(msg)
    quant_t = tensor_t.Quantization()
    if quant_t is None:
        return

    # In FlatBuffers, vtable field offsets start at 4 and increase by 2 bytes per field.
    # From schema.fbs: table QuantizationParameters
    #   min:[float];        // Field 0 -> offset 4 // can be ignored
    #   max:[float];        // Field 1 -> offset 6 // can be ignored
    #   scale:[float];      // Field 2 -> offset 8
    #   zero_point:[long];  // Field 3 -> offset 10
    #
    # see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs#L101-L125
    o_scale = flatbuffers.number_types.UOffsetTFlags.py_type(quant_t._tab.Offset(8))  # noqa: SLF001
    if o_scale != 0:
        offset = quant_t._tab.Vector(o_scale)  # noqa: SLF001
        length = quant_t.ScaleLength()
        if len(write_op.scales) != length:
            msg = f"Scale length mismatch in tensor {write_op.tensor_index}"
            raise ValueError(msg)
        new_data = np.array(write_op.scales, dtype=np.float32).tobytes()
        buf[offset : offset + len(new_data)] = new_data

    o_zp = flatbuffers.number_types.UOffsetTFlags.py_type(quant_t._tab.Offset(10))  # noqa: SLF001
    if o_zp != 0:
        offset = quant_t._tab.Vector(o_zp)  # noqa: SLF001
        length = quant_t.ZeroPointLength()
        if len(write_op.zero_points) != length:
            msg = f"Zero point length mismatch in tensor {write_op.tensor_index}"
            raise ValueError(msg)
        new_data = np.array(write_op.zero_points, dtype=np.int64).tobytes()
        buf[offset : offset + len(new_data)] = new_data
