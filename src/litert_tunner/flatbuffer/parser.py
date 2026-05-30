"""Flatbuffer parser for litert_tunner.

Uses the tflite python package to parse TFLite models
into an internal GraphDef structure.
"""

from pathlib import Path

import numpy as np
import tflite

from litert_tunner.graph import types

# Mapping from tflite TensorType codes to internal dtype strings.
_TENSOR_TYPE_MAP: dict[int, str] = {
    tflite.TensorType.FLOAT16: types.DTYPE_FLOAT16,
    tflite.TensorType.FLOAT32: types.DTYPE_FLOAT32,
    tflite.TensorType.FLOAT64: types.DTYPE_FLOAT64,
    tflite.TensorType.INT8: types.DTYPE_INT8,
    tflite.TensorType.INT16: types.DTYPE_INT16,
    tflite.TensorType.INT32: types.DTYPE_INT32,
    tflite.TensorType.INT64: types.DTYPE_INT64,
    tflite.TensorType.UINT8: types.DTYPE_UINT8,
    tflite.TensorType.UINT16: types.DTYPE_UINT16,
    tflite.TensorType.UINT32: types.DTYPE_UINT32,
    tflite.TensorType.UINT64: types.DTYPE_UINT64,
    tflite.TensorType.BOOL: types.DTYPE_BOOL,
}

# Mapping from tflite TensorType codes to numpy dtypes.
_NUMPY_DTYPE_MAP: dict[int, np.dtype] = {
    tflite.TensorType.FLOAT16: np.dtype(np.float16),
    tflite.TensorType.FLOAT32: np.dtype(np.float32),
    tflite.TensorType.FLOAT64: np.dtype(np.float64),
    tflite.TensorType.INT8: np.dtype(np.int8),
    tflite.TensorType.INT16: np.dtype(np.int16),
    tflite.TensorType.INT32: np.dtype(np.int32),
    tflite.TensorType.INT64: np.dtype(np.int64),
    tflite.TensorType.UINT8: np.dtype(np.uint8),
    tflite.TensorType.UINT16: np.dtype(np.uint16),
    tflite.TensorType.UINT32: np.dtype(np.uint32),
    tflite.TensorType.UINT64: np.dtype(np.uint64),
    tflite.TensorType.BOOL: np.dtype(np.bool_),
}

# Mapping from BuiltinOptions type codes to their tflite option classes.
# Covers ops from Phases 1–5 (Dense, Conv, Pooling, Reshape, Normalization,
# Skip connections, Softmax, Concatenation, etc.) plus common utility ops.
_BUILTIN_OPTIONS_MAP: dict[int, type] = {
    # Phase 1: Dense / FullyConnected
    tflite.BuiltinOptions.FullyConnectedOptions: tflite.FullyConnectedOptions,
    # Phase 2: Convolutions
    tflite.BuiltinOptions.Conv2DOptions: tflite.Conv2DOptions,
    tflite.BuiltinOptions.DepthwiseConv2DOptions: tflite.DepthwiseConv2DOptions,
    tflite.BuiltinOptions.TransposeConvOptions: tflite.TransposeConvOptions,
    tflite.BuiltinOptions.Conv3DOptions: tflite.Conv3DOptions,
    # Phase 3: Pooling & Reshape
    tflite.BuiltinOptions.Pool2DOptions: tflite.Pool2DOptions,
    tflite.BuiltinOptions.ReshapeOptions: tflite.ReshapeOptions,
    tflite.BuiltinOptions.SqueezeOptions: tflite.SqueezeOptions,
    tflite.BuiltinOptions.ExpandDimsOptions: tflite.ExpandDimsOptions,
    tflite.BuiltinOptions.SliceOptions: tflite.SliceOptions,
    tflite.BuiltinOptions.StridedSliceOptions: tflite.StridedSliceOptions,
    tflite.BuiltinOptions.PackOptions: tflite.PackOptions,
    tflite.BuiltinOptions.UnpackOptions: tflite.UnpackOptions,
    tflite.BuiltinOptions.TransposeOptions: tflite.TransposeOptions,
    tflite.BuiltinOptions.TileOptions: tflite.TileOptions,
    tflite.BuiltinOptions.GatherOptions: tflite.GatherOptions,
    tflite.BuiltinOptions.GatherNdOptions: tflite.GatherNdOptions,
    # Phase 4: Normalization & Skip connections
    tflite.BuiltinOptions.AddOptions: tflite.AddOptions,
    tflite.BuiltinOptions.SubOptions: tflite.SubOptions,
    tflite.BuiltinOptions.MulOptions: tflite.MulOptions,
    tflite.BuiltinOptions.DivOptions: tflite.DivOptions,
    tflite.BuiltinOptions.L2NormOptions: tflite.L2NormOptions,
    tflite.BuiltinOptions.ReducerOptions: tflite.ReducerOptions,
    tflite.BuiltinOptions.BatchToSpaceNDOptions: tflite.BatchToSpaceNDOptions,
    tflite.BuiltinOptions.SpaceToBatchNDOptions: tflite.SpaceToBatchNDOptions,
    tflite.BuiltinOptions.SpaceToDepthOptions: tflite.SpaceToDepthOptions,
    tflite.BuiltinOptions.DepthToSpaceOptions: tflite.DepthToSpaceOptions,
    # Phase 5: Advanced
    tflite.BuiltinOptions.SoftmaxOptions: tflite.SoftmaxOptions,
    tflite.BuiltinOptions.ConcatenationOptions: tflite.ConcatenationOptions,
    tflite.BuiltinOptions.PadOptions: tflite.PadOptions,
    tflite.BuiltinOptions.PadV2Options: tflite.PadV2Options,
    tflite.BuiltinOptions.MirrorPadOptions: tflite.MirrorPadOptions,
    tflite.BuiltinOptions.SplitOptions: tflite.SplitOptions,
    tflite.BuiltinOptions.SplitVOptions: tflite.SplitVOptions,
    tflite.BuiltinOptions.ResizeBilinearOptions: tflite.ResizeBilinearOptions,
    tflite.BuiltinOptions.ResizeNearestNeighborOptions: tflite.ResizeNearestNeighborOptions,
    tflite.BuiltinOptions.CastOptions: tflite.CastOptions,
    tflite.BuiltinOptions.MaximumMinimumOptions: tflite.MaximumMinimumOptions,
    tflite.BuiltinOptions.LeakyReluOptions: tflite.LeakyReluOptions,
    tflite.BuiltinOptions.ArgMaxOptions: tflite.ArgMaxOptions,
    tflite.BuiltinOptions.ArgMinOptions: tflite.ArgMinOptions,
    tflite.BuiltinOptions.BatchMatMulOptions: tflite.BatchMatMulOptions,
    # Quantization-related
    tflite.BuiltinOptions.QuantizeOptions: tflite.QuantizeOptions,
    tflite.BuiltinOptions.DequantizeOptions: tflite.DequantizeOptions,
    tflite.BuiltinOptions.FakeQuantOptions: tflite.FakeQuantOptions,
}


def get_op_code_name(opcode: tflite.OperatorCode) -> str:
    """Map operator code to string name.

    Args:
        opcode: A parsed OperatorCode flatbuffer object.

    Returns:
        The human-readable name of the operator (e.g. "FULLY_CONNECTED").
    """
    builtin_code = opcode.BuiltinCode()
    for attr in dir(tflite.BuiltinOperator):
        if attr.startswith("_"):
            continue
        val = getattr(tflite.BuiltinOperator, attr)
        if val == builtin_code:
            return attr
    custom_code = opcode.CustomCode()
    if custom_code:
        if isinstance(custom_code, bytes):
            return custom_code.decode("utf-8")
        return str(custom_code)
    return f"UNKNOWN_{builtin_code}"


def _map_tensor_type(tensor_type_code: int) -> str:
    """Map TFLite tensor type code to string.

    Args:
        tensor_type_code: Integer type code from the flatbuffer.

    Returns:
        One of types.DTYPE_INT8, types.DTYPE_FLOAT32, types.DTYPE_INT32.

    Raises:
        ValueError: If the tensor type is not supported.
    """
    if tensor_type_code in _TENSOR_TYPE_MAP:
        return _TENSOR_TYPE_MAP[tensor_type_code]
    raise ValueError(f"Unsupported tensor type: {tensor_type_code}")


def get_numpy_dtype(tensor_type_code: int) -> np.dtype:
    """Map TFLite tensor type code to numpy dtype.

    Args:
        tensor_type_code: Integer type code from the flatbuffer.

    Returns:
        The corresponding numpy dtype.

    Raises:
        ValueError: If the tensor type is not supported.
    """
    if tensor_type_code in _NUMPY_DTYPE_MAP:
        return _NUMPY_DTYPE_MAP[tensor_type_code]
    raise ValueError(f"Unsupported tensor type: {tensor_type_code}")


def _parse_builtin_options(op: tflite.Operator) -> dict:
    """Extract builtin options from an operator as a plain dict.

    Args:
        op: A parsed Operator flatbuffer object.

    Returns:
        A dict of option-name → value. Keys use snake_case for
        ``FusedActivationFunction`` (→ ``fused_activation_function``).
    """
    options: dict = {}
    opt_type = op.BuiltinOptionsType()
    if opt_type == tflite.BuiltinOptions.NONE:
        return options

    opt_cls = _BUILTIN_OPTIONS_MAP.get(opt_type)
    if opt_cls is None:
        return options

    opt_obj = opt_cls()
    opt_obj.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    for attr in dir(opt_obj):
        if attr.startswith("_") or attr in ("Init", "GetRootAs"):
            continue
        val = getattr(opt_obj, attr)
        if callable(val):
            try:
                resolved = val()
            except TypeError:
                continue
            key = "fused_activation_function" if attr == "FusedActivationFunction" else attr
            options[key] = resolved

    return options


def parse_tflite(path: str | Path) -> types.GraphDef:
    """Parse a .tflite file into an internal graph representation.

    Args:
        path: Path to the .tflite file.

    Returns:
        A fully-parsed GraphDef containing tensors, operators, and graph I/O
        indices.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the model has no subgraphs or contains unsupported types.
    """
    with open(path, "rb") as f:
        raw_bytes = f.read()

    buf = bytearray(raw_bytes)
    model = tflite.Model.GetRootAs(buf, 0)

    if model.SubgraphsLength() == 0:
        raise ValueError("Model has no subgraphs")
    subgraph = model.Subgraphs(0)

    # 1. Parse Tensors
    tensors: list[types.TensorInfo] = []
    for idx in range(subgraph.TensorsLength()):
        tensor_t = subgraph.Tensors(idx)
        name = tensor_t.Name()
        if not isinstance(name, str):
            name = name.decode("utf-8")
        shape_np = tensor_t.ShapeAsNumpy()
        shape = tuple(shape_np) if not isinstance(shape_np, int) else ()
        dtype_str = _map_tensor_type(tensor_t.Type())

        # Quantization params
        quant_params = tensor_t.Quantization()
        quant = None
        if quant_params is not None:
            scales = quant_params.ScaleAsNumpy()
            zero_points = quant_params.ZeroPointAsNumpy()
            quantized_dimension = quant_params.QuantizedDimension()

            if not isinstance(scales, int) and len(scales) > 0:
                scales_arr = np.array(scales, dtype=np.float32)
                if not isinstance(zero_points, int):
                    zp_arr = np.array(zero_points, dtype=np.int32)
                else:
                    zp_arr = np.zeros(len(scales), dtype=np.int32)
                quant = types.QuantizationParams(
                    scales=scales_arr,
                    zero_points=zp_arr,
                    quantized_dimension=quantized_dimension or 0,
                )

        # Buffer data
        buffer_idx = tensor_t.Buffer()
        buffer_t = model.Buffers(buffer_idx)
        tensor_data = None

        data_np = buffer_t.DataAsNumpy()
        if not isinstance(data_np, int) and len(data_np) > 0:
            try:
                dtype = get_numpy_dtype(tensor_t.Type())
                tensor_data = np.frombuffer(data_np.tobytes(), dtype=dtype).copy()
                if shape:
                    tensor_data = tensor_data.reshape(shape)
            except Exception:
                pass

        tensors.append(
            types.TensorInfo(
                name=name,
                index=idx,
                shape=shape,
                dtype=dtype_str,
                quantization=quant,
                buffer_index=buffer_idx,
                data=tensor_data,
            )
        )

    # 2. Parse Operators
    operators: list[types.OperatorInfo] = []
    for idx in range(subgraph.OperatorsLength()):
        op_t = subgraph.Operators(idx)
        opcode_t = model.OperatorCodes(op_t.OpcodeIndex())
        op_type = get_op_code_name(opcode_t)

        input_indices = tuple(op_t.InputsAsNumpy())
        output_indices = tuple(op_t.OutputsAsNumpy())
        options = _parse_builtin_options(op_t)

        operators.append(
            types.OperatorInfo(
                op_type=op_type,
                input_indices=input_indices,
                output_indices=output_indices,
                options=options,
            )
        )

    input_indices = tuple(subgraph.InputsAsNumpy()) if not subgraph.InputsIsNone() else ()
    output_indices = tuple(subgraph.OutputsAsNumpy()) if not subgraph.OutputsIsNone() else ()

    return types.GraphDef(
        tensors=tuple(tensors),
        operators=tuple(operators),
        input_indices=input_indices,
        output_indices=output_indices,
        raw_model_bytes=raw_bytes,
    )
