"""Graph type definitions for litert_tunner.

Frozen dataclasses representing the parsed TFLite graph in a
framework-agnostic way. These are the bridge between flatbuffer
parsing and Keras model building.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

# TFLite dtype string constants
DTYPE_FLOAT16 = "float16"
DTYPE_FLOAT32 = "float32"
DTYPE_FLOAT64 = "float64"
DTYPE_INT8 = "int8"
DTYPE_INT16 = "int16"
DTYPE_INT32 = "int32"
DTYPE_INT64 = "int64"
DTYPE_UINT8 = "uint8"
DTYPE_UINT16 = "uint16"
DTYPE_UINT32 = "uint32"
DTYPE_UINT64 = "uint64"
DTYPE_BOOL = "bool"

# INT8 quantization range
INT8_MIN = -128
INT8_MAX = 127


@dataclass(frozen=True)
class QuantizationParams:
    """Quantization parameters for a tensor.

    Attributes:
        scales: Per-channel or per-tensor scale factors (float32).
        zero_points: Per-channel or per-tensor zero points (int32).
        quantized_dimension: Axis for per-channel quantization (0 for per-tensor).
    """

    scales: np.ndarray
    zero_points: np.ndarray
    quantized_dimension: int = 0


@dataclass(frozen=True)
class TensorInfo:
    """Metadata and optional data for a single tensor in the graph.

    Attributes:
        name: Human-readable tensor name from the flatbuffer.
        index: Tensor index within the subgraph.
        shape: Tensor shape as a tuple of ints.
        dtype: Data type string ("int8", "int32", "float32").
        quantization: Quantization parameters, or None if unquantized.
        buffer_index: Index into the model's buffer list.
        data: Tensor data as a numpy array (weights/biases), or None for activations.
    """

    name: str
    index: int
    shape: tuple[int, ...]
    dtype: str
    quantization: QuantizationParams | None
    buffer_index: int
    data: np.ndarray | None = None


@dataclass(frozen=True)
class OperatorInfo:
    """Metadata for a single operator in the graph.

    Attributes:
        op_type: TFLite operator type string (e.g., "FULLY_CONNECTED").
        input_indices: Indices into the GraphDef.tensors list for inputs.
        output_indices: Indices into the GraphDef.tensors list for outputs.
        options: Op-specific options (fused activation, padding, etc.).
    """

    op_type: str
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    options: dict[str, Any]


@dataclass(frozen=True)
class BufferWriteOp:
    """Instruction to overwrite a tensor's buffer data in the flatbuffer.

    Attributes:
        tensor_index: Index of the tensor whose buffer should be updated.
        data: New raw bytes to write into the buffer.
    """

    tensor_index: int
    data: bytes


@dataclass(frozen=True)
class QuantizationWriteOp:
    """Instruction to overwrite a tensor's quantization params in the flatbuffer.

    Attributes:
        tensor_index: Index of the tensor whose quantization params should be updated.
        scales: New scale values (float32).
        zero_points: New zero-point values (int64).
    """

    tensor_index: int
    scales: list[float]
    zero_points: list[int]


@dataclass(frozen=True)
class GraphDef:
    """Complete parsed graph definition from a .tflite file.

    Attributes:
        tensors: All tensors in the subgraph.
        operators: All operators in topological order.
        input_indices: Tensor indices for graph-level inputs.
        output_indices: Tensor indices for graph-level outputs.
        raw_model_bytes: Original flatbuffer bytes for save round-trip.
    """

    tensors: tuple[TensorInfo, ...]
    operators: tuple[OperatorInfo, ...]
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    raw_model_bytes: bytes


def is_quantized(tensor: TensorInfo) -> bool:
    """Check whether a tensor has quantization parameters.

    Args:
        tensor: The tensor to check.

    Returns:
        True if the tensor has non-None quantization params.
    """
    return tensor.quantization is not None


@runtime_checkable
class Writable(Protocol):
    """Protocol for Keras layers that know how to serialize themselves back to a flatbuffer.

    Any layer that needs to persist updated parameters must implement this
    protocol by providing a ``collect_write_ops`` method. The writer calls it
    once per layer and applies the returned instructions to the flatbuffer.
    """

    def collect_write_ops(
        self,
        op: OperatorInfo,
    ) -> tuple[list[BufferWriteOp], list[QuantizationWriteOp]]:
        """Return flatbuffer write instructions for this layer.

        Args:
            op: The OperatorInfo that this layer was built from.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        ...
