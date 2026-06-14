"""Reusable test utilities for op builder tests.

Provides factories for creating ``TensorInfo`` / ``OperatorInfo`` fixtures and
assertion helpers that verify the standard contract every registered op must
satisfy:

    1. **build** — The builder returns a valid Keras layer.
    2. **call** — The layer produces output with the expected shape/dtype.
    3. **collect_write_ops** — ``Writable`` layers emit correct write ops.
    4. **trainable_weights** — The layer exposes the expected trainable params.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pathlib
    import typing

import keras
import numpy as np

import litert_tunner
from litert_tunner import flatbuffer
from litert_tunner.graph import types
from litert_tunner.ops import registry

# ---------------------------------------------------------------------------
# Tensor / operator factory helpers
# ---------------------------------------------------------------------------


def make_quant_params(
    scales: list[float] | None = None,
    zero_points: list[int] | None = None,
    quantized_dimension: int = 0,
) -> types.QuantizationParams:
    """Create a ``QuantizationParams`` with sensible defaults.

    Args:
        scales: Scale values (default ``[0.05]``).
        zero_points: Zero-point values (default ``[0]``).
        quantized_dimension: Channel axis for per-channel quant.

    Returns:
        A frozen ``QuantizationParams`` instance.
    """
    scales = scales if scales is not None else [0.05]
    zero_points = zero_points if zero_points is not None else [0]
    return types.QuantizationParams(
        scales=np.array(scales, dtype=np.float32),
        zero_points=np.array(zero_points, dtype=np.int32),
        quantized_dimension=quantized_dimension,
    )


def make_tensor(
    *,
    name: str = "tensor",
    index: int = 0,
    shape: tuple[int, ...] = (1, 4),
    dtype: str = types.DTYPE_INT8,
    quantization: types.QuantizationParams | None = None,
    buffer_index: int = 0,
    data: np.ndarray | None = None,
) -> types.TensorInfo:
    """Create a ``TensorInfo`` with sensible defaults.

    By default the tensor is a quantized INT8 activation tensor (no data)
    with a single-element quantization param.

    Args:
        name: Human-readable tensor name.
        index: Tensor index within the subgraph.
        shape: Tensor shape.
        dtype: Data type string.
        quantization: Quantization params (``None`` means unquantized).
        buffer_index: Index into the model's buffer list.
        data: Static weight / bias data (``None`` for activations).

    Returns:
        A frozen ``TensorInfo`` instance.
    """
    return types.TensorInfo(
        name=name,
        index=index,
        shape=shape,
        dtype=dtype,
        quantization=quantization,
        buffer_index=buffer_index,
        data=data,
    )


def make_operator(
    *,
    op_type: str,
    input_indices: tuple[int, ...],
    output_indices: tuple[int, ...],
    options: dict | None = None,
) -> types.OperatorInfo:
    """Create an ``OperatorInfo``.

    Args:
        op_type: TFLite operator type string (e.g. ``"QUANTIZE"``).
        input_indices: Input tensor indices.
        output_indices: Output tensor indices.
        options: Op-specific options dict (defaults to ``{}``).

    Returns:
        A frozen ``OperatorInfo`` instance.
    """
    return types.OperatorInfo(
        op_type=op_type,
        input_indices=input_indices,
        output_indices=output_indices,
        options=options or {},
    )


# ---------------------------------------------------------------------------
# Build + call helpers
# ---------------------------------------------------------------------------


def build_layer_from_registry(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Look up the registered builder and invoke it.

    Args:
        op: Operator info.
        tensors: All tensors in the graph.

    Returns:
        The constructed Keras layer.
    """
    builder_fn = registry.get_op_builder(op.op_type)
    return builder_fn(op, tensors)


def build_and_call(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    input_data: np.ndarray | list[np.ndarray],
) -> tuple[keras.Layer, np.ndarray]:
    """Build a layer from the registry, call it, and return both.

    The layer is built via ``build_layer_from_registry`` and then called with
    ``input_data``.

    Args:
        op: Operator info describing the op.
        tensors: Full tensor table.
        input_data: Input numpy array (float32) or list of arrays.

    Returns:
        A tuple of ``(layer, output_numpy)``.
    """
    layer = build_layer_from_registry(op, tensors)
    if isinstance(input_data, list):
        keras_input = [keras.ops.convert_to_tensor(x) for x in input_data]
    else:
        keras_input = keras.ops.convert_to_tensor(input_data)
    output = layer(keras_input)
    output_np = np.asarray(keras.ops.convert_to_numpy(output))
    return layer, output_np


# ---------------------------------------------------------------------------
# Mixed precision helpers
# ---------------------------------------------------------------------------


# Default tolerance for float32 policy (0.1% relative error)
_ATOL_FLOAT32 = 0.001

# Relaxed tolerance for mixed_float16 policy (~3 digits of float16 precision)
_ATOL_MIXED_FLOAT16 = 0.01


def get_default_atol(dtype_policy: str) -> float:
    """Return the default comparison tolerance for a given Keras dtype policy.

    Float16 compute has ~3 decimal digits of precision, so quantization
    simulation accumulates more rounding error than float32.

    Args:
        dtype_policy: Keras dtype policy name (e.g. ``"float32"``, ``"mixed_float16"``).

    Returns:
        Absolute tolerance suitable for ``np.testing.assert_allclose``.
    """
    if "float16" in dtype_policy:
        return _ATOL_MIXED_FLOAT16
    return _ATOL_FLOAT32


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_layer_is_writable(layer: keras.Layer) -> None:
    """Assert that the layer implements the ``Writable`` protocol.

    Args:
        layer: The Keras layer to check.
    """
    assert isinstance(layer, types.Writable), (
        f"Layer {layer.name!r} ({type(layer).__name__}) does not implement "
        f"the Writable protocol (missing collect_write_ops method)."
    )


def assert_layer_not_writable(layer: keras.Layer) -> None:
    """Assert that the layer does NOT implement the ``Writable`` protocol.

    This is useful for layers that shouldn't persist any state (e.g.,
    reshape, pooling) — they must not accidentally implement ``Writable``.

    Args:
        layer: The Keras layer to check.
    """
    assert not isinstance(layer, types.Writable), (
        f"Layer {layer.name!r} ({type(layer).__name__}) unexpectedly implements "
        f"the Writable protocol."
    )


def assert_trainable_weight_names(
    layer: keras.Layer,
    expected_names: set[str],
) -> None:
    """Assert that the layer's trainable weights match the expected set.

    Compares only the *variable name suffix* (after the last ``/``).

    Args:
        layer: Built Keras layer.
        expected_names: Expected set of trainable weight base names.
    """
    actual_names = {w.name.split("/")[-1] for w in layer.trainable_weights}
    assert actual_names == expected_names, (
        f"Trainable weight mismatch for {layer.name!r}.\n"
        f"  Expected: {sorted(expected_names)}\n"
        f"  Actual:   {sorted(actual_names)}"
    )


def assert_non_trainable_weight_names(
    layer: keras.Layer,
    expected_names: set[str],
) -> None:
    """Assert that the layer's non-trainable weights match the expected set.

    Compares only the *variable name suffix* (after the last ``/``).

    Args:
        layer: Built Keras layer.
        expected_names: Expected set of non-trainable weight base names.
    """
    actual_names = {w.name.split("/")[-1] for w in layer.non_trainable_weights}
    assert actual_names == expected_names, (
        f"Non-trainable weight mismatch for {layer.name!r}.\n"
        f"  Expected: {sorted(expected_names)}\n"
        f"  Actual:   {sorted(actual_names)}"
    )


def assert_output_shape(
    output: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    """Assert the output tensor has the expected shape.

    Args:
        output: Output numpy array from the layer call.
        expected_shape: Expected shape tuple.
    """
    assert output.shape == expected_shape, (
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    )


def assert_collect_write_ops(
    layer: keras.Layer,
    op: types.OperatorInfo,
    *,
    expected_buffer_writes: int,
    expected_quant_writes: int,
) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
    """Call ``collect_write_ops`` and verify the number of emitted operations.

    Also verifies that the layer implements the ``Writable`` protocol.

    Args:
        layer: The Keras layer to call.
        op: The OperatorInfo used during build.
        expected_buffer_writes: Expected number of ``BufferWriteOp`` entries.
        expected_quant_writes: Expected number of ``QuantizationWriteOp`` entries.

    Returns:
        The ``(buffer_writes, quant_writes)`` tuple for further inspection.
    """
    assert_layer_is_writable(layer)
    buf_writes, quant_writes = layer.collect_write_ops(op)

    assert len(buf_writes) == expected_buffer_writes, (
        f"BufferWriteOp count mismatch for {layer.name!r}: "
        f"expected {expected_buffer_writes}, got {len(buf_writes)}"
    )
    assert len(quant_writes) == expected_quant_writes, (
        f"QuantizationWriteOp count mismatch for {layer.name!r}: "
        f"expected {expected_quant_writes}, got {len(quant_writes)}"
    )

    return buf_writes, quant_writes


def assert_quant_write_tensor_indices(
    quant_writes: list[types.QuantizationWriteOp],
    expected_indices: set[int],
) -> None:
    """Assert that quant writes target exactly the expected tensor indices.

    Args:
        quant_writes: List of ``QuantizationWriteOp`` to inspect.
        expected_indices: Set of tensor indices that should be written.
    """
    actual_indices = {qw.tensor_index for qw in quant_writes}
    assert actual_indices == expected_indices, (
        f"QuantizationWriteOp tensor index mismatch.\n"
        f"  Expected: {sorted(expected_indices)}\n"
        f"  Actual:   {sorted(actual_indices)}"
    )


def assert_buffer_write_tensor_indices(
    buffer_writes: list[types.BufferWriteOp],
    expected_indices: set[int],
) -> None:
    """Assert that buffer writes target exactly the expected tensor indices.

    Args:
        buffer_writes: List of ``BufferWriteOp`` to inspect.
        expected_indices: Set of tensor indices that should be written.
    """
    actual_indices = {bw.tensor_index for bw in buffer_writes}
    assert actual_indices == expected_indices, (
        f"BufferWriteOp tensor index mismatch.\n"
        f"  Expected: {sorted(expected_indices)}\n"
        f"  Actual:   {sorted(actual_indices)}"
    )


def verify_model_outputs(
    model_path: pathlib.Path | str,
    x_train: np.ndarray | list[np.ndarray],
    run_interpreter: typing.Callable,
    atol: float = 0.001,
) -> None:
    """Verify that the tunner Keras model output matches LiteRT Interpreter.

    This helper loads the exported TFLite model, runs it through Keras,
    compares predictions, saves it back, and compares predictions again.

    Args:
        model_path: Path to the .tflite model file.
        x_train: Input data for prediction.
        run_interpreter: The pytest fixture for running the LiteRT Interpreter.
        atol: allowed percentage of error for float outputs 0.001 means 0.1% error
    """
    # Get original LiteRT output
    litert_outputs = run_interpreter(model_path, x_train)

    # Compare original LiteRT model with Keras parsed
    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)

    # Under mixed precision, Keras outputs may be float16. Cast to float32
    # for comparison with interpreter outputs (always float32).
    if isinstance(keras_outputs, list):
        keras_outputs = [np.asarray(o, dtype=np.float32) for o in keras_outputs]
    else:
        keras_outputs = np.asarray(keras_outputs, dtype=np.float32)

    if isinstance(litert_outputs, list):
        assert isinstance(keras_outputs, list), "Expected list of Keras outputs"
        for litert_out, keras_out in zip(litert_outputs, keras_outputs, strict=True):
            litert_arr = cast("np.ndarray", litert_out)
            keras_arr = cast("np.ndarray", keras_out)
            max_val = np.max(np.abs(litert_arr))
            np.testing.assert_allclose(
                litert_arr / max_val,
                keras_arr / max_val,
                atol=atol,
            )
    else:
        assert isinstance(litert_outputs, np.ndarray), "Expected numpy array from LiteRT"
        max_val = np.max(np.abs(litert_outputs))
        np.testing.assert_allclose(
            cast("Any", litert_outputs / max_val),
            cast("Any", keras_outputs / max_val),
            atol=atol,
        )

    # Save the model and make sure the outputs are still the same
    litert_tunner.save_model(keras_model, str(model_path))
    litert_saved_outputs = run_interpreter(model_path, x_train)

    if isinstance(litert_saved_outputs, list):
        assert isinstance(keras_outputs, list), "Expected list of Keras outputs"
        for saved_out, keras_out in zip(litert_saved_outputs, keras_outputs, strict=True):
            np.testing.assert_allclose(cast("Any", keras_out), cast("Any", saved_out), atol=1e-5)
    else:
        assert isinstance(litert_saved_outputs, np.ndarray), "Expected numpy array from LiteRT"
        np.testing.assert_allclose(litert_outputs, litert_saved_outputs, atol=1e-5)


def verify_model_contains_operator(
    model_path: pathlib.Path | str,
    op_type: str,
) -> None:
    """Verify that the exported TFLite model contains the specified operator type.

    Args:
        model_path: Path to the .tflite model file.
        op_type: The expected operator type string (e.g. "ADD").
    """
    graph_def = flatbuffer.parse_tflite(model_path)
    op_types = {op.op_type for op in graph_def.operators}
    assert op_type in op_types, (
        f"Expected operator {op_type!r} not found in model.\nFound operators: {sorted(op_types)}"
    )
