"""Op registry for litert_tunner.

Maps TFLite operator type strings to builder functions that create
the corresponding Keras layers. New ops are registered via the
@register_op decorator.
"""

from __future__ import annotations

from collections.abc import Callable

import keras

from litert_tunner.graph import types

# Type alias for op builder functions.
# An op builder receives the OperatorInfo and a list of TensorInfos
# (all tensors in the graph) and returns a Keras layer.
OpBuilder = Callable[[types.OperatorInfo, tuple[types.TensorInfo, ...]], keras.Layer]

# Internal registry mapping op type string → builder function.
_REGISTRY: dict[str, OpBuilder] = {}


def register_op(op_type: str) -> Callable[[OpBuilder], OpBuilder]:
    """Decorator to register an op builder function.

    Args:
        op_type: TFLite operator type string (e.g., "FULLY_CONNECTED").

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If an op with the same type is already registered.

    Example:
        @register_op("FULLY_CONNECTED")
        def build_dense(op: OperatorInfo, tensors: tuple[TensorInfo, ...]) -> keras.Layer:
            ...
    """

    def decorator(fn: OpBuilder) -> OpBuilder:
        """Register the builder function for the specified op type."""
        if op_type in _REGISTRY:
            msg = f"Op '{op_type}' is already registered"
            raise ValueError(msg)
        _REGISTRY[op_type] = fn
        return fn

    return decorator


def get_op_builder(op_type: str) -> OpBuilder:
    """Look up a registered op builder by TFLite op name.

    Args:
        op_type: TFLite operator type string.

    Returns:
        The registered builder function.

    Raises:
        KeyError: If no builder is registered for the given op type.
    """
    if op_type not in _REGISTRY:
        registered = ", ".join(sorted(_REGISTRY.keys()))
        msg = f"Unsupported op type '{op_type}'. Registered ops: [{registered}]"
        raise KeyError(msg)
    return _REGISTRY[op_type]


def registered_ops() -> list[str]:
    """Return a sorted list of all registered op type strings."""
    return sorted(_REGISTRY.keys())


def reset_registry() -> None:
    """Clear all registered ops. Only use in tests."""
    _REGISTRY.clear()
