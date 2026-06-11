"""Graph builder for litert_tunner.

Converts a parsed GraphDef into a Keras Functional model using registered
op builders.
"""

import re

import keras

from litert_tunner.graph import types
from litert_tunner.ops import registry


def build_keras_model(graph_def: types.GraphDef) -> keras.Model:
    """Build a Keras model from the parsed graph definition.

    Args:
        graph_def: The parsed GraphDef.

    Returns:
        A trainable Keras Model replica of the TFLite graph.
    """
    # Import ops package to trigger registration of all ops
    import litert_tunner.ops  # noqa: F401, PLC0415

    tensor_symbols = {}

    # 1. Create Keras inputs for graph-level inputs
    for idx in graph_def.input_indices:
        tensor = graph_def.tensors[idx]
        # We use None for the batch dimension to allow flexible batch sizes during fine-tuning
        batch_shape = (None, *tensor.shape[1:]) if tensor.shape else (None,)

        # Sanitize tensor name to avoid invalid node name errors in TFLiteConverter
        safe_name = re.sub(r"[^A-Za-z0-9._/-]", "_", tensor.name) if tensor.name else None

        x = keras.Input(batch_shape=batch_shape, dtype="float32", name=safe_name)
        tensor_symbols[idx] = x

    # 2. Build layers and connect them topologically
    for op in graph_def.operators:
        builder = registry.get_op_builder(op.op_type)
        layer = builder(op, graph_def.tensors)

        layer_inputs = [
            tensor_symbols[idx]
            for idx in op.input_indices
            if idx >= 0 and graph_def.tensors[idx].data is None
        ]

        if not layer_inputs:
            msg = f"Operator {op.op_type} has no activation inputs"
            raise ValueError(msg)

        layer_output = layer(layer_inputs[0]) if len(layer_inputs) == 1 else layer(layer_inputs)

        # Assign output tensor symbols
        if len(op.output_indices) == 1:
            tensor_symbols[op.output_indices[0]] = layer_output
        else:
            if not isinstance(layer_output, (list, tuple)):
                msg = (
                    f"Operator {op.op_type} has multiple outputs but layer returned a single tensor"
                )
                raise ValueError(msg)
            for i, out_idx in enumerate(op.output_indices):
                tensor_symbols[out_idx] = layer_output[i]

    # 3. Construct Keras Model
    model_inputs = [tensor_symbols[idx] for idx in graph_def.input_indices]
    model_outputs = [tensor_symbols[idx] for idx in graph_def.output_indices]

    if len(model_outputs) == 1:
        model = keras.Model(inputs=model_inputs, outputs=model_outputs[0])
    else:
        model = keras.Model(inputs=model_inputs, outputs=model_outputs)

    # Attach GraphDef metadata to the model for later save
    model._graph_def = graph_def  # noqa: SLF001

    return model
