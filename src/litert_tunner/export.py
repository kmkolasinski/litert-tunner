from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, TypeAlias

import keras
import numpy as np
import tensorflow as tf

RepresentativeDataset: TypeAlias = Callable[
    [], tf.data.Dataset | Iterable[list[np.ndarray] | dict[str, np.ndarray]]
]


def export_litert_model(
    model: keras.Model,
    save_path: str | Path,
    *,
    quantization: Literal["int8", "float32"] = "int8",
    float_io: bool = True,
    representative_dataset: RepresentativeDataset | None = None,
    run_debugger: bool = False,
    denylisted_ops: list[str] | None = None,
    denylisted_nodes: list[str] | None = None,
) -> "tf.lite.experimental.QuantizationDebugger | None":
    """Export a Keras model to a TFLite model.

    Args:
        model: The Keras model to export.
        save_path: Full path where the exported model.tflite will be saved. Stats will
            be saved in the same directory.
        quantization: The quantization mode. Either "int8" or "float32". Defaults to "int8".
        float_io: If True, use float32 inputs/outputs even for int8 quantization. Defaults to True.
        representative_dataset: A generator yielding sample inputs for quantization.
            Required if quantization is "int8".
        run_debugger: If True, run the TFLite QuantizationDebugger and save stats.
            Only supported for int8 quantization. Defaults to False.
        denylisted_ops: List of ops to exclude from quantization.
        denylisted_nodes: List of nodes to exclude from quantization.

    Returns:
        The QuantizationDebugger instance (if run_debugger is True, otherwise None).
    """
    if (denylisted_ops or denylisted_nodes) and not run_debugger:
        raise ValueError(
            "denylisted_ops and denylisted_nodes can only be provided when run_debugger is True"
        )
    if quantization == "int8" and representative_dataset is None:
        raise ValueError("representative_dataset must be provided for int8 quantization")
    if quantization == "float32" and run_debugger:
        raise ValueError("Debugger only supported for int8 quantization")
    if quantization not in ("int8", "float32"):
        msg = f"Unknown quantization: {quantization}"
        raise ValueError(msg)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    match quantization:
        case "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            if denylisted_ops or denylisted_nodes:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        case "float32":
            pass

    io_type = tf.float32 if float_io or quantization == "float32" else tf.int8
    converter.inference_input_type = io_type
    converter.inference_output_type = io_type

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    debugger = None
    if run_debugger:
        debug_options = tf.lite.experimental.QuantizationDebugOptions(
            denylisted_ops=denylisted_ops,
            denylisted_nodes=denylisted_nodes,
        )
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=converter,
            debug_dataset=representative_dataset,
            debug_options=debug_options,
        )
        debugger.run()
        tflite_model = debugger.get_nondebug_quantized_model()

        with save_path.with_name("quantization_stats.csv").open("w") as f:
            debugger.layer_statistics_dump(f)
    else:
        tflite_model = converter.convert()

    save_path.write_bytes(tflite_model)

    return debugger
