from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, TypeAlias, overload

import keras
import numpy as np
import tensorflow as tf

RepresentativeDataset: TypeAlias = Callable[
    [], tf.data.Dataset | Iterable[list[np.ndarray] | dict[str, np.ndarray]]
]


@overload
def export_litert_model(
    model: keras.Model,
    export_dir: str | Path,
    *,
    quantization: Literal["int8", "float32"] = "int8",
    float_io: bool = True,
    representative_dataset: RepresentativeDataset | None = None,
    run_debugger: Literal[False] = False,
    denylisted_ops: None = None,
    denylisted_nodes: None = None,
) -> Path: ...


@overload
def export_litert_model(
    model: keras.Model,
    export_dir: str | Path,
    *,
    quantization: Literal["int8", "float32"] = "int8",
    float_io: bool = True,
    representative_dataset: RepresentativeDataset | None = None,
    run_debugger: Literal[True],
    denylisted_ops: list[str] | None = None,
    denylisted_nodes: list[str] | None = None,
) -> tuple[Path, "tf.lite.experimental.QuantizationDebugger"]: ...


def export_litert_model(
    model: keras.Model,
    export_dir: str | Path,
    *,
    quantization: Literal["int8", "float32"] = "int8",
    float_io: bool = True,
    representative_dataset: RepresentativeDataset | None = None,
    run_debugger: bool = False,
    denylisted_ops: list[str] | None = None,
    denylisted_nodes: list[str] | None = None,
) -> Path | tuple[Path, "tf.lite.experimental.QuantizationDebugger"]:
    """Export a Keras model to a TFLite model.

    Args:
        model: The Keras model to export.
        export_dir: Directory where the exported model and stats will be saved.
        quantization: The quantization mode. Either "int8" or "float32". Defaults to "int8".
        float_io: If True, use float32 inputs/outputs even for int8 quantization. Defaults to True.
        representative_dataset: A generator yielding sample inputs for quantization.
            Required if quantization is "int8".
        run_debugger: If True, run the TFLite QuantizationDebugger and save stats.
            Only supported for int8 quantization. Defaults to False.
        denylisted_ops: List of ops to exclude from quantization.
        denylisted_nodes: List of nodes to exclude from quantization.

    Returns:
        The path to the exported .tflite model. If run_debugger is True, also
        returns the QuantizationDebugger instance.
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

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

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

        with (export_dir / "quantization_stats.csv").open("w") as f:
            debugger.layer_statistics_dump(f)
    else:
        tflite_model = converter.convert()

    tflite_model_filepath = export_dir / "model.tflite"
    tflite_model_filepath.write_bytes(tflite_model)

    if run_debugger:
        return tflite_model_filepath, debugger
    return tflite_model_filepath
