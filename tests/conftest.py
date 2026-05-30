"""Shared test fixtures and helper utilities for litert_tunner."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Fixture returning a temporary directory for models."""
    return tmp_path


@pytest.fixture
def make_dense_tflite(temp_model_dir: Path) -> Callable:
    """Fixture returning a function to create fully quantized INT8 FullyConnected TFLite models."""

    def _make(
        num_features: int = 8,
        num_units: int = 1,
        use_bias: bool = True,
        activation: str | None = None,
        float_io: bool = False,
        seed: int = 42,
    ) -> Path:
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Build Keras model
        inputs = tf.keras.Input(shape=(num_features,))
        outputs = tf.keras.layers.Dense(
            units=num_units,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.5, 0.5),
            bias_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
        )(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Define representative dataset for quantization calibration
        def representative_dataset_gen():
            for _ in range(100):
                # Generates values within [-1.0, 1.0]
                yield [np.random.uniform(-1.0, 1.0, (1, num_features)).astype(np.float32)]

        # Convert to TFLite fully quantized INT8
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        if not float_io:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # Save to temp path
        output_path = temp_model_dir / f"dense_{num_units}_{activation}_{float_io}.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        return output_path

    return _make


@pytest.fixture
def run_interpreter() -> Callable:
    """Fixture returning a function to run LiteRT/TFLite Interpreter on a model."""

    def _run(model_path: Path | str, input_data: np.ndarray) -> np.ndarray:
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Handle INT8 input type conversion / scaling
        in_dtype = input_details[0]["dtype"]
        if in_dtype == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            if scale > 0:
                quant_input = np.round(input_data / scale) + zero_point
                quant_input = np.clip(quant_input, -128, 127).astype(np.int8)
                interpreter.set_tensor(input_details[0]["index"], quant_input)
            else:
                interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.int8))
        else:
            interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        return output_data

    return _run
