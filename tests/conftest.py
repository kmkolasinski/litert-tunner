"""Shared test fixtures and helper utilities for litert_tunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import keras
import numpy as np
import pytest
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter


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
        inputs = keras.Input(shape=(num_features,))
        outputs = keras.layers.Dense(
            units=num_units,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
            bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Save to temp path
        output_path = temp_model_dir / f"dense_{num_units}_{activation}_{float_io}.tflite"
        export_quantized_tflite_model((num_features,), model, float_io, output_path)

        return output_path

    return _make


@pytest.fixture
def make_mlp_tflite(temp_model_dir: Path) -> Callable:
    """Fixture returning a function to create fully quantized INT8 FullyConnected TFLite models."""

    def _make(
        input_size: int = 8,
        hidden_sizes: list[int] = [1],
        use_bias: bool = True,
        activation: str | None = None,
        float_io: bool = False,
        add_skip_connections: bool = False,
        add_batchnorm: bool = False,
        seed: int = 42,
    ) -> Path:
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Build Keras model
        inputs = keras.Input(shape=(input_size,))
        x = inputs
        num_layers = len(hidden_sizes)
        for layer_index, num_units in enumerate(hidden_sizes):
            residual: Any = x
            x = keras.layers.Dense(
                units=num_units,
                use_bias=use_bias,
                activation=None,
                kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
                bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
            )(x)
            if add_batchnorm:
                x = keras.layers.BatchNormalization()(x)
            # skip the last layer activation
            if layer_index != num_layers - 1:
                if activation is not None:
                    x = keras.layers.Activation(activation)(x)
            if add_skip_connections and residual.shape[-1] == num_units:
                x = keras.layers.Add()([residual, x])

        outputs = x
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        # Save to temp path
        sizes_str = "_".join(map(str, hidden_sizes))
        output_path = (
            temp_model_dir
            / f"mlp_{sizes_str}_{activation}_{float_io}_{add_skip_connections}_{add_batchnorm}.tflite"  # noqa: E501
        )
        export_quantized_tflite_model((input_size,), model, float_io, output_path)

        return output_path

    return _make


@pytest.fixture
def make_resnet_tflite(temp_model_dir: Path) -> Callable:
    """Fixture returning a function to create fully quantized INT8 ResNet-like CNN TFLite models."""

    def _make(
        input_shape: tuple[int, int, int] = (8, 8, 3),
        filters: list[int] = [8, 8],
        kernel_size: int | tuple[int, int] = 3,
        use_bias: bool = True,
        activation: str | None = None,
        float_io: bool = False,
        add_skip_connections: bool = True,
        add_batchnorm: bool = False,
        pooling_type: str | None = None,
        seed: int = 42,
    ) -> Path:
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Build Keras model
        inputs = keras.Input(shape=input_shape)
        x = inputs

        # First Conv layer
        x = keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
            bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
        )(x)
        if add_batchnorm:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)

        if pooling_type == "max":
            x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        elif pooling_type == "avg":
            x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

        # Build subsequent layers
        for f in filters[1:]:
            residual = x

            # Conv block
            x = keras.layers.Conv2D(
                filters=f,
                kernel_size=kernel_size,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
                bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
            )(x)
            if add_batchnorm:
                x = keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = keras.layers.Activation(activation)(x)

            x = keras.layers.Conv2D(
                filters=f,
                kernel_size=kernel_size,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
                bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
            )(x)
            if add_batchnorm:
                x = keras.layers.BatchNormalization()(x)

            if add_skip_connections:
                # If channel dimensions differ, project residual
                if residual.shape[-1] != f:
                    residual = keras.layers.Conv2D(
                        filters=f,
                        kernel_size=1,
                        padding="same",
                        use_bias=use_bias,
                        kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
                        bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
                    )(residual)
                    if add_batchnorm:
                        residual = keras.layers.BatchNormalization()(residual)
                x = keras.layers.Add()([residual, x])

            if activation is not None:
                x = keras.layers.Activation(activation)(x)

        # Classifier head: global pooling + Dense to logits
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(
            units=10,
            use_bias=use_bias,
            kernel_initializer=cast(Any, keras.initializers.RandomUniform(-0.5, 0.5)),
            bias_initializer=cast(Any, keras.initializers.RandomUniform(-0.1, 0.1)),
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        # Save to temp path
        filters_str = "_".join(map(str, filters))
        shape_str = "_".join(map(str, input_shape))
        output_path = (
            temp_model_dir
            / f"resnet_{shape_str}_{filters_str}_{activation}_{float_io}_{add_skip_connections}_{add_batchnorm}_{pooling_type}.tflite"  # noqa: E501
        )
        export_quantized_tflite_model(input_shape, model, float_io, output_path)

        return output_path

    return _make


def export_quantized_tflite_model(
    input_shape: tuple[int, ...], model: keras.Model, float_io: bool, output_path: Path
):
    def representative_dataset_gen():
        # Get shape from model input shape, replacing None or dynamic dimensions with 1
        rep_shape = [1 if d is None else d for d in model.input_shape]
        for _ in range(100):
            # Generates values within [-1.0, 1.0]
            yield [np.random.uniform(-1.0, 1.0, rep_shape).astype(np.float32)]

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
    with open(output_path, "wb") as f:
        f.write(tflite_model)


@pytest.fixture
def run_interpreter() -> Callable:
    """Fixture returning a function to run LiteRT/TFLite Interpreter on a model."""

    def _run(model_path: Path | str, input_data: np.ndarray) -> np.ndarray:
        interpreter = Interpreter(model_path=str(model_path))
        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(input_details[0]["index"], list(input_data.shape))
        interpreter.allocate_tensors()

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
