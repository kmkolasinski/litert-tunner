"""Tests for operations with constant inputs."""

from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np

import litert_tunner
from litert_tunner import export, testing_utils
from tests import conftest


def test__mul_with_constant_input(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a model with MUL having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs * constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "mul_constant.tflite"
    conftest.export_quantized_tflite_model((4,), model, float_io=True, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)


def test__add_with_constant_input(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a model with ADD having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs + constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "add_constant.tflite"
    conftest.export_quantized_tflite_model((4,), model, float_io=True, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)


def test__mul_with_constant_input_float32(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a float32 model with MUL having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs * constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "mul_constant_f32.tflite"
    conftest.export_float32_tflite_model((4,), model, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-5)


def test__add_with_constant_input_float32(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a float32 model with ADD having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs + constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "add_constant_f32.tflite"
    conftest.export_float32_tflite_model((4,), model, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-5)


def test__concat_with_constant_input(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a model with CONCATENATION having a constant input."""
    inputs = keras.Input(shape=(4, 4, 1), batch_size=1)
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")(inputs)
    constant_val = np.array([[[[0.5, 1.5]]]], dtype=np.float32)
    constant_val = np.broadcast_to(constant_val, (1, 4, 4, 2))
    outputs = keras.layers.Concatenate(axis=-1)([x, constant_val])
    model = keras.Model(inputs=inputs, outputs=outputs)

    model_path = temp_model_dir / "concat_constant.tflite"
    conftest.export_quantized_tflite_model(
        (1, 4, 4, 1), model, float_io=True, output_path=model_path
    )

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (1, 4, 4, 1)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-5)

    def dummy_dataset():
        yield [np.zeros((1, 4, 4, 1), dtype=np.float32)]

    export.export_litert_model(
        keras_model,
        save_path=str(temp_model_dir / "exported.tflite"),
        quantization="int8",
        representative_dataset=dummy_dataset,
        run_debugger=False,
    )


def test__concat_with_constant_input_float32(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a float32 model with CONCATENATION having a constant input."""
    inputs = keras.Input(shape=(4, 4, 1), batch_size=1)
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")(inputs)
    constant_val = np.array([[[[0.5, 1.5]]]], dtype=np.float32)
    constant_val = np.broadcast_to(constant_val, (1, 4, 4, 2))
    outputs = keras.layers.Concatenate(axis=-1)([x, constant_val])
    model = keras.Model(inputs=inputs, outputs=outputs)

    model_path = temp_model_dir / "concat_constant_f32.tflite"
    conftest.export_float32_tflite_model((1, 4, 4, 1), model, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (1, 4, 4, 1)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-5)
