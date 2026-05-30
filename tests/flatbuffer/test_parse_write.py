import keras
import numpy as np
import pytest
import tensorflow as tf

from litert_tunner.flatbuffer import parser, writer


def create_mlp_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def create_cnn_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(16, (3, 3), activation="relu")(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def create_cnn_skip_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    # Skip connection
    y = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.Add()([x, y])
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def create_efficientnet_model():
    return keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,  # type: ignore
        input_shape=(32, 32, 3),
        classes=10,
    )


def convert_to_tflite_int8(model, path):
    def representative_dataset():
        input_shape = [1 if d is None else d for d in model.input_shape]
        for _ in range(10):
            yield [np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(path, "wb") as f:
        f.write(tflite_model)


@pytest.fixture(scope="module")
def mlp_tflite_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "mlp.tflite"
    convert_to_tflite_int8(create_mlp_model(), path)
    return path


@pytest.fixture(scope="module")
def cnn_tflite_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "cnn.tflite"
    convert_to_tflite_int8(create_cnn_model(), path)
    return path


@pytest.fixture(scope="module")
def cnn_skip_tflite_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "cnn_skip.tflite"
    convert_to_tflite_int8(create_cnn_skip_model(), path)
    return path


@pytest.fixture(scope="module")
def efficientnet_tflite_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "efficientnet.tflite"
    convert_to_tflite_int8(create_efficientnet_model(), path)
    return path


@pytest.mark.parametrize(
    "model_fixture",
    [
        "mlp_tflite_path",
        "cnn_tflite_path",
        "cnn_skip_tflite_path",
        "efficientnet_tflite_path",
    ],
)
def test_parse_write_idempotent(model_fixture, request, tmp_path):
    tflite_path = request.getfixturevalue(model_fixture)

    # 1. Parse TFLite to get GraphDef
    graph_def = parser.parse_tflite(tflite_path)

    # 2. Create a dummy Keras model to hold the _graph_def
    # This simulates a model that has no writable layers
    dummy_model = keras.Model()
    dummy_model._graph_def = graph_def

    # 3. Write it back
    out_path = tmp_path / "out.tflite"
    writer.save_tflite(dummy_model, out_path)

    # 4. Verify identical bytes
    original_bytes = tflite_path.read_bytes()
    saved_bytes = out_path.read_bytes()

    assert original_bytes == saved_bytes, f"Parse+Write is not idempotent for {model_fixture}"
