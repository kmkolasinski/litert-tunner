from collections.abc import Callable

import numpy as np

import litert_tunner


def test__mlp_single_layer_forward(make_mlp_tflite: Callable, run_interpreter: Callable):
    model_path = make_mlp_tflite(
        input_size=4,
        hidden_sizes=[8],
        use_bias=True,
        activation="relu",
        float_io=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (32, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    # compare original LiteRT model with keras parsed
    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)

    # save the model and make sure the outputs are still the same
    litert_tunner.save_model(keras_model, str(model_path))
    litert_saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(keras_outputs, litert_saved_outputs, atol=1e-3)


def test__mlp_multiple_layers_forward(make_mlp_tflite: Callable, run_interpreter: Callable):
    model_path = make_mlp_tflite(
        input_size=32,
        hidden_sizes=[32, 16, 8],
        use_bias=True,
        activation="swish",
        float_io=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (22, 32)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    # compare original LiteRT model with keras parsed
    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)

    # save the model and make sure the outputs are still the same
    litert_tunner.save_model(keras_model, str(model_path))
    litert_saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(keras_outputs, litert_saved_outputs, atol=1e-3)


def test__mlp_multiple_layers_batchnorm_forward(
    make_mlp_tflite: Callable, run_interpreter: Callable
):
    """Test forward pass of MLP with Batch Normalization folding."""
    model_path = make_mlp_tflite(
        input_size=32,
        hidden_sizes=[32, 16, 8],
        use_bias=True,
        activation="relu",
        float_io=True,
        add_batchnorm=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (22, 32)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    # compare original LiteRT model with keras parsed
    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)

    # save the model and make sure the outputs are still the same
    litert_tunner.save_model(keras_model, str(model_path))
    litert_saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(keras_outputs, litert_saved_outputs, atol=1e-3)


def test__mlp_multiple_layers_with_skip_connections_forward(
    make_mlp_tflite: Callable, run_interpreter: Callable
):
    model_path = make_mlp_tflite(
        input_size=32,
        hidden_sizes=[32, 16, 8],
        use_bias=True,
        activation="relu",
        float_io=True,
        add_skip_connections=True,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (22, 32)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_train)

    # compare original LiteRT model with keras parsed
    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)

    # save the model and make sure the outputs are still the same
    litert_tunner.save_model(keras_model, str(model_path))
    litert_saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(keras_outputs, litert_saved_outputs, atol=1e-3)
