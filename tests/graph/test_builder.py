import tensorflow as tf

import litert_tunner


def test__load_model_sanitizes_tensor_names(make_dense_tflite):
    """Verify loaded models have sanitized input names, preventing TFLiteConverter crashes."""
    # make_dense_tflite naturally generates invalid input names like serving_default
    model_path = make_dense_tflite(num_features=4, num_units=2)

    # Load the model
    tunner_model = litert_tunner.load_model(str(model_path))

    # Verify that the input name was sanitized (no colons)
    for inp in tunner_model.inputs:
        assert ":" not in inp.name

    # Verify that from_keras_model does not crash
    # (this used to raise TypeError due to the invalid name)
    converter = tf.lite.TFLiteConverter.from_keras_model(tunner_model)
    tflite_model = converter.convert()
    assert tflite_model is not None
