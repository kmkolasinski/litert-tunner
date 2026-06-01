"""Tests for RESIZE_NEAREST_NEIGHBOR operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.conftest import export_quantized_tflite_model
from tests.ops import op_test_utils


@pytest.fixture
def resize_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal RESIZE_NEAREST_NEIGHBOR op."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 2, 2, 3),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    size_data = np.array([4, 4], dtype=np.int32)
    size_tensor = op_test_utils.make_tensor(
        name="size_int32", index=1, shape=(2,), dtype=types.DTYPE_INT32, data=size_data
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=2,
        shape=(1, 4, 4, 3),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, size_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="RESIZE_NEAREST_NEIGHBOR",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestResizeNearestNeighborBuild:
    def test__resize_is_registered(self):
        assert "RESIZE_NEAREST_NEIGHBOR" in registry.registered_ops()

    def test__build_returns_keras_layer(self, resize_setup):
        op, tensors = resize_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, resize_setup):
        op, tensors = resize_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")

    def test__build_raises_without_constant_size(self, resize_setup):
        op, tensors = resize_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="size_int32",
            index=1,
            shape=(2,),
            dtype=types.DTYPE_INT32,
            data=None,
        )
        with pytest.raises(ValueError, match="requires a constant size tensor"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


class TestResizeNearestNeighborCall:
    def test__output_shape_matches_expected(self, resize_setup):
        op, tensors = resize_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 2, 2, 3)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4, 4, 3))

    def test__resize_formula_matches_expected(self, resize_setup):
        op, tensors = resize_setup
        # Input 2x2, resize to 4x4 using nearest neighbor. Each pixel is 2x2.
        input_data = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
        # shape of tensor is (1, 2, 2, 1)
        # Change setup tensors to fit input shape for testing formula
        tensors_list = list(tensors)
        tensors_list[0] = op_test_utils.make_tensor(
            name="input_int8",
            index=0,
            shape=(1, 2, 2, 1),
            dtype=types.DTYPE_INT8,
            quantization=tensors[0].quantization,
        )
        _, output = op_test_utils.build_and_call(op, tuple(tensors_list), input_data)

        expected = np.array(
            [
                [
                    [[1.0], [1.0], [2.0], [2.0]],
                    [[1.0], [1.0], [2.0], [2.0]],
                    [[3.0], [3.0], [4.0], [4.0]],
                    [[3.0], [3.0], [4.0], [4.0]],
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test__not_writable(self, resize_setup):
        op, tensors = resize_setup
        input_data = np.zeros((1, 2, 2, 3), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_layer_not_writable(layer)


def test__resize_nearest_neighbor_integration(temp_model_dir, run_interpreter):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(2, 2, 3))
    # Use keras.ops.image.resize to map directly to RESIZE_NEAREST_NEIGHBOR without EXPAND_DIMS
    outputs = keras.layers.Lambda(
        lambda x: keras.ops.image.resize(x, [4, 4], interpolation="nearest")
    )(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 2, 2, 3)

    output_path = temp_model_dir / "resize_integration.tflite"
    export_quantized_tflite_model(input_shape[1:], model, True, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)
    op_test_utils.verify_model_contains_operator(output_path, "RESIZE_NEAREST_NEIGHBOR")
