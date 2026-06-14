"""Tests for pooling operators (MAX_POOL_2D)."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils

# ---------------------------------------------------------------------------
# Fixtures — MAX_POOL_2D
# ---------------------------------------------------------------------------


@pytest.fixture
def max_pool_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal MAX_POOL_2D op."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=1,
        shape=(1, 2, 2, 2),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="MAX_POOL_2D",
        input_indices=(0,),
        output_indices=(1,),
        options={
            "FilterHeight": 2,
            "FilterWidth": 2,
            "StrideH": 2,
            "StrideW": 2,
            "Padding": 1,  # VALID
        },
    )
    return op, tensors


# ===================================================================
# MAX_POOL_2D build tests
# ===================================================================


class TestMaxPool2DBuild:
    """Tests for the MAX_POOL_2D op builder."""

    def test__max_pool_2d_is_registered(self):
        """MAX_POOL_2D must be present in the op registry."""
        assert "MAX_POOL_2D" in registry.registered_ops()

    def test__build_returns_keras_layer(self, max_pool_setup):
        """The builder must return a Keras layer."""
        op, tensors = max_pool_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, max_pool_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = max_pool_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )


# ===================================================================
# MAX_POOL_2D call tests
# ===================================================================


class TestMaxPool2DCall:
    """Tests for calling the MAX_POOL_2D layer."""

    def test__output_shape_matches_expected(self, max_pool_setup):
        """Output shape must match expected pooling output shape."""
        op, tensors = max_pool_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 2, 2, 2))

    def test__max_pool_selects_maximum(self, max_pool_setup):
        """Max pool must select the maximum value in each window."""
        op, tensors = max_pool_setup
        # Create input where max values are known
        input_data = np.array(
            [
                [
                    [[1, 2], [3, 4], [5, 6], [7, 8]],
                    [[9, 10], [11, 12], [13, 14], [15, 16]],
                    [[17, 18], [19, 20], [21, 22], [23, 24]],
                    [[25, 26], [27, 28], [29, 30], [31, 32]],
                ]
            ],
            dtype=np.float32,
        )
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        # With 2x2 pool and stride 2 (VALID):
        # top-left  window: max([1,3,9,11], [2,4,10,12]) = [11, 12]
        # top-right window: max([5,7,13,15], [6,8,14,16]) = [15, 16]
        # bot-left  window: max([17,19,25,27], [18,20,26,28]) = [27, 28]
        # bot-right window: max([21,23,29,31], [22,24,30,32]) = [31, 32]
        expected = np.array(
            [[[[11, 12], [15, 16]], [[27, 28], [31, 32]]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected, atol=1e-5)


# ===================================================================
# MAX_POOL_2D not-writable test
# ===================================================================


class TestMaxPool2DNotWritable:
    """Tests that MAX_POOL_2D does not implement Writable."""

    def test__is_not_writable(self, max_pool_setup):
        """MAX_POOL_2D layer must NOT implement the Writable protocol."""
        op, tensors = max_pool_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_layer_not_writable(layer)

    def test__no_trainable_weights(self, max_pool_setup):
        """MAX_POOL_2D layer must have no trainable weights."""
        op, tensors = max_pool_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        assert len(layer.trainable_weights) == 0


@pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__pool_integration(temp_model_dir, run_interpreter, quantization: str, dtype_policy: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(8, 8, 3))
    outputs = keras.layers.MaxPooling2D()(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 8, 8, 3)

    output_path = temp_model_dir / f"{quantization}_{dtype_policy}_pool_integration.tflite"
    conftest.export_tflite_model(
        input_shape=input_shape[1:],
        model=model,
        quantization=quantization,
        float_io=True,
        output_path=output_path,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    original_policy = keras.config.dtype_policy()
    try:
        keras.config.set_dtype_policy(dtype_policy)
        atol = op_test_utils.get_default_atol(dtype_policy)
        op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter, atol=atol)
    finally:
        keras.config.set_dtype_policy(original_policy)

    op_test_utils.verify_model_contains_operator(output_path, "MAX_POOL_2D")
