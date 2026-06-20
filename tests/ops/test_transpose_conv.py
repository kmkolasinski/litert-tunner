"""Tests for TRANSPOSE_CONV operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils

# ---------------------------------------------------------------------------
# Fixtures — TRANSPOSE_CONV (quantized)
# ---------------------------------------------------------------------------


@pytest.fixture
def transpose_conv_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal TRANSPOSE_CONV op with INT8 I/O and per-channel quantization.

    TRANSPOSE_CONV input indices:
        [0] output_shape (INT32 constant — has data in unit tests)
        [1] weights (INT8), shape (out_ch, kH, kW, in_ch)
        [2] input activation (INT8)
        [3] bias (INT32, optional)
    """
    # Output shape tensor (constant INT32, has data in flatbuffer)
    output_shape_data = np.array([1, 4, 4, 3], dtype=np.int32)
    output_shape_tensor = op_test_utils.make_tensor(
        name="output_shape",
        index=0,
        shape=(4,),
        dtype=types.DTYPE_INT32,
        data=output_shape_data,
    )

    # Per-channel weight quantization: 3 output channels
    weight_quant = op_test_utils.make_quant_params(
        scales=[0.2, 0.3, 0.1],
        zero_points=[0, 0, 0],
        quantized_dimension=0,
    )
    # Weight shape: (out_ch=3, kH=1, kW=1, in_ch=2) — 1x1 for simplicity
    weight_data = np.array(
        [
            [[[10, 20]]],
            [[[-10, -20]]],
            [[[5, 5]]],
        ],
        dtype=np.int8,
    )
    weight_tensor = op_test_utils.make_tensor(
        name="weight_int8",
        index=1,
        shape=(3, 1, 1, 2),
        dtype=types.DTYPE_INT8,
        quantization=weight_quant,
        data=weight_data,
    )

    # Input activation
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=2,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    # Bias: per-channel, INT32
    bias_data = np.array([1, -1, 0], dtype=np.int32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_int32", index=3, shape=(3,), dtype=types.DTYPE_INT32, data=bias_data
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=4,
        shape=(1, 4, 4, 3),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (output_shape_tensor, weight_tensor, input_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="TRANSPOSE_CONV",
        input_indices=(0, 1, 2, 3),
        output_indices=(4,),
        options={"Padding": 0, "StrideH": 1, "StrideW": 1},
    )
    return op, tensors


# ---------------------------------------------------------------------------
# Fixtures — TRANSPOSE_CONV (float32)
# ---------------------------------------------------------------------------


@pytest.fixture
def float_transpose_conv_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal TRANSPOSE_CONV op with float32 I/O (no quantization)."""
    # Output shape tensor (constant INT32, has data in unit tests)
    output_shape_data = np.array([1, 4, 4, 3], dtype=np.int32)
    output_shape_tensor = op_test_utils.make_tensor(
        name="output_shape",
        index=0,
        shape=(4,),
        dtype=types.DTYPE_INT32,
        data=output_shape_data,
    )

    weight_data = np.array(
        [
            [[[1.0, 2.0]]],
            [[[-1.0, -2.0]]],
            [[[0.5, 0.5]]],
        ],
        dtype=np.float32,
    )
    weight_tensor = op_test_utils.make_tensor(
        name="weight_f32",
        index=1,
        shape=(3, 1, 1, 2),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
        data=weight_data,
    )

    input_tensor = op_test_utils.make_tensor(
        name="input_f32",
        index=2,
        shape=(1, 4, 4, 2),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
    )

    bias_data = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    bias_tensor = op_test_utils.make_tensor(
        name="bias_f32", index=3, shape=(3,), dtype=types.DTYPE_FLOAT32, data=bias_data
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32",
        index=4,
        shape=(1, 4, 4, 3),
        dtype=types.DTYPE_FLOAT32,
        quantization=None,
    )

    tensors = (output_shape_tensor, weight_tensor, input_tensor, bias_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="TRANSPOSE_CONV",
        input_indices=(0, 1, 2, 3),
        output_indices=(4,),
        options={"Padding": 0, "StrideH": 1, "StrideW": 1},
    )
    return op, tensors


# ===================================================================
# TRANSPOSE_CONV build tests (quantized)
# ===================================================================


class TestTransposeConvBuild:
    """Tests for the TRANSPOSE_CONV op builder."""

    def test__transpose_conv_is_registered(self):
        """TRANSPOSE_CONV must be present in the op registry."""
        assert "TRANSPOSE_CONV" in registry.registered_ops()

    def test__build_returns_keras_layer(self, transpose_conv_setup):
        """The builder must return a Keras layer."""
        op, tensors = transpose_conv_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, transpose_conv_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = transpose_conv_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_weights(self, transpose_conv_setup):
        """Builder must raise if the weight tensor has no data."""
        op, tensors = transpose_conv_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_int8",
            index=1,
            shape=(3, 1, 1, 2),
            dtype=types.DTYPE_INT8,
            quantization=tensors[1].quantization,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


# ===================================================================
# TRANSPOSE_CONV call tests (quantized)
# ===================================================================


class TestTransposeConvCall:
    """Tests for calling the TRANSPOSE_CONV layer."""

    def test__output_shape_matches_expected(self, transpose_conv_setup):
        """Output shape must match expected transpose conv output shape."""
        op, tensors = transpose_conv_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 4, 4, 3))

    def test__output_values_in_int8_range(self, transpose_conv_setup):
        """Output values must be in the INT8 range [-128, 127]."""
        op, tensors = transpose_conv_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-10.0, 10.0, (1, 4, 4, 2)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        assert output.min() >= -128.0
        assert output.max() <= 127.0


# ===================================================================
# TRANSPOSE_CONV trainable weight tests (quantized)
# ===================================================================


class TestTransposeConvTrainableWeights:
    """Tests for TRANSPOSE_CONV layer trainable parameters."""

    def test__trainable_weights(self, transpose_conv_setup):
        """TRANSPOSE_CONV layer must have trainable bias, weight_int8, weight_scale."""
        op, tensors = transpose_conv_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_trainable_weight_names(layer, {"bias", "weight_int8", "weight_scale"})

    def test__non_trainable_weights(self, transpose_conv_setup):
        """TRANSPOSE_CONV layer must have frozen weights and I/O scales/zps."""
        op, tensors = transpose_conv_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input_scale",
                "input_zero_point",
                "output_scale",
                "output_zero_point",
                "weight_zero_point",
            },
        )


# ===================================================================
# TRANSPOSE_CONV write ops tests (quantized)
# ===================================================================


class TestTransposeConvWriteOps:
    """Tests for TRANSPOSE_CONV layer collect_write_ops."""

    def test__is_writable(self, transpose_conv_setup):
        """TRANSPOSE_CONV layer must implement the Writable protocol."""
        op, tensors = transpose_conv_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, transpose_conv_setup):
        """TRANSPOSE_CONV must emit 2 buffer writes (weight, bias) and 4 quant writes."""
        op, tensors = transpose_conv_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=4,
        )

    def test__write_ops_buffer_indices(self, transpose_conv_setup):
        """Buffer writes must target weight and bias tensor indices."""
        op, tensors = transpose_conv_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        buffer_writes, _ = layer.collect_write_ops(op)
        # TRANSPOSE_CONV: weight at index 1, bias at index 3
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[3]}
        )

    def test__write_ops_quant_indices(self, transpose_conv_setup):
        """Quant writes must target input, weight, bias, and output tensor indices."""
        op, tensors = transpose_conv_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 2), dtype=np.float32)
        )
        _, quant_writes = layer.collect_write_ops(op)
        # TRANSPOSE_CONV: input at index 2, weight at index 1, bias at index 3, output at index 4
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes,
            {op.input_indices[1], op.input_indices[2], op.input_indices[3], op.output_indices[0]},
        )


# ===================================================================
# Float TRANSPOSE_CONV build tests
# ===================================================================


class TestFloatTransposeConvBuild:
    def test__float_transpose_conv_build_returns_keras_layer(self, float_transpose_conv_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_transpose_conv_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_transpose_conv_layer_name_contains_output_index(
        self, float_transpose_conv_setup
    ):
        """Layer name must end with output tensor index."""
        op, tensors = float_transpose_conv_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")

    def test__float_transpose_conv_build_raises_without_weights(self, float_transpose_conv_setup):
        """Builder must raise if weight tensor has no data."""
        op, tensors = float_transpose_conv_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="weight_f32",
            index=1,
            shape=(3, 1, 1, 2),
            dtype=types.DTYPE_FLOAT32,
            quantization=None,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


# ===================================================================
# Float TRANSPOSE_CONV call tests
# ===================================================================


class TestFloatTransposeConvCall:
    def test__float_transpose_conv_output_shape(self, float_transpose_conv_setup):
        """Output shape must match expected shape."""
        op, tensors = float_transpose_conv_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 4, 4, 3))

    def test__float_transpose_conv_formula_matches_numpy(self, float_transpose_conv_setup):
        """Float32 op output must match numpy reference computation."""
        op, tensors = float_transpose_conv_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4, 4, 2)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)

        # 1x1 transpose conv with stride=1, same padding is equivalent to
        # matrix multiply along last axis: input @ kernel for each spatial position
        # TFLite weight: (out_ch=3, kH=1, kW=1, in_ch=2)
        weight_data = tensors[1].data
        bias_data = tensors[3].data
        # For 1x1 transpose conv: (batch, H, W, in_ch) @ (in_ch, out_ch) + bias
        w = weight_data.reshape(3, 2).T  # (in_ch=2, out_ch=3)
        expected = np.dot(input_data, w) + bias_data
        np.testing.assert_allclose(output, expected, atol=1e-5)


# ===================================================================
# Float TRANSPOSE_CONV trainable weight tests
# ===================================================================


class TestFloatTransposeConvTrainableWeights:
    def test__float_transpose_conv_trainable_weights(self, float_transpose_conv_setup):
        op, tensors = float_transpose_conv_setup
        dummy_input = np.zeros((1, 4, 4, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_trainable_weight_names(layer, {"kernel", "bias"})

    def test__float_transpose_conv_non_trainable_weights(self, float_transpose_conv_setup):
        op, tensors = float_transpose_conv_setup
        dummy_input = np.zeros((1, 4, 4, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_non_trainable_weight_names(layer, set())


# ===================================================================
# Float TRANSPOSE_CONV write ops tests
# ===================================================================


class TestFloatTransposeConvWriteOps:
    def test__float_transpose_conv_is_writable(self, float_transpose_conv_setup):
        op, tensors = float_transpose_conv_setup
        dummy_input = np.zeros((1, 4, 4, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_layer_is_writable(layer)

    def test__float_transpose_conv_write_ops_counts(self, float_transpose_conv_setup):
        op, tensors = float_transpose_conv_setup
        dummy_input = np.zeros((1, 4, 4, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=2,
            expected_quant_writes=0,
        )

    def test__float_transpose_conv_write_ops_buffer_indices(self, float_transpose_conv_setup):
        op, tensors = float_transpose_conv_setup
        dummy_input = np.zeros((1, 4, 4, 2), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        buffer_writes, _ = layer.collect_write_ops(op)
        # TRANSPOSE_CONV: weight at index 1, bias at index 3
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[3]}
        )


# ===================================================================
# Integration tests
# ===================================================================


@pytest.mark.parametrize("quantization", ["int8", "float32"])
@pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
def test__transpose_conv_integration(
    temp_model_dir, run_interpreter, quantization: str, dtype_policy: str
):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4, 4, 3))
    outputs = keras.layers.Conv2DTranspose(8, 3, padding="same")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4, 4, 3)

    output_path = (
        temp_model_dir / f"{quantization}_{dtype_policy}_transpose_conv_integration.tflite"
    )
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
        # Transpose conv has extra upstream ops (SHAPE→STRIDED_SLICE→PACK) that
        # accumulate more float16 rounding error than simple conv2d
        atol = op_test_utils.get_default_atol(dtype_policy)
        if "float16" in dtype_policy and quantization == "int8":
            atol = 0.02
        op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter, atol=atol)
    finally:
        keras.config.set_dtype_policy(original_policy)

    op_test_utils.verify_model_contains_operator(output_path, "TRANSPOSE_CONV")
