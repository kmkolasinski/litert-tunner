"""Tests for MEAN operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests.ops import op_test_utils

# ---------------------------------------------------------------------------
# Fixtures — MEAN
# ---------------------------------------------------------------------------


@pytest.fixture
def mean_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal MEAN op that reduces spatial dims [1, 2] (GlobalAvgPool)."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 4, 4, 3),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    # Axis tensor: constant INT32 with values [1, 2]
    axis_data = np.array([1, 2], dtype=np.int32)
    axis_tensor = op_test_utils.make_tensor(
        name="axis",
        index=1,
        shape=(2,),
        dtype=types.DTYPE_INT32,
        data=axis_data,
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.2], zero_points=[3])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=2,
        shape=(1, 3),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, axis_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="MEAN",
        input_indices=(0, 1),
        output_indices=(2,),
        options={"KeepDims": False},
    )
    return op, tensors


@pytest.fixture
def mean_keepdims_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a MEAN op with KeepDims=True."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[0])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 4, 4, 3),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    axis_data = np.array([1, 2], dtype=np.int32)
    axis_tensor = op_test_utils.make_tensor(
        name="axis",
        index=1,
        shape=(2,),
        dtype=types.DTYPE_INT32,
        data=axis_data,
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[0])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=2,
        shape=(1, 1, 1, 3),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, axis_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="MEAN",
        input_indices=(0, 1),
        output_indices=(2,),
        options={"KeepDims": True},
    )
    return op, tensors


# ===================================================================
# MEAN build tests
# ===================================================================


class TestMeanBuild:
    """Tests for the MEAN op builder."""

    def test__mean_is_registered(self):
        """MEAN must be present in the op registry."""
        assert "MEAN" in registry.registered_ops()

    def test__build_returns_keras_layer(self, mean_setup):
        """The builder must return a Keras layer."""
        op, tensors = mean_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, mean_setup):
        """Layer name must contain the output tensor index for writer lookup."""
        op, tensors = mean_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}"), (
            f"Layer name {layer.name!r} must end with '_{output_idx}'"
        )

    def test__build_raises_without_axis_data(self, mean_setup):
        """Builder must raise if the axis tensor has no data."""
        op, tensors = mean_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="axis",
            index=1,
            shape=(2,),
            dtype=types.DTYPE_INT32,
            data=None,
        )
        with pytest.raises(ValueError, match="has no data"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


# ===================================================================
# MEAN call tests
# ===================================================================


class TestMeanCall:
    """Tests for calling the MEAN layer."""

    def test__output_shape_no_keepdims(self, mean_setup):
        """Output shape with KeepDims=False reduces spatial dims."""
        op, tensors = mean_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 3))

    def test__output_shape_keepdims(self, mean_keepdims_setup):
        """Output shape with KeepDims=True retains spatial dims as 1."""
        op, tensors = mean_keepdims_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (2, 1, 1, 3))

    def test__output_values_in_int8_range(self, mean_setup):
        """Output values must be in the INT8 range [-128, 127]."""
        op, tensors = mean_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-10.0, 10.0, (1, 4, 4, 3)).astype(np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        assert output.min() >= -128.0
        assert output.max() <= 127.0


# ===================================================================
# MEAN trainable weight tests
# ===================================================================


class TestMeanTrainableWeights:
    """Tests for MEAN layer trainable parameters."""

    def test__trainable_weights(self, mean_setup):
        """MEAN layer must have trainable output_scale, output_zero_point."""
        op, tensors = mean_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 3), dtype=np.float32)
        )
        op_test_utils.assert_trainable_weight_names(layer, {"output_scale", "output_zero_point"})

    def test__non_trainable_weights(self, mean_setup):
        """MEAN layer must have frozen input_scale, input_zero_point."""
        op, tensors = mean_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 3), dtype=np.float32)
        )
        op_test_utils.assert_non_trainable_weight_names(layer, {"input_scale", "input_zero_point"})


# ===================================================================
# MEAN write ops tests
# ===================================================================


class TestMeanWriteOps:
    """Tests for MEAN layer collect_write_ops."""

    def test__is_writable(self, mean_setup):
        """MEAN layer must implement the Writable protocol."""
        op, tensors = mean_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 3), dtype=np.float32)
        )
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, mean_setup):
        """MEAN must emit 0 buffer writes and 2 quant writes (input + output)."""
        op, tensors = mean_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 3), dtype=np.float32)
        )
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(self, mean_setup):
        """Quant writes must target input and output tensor indices."""
        op, tensors = mean_setup
        layer, _ = op_test_utils.build_and_call(
            op, tensors, np.zeros((1, 4, 4, 3), dtype=np.float32)
        )
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )
