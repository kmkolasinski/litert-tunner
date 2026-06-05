# Float32 Layer Support — Implementation Recipe

This document is a self-contained recipe for adding float32 TFLite model support
to any op in `litert-tunner`. Follow it step-by-step for each layer migration.

## Context

`litert-tunner` originally supported only **fully-quantized INT8** TFLite models.
We are extending every op to also handle **float32 (unquantized)** TFLite models.

### What already exists

- **`types.is_quantized(tensor)`** — helper that checks if a `TensorInfo` has
  quantization params. Use this in builder dispatch.
- **`utils.get_float32_bias(op, tensors, output_units)`** — extracts float32 bias
  directly from a float32 model (no INT32 dequantization). Use for weighted ops.
- **`conftest.export_float32_tflite_model(input_shape, model, output_path)`** —
  exports a Keras model to an unquantized float32 `.tflite` file. Use in
  integration tests.
- **Reference implementation**: `FloatDense` in
  \[dense.py\](file:///home/krzysiek/DATA/GITHUB/litert-tunner/src/litert_tunner/ops/dense.py)
  and its tests in
  \[test_dense.py\](file:///home/krzysiek/DATA/GITHUB/litert-tunner/tests/ops/test_dense.py).

### Design principle: Builder-level dispatch, separate classes

Each op keeps its existing `Quantized*` layer untouched and gets a new `Float*`
layer. The registered builder function dispatches based on `types.is_quantized()`:

```python
@registry.register_op("OP_TYPE")
def build_op(op, tensors):
    input_tensor = tensors[op.input_indices[0]]
    if types.is_quantized(input_tensor):
        return _build_quantized_op(op, tensors)
    return _build_float_op(op, tensors)
```

This ensures **zero changes** to the existing INT8 path.

______________________________________________________________________

## Op Categories & What Each Float32 Layer Needs

There are 3 categories of ops. Each has different requirements:

### Category A: Weighted ops (have kernel/bias buffers)

**Ops**: `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`

These ops store weight data in the flatbuffer and need `Writable` to save
updated weights back.

| Aspect            | Quantized layer                                            | Float32 layer                                                               |
| ----------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| **`__init__`**    | Stores INT8 weights, quant params                          | Stores float32 kernel + bias + their original dtypes                        |
| **`build`**       | `add_weight` for INT8 weights + QuantizationVars           | `add_weight` for kernel (trainable) + bias (trainable)                      |
| **`call`**        | dequant → compute → fused activation → quant               | compute → fused activation (no quant)                                       |
| **`Writable`**    | Yes — emits `BufferWriteOp` (INT8) + `QuantizationWriteOp` | Yes — emits `BufferWriteOp` (original dtype) only, no `QuantizationWriteOp` |
| **Trainable**     | bias, weight_scale                                         | kernel, bias                                                                |
| **Non-trainable** | weight_int8, input/output/weight quant vars                | (none)                                                                      |

**Builder changes**:

- Move existing quant-param extraction into `_build_quantized_<op>()`
- New `_build_float_<op>()` uses `utils.get_float32_bias()` for bias
- Top-level builder dispatches on `types.is_quantized(input_tensor)`
- Weight data `None` check stays in top-level builder (before dispatch)
- Add `assert weight_tensor.data is not None  # noqa: S101` in private builders
  for type narrowing

**`collect_write_ops` for float32**:

```python
def collect_write_ops(self, op):
    buffer_writes = []
    op_inputs = op.input_indices

    # Write kernel with original dtype
    kernel_np = typing.cast("np.ndarray", ops.convert_to_numpy(self.kernel)).astype(
        self._kernel_dtype
    )
    buffer_writes.append(
        types.BufferWriteOp(tensor_index=op_inputs[1], data=bytes(kernel_np.tobytes()))
    )

    # Write bias (if present) with original dtype
    bias_index = 2
    if len(op_inputs) > bias_index and op_inputs[bias_index] >= 0:
        bias_np = typing.cast("np.ndarray", ops.convert_to_numpy(self.bias)).astype(
            self._bias_dtype
        )
        buffer_writes.append(
            types.BufferWriteOp(
                tensor_index=op_inputs[2], data=bytes(bias_np.tobytes())
            )
        )

    return buffer_writes, []  # empty quant writes
```

### Category B: Element-wise / binary ops (no weight buffers)

**Ops**: `ADD`, `SUB`, `MUL`, `DIV`, `SQUARED_DIFFERENCE`

These take 2 inputs and produce 1 output. Some may have a constant input.

| Aspect         | Quantized layer                                         | Float32 layer                                |
| -------------- | ------------------------------------------------------- | -------------------------------------------- |
| **`__init__`** | Stores quant params, optional constant input            | Stores optional constant float32 input       |
| **`build`**    | QuantizationVars for all I/O + optional constant weight | Optional constant weight only                |
| **`call`**     | dequant → op → fused activation → quant                 | op → fused activation                        |
| **`Writable`** | Yes — emits QuantizationWriteOp only                    | **No** — no buffers or quant params to write |
| **Trainable**  | (none by default)                                       | (none)                                       |

**Constant inputs**: In quantized models, constant data is INT8. In float32
models, constant data is already float32 — just store it directly via
`add_weight`. Check `tensor.data is not None` to detect constants.

**Float32 constant extraction pattern**:

```python
def _extract_float_constant(
    input1_tensor: types.TensorInfo,
    input2_tensor: types.TensorInfo,
) -> tuple[np.ndarray | None, int]:
    """Extract a constant float32 input tensor."""
    for idx, tensor in enumerate([input1_tensor, input2_tensor]):
        if tensor.data is not None:
            return tensor.data.astype(np.float32), idx
    return None, -1
```

> **TIP**: `ADD`, `SUB`, `MUL`, `DIV` follow identical patterns. Consider a
> shared `FloatBinaryOp` base class parameterized by the operation function
> (`ops.add`, `ops.subtract`, etc.).

### Category C: Unary ops (activation, reduction, misc)

**Ops**: `RELU`, `GELU`, `LOGISTIC`, `SOFTMAX`, `MEAN`, `NEG`, `RSQRT`,
`CONCATENATION`, `RESIZE_NEAREST_NEIGHBOR`

These apply a simple function to their input(s).

| Aspect         | Quantized layer                           | Float32 layer                   |
| -------------- | ----------------------------------------- | ------------------------------- |
| **`__init__`** | Stores quant params + op-specific options | Stores op-specific options only |
| **`build`**    | QuantizationVars for I/O                  | Nothing (or `super().build()`)  |
| **`call`**     | dequant → op → quant                      | just the op                     |
| **`Writable`** | Yes — emits QuantizationWriteOp only      | **No** — nothing to write       |
| **Trainable**  | (none by default)                         | (none)                          |

These are the simplest to migrate. Example for RELU:

```python
class FloatRelu(keras.Layer):
    """Float32 RELU — just applies ops.relu."""

    def call(self, x):
        return ops.relu(x)
```

**Note**: Float32 unary/activation layers do **not** implement `Writable` and
do **not** extend `types.Writable`. They have no state to persist.

### Ops that already work (no changes needed)

These are already dtype-agnostic passthroughs:

- `RESHAPE`, `TRANSPOSE`, `PACK`, `STRIDED_SLICE`, `SHAPE` — shape ops
- `MAX_POOL_2D` — pooling passthrough
- `QUANTIZE`, `DEQUANTIZE` — only appear in quantized models

______________________________________________________________________

## Naming Conventions

| Item            | Pattern                                | Example                           |
| --------------- | -------------------------------------- | --------------------------------- |
| Float32 class   | `Float<OpName>`                        | `FloatDense`, `FloatConv2D`       |
| Layer name      | `f"float_<op>_{op.output_indices[0]}"` | `float_dense_3`, `float_conv2d_5` |
| Quantized class | `Quantized<OpName>` (unchanged)        | `QuantizedDense`                  |
| Private builder | `_build_float_<op>`                    | `_build_float_dense`              |

The layer name **must end with** `_{output_tensor_index}` — the writer uses
this suffix to match layers to operators.

______________________________________________________________________

## Step-by-Step Recipe for Each Op

### Step 1: Add the Float32 Layer Class

In `src/litert_tunner/ops/<op>.py`, add a new class after the existing
`Quantized*` class:

- **Class declaration**: `class Float<Op>(keras.Layer):` for ops without
  persistent weights, or `class Float<Op>(keras.Layer, types.Writable):` for
  weighted ops
- **`__init__`**: Accept float32 data + original dtypes (for float16 support) + op-specific options (padding,
  strides, etc.). No quant params.
- **`build`**: Create `add_weight` for any data that needs to be persisted
  (kernel, bias, constant inputs). All are trainable unless constant.
- **`call`**: The core computation — same as the quantized version but without
  the dequant/quant wrapping. Just the raw float32 math.
- **`get_config`**: Return serialization config (op-specific options + dtypes).
- **`collect_write_ops`** (only for `Writable` layers): Cast weights back to their
  original dtypes (e.g. `self._kernel_dtype`), emit `BufferWriteOp` with those bytes.
  Return empty list for `QuantizationWriteOp`.

### Step 2: Refactor the Builder Function

- **Extract** the existing quantized builder logic into a private
  `_build_quantized_<op>()` function
- **Add** a new `_build_float_<op>()` function
- **Modify** the registered `build_<op>()` to dispatch:

```python
@registry.register_op("OP_TYPE")
def build_op(op, tensors):
    # Common validation (e.g., weight data check) stays here
    input_tensor = tensors[op.input_indices[0]]
    if types.is_quantized(input_tensor):
        return _build_quantized_op(op, tensors)
    return _build_float_op(op, tensors)
```

- In `_build_float_<op>()`, remember to extract `bias_dtype` (if applicable) and pass the original `weight_tensor.dtype` and `bias_dtype` into the `Float<Op>` constructor.

- In private builders, add type narrowing for `weight_tensor.data`:
  `assert weight_tensor.data is not None  # noqa: S101`

### Step 3: Update the Module Docstring

Update the file's module docstring to mention both quantized and float32 support.

______________________________________________________________________

## Testing Recipe

Each migrated op needs the following test additions in its existing test file
(`tests/ops/test_<op>.py`). **Do not create separate test files for float32
variants — add them to the same file.**

### 3.1 Setup Fixture

Add a `float_<op>_setup` fixture that creates float32 tensors (no quant params):

```python
@pytest.fixture
def float_<op>_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal <OP_TYPE> op with float32 I/O (no quantization)."""
    rng = np.random.default_rng(42)

    input_tensor = op_test_utils.make_tensor(
        name="input_f32", index=0, shape=<INPUT_SHAPE>,
        dtype=types.DTYPE_FLOAT32, quantization=None
    )

    # For weighted ops: add kernel/bias tensors with data
    # For element-wise ops: add second input tensor
    # For unary ops: just input + output

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=<N>, shape=<OUTPUT_SHAPE>,
        dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input_tensor, ..., output_tensor)
    op = op_test_utils.make_operator(
        op_type="<OP_TYPE>",
        input_indices=(...),
        output_indices=(<N>,),
    )
    return op, tensors
```

### 3.2 Build Tests (`TestFloat<Op>Build`)

```python
class TestFloat<Op>Build:
    def test__float_<op>_build_returns_keras_layer(self, float_<op>_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_<op>_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_<op>_layer_name_contains_output_index(self, float_<op>_setup):
        """Layer name must end with output tensor index."""
        op, tensors = float_<op>_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")

    # For weighted ops only:
    def test__float_<op>_build_raises_without_weights(self, float_<op>_setup):
        """Builder must raise if weight tensor has no data."""
        ...
```

### 3.3 Call Tests (`TestFloat<Op>Call`)

```python
class TestFloat<Op>Call:
    def test__float_<op>_output_shape(self, float_<op>_setup):
        """Output shape must match expected shape."""
        op, tensors = float_<op>_setup
        input_data = ...  # appropriate random input
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, <EXPECTED_SHAPE>)

    def test__float_<op>_formula_matches_numpy(self, float_<op>_setup):
        """Float32 op output must match numpy reference computation."""
        op, tensors = float_<op>_setup
        input_data = ...
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Compute expected with numpy
        expected = ...  # e.g., input_data @ kernel.T + bias
        np.testing.assert_allclose(output, expected, atol=1e-5)
```

### 3.4 Trainable Weight Tests (`TestFloat<Op>TrainableWeights`)

```python
class TestFloat<Op>TrainableWeights:
    def test__float_<op>_trainable_weights(self, float_<op>_setup):
        op, tensors = float_<op>_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        # Weighted ops: {"kernel", "bias"}
        # Non-weighted ops: set()
        op_test_utils.assert_trainable_weight_names(layer, <EXPECTED_NAMES>)

    def test__float_<op>_non_trainable_weights(self, float_<op>_setup):
        op, tensors = float_<op>_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        # Weighted ops: set()
        # Non-weighted ops with constants: {"constant_input"}
        # Non-weighted ops without constants: set()
        op_test_utils.assert_non_trainable_weight_names(layer, <EXPECTED_NAMES>)
```

### 3.5 Write Ops Tests (`TestFloat<Op>WriteOps`)

**Only for Writable layers (Category A weighted ops):**

```python
class TestFloat<Op>WriteOps:
    def test__float_<op>_is_writable(self, float_<op>_setup):
        op, tensors = float_<op>_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_layer_is_writable(layer)

    def test__float_<op>_write_ops_counts(self, float_<op>_setup):
        op, tensors = float_<op>_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_collect_write_ops(
            layer, op,
            expected_buffer_writes=2,  # kernel + bias
            expected_quant_writes=0,   # always 0 for float32
        )

    def test__float_<op>_write_ops_buffer_indices(self, float_<op>_setup):
        op, tensors = float_<op>_setup
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        buffer_writes, _ = layer.collect_write_ops(op)
        op_test_utils.assert_buffer_write_tensor_indices(
            buffer_writes, {op.input_indices[1], op.input_indices[2]}
        )
```

**For non-Writable layers (Categories B and C):**

```python
def test__float_<op>_not_writable(self, float_<op>_setup):
    op, tensors = float_<op>_setup
    layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
    op_test_utils.assert_layer_not_writable(layer)
```

### 3.6 Integration Test

One integration test per op that builds a real Keras model, exports to float32
TFLite, loads with `litert_tunner`, and compares outputs with Interpreter:

```python
def test__float32_<op>_integration(temp_model_dir, run_interpreter):
    """Float32 <op>: load → predict → save → reload → compare."""
    from tests.conftest import export_float32_tflite_model

    keras.utils.set_random_seed(42)

    # Build a minimal Keras model that uses this op
    inputs = keras.Input(shape=<INPUT_SHAPE>)
    outputs = keras.layers.<KerasLayer>(...)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    output_path = temp_model_dir / "float32_<op>_integration.tflite"
    export_float32_tflite_model(<INPUT_SHAPE>, model, output_path)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, (1, *<INPUT_SHAPE>)).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)
    op_test_utils.verify_model_contains_operator(output_path, "<OP_TYPE>")
```

`verify_model_outputs` does:

- Run interpreter on original model
- Load with `litert_tunner`, run `predict`, compare
- Save model back to `.tflite`
- Re-run interpreter, compare with original — verifies save roundtrip

______________________________________________________________________

## Checklist Per Op

- [ ] Add `Float<Op>` class in `src/litert_tunner/ops/<op>.py`
- [ ] Refactor builder: extract `_build_quantized_<op>()`, add `_build_float_<op>()`,
  dispatch with `types.is_quantized()`
- [ ] Update module docstring to mention float32 support
- [ ] Add `float_<op>_setup` fixture in `tests/ops/test_<op>.py`
- [ ] Add `TestFloat<Op>Build` tests (layer type, name, error on missing data)
- [ ] Add `TestFloat<Op>Call` tests (output shape, formula matches numpy)
- [ ] Add `TestFloat<Op>TrainableWeights` tests
- [ ] Add `TestFloat<Op>WriteOps` tests (or `not_writable` assertion)
- [ ] Add `test__float32_<op>_integration` test
- [ ] Run: `.venv/bin/python -m pytest tests/ops/test_<op>.py -v`
- [ ] Run: `make precommit`
- [ ] Verify existing INT8 tests still pass

______________________________________________________________________

## Remaining Ops Migration Tracker

### Category A: Weighted ops

- [x] `FULLY_CONNECTED` — done (reference implementation)
- [ ] `CONV_2D`
- [ ] `DEPTHWISE_CONV_2D`

### Category B: Element-wise binary ops

- [ ] `ADD`
- [ ] `SUB`
- [ ] `MUL`
- [ ] `DIV`

### Category C: Unary / activation / reduction ops

- [ ] `RELU`
- [ ] `GELU`
- [ ] `LOGISTIC`
- [ ] `SOFTMAX`
- [ ] `MEAN`
- [ ] `CONCATENATION`
- [ ] `NEG`
- [ ] `RSQRT`
- [ ] `SQUARED_DIFFERENCE`
- [ ] `RESIZE_NEAREST_NEIGHBOR`
- [ ] `AVERAGE_POOL_2D` (if it exists as separate from pooling passthrough)

### Already working (no changes needed)

- [x] `RESHAPE`
- [x] `TRANSPOSE`
- [x] `PACK`
- [x] `STRIDED_SLICE`
- [x] `SHAPE`
- [x] `MAX_POOL_2D`

______________________________________________________________________

## Common Pitfalls

- **Ruff S101**: Don't use bare `assert` — add `# noqa: S101` when using
  assert for type narrowing, or use `typing.cast`
- **Ruff PLC0415**: All imports at top of file, not inside functions
- **`weight_tensor.data` is `Optional`**: The parent builder validates
  `data is not None` but private builders need their own type narrowing
- **Layer name suffix**: Must end with `_{output_tensor_index}` or the writer
  won't find the layer. Use pattern: `name=f"float_<op>_{op.output_indices[0]}"`
- **No `QuantizationWriteOp` for float32**: Always return empty list `[]`
- **TFLite weight layout**: Conv2D weights are `(out_ch, kH, kW, in_ch)` in
  TFLite but Keras expects `(kH, kW, in_ch, out_ch)` — transpose in `call()`.
  Same issue applies to DepthwiseConv2D.
- **Constant inputs in binary ops**: In float32 models, constant data is
  already float32. Don't call `extract_constant_input` (that assumes INT8) —
  write a float32 variant or just read `tensor.data` directly.
- **`get_float32_bias`** vs **`get_bias_float32`**: Use `get_float32_bias` for
  float32 models (reads data as-is). Use `get_bias_float32` for INT8 models
  (dequantizes INT32 bias using scales).
