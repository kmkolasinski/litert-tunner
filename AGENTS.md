# AGENTS.md — Guidelines for AI Agents Working on `litert-tunner`

## 1. Project Vision

**litert-tunner** is a Python library that lets users fine-tune fully-quantized
INT8 LiteRT (TFLite) models *after* export.

The core idea is an alternative to Quantization-Aware Training (QAT): instead of
adding fake-quantization nodes during training, we parse an already-exported
INT8 graph, reconstruct it as a Keras 3 model with differentiable quantization
simulation, fine-tune the float32 parameters (biases, scales, zero-points), and
write the updated parameters back into the flatbuffer — without altering the
graph topology.

## 2. User Flow

```text
┌─────────────────────────────────────┐
│  1. Train a Keras model normally    │
│  2. Export to LiteRT INT8 format    │
│     (full integer quantization)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. litert_tunner.load_model(path)  │
│     - Parse flatbuffer graph        │
│     - Reconstruct as Keras model    │
│       with fake-quantization nodes  │
│     - INT8 weights, float32 biases, │
│       scales, and zero-points       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. tunner_model.predict(inputs)    │
│     Must exactly match the LiteRT   │
│     Interpreter output (within      │
│     numerical noise)                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. tunner_model.fit(...)           │
│     Fine-tune biases, scales, etc.  │
│     Gradients flow through float32  │
│     parameters only; INT8 weights   │
│     are frozen (configurable)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. litert_tunner.save_model(       │
│        tunner_model, path)          │
│     Update flatbuffer with new      │
│     parameter values — graph        │
│     topology must remain identical  │
└─────────────────────────────────────┘
```

## 3. Target API

```python
import litert_tunner

# Load an INT8 LiteRT model and get a trainable Keras replica
tunner_model = litert_tunner.load_model("model_int8.tflite")

# Inference — should match LiteRT Interpreter output
predictions = tunner_model.predict(inputs)

# Fine-tune (standard Keras training loop)
tunner_model.compile(optimizer=..., loss=..., metrics=...)
tunner_model.fit(train_ds, validation_data=val_ds, epochs=5)

# Export — writes updated parameters back into the flatbuffer
litert_tunner.save_model(tunner_model, "model_int8_finetuned.tflite")
```

## 4. Technical Details

### 4.1 Quantization Representation

| Component   | LiteRT storage | Tunner Keras representation | Trainable by default |
| ----------- | -------------- | --------------------------- | -------------------- |
| Weights     | INT8           | INT8 (stored as-is)         | No                   |
| Biases      | INT32          | Float32                     | Yes                  |
| Scales      | Float32        | Float32                     | Yes                  |
| Zero-points | INT8/INT32     | Float32                     | Yes (configurable)   |

- **Fake quantization nodes** simulate quantize → dequantize round-trips so
  forward pass matches the integer arithmetic of the real graph.
- **Gradients** flow through float32 parameters only. Straight-Through
  Estimator (STE) is used for rounding and clipping operations (see
  `ops/utils.py`: `quantize_ste`, `dequantize_ste`).

### 4.2 Quantization Formulas

All fake-quant simulation is based on the standard TFLite affine quantization
scheme. Agents implementing ops must use these formulas exactly.

**Dequantize (INT8 → Float32):**

```text
real_value = scale * (int8_value - zero_point)
```

**Quantize (Float32 → INT8):**

```text
int8_value = clamp(round(real_value / scale) + zero_point, -128, 127)
```

**Per-channel vs per-tensor quantization:**

- **Activations**: Always per-tensor (one scale, one zero-point per tensor).
- **Weights**: Per-channel for Conv2D/DepthwiseConv2D (one scale per output
  channel), per-tensor for Dense/FullyConnected.
- **Biases**: Per-channel when weights are per-channel. Bias scale =
  `input_scale * weight_scale[channel]`. Bias zero-point is always 0.

### 4.3 Fused Activations

TFLite commonly fuses activations into the preceding op (e.g., `Conv2D+ReLU`
is a single op with `fused_activation_function=RELU`). Supported fused
activations are defined in `ops/utils.py` as constants:

- `FUSED_ACTIVATION_NONE = 0`
- `FUSED_ACTIVATION_RELU = 1`
- `FUSED_ACTIVATION_RELU_N1_TO_1 = 2`
- `FUSED_ACTIVATION_RELU6 = 3`

The `apply_fused_activation()` helper applies the activation after the
accumulation and before requantization.

### 4.4 Flatbuffer Handling

- **Load**: Parse the `.tflite` flatbuffer, extract graph topology, tensor
  metadata (shapes, types, quantization params), and buffer data.
- **Save**: Update *only* the buffer data (weights, biases, scales,
  zero-points) in the existing flatbuffer. **Never modify the graph topology**
  — this guarantees the output file is always a valid LiteRT model.

Uses the **`tflite` PyPI package** with the Object API (`Model`) for
programmatic parse → modify → re-serialize, entirely in Python.

### 4.5 Supported Operations

Currently implemented operations (one file per op in `src/litert_tunner/ops/`):

| Op file             | TFLite op type(s)                      | Writable | Notes                             |
| ------------------- | -------------------------------------- | -------- | --------------------------------- |
| `dense.py`          | `FULLY_CONNECTED`                      | ✅       | Per-tensor weight quant           |
| `conv2d.py`         | `CONV_2D`                              | ✅       | Per-channel weight quant          |
| `depthwise_conv2d.py` | `DEPTHWISE_CONV_2D`                  | ✅       | Per-channel weight quant          |
| `add.py`            | `ADD`                                  | ✅       | Requantizes both inputs           |
| `mul.py`            | `MUL`                                  | ✅       | Requantizes both inputs           |
| `logistic.py`       | `LOGISTIC`                             | ✅       | Sigmoid activation                |
| `mean.py`           | `MEAN`                                 | ✅       | Reduce mean                       |
| `quantize_op.py`    | `QUANTIZE`, `DEQUANTIZE`               | ✅       | Quantize/dequantize passthrough   |
| `pool.py`           | `MAX_POOL_2D`, `AVERAGE_POOL_2D`       | ❌       | No trainable params               |
| `reshape.py`        | `RESHAPE`                              | ❌       | Static or dynamic shape           |
| `pack.py`           | `PACK`                                 | ❌       | Constant tensor packing           |
| `shape_op.py`       | `SHAPE`                                | ❌       | Returns tensor shape              |
| `strided_slice.py`  | `STRIDED_SLICE`                        | ❌       | Tensor slicing                    |

**Adding a new op:**

1. Create `src/litert_tunner/ops/<op_name>.py` with a `keras.Layer` subclass.
2. Decorate the builder function with `@registry.register_op("OP_TYPE")`.
3. Add the import to `src/litert_tunner/ops/__init__.py`.
4. Write tests in `tests/ops/test_<op_name>.py`.

Ops that persist parameters must implement the `Writable` protocol (see
`graph/types.py`) and its `collect_write_ops()` method. Passthrough ops
(reshape, pooling, etc.) do not implement `Writable`.

### 4.6 Op Implementation Pattern

Each quantized op follows the same pattern:

1. **Keras Layer class** (`QuantizedDense`, `QuantizedConv2D`, etc.):
   - `__init__`: Stores raw numpy data for weights, bias, and quant params.
   - `build`: Creates Keras weights via `self.add_weight(...)` — frozen for
     INT8 data, trainable for bias/scales/zero-points.
   - `call`: Dequantize inputs → compute in float32 → apply fused activation
     → quantize output (using STE for gradient flow).
   - `collect_write_ops`: Returns `BufferWriteOp` and `QuantizationWriteOp`
     instructions for the flatbuffer writer.

2. **Builder function** (decorated with `@registry.register_op`):
   - Extracts tensors, quantization params, and options from `OperatorInfo`.
   - Uses shared helpers from `ops/utils.py` (e.g., `get_bias_float32`,
     `get_quant_param_value`, `apply_fused_activation`).
   - Returns a configured layer instance.

### 4.7 Shared Utilities (`ops/utils.py`)

Key functions used across all op implementations:

- `dequantize_ste` / `quantize_ste` / `_fake_quantize` — STE-based quant/dequant
- `apply_fused_activation` — TFLite fused activation dispatch
- `get_bias_float32` — Extract and dequantize INT32 bias to float32
- `get_quant_param_value` — Extract scalar or array quant params
- `expand_dims_if_not_scalar` — Reshape for per-channel broadcasting
- `quantize_to_int8` / `quantize_bias_to_int32` — Convert back for writing
- `make_quant_write_op` / `to_float_list` / `to_int_list` — Write-op helpers
- `compute_requantize_multiplier` — `(input_scale * weight_scale) / output_scale`

## 5. Architecture & Code Organization

```text
src/litert_tunner/
├── __init__.py              # Public API: load_model, save_model
├── flatbuffer/              # Flatbuffer parsing & serialization
│   ├── __init__.py          # Re-exports parse_tflite, save_tflite
│   ├── parser.py            # Parse .tflite → GraphDef
│   └── writer.py            # Write updated params back to .tflite
├── graph/                   # Internal graph representation
│   ├── __init__.py          # Re-exports types and build_keras_model
│   ├── types.py             # TensorInfo, OperatorInfo, GraphDef, Writable protocol
│   └── builder.py           # Convert GraphDef → Keras Functional model
└── ops/                     # Operation registry & implementations
    ├── __init__.py           # Imports all ops to trigger registration
    ├── registry.py           # @register_op decorator, get_op_builder()
    ├── utils.py              # Shared helpers: STE quant/dequant, fused activation, etc.
    ├── dense.py              # FULLY_CONNECTED
    ├── conv2d.py             # CONV_2D
    ├── depthwise_conv2d.py   # DEPTHWISE_CONV_2D
    ├── add.py                # ADD
    ├── mul.py                # MUL
    ├── logistic.py           # LOGISTIC
    ├── mean.py               # MEAN
    ├── quantize_op.py        # QUANTIZE, DEQUANTIZE
    ├── pool.py               # MAX_POOL_2D, AVERAGE_POOL_2D
    ├── reshape.py            # RESHAPE
    ├── pack.py               # PACK
    ├── shape_op.py           # SHAPE
    └── strided_slice.py      # STRIDED_SLICE
```

```text
tests/
├── conftest.py              # Shared fixtures: make_dense_tflite, make_mlp_tflite,
│                            #   make_resnet_tflite, make_efficientnetb0_tflite,
│                            #   run_interpreter, export_quantized_tflite_model
├── test_load_save_roundtrip.py  # Load → save → verify bit-exact identity
├── test_finetuning_e2e.py       # Fine-tune bias → verify loss decreases → save/reload
├── flatbuffer/
│   ├── test_parser.py       # Flatbuffer parser unit tests
│   ├── test_writer.py       # Flatbuffer writer unit tests
│   └── test_parse_write.py  # Parser + writer integration tests
├── ops/
│   ├── op_test_utils.py     # Test helpers: make_tensor, build_and_call, assertions
│   ├── test_utils.py        # Tests for ops/utils.py helpers
│   ├── test_dense.py        # FullyConnected unit tests
│   ├── test_conv2d.py       # Conv2D unit tests
│   ├── test_depthwise_conv2d.py  # DepthwiseConv2D unit tests
│   ├── test_add.py          # Add unit tests
│   ├── test_mul.py          # Mul unit tests
│   ├── test_logistic.py     # Logistic unit tests
│   ├── test_mean.py         # Mean unit tests
│   ├── test_pool.py         # Pool unit tests
│   ├── test_quantize_op.py  # Quantize/Dequantize unit tests
│   └── test_passthrough_ops.py  # Reshape, Pack, Shape, StridedSlice tests
└── networks/
    ├── test_mlp.py          # Multi-layer MLP forward-pass tests
    ├── test_resnet.py       # ResNet-like CNN forward-pass tests
    └── test_efficientnet.py # EfficientNetB0 forward-pass test
```

### Design Principles

- **Modular**: Each op is a self-contained module. Adding a new op requires
  only: create the file, decorate with `@register_op`, add import to
  `ops/__init__.py`.
- **Composable**: The graph builder composes Keras layers from the op registry.
  Operations are independent building blocks.
- **Clean separation of concerns**: Flatbuffer I/O knows nothing about Keras.
  The graph builder knows nothing about flatbuffers.

## 6. Coding Standards

### 6.1 General

- **Python ≥ 3.11** — use modern syntax (type unions with `|`, `match`, etc.).
- **Type hints everywhere** — all function signatures must be fully typed.
- **Docstrings** — Google-style docstrings on all public functions and classes.
- **Line length** — 100 characters max (configured in `ruff`).
- **Linting** — must pass `ruff check` and `ruff format` (configured in
  `pyproject.toml`). Use `ALL` rule selection with explicit ignores.
- **No magic numbers** — use named constants or enums.
- **Random number generation** — always use the modern
  `np.random.default_rng(seed)` API to create a `Generator` instance
  (e.g. `rng.uniform(...)`). Avoid legacy `np.random.seed` / `np.random.uniform`.

### 6.2 Import Style (Google-style)

Use **Google-style imports** throughout the project. Import modules, not
individual symbols from modules.

```python
# ✅ Correct — import the module, use dotted access
from litert_tunner import flatbuffer
from litert_tunner import graph

graph_def = flatbuffer.parse_tflite("model.tflite")
model = graph.build_keras_model(graph_def)
flatbuffer.save_tflite(model, "out.tflite")

# ✅ Also correct — import submodule
from litert_tunner.graph import types

tensor = types.TensorInfo(...)

# ❌ Wrong — importing symbols directly
from litert_tunner.flatbuffer.parser import parse_tflite  # NO
from litert_tunner.graph.types import TensorInfo  # NO
```

Each package's `__init__.py` must re-export the public API of its submodules
so that `from litert_tunner import flatbuffer` then `flatbuffer.parse_tflite()`
works. Keep `__init__.py` files minimal — only re-exports, no logic.

### 6.3 Test Naming Convention

All test functions must use a **double underscore** prefix to visually separate
the `test` keyword from the descriptive name:

```python
# ✅ Correct
def test__dense_output_matches_interpreter(): ...
def test__load_save_identity(): ...
def test__quantize_dequantize_roundtrip(): ...


# ❌ Wrong — single underscore
def test_dense_output_matches_interpreter(): ...
```

### 6.4 Dataclass Style

Use `frozen=True` for all immutable data containers (graph types, quantization
params, tensor info, etc.). Only use mutable dataclasses when mutation is
explicitly required.

```python
@dataclass(frozen=True)
class QuantizationParams:
    scales: np.ndarray
    zero_points: np.ndarray
    quantized_dimension: int
```

### 6.5 Keras 3 Backend-Agnostic Code

The tunner model must be **backend-agnostic** using Keras 3:

- **Use `keras.ops`** for all numerical operations — never use `tf.`, `jax.`,
  or `torch.` directly in production code.

- **Use `keras.Layer`** subclasses for custom layers (fake-quant nodes, op
  implementations).

- **TF-specific code is allowed only in tests** — for `tf.lite.TFLiteConverter`
  and test model export. Use `ai_edge_litert.interpreter.Interpreter` for
  inference in tests.

- **Import pattern**:

  ```python
  # ✅ Production code
  import keras
  from keras import ops

  # ✅ Test code only
  import tensorflow as tf
  from ai_edge_litert.interpreter import Interpreter
  ```

### 6.6 Dependencies

Core dependencies (in `pyproject.toml`):

- `keras >= 3.0` — model building and training (backend-agnostic)
- `numpy` — numerical operations
- `tflite >= 2.18.0` — TFLite FlatBuffer schema parsing
- `flatbuffers` — FlatBuffer serialization/deserialization

Test / dev dependencies:

- `tensorflow >= 2.18.0` — for `tf.lite.TFLiteConverter` (tests only)
- `ai-edge-litert` — LiteRT runtime (`Interpreter`)
- `pytest`, `pytest-xdist`, `pytest-forked`, `pytest-cov`, `pytest-randomly`,
  `pytest-dotenv`, `ruff`, etc.

Keep the dependency footprint minimal. Do not add unnecessary libraries.

### 6.7 Environment

- **Always check** the active environment before running any Python code,
  tests, or scripts.
- All tests and Python commands should be run within the virtual environment
  activated using `source .venv/bin/activate`.
- If the environment is not clear, ask the user.
- **Command Execution Rules**:
  - **Tooling**: The project uses `uv` for package/environment management.
    Do not assume `pip` or `.venv/bin/pip` is present; use `uv pip` instead.
  - **Testing**: Always run tests using
    `.venv/bin/python -m pytest <path_to_test>` rather than directly invoking
    `.venv/bin/pytest` or `pytest`. This guarantees that Python resolves the
    root `tests` module correctly without raising `ModuleNotFoundError`.
  - **Type Checking (Pyright)**: Pyright is run via Node/npm. Always run it
    non-interactively using `npx -y pyright` to prevent blocking on interactive
    npm prompts. To configure it to use the project's virtual environment, pass
    the `--pythonpath .venv/bin/python` argument (do not use `--venv`).

## 7. Testing Pipeline

### 7.1 Test Fixtures (in `tests/conftest.py`)

Tests use pytest fixtures to generate quantized TFLite models on the fly:

- **`make_dense_tflite`** — creates a single Dense layer model.
- **`make_mlp_tflite`** — creates multi-layer MLPs with optional skip
  connections, batch normalization, and various activations.
- **`make_resnet_tflite`** — creates ResNet-like CNN models with Conv2D,
  residual connections, pooling, and batch normalization.
- **`make_efficientnetb0_tflite`** — creates an EfficientNetB0-based model.
- **`run_interpreter`** — runs a `.tflite` model through the LiteRT Interpreter
  and returns outputs. Handles INT8/float32 input type conversion.
- **`export_quantized_tflite_model`** — helper that converts a Keras model to
  a fully-quantized INT8 TFLite model using `tf.lite.TFLiteConverter` with
  a representative dataset.

### 7.2 Op-Level Tests (`tests/ops/`)

Each op has unit tests that verify the builder, layer call, trainable weights,
and `Writable` protocol. The shared framework `tests/ops/op_test_utils.py`
provides:

- `make_tensor`, `make_operator`, `make_quant_params` — fixture factories
- `build_and_call` — build from registry + call in one step
- `assert_trainable_weight_names` / `assert_non_trainable_weight_names`
- `assert_layer_is_writable` / `assert_layer_not_writable`
- `assert_collect_write_ops` — verify write-op counts and tensor indices

### 7.3 Network-Level Forward-Pass Tests (`tests/networks/`)

These are the primary correctness tests. The pattern for every network test:

```python
def test__mlp_single_layer_forward(make_mlp_tflite, run_interpreter):
    # 1. Create a quantized TFLite model via fixture
    model_path = make_mlp_tflite(input_size=4, hidden_sizes=[8], ...)

    # 2. Run inference through the LiteRT Interpreter
    litert_outputs = run_interpreter(model_path, x_train)

    # 3. Load with litert_tunner and run Keras prediction
    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_train)

    # 4. Compare — forward propagation must match within tolerance
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)

    # 5. Save and verify round-trip
    litert_tunner.save_model(keras_model, str(model_path))
    litert_saved_outputs = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(keras_outputs, litert_saved_outputs, atol=1e-3)
```

Key points:
- **Forward propagation comparison** between the Keras model output and the
  LiteRT Interpreter output is the primary validation method.
- Typical tolerance: `atol=1e-3` for float I/O models.
- Every test also verifies the **save round-trip**: save the model back to
  `.tflite`, re-run through the Interpreter, confirm outputs still match.

### 7.4 Integration Tests

- **`test_load_save_roundtrip.py`** — Verifies load → save → reload produces
  **bit-exact identical** outputs. Also tests that manual parameter modification
  (e.g., shifting bias) results in a valid, loadable `.tflite` file.
- **`test_finetuning_e2e.py`** — End-to-end smoke test: load model → freeze
  all params except bias → fine-tune on shifted targets → verify loss decreases
  → save → verify the Interpreter also shows improved predictions.

### 7.5 Running Tests

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run all tests with coverage (via Makefile)
make test

# Run a specific test file
.venv/bin/python -m pytest tests/ops/test_dense.py -v

# Run a specific test directory
.venv/bin/python -m pytest tests/networks/ -v

# Run linting
ruff check src/ tests/
ruff format --check src/ tests/
```

## 8. Agent Workflow Checklist

When an agent is asked to implement a new feature (e.g., a new op), follow this
checklist:

- [ ] **Understand** the LiteRT op specification (input/output tensors,
  attributes, quantization behavior).
- [ ] **Implement** the op as a `keras.Layer` subclass in
  `src/litert_tunner/ops/<op_name>.py`. Use the shared helpers from
  `ops/utils.py`.
- [ ] **Register** the op with `@registry.register_op("OP_TYPE")` in the same
  file.
- [ ] **Add import** in `src/litert_tunner/ops/__init__.py` to trigger
  registration.
- [ ] **Implement `Writable`** if the op has trainable parameters that need to
  be written back to the flatbuffer.
- [ ] **Write op-level unit tests** in `tests/ops/test_<op_name>.py` using the
  `op_test_utils` framework.
- [ ] **Write or extend a network-level test** in `tests/networks/` that
  exercises the op in a real model, comparing Keras vs Interpreter outputs.
- [ ] **Run linting**: `ruff check src/ tests/` and `ruff format src/ tests/`.
- [ ] **Run all tests**: `.venv/bin/python -m pytest` — ensure nothing is
  broken.
- [ ] **Update this file** if the change affects architecture or conventions.

## 9. What NOT to Do

- **Do not modify the graph topology on save.** The flatbuffer graph must
  remain structurally identical — only buffer data changes.
- **Do not add ops without tests.** Untested ops will cause silent correctness
  bugs.
- **Do not hard-code model-specific logic.** Everything must be generic and
  driven by the parsed graph.
- **Do not introduce heavy dependencies.** Keep the library lightweight.
- **Do not write monolithic code.** Each module should be small, focused, and
  independently testable.
- **Do not skip type hints or docstrings.** Code must be self-documenting.
- **Do not use TF/JAX/PyTorch ops directly in production code.** Always use
  `keras.ops` for backend-agnostic compatibility.
- **Do not use legacy `np.random.seed` or `np.random.uniform`.** Always use
  modern `np.random.default_rng(seed)` to obtain a generator instance
  (`Generator`) and call its methods.
- **Do not import symbols directly from submodules.** Use Google-style imports
  (import the module, use dotted access).
