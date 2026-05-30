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
  Estimator (STE) may be used where needed for rounding operations.

#### Why Not Use Keras 3 Built-in Quantization?

Keras 3 has its own quantization API (`model.quantize("int8")`), but it is
**NOT suitable** for this project because:

- It uses **symmetric quantization** (scale only, no zero-point) — TFLite uses
  **affine quantization** (scale + zero-point).
- It's **one-way PTQ** — Keras explicitly states: "you cannot train a model
  after quantizing it to INT8."
- It operates on **Keras model weights**, not TFLite flatbuffer graphs.
- It uses **dynamic AbsMax** activation scaling at runtime, whereas TFLite
  pre-computes fixed activation ranges during calibration.

Our library does the opposite: we take an already-quantized TFLite graph and
make it trainable again by wrapping the integer arithmetic in differentiable
simulation layers.

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

**Fully-Connected / Conv2D integer arithmetic:**

```text
# Accumulation in INT32:
acc_int32 = sum(input_int8[i] * weight_int8[i]) + bias_int32

# Requantize to output INT8:
# multiplier = input_scale * weight_scale / output_scale
# This is computed as a fixed-point multiply + shift in real TFLite,
# but in fake-quant simulation we use float32:
output_float = acc_int32 * (input_scale * weight_scale / output_scale)
output_int8 = clamp(round(output_float) + output_zero_point, -128, 127)
```

**Per-channel vs per-tensor quantization:**

- **Activations**: Always per-tensor (one scale, one zero-point per tensor).
- **Weights**: Per-channel for Conv2D/DepthwiseConv2D (one scale per output
  channel), per-tensor for Dense/FullyConnected.
- **Biases**: Per-channel when weights are per-channel. Bias scale =
  `input_scale * weight_scale[channel]`. Bias zero-point is always 0.

### 4.3 Fused Activations

TFLite commonly fuses activations into the preceding op (e.g., `Conv2D+ReLU`
is a single op with `fused_activation_function=RELU`). In the quantized graph,
the activation is applied **in the INT32 accumulator space before
requantization** to INT8. The fake-quant simulation must replicate this order:

```text
1. Compute INT32 accumulator (dot product + bias)
2. Apply fused activation (e.g., ReLU clamp in INT32 space)
3. Requantize to output INT8
```

### 4.4 Flatbuffer Handling

- **Load**: Parse the `.tflite` flatbuffer, extract graph topology, tensor
  metadata (shapes, types, quantization params), and buffer data.
- **Save**: Update *only* the buffer data (weights, biases, scales,
  zero-points) in the existing flatbuffer. **Never modify the graph topology**
  — this guarantees the output file is always a valid LiteRT model.

#### Flatbuffer Read/Write Strategy

For parsing and modifying `.tflite` files, use the **`tflite` PyPI package**
which provides pre-generated Python classes from the TFLite FlatBuffer schema.
This allows programmatic parsing → modification → re-serialization entirely in
Python.

**Why this approach (not JSON via `flatc`):**

| Approach                                            | Verdict                 | Reason                                                                                                                             |
| --------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `flatc` JSON round-trip                             | ❌ Rejected             | Requires external `flatc` binary (not pip-installable), bloats large weight buffers into massive JSON, schema versioning fragility |
| Binary surgery (overwrite bytes at offsets)         | ⚠️ Possible but fragile | Fast and topology-preserving, but requires offset tracking and is error-prone                                                      |
| **`tflite` Python package (parse → modify → Pack)** | ✅ Recommended          | Pure Python, pip-installable, uses Object API (`ModelT`) for clean read-modify-write, no external tools needed                     |

**Workflow:**

```python
import tflite.Model
import flatbuffers

# 1. Parse
with open("model.tflite", "rb") as f:
    buf = bytearray(f.read())
model_obj = tflite.Model.Model.GetRootAs(buf, 0)
model = tflite.Model.ModelT.InitFromObj(model_obj)  # Mutable object

# 2. Modify buffer data (e.g., update bias values)
model.buffers[bias_buffer_idx].data = new_bias_bytes

# 3. Re-serialize
builder = flatbuffers.Builder(len(buf))
model_offset = model.Pack(builder)
builder.Finish(model_offset)
new_buf = builder.Output()

with open("modified.tflite", "wb") as f:
    f.write(bytes(new_buf))
```

**Important**: Ensure the `tflite` package version matches the TensorFlow
version used to create the model. Pin versions in `pyproject.toml`.

### 4.5 Supported Operations (Incremental)

Agents will add support for operations incrementally. Start with the simplest
ops and expand over time:

1. **Phase 1 — Bootstrap**: Full infrastructure + `FullyConnected` (Dense)
   with no activation. The simplest possible model: a single `Dense(1)` layer,
   INT8 in / INT8 out. Get this matching bit-for-bit before anything else.
1. **Phase 1b — Activations**: Add fused activation support (ReLU, ReLU6).
   Test with `Dense(..., activation="relu")`.
1. **Phase 2 — Convolutions**: Conv2D, DepthwiseConv2D (per-channel quant).
1. **Phase 3 — Pooling & Reshape**: MaxPool2D, AveragePool2D, Reshape, Flatten.
1. **Phase 4 — Normalization & Skip**: BatchNormalization (fused), Add
   (residual connections — requires input requantization).
1. **Phase 5 — Advanced**: Softmax (fixed-point LUT — hard to match exactly),
   Concatenation, Transpose, Pad, etc.

Each new operation must include its own unit tests before being considered
complete.

## 5. Architecture & Code Organization

```text
src/litert_tunner/
├── __init__.py              # Public API: load_model, save_model
├── flatbuffer/              # Flatbuffer parsing & serialization
│   ├── __init__.py
│   ├── parser.py            # Parse .tflite into internal graph representation
│   └── writer.py            # Write updated params back to .tflite
├── graph/                   # Internal graph representation
│   ├── __init__.py
│   ├── types.py             # Dataclasses: TensorInfo, OperatorInfo, GraphDef
│   └── builder.py           # Convert GraphDef → Keras model
├── ops/                     # Operation registry & implementations
│   ├── __init__.py
│   ├── registry.py          # Op name → builder function mapping
│   ├── dense.py             # FullyConnected implementation
│   ├── conv2d.py            # Conv2D implementation
│   └── ...                  # One file per op (or family of ops)
├── quantization/            # Fake-quant simulation layers
│   ├── __init__.py
│   ├── fake_quant.py        # Keras layers for quantize/dequantize
│   └── numerics.py          # Int8 arithmetic helpers
└── utils.py                 # Shared utilities
```

```text
tests/
├── test_flatbuffer_parser.py
├── test_flatbuffer_writer.py
├── test_graph_builder.py
├── test_ops/
│   ├── test_dense.py
│   ├── test_conv2d.py
│   └── ...
├── test_quantization.py
├── test_load_save_roundtrip.py
└── test_finetuning_e2e.py
```

### Design Principles

- **Modular**: Each op is a self-contained module. Adding a new op should
  require no changes to existing modules — only registering it in the registry.
- **Composable**: The graph builder composes Keras layers from the op registry.
  Operations are independent building blocks.
- **Easy to extend**: The `ops/registry.py` pattern makes it trivial for an
  agent to add a new op: implement a builder function, register it, add tests.
- **Clean separation of concerns**: Flatbuffer I/O knows nothing about Keras.
  The graph builder knows nothing about flatbuffers. Quantization layers are
  reusable.

## 6. Coding Standards

### 6.1 General

- **Python ≥ 3.11** — use modern syntax (type unions with `|`, `match`, etc.).
- **Type hints everywhere** — all function signatures must be fully typed.
- **Docstrings** — Google-style docstrings on all public functions and classes.
- **Line length** — 100 characters max (configured in `ruff`).
- **Linting** — must pass `ruff check` and `ruff format` (configured in
  `pyproject.toml`).
- **No magic numbers** — use named constants or enums.
- **Random number generation** — always use the modern `np.random.default_rng(seed)` API to create a `Generator` instance for random arrays (e.g. `rng.uniform(...)`), and avoid legacy APIs like `np.random.seed` and `np.random.uniform` to prevent `NPY002` violations.

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

- **TF-specific code is allowed only in tests** — for `tf.lite.Interpreter`,
  `tf.lite.TFLiteConverter`, and test model export. These should be isolated
  in test utilities.

- **Import pattern**:

  ```python
  # ✅ Production code
  import keras
  from keras import ops

  # ✅ Test code only
  import tensorflow as tf
  ```

### 6.6 Dependencies

Core dependencies (in `pyproject.toml`):

- `keras >= 3.0` — model building and training (backend-agnostic)
- `numpy` — numerical operations
- `tflite` — TFLite FlatBuffer schema parsing (pre-generated Python classes)
- `flatbuffers` — FlatBuffer serialization/deserialization

Test / dev dependencies:

- `tensorflow` — for `tf.lite.Interpreter` and `tf.lite.TFLiteConverter`
  (used in tests only)
- `ai-edge-litert` — modern LiteRT runtime (alternative to `tflite-runtime`,
  use `from ai_edge_litert.interpreter import Interpreter`)
- `pytest`, `ruff`, etc. — already configured

Keep the dependency footprint minimal. Do not add unnecessary libraries.

### 6.7 Environment

- **Always check** the active environment before running any Python code,
  tests, or scripts.
- All tests and Python commands should be run within the virtual environment activated using `source .venv/bin/activate`.
- If using conda, check the active conda environment with `echo $CONDA_DEFAULT_ENV`.
- If the environment is not clear, ask the user.
- **Command Execution Rules**:
  - **Tooling**: The project uses `uv` for package/environment management. Do not assume `pip` or `.venv/bin/pip` is present; use `uv pip` instead.
  - **Testing**: Always run tests using `.venv/bin/python -m pytest <path_to_test>` rather than directly invoking `.venv/bin/pytest` or `pytest`. This guarantees that Python resolves the root `tests` module correctly without raising `ModuleNotFoundError`.
  - **Type Checking (Pyright)**: Pyright is run via Node/npm. Always run it non-interactively using `npx -y pyright` to prevent blocking on interactive npm prompts. To configure it to use the project's virtual environment, pass the `--pythonpath .venv/bin/python` argument (do not use `--venv`).

## 7. Testing Pipeline

Every feature must be validated by the following pipeline. Tests can (and
should) be split into independent steps.

### 7.1 Unit Tests (per operation)

For each new op, the agent must:

1. **Define a minimal Keras model** using that op (e.g., a single Dense layer).
1. **Export it** to LiteRT INT8 using `tf.lite.TFLiteConverter` with
   representative dataset calibration.
1. **Load it** with `litert_tunner.load_model()`.
1. **Compare outputs**: `tunner_model.predict(inputs)` vs
   `tf.lite.Interpreter` — must match within numerical noise
   (typically `atol=1, rtol=0` for int8 output comparison, or appropriate
   tolerance for float comparisons).

### 7.2 Load/Save Round-trip

1. Load a `.tflite` model with `litert_tunner.load_model()`.
1. Save it back with `litert_tunner.save_model()` (no fine-tuning).
1. Load the saved model with `tf.lite.Interpreter`.
1. Compare outputs to the original — must be **bit-exact identical**.

### 7.3 Fine-tuning Smoke Test

1. Train a small Keras model → export INT8.
1. Load with `litert_tunner.load_model()`.
1. Measure initial gap: `|tunner_model.predict(x) - original_model.predict(x)|`.
1. Fine-tune the tunner model using the original model outputs as targets.
1. Verify the gap decreases.
1. Save and re-load — verify the saved model also shows the improved gap.

### 7.4 Running Tests

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run all tests
make test
# or
pytest

# Run a specific test file
pytest tests/test_ops/test_dense.py -v

# Run with coverage (when configured)
pytest --cov=litert_tunner
```

## 8. Agent Workflow Checklist

When an agent is asked to implement a new feature (e.g., a new op), follow this
checklist:

- [ ] **Understand** the LiteRT op specification (input/output tensors,
  attributes, quantization behavior).
- [ ] **Implement** the op builder in `src/litert_tunner/ops/<op_name>.py`.
- [ ] **Register** the op in `src/litert_tunner/ops/registry.py`.
- [ ] **Implement** any new fake-quantization behavior if needed.
- [ ] **Write unit tests** following the testing pipeline (Section 7).
- [ ] **Run linting**: `ruff check src/ tests/` and `ruff format src/ tests/`.
- [ ] **Run all tests**: `pytest` — ensure nothing is broken.
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
- **Do not use `flatc` CLI or JSON conversion for flatbuffer I/O.** Use the
  `tflite` Python package with the Object API for programmatic read/write.
- **Do not use Keras 3 built-in quantization API.** It uses symmetric
  quantization, is one-way PTQ (not trainable), and doesn't parse TFLite
  flatbuffers. Our library has fundamentally different goals.
- **Do not use legacy `np.random.seed` or `np.random.uniform`.** Always use modern `np.random.default_rng(seed)` to obtain a generator instance (`Generator`) and call its methods.
- **Do not import symbols directly from submodules.** Use Google-style imports
  (import the module, use dotted access).
