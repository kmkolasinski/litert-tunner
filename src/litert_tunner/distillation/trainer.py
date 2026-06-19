import re
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import keras
import numpy as np
from keras import ops

from litert_tunner import logging
from litert_tunner.distillation import losses

logger = logging.get_logger()


# Generic types to document Keras backend-agnostic tensors and data structures
TensorLike: TypeAlias = keras.KerasTensor | np.ndarray | float | int
DataStruct: TypeAlias = TensorLike | dict[str, Any] | list[Any] | tuple[Any, ...]
MetricFunc: TypeAlias = Callable[[keras.KerasTensor, keras.KerasTensor], keras.KerasTensor]


class Trainer(keras.Model):
    """Generic fine-tuning distillation trainer for quantized models.

    This trainer computes a distillation loss between a base model (teacher) and
    a LiteRT Tunner model (student), along with an L2 weight drift loss to prevent
    the fine-tuned parameters from diverging too much from their original values.

    The model expects a dataset yielding `x` (or `(x, y)` where `y` is ignored for
    distillation) and computes the loss natively.
    """

    def __init__(
        self,
        student_model: keras.Model,
        teacher_model: keras.Model,
        distillation_loss_fn: MetricFunc | None = None,
        l2_weight_decay: float = 0.01,
        extra_metrics: Mapping[str, MetricFunc] | None = None,
        **kwargs,
    ):
        """Initializes the trainer.

        Args:
            student_model: The student model (typically the quantized model from litert_tunner).
            teacher_model: The teacher model (the original unquantized model).
            distillation_loss_fn: A callable to compute the loss between the student
                and teacher outputs. Defaults to Mean Squared Error (MSE).
            l2_weight_decay: Strength of the L2 weight drift penalty.
            extra_metrics: Dictionary mapping metric names to callables computing
                additional metrics between student and teacher outputs.
            **kwargs: Additional arguments passed to `keras.Model`.
        """
        super().__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.trainable = False

        self.distillation_loss_fn = distillation_loss_fn or losses.mse_loss
        self.l2_weight_decay = l2_weight_decay
        self.extra_metrics_fns = extra_metrics or {}

        # Store original trainable variables for L2 weight drift loss
        self.original_variables = {
            v.path: v.numpy().copy() for v in self.student_model.trainable_weights
        }

        # Metric trackers
        self.distill_loss_tracker = keras.metrics.Mean(name="distill_loss")
        self.l2_loss_tracker = keras.metrics.Mean(name="l2_loss")
        self.extra_metric_trackers = {
            name: keras.metrics.Mean(name=name) for name in self.extra_metrics_fns
        }

    @property
    def metrics(self):
        """Returns all metric trackers."""
        # Include base class metrics (like the compiled 'loss' tracker) if any
        base_metrics = super().metrics
        return [
            *base_metrics,
            self.distill_loss_tracker,
            self.l2_loss_tracker,
            *self.extra_metric_trackers.values(),
        ]

    def call(self, inputs: DataStruct, training: bool = False) -> DataStruct:  # noqa: FBT001, FBT002
        """Forward pass of the student model."""
        return self.student_model(inputs, training=training)

    def compute_loss(
        self,
        x: DataStruct | None = None,
        y: DataStruct | None = None,
        y_pred: TensorLike | None = None,
        sample_weight: DataStruct | None = None,
    ) -> TensorLike:
        """Computes the total loss including distillation and L2 weight drift."""
        teacher_outputs = ops.stop_gradient(self.teacher_model(x, training=False))

        # Cache teacher outputs for compute_metrics to reuse.
        # Under JAX JIT tracing, calling the teacher model a second time
        # (in compute_metrics) causes non-trainable layer weights to resolve
        # to None, triggering "'NoneType' object is not callable" errors.
        self._cached_teacher_outputs = teacher_outputs

        # Compute distillation loss
        distill_loss = ops.cast(self.distillation_loss_fn(y_pred, teacher_outputs), "float32")  # pyright: ignore[reportArgumentType]

        # Compute L2 weight drift loss
        l2_loss = 0.0
        if self.l2_weight_decay > 0.0:
            for v in self.student_model.trainable_weights:
                if v.path in self.original_variables:
                    original_v = self.original_variables[v.path]
                    l2_loss += self.l2_weight_decay * ops.mean(
                        ops.square(ops.subtract(v, original_v))
                    )

        l2_loss = ops.cast(l2_loss, "float32")

        # Track the components
        self.distill_loss_tracker.update_state(distill_loss)
        self.l2_loss_tracker.update_state(l2_loss)

        total_loss = distill_loss + l2_loss

        # Call super to include any compiled loss (if the user provided one)
        try:
            compiled_loss = super().compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )
            if compiled_loss is not None:
                total_loss += ops.cast(compiled_loss, "float32")
        except ValueError:
            pass

        return total_loss  # pyright: ignore[reportReturnType]

    def compute_metrics(
        self,
        x: DataStruct | None = None,
        y: DataStruct | None = None,
        y_pred: TensorLike | None = None,
        sample_weight: DataStruct | None = None,
    ) -> TensorLike:
        """Computes all metrics, including extra custom metrics."""
        # Reuse teacher outputs cached by compute_loss to avoid calling the
        # teacher model twice per step (breaks under JAX JIT tracing and is
        # wasteful even on other backends).
        if self.extra_metrics_fns:
            teacher_outputs = self._cached_teacher_outputs
            for name, metric_fn in self.extra_metrics_fns.items():
                val = metric_fn(y_pred, teacher_outputs)  # pyright: ignore[reportArgumentType]
                self.extra_metric_trackers[name].update_state(val)

        # Let the base class update standard compiled metrics (like the total loss)
        return super().compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)  # pyright: ignore[reportReturnType]


def prepare_for_finetuning(
    model: keras.Model, trainable_pattern: str = r".*(bias|weight_scale)$"
) -> None:
    """Freezes all model variables except those matching the pattern.

    Args:
        model: The Keras model to prepare for fine-tuning.
        trainable_pattern: Regex pattern matching the paths of variables that should
            remain trainable. By default matches biases and weight scales. To also
            train INT8 weights, use ``r".*(bias|weight_scale|weight_int8)$"``.
    """
    total_vars = 0
    trainable_vars = 0
    total_params = 0
    trainable_params = 0

    pattern = re.compile(trainable_pattern)
    for v in model.variables:
        # Calculate size of the variable
        size = 1
        if v.shape:
            for d in v.shape:
                if d is not None:
                    size *= d

        total_vars += 1
        total_params += size

        if pattern.search(v.path):
            v.trainable = True
            trainable_vars += 1
            trainable_params += size
            logger.info(
                "Variable taken for training: path=%s, dtype=%s, shape=%s",
                v.path,
                v.dtype,
                v.shape,
            )
        else:
            v.trainable = False

    logger.info(
        "Finetuning statistics:\n"
        "  Trainable variables: %d / %d (%.2f%%)\n"
        "  Trainable parameters: %d / %d (%.2f%%)",
        trainable_vars,
        total_vars,
        (trainable_vars / total_vars * 100.0) if total_vars > 0 else 0.0,
        trainable_params,
        total_params,
        (trainable_params / total_params * 100.0) if total_params > 0 else 0.0,
    )
