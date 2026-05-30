import re
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import keras
import numpy as np
from keras import ops

# Generic types to document Keras backend-agnostic tensors and data structures
TensorLike: TypeAlias = keras.KerasTensor | np.ndarray | float | int
DataStruct: TypeAlias = TensorLike | dict[str, Any] | list[Any] | tuple[Any, ...]


def prepare_for_finetuning(
    model: keras.Model, trainable_pattern: str = r".*(bias|weight_scale)$"
) -> None:
    """Freezes all model variables except those matching the pattern.

    Args:
        model: The Keras model to prepare for fine-tuning.
        trainable_pattern: Regex pattern matching the paths of variables that should
            remain trainable (e.g., biases and scales).
    """
    pattern = re.compile(trainable_pattern)
    for v in model.variables:
        if pattern.search(v.path):
            v.trainable = True
        else:
            v.trainable = False


class Trainer(keras.Model):
    """Generic fine-tuning distillation trainer for quantized models.

    This trainer computes a distillation loss between a base model (teacher) and
    a LiteRT tunner model (student), along with an L2 weight drift loss to prevent
    the fine-tuned parameters from diverging too much from their original values.

    The model expects a dataset yielding `x` (or `(x, y)` where `y` is ignored for
    distillation) and computes the loss natively.
    """

    def __init__(
        self,
        litert_model: keras.Model,
        base_model: keras.Model,
        distillation_loss_fn: Callable[[TensorLike, TensorLike], keras.KerasTensor] | None = None,
        l2_weight_decay: float = 0.1,
        extra_metrics: Mapping[str, Callable[[TensorLike, TensorLike], keras.KerasTensor]]
        | None = None,
        **kwargs,
    ):
        """Initializes the trainer.

        Args:
            litert_model: The student model (typically the quantized model from litert_tunner).
            base_model: The teacher model (the original unquantized model).
            distillation_loss_fn: A callable to compute the loss between the student
                and teacher outputs. Defaults to Mean Squared Error (MSE).
            l2_weight_decay: Strength of the L2 weight drift penalty.
            extra_metrics: Dictionary mapping metric names to callables computing
                additional metrics between student and teacher outputs.
            **kwargs: Additional arguments passed to `keras.Model`.
        """
        super().__init__(**kwargs)
        self.litert_model = litert_model
        self.base_model = base_model
        self.base_model.trainable = False

        self.distillation_loss_fn = distillation_loss_fn or self._default_mse_loss
        self.l2_weight_decay = l2_weight_decay
        self.extra_metrics_fns = extra_metrics or {}

        # Store original trainable variables for L2 weight drift loss
        self.original_variables = {
            v.path: ops.convert_to_tensor(v.numpy()) for v in self.litert_model.trainable_weights
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

    def _default_mse_loss(
        self, student_outputs: TensorLike, teacher_outputs: TensorLike
    ) -> keras.KerasTensor:
        return ops.mean(ops.square(ops.subtract(student_outputs, teacher_outputs)))  # pyright: ignore[reportReturnType]

    def call(self, inputs: DataStruct, training: bool = False) -> DataStruct:  # noqa: FBT001, FBT002
        """Forward pass of the student model."""
        return self.litert_model(inputs, training=training)

    def compute_loss(
        self,
        x: DataStruct | None = None,
        y: DataStruct | None = None,
        y_pred: TensorLike | None = None,
        sample_weight: DataStruct | None = None,
    ) -> TensorLike:
        """Computes the total loss including distillation and L2 weight drift."""
        teacher_outputs = ops.stop_gradient(self.base_model(x, training=False))

        # Compute distillation loss
        distill_loss = self.distillation_loss_fn(y_pred, teacher_outputs)  # pyright: ignore[reportArgumentType]

        # Compute L2 weight drift loss
        l2_loss = 0.0
        if self.l2_weight_decay > 0.0:
            for v in self.litert_model.trainable_weights:
                if v.path in self.original_variables:
                    original_v = self.original_variables[v.path]
                    l2_loss += self.l2_weight_decay * ops.mean(
                        ops.square(ops.subtract(v, original_v))
                    )

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
                total_loss += compiled_loss
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
        # Compute custom extra metrics
        if self.extra_metrics_fns:
            teacher_outputs = ops.stop_gradient(self.base_model(x, training=False))
            for name, metric_fn in self.extra_metrics_fns.items():
                val = metric_fn(y_pred, teacher_outputs)  # pyright: ignore[reportArgumentType]
                self.extra_metric_trackers[name].update_state(val)

        # Let the base class update standard compiled metrics (like the total loss)
        return super().compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)  # pyright: ignore[reportReturnType]
