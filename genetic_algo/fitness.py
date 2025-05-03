from enum import Enum, auto

import numpy as np
from nml.tensor import Tensor


class Metric(Enum):
    """Enum for fitness metrics"""

    ACCURACY = auto()
    CROSS_ENTROPY = auto()
    MEAN_PROBABILITY = auto()
    COMBINED = auto()


class FitnessEvaluator():
    """
    Class for evaluating fitness of models using specific metrics.

    Examples:
        # Basic usage with accuracy metric
        evaluator = FitnessEvaluator(metric=Metric.ACCURACY)
        fitness_score = evaluator(model_predictions, expected_labels)

        # Using combined metric with custom weights
        evaluator = FitnessEvaluator(
            metric=Metric.COMBINED,
            weight_accuracy=0.6,  # 60% weight for accuracy
            weight_prob=0.4       # 40% weight for probability
        )
        fitness_score = evaluator(model_predictions, expected_labels)
    """

    def __init__(self, metric: Metric = Metric.ACCURACY, **kwargs):
        """
        Initialize the fitness evaluator.

        Args:
            metric: Metric to use for fitness evaluation (from Metric enum)
            **kwargs: Additional arguments for specific metrics, such as:
                - For COMBINED metric:
                    - weight_accuracy: Weight for accuracy component (default: 0.7)
                    - weight_prob: Weight for probability component (default: 0.3)
                - For all metrics:
                    - num_classes: Number of classes in the dataset (default: 10)
        """
        self.metric = metric
        self.kwargs = kwargs

        self.metric_functions = {
            Metric.ACCURACY: self.accuracy,
            Metric.CROSS_ENTROPY: self.cross_entropy_loss,
            Metric.MEAN_PROBABILITY: self.mean_correct_probability,
            Metric.COMBINED: self.combined_metric,
        }

        self.num_classes = kwargs.get("num_classes", 10)

    def __call__(self, predictions: Tensor, labels_expected: Tensor) -> float:
        """
        Evaluate fitness using the selected metric
        """
        pred_array = predictions.array
        labels_array = np.argmax(labels_expected.array, axis=1)

        return self.metric_functions[self.metric](pred_array, labels_array)

    def _ensure_2d_array(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions)

        if predictions.ndim == 1:
            if len(predictions) == self.num_classes:
                return predictions.reshape(1, -1)
            else:
                batch_size = len(predictions)
                result = np.zeros((batch_size, self.num_classes))
                result[np.arange(batch_size), predictions.astype(int)] = 1
                return result

        return predictions

    def accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        predictions = self._ensure_2d_array(predictions)
        predicted_labels = np.argmax(predictions, axis=1)
        correct = np.sum(predicted_labels == labels)
        return float(correct) / len(labels)

    def cross_entropy_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        predictions = self._ensure_2d_array(predictions)
        batch_size = predictions.shape[0]
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        one_hot_labels = np.zeros_like(predictions)
        one_hot_labels[np.arange(batch_size), labels] = 1
        loss = -np.sum(one_hot_labels * np.log(predictions)) / batch_size

        return -loss

    def mean_correct_probability(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        return np.mean(predictions[np.arange(len(labels)), labels])

    def combined_metric(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        weight_accuracy = self.kwargs.get("weight_accuracy", 0.7)
        weight_prob = self.kwargs.get("weight_prob", 0.3)

        acc = self.accuracy(predictions, labels)
        mean_prob = self.mean_correct_probability(predictions, labels)

        return weight_accuracy * acc + weight_prob * mean_prob
