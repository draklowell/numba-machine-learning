from enum import Enum, auto

import numpy as np


class Metric(Enum):
    """Enum for fitness metrics"""

    ACCURACY = auto()
    CROSS_ENTROPY = auto()
    MEAN_PROBABILITY = auto()
    COMBINED = auto()


class FitnessEvaluator:
    """
    Class for evaluating fitness of models.
    """

    def __init__(self, data_manager, metric: Metric = Metric.ACCURACY, **kwargs):
        """
        Initialize the fitness evaluator.

        Args:
            data_manager: Data manager to get samples from
            metric: Metric to use for fitness evaluation (from Metric enum)
            **kwargs: Additional arguments for specific metrics (e.g., weights for combined)
        """
        self.data_manager = data_manager
        self.metric = metric
        self.kwargs = kwargs

        self.metric_functions = {
            Metric.ACCURACY: self._accuracy,
            Metric.CROSS_ENTROPY: self._cross_entropy_loss,
            Metric.MEAN_PROBABILITY: self._mean_correct_probability,
            Metric.COMBINED: self._combined_metric,
        }

        self.num_classes = kwargs.get("num_classes", 10)

    def _ensure_2d_array(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions)

        # If input is 1D, reshape it to 2D assuming it's a single sample
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
        # Add small epsilon to avoid log(0)
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

        acc = self._accuracy(predictions, labels)
        mean_prob = self._mean_correct_probability(predictions, labels)

        return weight_accuracy * acc + weight_prob * mean_prob
