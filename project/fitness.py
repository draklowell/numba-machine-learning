from enum import Enum, auto

import numpy as np

from nml import Tensor


class FitnessMetric(Enum):
    """Enum for fitness metrics"""

    ACCURACY = auto()
    CROSS_ENTROPY = auto()
    MEAN_PROBABILITY = auto()
    COMBINED = auto()
    BALANCED_ACCURACY = auto()


class FitnessEvaluator:
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

    def __init__(self, metric: FitnessMetric = FitnessMetric.ACCURACY, **kwargs):
        """
        Initialize the fitness evaluator.

        Args:
            metric: Metric to use for fitness evaluation (from FitnessMetric enum)
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
            FitnessMetric.ACCURACY: self.accuracy,
            FitnessMetric.CROSS_ENTROPY: self.cross_entropy_loss,
            FitnessMetric.MEAN_PROBABILITY: self.mean_correct_probability,
            FitnessMetric.COMBINED: self.combined_metric,
            FitnessMetric.BALANCED_ACCURACY: self.balanced_accuracy,
        }

        self.num_classes = kwargs.get("num_classes", 10)

    def __call__(self, predictions: Tensor, labels_expected: Tensor) -> float:
        """
        Evaluate fitness using the selected metric by comparing prediction vectors
        with expected label vectors.
        """
        pred_array = predictions.array
        labels_array = labels_expected.array

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

    def accuracy(self, predictions: np.ndarray, one_hot_labels: np.ndarray) -> float:
        """Calculate accuracy with one-hot encoded labels"""
        predictions = self._ensure_2d_array(predictions)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(one_hot_labels, axis=1)
        correct = np.sum(predicted_classes == true_classes)
        return float(correct) / len(true_classes)

    def balanced_accuracy(
        self, predictions: np.ndarray, one_hot_labels: np.ndarray
    ) -> float:
        """Calculate balanced accuracy with one-hot encoded labels"""
        predictions = self._ensure_2d_array(predictions)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(one_hot_labels, axis=1)

        # Calculate the accuracy for each class
        accuracies = []
        for i in range(self.num_classes):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                accuracies.append(np.mean(predicted_classes[class_mask] == i))

        return np.mean(accuracies) if accuracies else 0.0

    def cross_entropy_loss(
        self, predictions: np.ndarray, one_hot_labels: np.ndarray
    ) -> float:
        """Calculate cross entropy loss with one-hot encoded labels"""
        predictions = self._ensure_2d_array(predictions)
        batch_size = predictions.shape[0]
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.sum(one_hot_labels * np.log(predictions)) / batch_size

        return -loss

    def mean_correct_probability(
        self, predictions: np.ndarray, one_hot_labels: np.ndarray
    ) -> float:
        """Calculate mean probability of correct class with one-hot encoded labels"""
        return np.sum(predictions * one_hot_labels) / len(predictions)

    def combined_metric(
        self, predictions: np.ndarray, one_hot_labels: np.ndarray
    ) -> float:
        weight_accuracy = self.kwargs.get("weight_accuracy", 0.7)
        weight_prob = self.kwargs.get("weight_prob", 0.3)

        acc = self.accuracy(predictions, one_hot_labels)
        mean_prob = self.mean_correct_probability(predictions, one_hot_labels)

        return weight_accuracy * acc + weight_prob * mean_prob
