import csv
import pickle
from io import TextIOWrapper

import numpy as np

from nml import Device, Tensor, copy_to, save_weights
from project.generation_handler import GenerationHandler


def normalized_entropy(labels: np.ndarray) -> float:
    labels = np.argmax(labels, axis=1)
    class_counts = np.bincount(labels)
    probabilities = class_counts[class_counts > 0] / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(len(np.unique(labels)))  # log2(K)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def imbalance_ratio(labels: np.ndarray) -> float:
    labels = np.argmax(labels, axis=1)
    class_counts = np.bincount(labels)
    max_count = np.max(class_counts)
    min_count = np.min(class_counts[class_counts > 0])
    return max_count / min_count if min_count > 0 else 0.0


class VerboseHandler(GenerationHandler):
    """
    A class that handles the generation callback of a genetic algorithm.
    It saves the best genome and logs the generation information.

    Parameters:
        save_path: The path to save the best genome.
        save_period: The period (in generations) to save the genome.
        log_file: The file to log generation information.
        log_period: The period (in generations) to log information.
        profile_file: The file to log profiling information.
        profile_period: The period (in generations) to log profiling information.
    """

    save_path: str
    save_period: int
    log_period: int
    profile_period: int

    def __init__(
        self,
        save_path: str,
        save_period: int,
        log_file: TextIOWrapper,
        log_period: int,
        profile_file: TextIOWrapper,
        profile_period: int,
    ):
        self.save_path = save_path
        self.save_period = save_period
        self.log_file = csv.writer(log_file)
        self.log_file.writerow(
            [
                "generation",
                "fitness_min",
                "fitness_mean",
                "fitness_max",
                "fitness_std",
                "labels_entropy",
                "labels_imbalance",
            ]
        )
        self.log_period = log_period
        self.profile_file = csv.writer(profile_file)
        self.profile_file.writerow(
            [
                "generation",
                "fitness_evaluation_time",
                "genome_generation",
                "log",
                "total",
            ]
        )
        self.profile_period = profile_period

    def on_generation(
        self,
        population: list[tuple[dict[str, Tensor]], float],
        labels: Tensor,
        generation: int,
        is_last: bool,
    ) -> bool:
        if generation % self.log_period == 0 or is_last:
            labels = copy_to(labels, Device.CPU).array
            results = np.array([x[1] for x in population])

            self.log_file.writerow(
                [
                    generation,
                    np.min(results),
                    np.mean(results),
                    np.max(results),
                    np.std(results),
                    normalized_entropy(labels),
                    imbalance_ratio(labels),
                ]
            )

        if generation % self.save_period == 0 or is_last:
            with open(self.save_path.format(generation=generation), "wb") as file:
                pickle.dump([(save_weights(x), y) for x, y in population], file)

        return False

    def on_profile(self, profile: dict[str, float], generation: int):
        if generation % self.profile_period == 0:
            fitness_evaluation_time = profile["fitness"] - profile["start"]
            genome_generation_time = profile["pipeline"] - profile["fitness"]
            log_time = profile["start"] - profile["last_generation"]
            total_time = profile["pipeline"] - profile["last_generation"]

            self.profile_file.writerow(
                [
                    generation,
                    fitness_evaluation_time,
                    genome_generation_time,
                    log_time,
                    total_time,
                ]
            )
