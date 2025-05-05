import pickle
from io import TextIOWrapper

from nml import Tensor


class GenerationHandler:
    """
    A class that handles the generation callback of a genetic algorithm.
    It saves the best genome and logs the generation information.

    Parameters:
        save_path: The path to save the best genome.
        save_period: The period (in generations) to save the genome.
        log_file: The file to log generation information.
        log_period: The period (in generations) to log information.
    """

    save_path: str
    save_period: int
    log_file: TextIOWrapper
    log_period: int

    def __init__(
        self, save_path: str, save_period: int, log_file: TextIOWrapper, log_period: int
    ):
        self.save_path = save_path
        self.save_period = save_period
        self.log_file = log_file
        self.log_period = log_period

    def __call__(
        self,
        population: list[tuple[dict[str, Tensor]], float],
        generation: int,
        is_last: bool,
    ) -> bool:
        """
        Called at the end of each fitness evaluation.

        Parameters:
            population: The population of genomes and their fitness scores.
            generation: The current generation number.
            is_last: Whether this is the last generation.

        Returns:
            True if the genetic algorithm should be stopped, False otherwise.
        """
        if generation % self.log_period == 0 or is_last:
            population = sorted(population, key=lambda x: x[1], reverse=True)
            self.log_file.write(
                f"Generation {generation}: {population[0][1]:.4f}/{population[-1][1]:.4f}\n"
            )

        if generation % self.save_period == 0 or is_last:
            # Extract the best genome
            best = max(population, key=lambda x: x[1])

            self.log_file.write(f"Saving generation {generation} ({best[1]:.4f})...\n")

            with open(self.save_path.format(generation=generation), "wb") as file:
                pickle.dump(best[0], file)

            self.log_file.write(f"Generation {generation} saved.\n")

        return False
