from nml import Tensor
from project.generation_handler import GenerationHandler


class PrintHandler(GenerationHandler):
    """
    A class that handles the generation callback of a genetic algorithm.
    It prints generation information and profiling information.

    Parameters:
        period: The period (in generations) to log information.
    """

    period: int

    def __init__(self, period: int):
        self.period = period

    def on_generation(
        self,
        population: list[tuple[dict[str, Tensor]], float],
        labels: Tensor,
        generation: int,
        is_last: bool,
    ) -> bool:
        if generation % self.period != 0 and not is_last:
            return False

        population = sorted(population, key=lambda x: x[1], reverse=True)
        print(
            f"Generation {generation}: {population[0][1]:.4f}/{population[-1][1]:.4f} fitness"
        )
        return False

    def on_profile(self, profile: dict[str, float], generation: int):
        if generation % self.period != 0:
            return

        fitness_evaluation_time = profile["fitness"] - profile["start"]
        genome_generation_time = profile["pipeline"] - profile["fitness"]
        log_time = profile["start"] - profile["last_generation"]
        total_time = profile["pipeline"] - profile["last_generation"]

        print(
            f"Generation {generation} profile:\n"
            f"  Fitness evaluation: {fitness_evaluation_time:.4f}s\n"
            f"  Genome generation: {genome_generation_time:.4f}s\n"
            f"  Log time: {log_time:.4f}s\n"
            f"  Total time: {total_time:.4f}s"
        )
